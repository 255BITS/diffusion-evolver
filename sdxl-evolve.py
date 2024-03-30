import anthropic
import argparse
import asyncio
import base64
import evolve
import logging
import time
import os
import random
import torch

from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from pathlib import Path
from io import BytesIO
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evolutionary merge")
    parser.add_argument('model_list', type=str, help='yml file containing list of models in the initial population')
    parser.add_argument('-seed', dest='seed', type=int, default=None, help='Random seed')
    parser.add_argument('-cycles', dest='cycles', type=int, default=10, help='Number of evolutionary cycles')
    parser.add_argument('-elite', dest='elite', type=int, default=10, help='Number of elite candidates to keep every iteration')
    parser.add_argument('-parents', dest='parents', type=int, default=2, help='Number of parents for each child')
    parser.add_argument('-population', dest='population', type=int, default=50, help='Size of population')
    parser.add_argument('-mutation', dest='mutation', type=float, default=0.05, help='Chance of mutation')
    parser.add_argument('-output_path', dest='output_path', type=str, default="evolve_output", help='Directory to save results')
    parser.add_argument('-prompt', dest='prompt', type=str, default='Which candidate generated more engaging images?', help='A prompt to inject into the decision making for claude. A question about which candidate did better.')
    parser.add_argument('-eval_file', dest='eval_file', type=str, default='evals.txt', help='A txt file containing a newline delimited list of prompts to evaluation against')
    parser.add_argument('-eval_samples', dest='eval_samples', type=int, default=3, help='The number of samples to evaluate between candidates')
    parser.add_argument('-device', dest='device', type=str, default="cuda:0", help='The device to run on')
    return parser.parse_args()

global_cache = {}
global_device = None
global_prompt = None
def generate_images(file_path, evals):
    global global_cache
    if file_path in global_cache:
        return global_cache[file_path]
    images = []
    logging.info(f"Loading {file_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(file_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(global_device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    for i, evl in enumerate(evals):
        image = pipe(evl['prompt'], num_inference_steps=8, guidance_scale=1, generator=torch.manual_seed(evl['seed'])).images[0]
        image = image.resize((512, 512))
        image.save(f"output-evolve-{file_path.split('/')[-1]}-{i}.png")
        images.append(image)

    del pipe
    global_cache[file_path] = images
    return images

def generate_b64_images(*args):
    # Assuming generate_images() returns a list of PIL Image objects
    images = generate_images(*args)
    b64_images = []

    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        b64_images.append(img_str)

    return b64_images

def load_random_evals(file_path, count):
    evals = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    count = min(count, len(lines))
    selected_lines = random.sample(lines, count)

    for line in selected_lines:
        evals.append({"prompt": line.strip(), "seed": random.randint(0, 2**32)})

    return evals

global_evals = []

def get_evals():
    return global_evals

def set_evals(evals):
    global global_evals
    global global_cache
    global_evals = evals
    global_cache = {}

def vlm_judge(prompts, b64_images_a, b64_images_b):
    client = anthropic.Anthropic()
    media_type = "image/png"
    prompts_list = "\n".join(["{0}. {1}".format(i, prompt) for i, prompt in enumerate(prompts)])
    begin_text = f"""
Here are the prompts for the generations:
{prompts_list}

Each candidate will be given these prompts to generate images. First you will receive candidate 1 generations based on these prompts, then candidate 2.

Candidate 1 generations:
""".strip()
    end_text = """
{global_prompt} If candidate 1 did better, simply output '1'. If candidate 2 did better, output '2'. If any of the candidates images are broken then disqualify it.
This is automated so simply output 1 or 2 based on comparing the images you've seen.
""".strip()
    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": begin_text
                    },
                    *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    } for b64_image in b64_images_a],
                    {
                        "type": "text",
                        "text": "Candidate 2 generations:"
                    },
                    *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_image,
                        },
                    } for b64_image in b64_images_b],
                    {
                        "type": "text",
                        "text": end_text 
                    }
                ],
            }
        ]
    print(messages)

    model = "claude-3-haiku-20240307"
    message = client.messages.create(
        model=model,
        max_tokens=128,
        system="You are a critical AI judge for text-to-image diffusion models. You will be presented images from both models with the same prompt and seed. At the end you will give your judgement. You love high quality generations, prompt adherence, and creativity.",
        messages=messages,
    )
    text = message.content[0].text
    if text[0] == "1" or text[0] == "2":
        return int(text)
    else:
        logging.info("wtf  bad output", text)
        raise "error"
        #return 1

def vlm_judge_with_retry(*args, max_retries=3, initial_wait=1, max_wait=10):
    for attempt in range(max_retries):
        try:
            return vlm_judge(*args)
        except Exception as e:
            wait_time = min(max_wait, initial_wait * 2 ** attempt)
            wait_time += random.uniform(0, wait_time * 0.2)  # Adding random jitter
            if attempt < max_retries - 1:
                # Log the full stack trace before retrying
                logging.exception(f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # Log the full stack trace before raising the exception after all retries have failed
                logging.exception("All attempts failed. Raising exception.")
                raise

global_comparisons = 0
global_yays = 0
global_nays = 0

async def compare(a: evolve.Candidate, b:evolve.Candidate):
    global global_comparisons, global_yays, global_nays
    reverse = random.random() > 0.5
    b64_images_a = generate_b64_images(a.file_path, get_evals())
    b64_images_b = generate_b64_images(b.file_path, get_evals())
    prompts = [evl["prompt"] for evl in get_evals()]
    if reverse:
        judgement = vlm_judge_with_retry(prompts, b64_images_b, b64_images_a)
        judgement = (1 if judgement == 2 else 2)
    else:
        judgement = vlm_judge_with_retry(prompts, b64_images_a, b64_images_b)
    global_comparisons += 1

    if judgement == 1:
        global_yays += 1
    else:
        global_nays += 1
    logging.info(f"Number of comparisons Total: {global_comparisons} Yay: {global_yays} Nay: {global_nays}")

    if judgement == 1:
        return 1
    return -1

async def main():
    global global_device, global_prompt
    # Parse command-line arguments
    args = parse_arguments()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    global_device = args.device
    global_prompt = args.prompt
    os.makedirs(args.output_path, exist_ok=True)
    population = evolve.load_candidates(args.model_list)
    evolve.write_yaml(population, Path(args.output_path) / "initial.yaml")
    logging.info("Beginning evolution")
    async for i in tqdm(range(args.cycles), desc='Evolving'):
        set_evals(load_random_evals(args.eval_file, args.eval_samples))
        population = await evolve.run_evolution(population, args.elite, args.parents, args.population, args.mutation, args.output_path, compare)
        evolve.write_yaml(population, Path(args.output_path) / f"step-{i}.yaml")

    logging.info("Resulting population:")
    evolve.log_candidates(population)
if __name__ == "__main__":
    asyncio.run(main())
