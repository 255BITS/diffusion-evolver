import anthropic
import argparse
import asyncio
import base64
import evolve
import os
import random
import torch

from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from io import BytesIO
from tqdm.asyncio import tqdm

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
    parser.add_argument('-criteria', dest='criteria', type=str, default='Which image has the highest quality?', help='Prompt for vlm evaluation')
    parser.add_argument('-eval_file', dest='eval_file', type=str, default='evals.txt', help='A txt file containing a newline delimited list of prompts to evaluation against')
    parser.add_argument('-eval_samples', dest='eval_samples', type=int, default=3, help='The number of samples to evaluate between candidates')
    return parser.parse_args()

global_cache = {}
def generate_images(file_path, evals):
    global global_cache
    if file_path in global_cache:
        return global_cache[file_path]
    images = []
    print(f"Loading {file_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(file_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, local_files_only=True).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    #pipe.load_lora_weights(
    #    hf_hub_download(
    #        repo_id="jiaxiangc/res-adapter",
    #        subfolder="sdxl-i",
    #        filename="resolution_lora.safetensors",
    #    ),
    #    adapter_name="res_adapter",
    #)
    #pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])

    for i, evl in enumerate(evals):
        #image = pipe(evl['prompt'], num_inference_steps=8, width=512, height=512, guidance_scale=1, generator=torch.manual_seed(evl['seed'])).images[0]
        image = pipe(evl['prompt'], num_inference_steps=8, guidance_scale=1, generator=torch.manual_seed(evl['seed'])).images[0]
        image.resize((512, 512))
        image.save(f"output-evolve-{i}-{file_path.split('/')[-1]}.png")
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
Which candidate generated more engaging images? If candidate 1 did better, simply output '1'. If candidate 2 did better, output '2'.
This is automated so simply output 1 or 2.
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

    model = "claude-3-haiku-20240307"
    message = client.messages.create(
        model=model,
        max_tokens=128,
        system="You are a critical AI judge for text-to-image diffusion models. You will be presented images from both models with the same prompt and seed. At the end you will give your finest judgement. You love high quality generations.",
        messages=messages,
    )
    text = message.content[0].text
    if text[0] == "1" or text[0] == "2":
        return int(text)
    else:
        print("wtf  bad output", text)
        raise "error"
        #return 1

global_comparisons = 0
async def compare(a: evolve.Candidate, b:evolve.Candidate):
    global global_comparisons
    b64_images_a = generate_b64_images(a.file_path, get_evals())
    b64_images_b = generate_b64_images(b.file_path, get_evals())
    prompts = [evl["prompt"] for evl in get_evals()]
    judgement = vlm_judge(prompts, b64_images_a, b64_images_b)
    global_comparisons += 1
    print("Number of comparisons", global_comparisons)

    if judgement == 1:
        return 1
    return -1

async def main():
    # Parse command-line arguments
    args = parse_arguments()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    population = evolve.load_candidates(args.model_list)
    print("Beginning evolution")
    async for i in tqdm(range(args.cycles), desc='Evolving'):
        set_evals(load_random_evals(args.eval_file, args.eval_samples))
        population = await evolve.run_evolution(population, args.elite, args.parents, args.population, args.mutation, compare)

    print("Resulting population:")
    evolve.log_candidates(population)
if __name__ == "__main__":
    asyncio.run(main())
