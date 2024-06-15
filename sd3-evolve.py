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
import uuid

from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from PIL import Image
from dataclasses import dataclass
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
from io import BytesIO
from pathlib import Path
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evolutionary merge")
    parser.add_argument('model_list', type=str, help='yml file containing list of models in the initial population')
    parser.add_argument('-seed', dest='seed', type=int, default=None, help='Random seed')
    parser.add_argument('-cycles', dest='cycles', type=int, default=10, help='Number of evolutionary cycles')
    parser.add_argument('-elite', dest='elite', type=int, default=5, help='Number of elite candidates to keep every iteration')
    parser.add_argument('-parents', dest='parents', type=int, default=2, help='Number of parents for each child')
    parser.add_argument('-population', dest='population', type=int, default=20, help='Size of population')
    parser.add_argument('-mutation', dest='mutation', type=float, default=0.05, help='Chance of mutation')
    parser.add_argument('-output_path', dest='output_path', type=str, default="evolve_output", help='Directory to save results')
    parser.add_argument('-criteria', dest='criteria', type=str, default='Which candidate generated more colorful images?', help='Criteria for decision making in the VLM.')
    parser.add_argument('-eval_file', dest='eval_file', type=str, default='evals.txt', help='A txt file containing a newline delimited list of prompts to evaluation against')
    parser.add_argument('-eval_samples', dest='eval_samples', type=int, default=2, help='The number of samples to evaluate between candidates')
    parser.add_argument('-device', dest='device', type=str, default="cuda:0", help='The device to run on')
    parser.add_argument('-reintroduction_threshold', dest='reintroduction_threshold', type=float, default=0, help='The chance to reintroduce an initial model back into the elite population. Can help with solution diversity.')
    parser.add_argument('-vlm', dest='vlm', type=str, default="claude", help='The vlm to use. claude or llava')
    parser.add_argument('-append_prompt', dest='append_prompt', type=str, default="", help='Appends to the prompt')
    parser.add_argument('-negative_prompt', dest='negative_prompt', type=str, default="", help='Set the negative prompt')
    parser.add_argument('-guidance_scale', dest='guidance_scale', type=float, default=1, help='The guidance scale to use')
    parser.add_argument('-scheduler', dest='scheduler', type=str, default="sgm_uniform", help='The diffusion scheduler to use')
    parser.add_argument('-diffusion_steps', dest='diffusion_steps', type=int, default=8, help='The number of diffusion steps to run')
    parser.add_argument('-diffusion_prompt_change', dest='diffusion_prompt_change', type=str, choices=["every_cycle", "never"], default="cycle", help='The type of generation cache to use. Controls when vlm image prompts are changed. Choices: never, every_cycle')
    parser.add_argument("-width", dest='width', type=int, default=1024, help='Width of diffusion samples to generate')
    parser.add_argument("-height", dest='height', type=int, default=1024, help='Height of diffusion samples to generate')
    parser.add_argument("-resize_width", dest='resize_width', type=int, default=512, help='Width to resized diffusion samples before sending to the VLM')
    parser.add_argument("-resize_height", dest='resize_height', type=int, default=512, help='Height to resize diffusion samples before sending to the VLM')
    parser.add_argument("-vae", dest='vae', type=str, default=None, help='Custom VAE to use during sampling')
    parser.add_argument("-perturb_seed_population", dest='perturb_seed_population', type=int, default=0, help='Build this many children of the seed population')
    return parser.parse_args()

def generate_images(file_path, evals, device, cache, settings):
    if file_path in cache:
        return cache[file_path]
    images = []
    logging.info(f"Loading {file_path}")

    dtype = torch.bfloat16
    model_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    pipe = StableDiffusion3Pipeline.from_single_file(file_path, text_encoder = None, text_encoder_2 = None, text_encoder_3 = None, device=device)
    pipe.text_encoder = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder")
    pipe.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
    #pipe.text_encoder_3 = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder_3")
    pipe.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    pipe.tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    #pipe.tokenizer_3 = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3")
    pipe = pipe.to(dtype).to(device)

    for i, evl in enumerate(evals):
        image = pipe(evl['prompt']+settings.append_prompt, width=settings.width, height=settings.height, negative_prompt=settings.negative_prompt, num_inference_steps=settings.diffusion_steps, guidance_scale=settings.guidance_scale, generator=torch.manual_seed(evl['seed'])).images[0]
        if settings.resize_width != settings.width:
            image = image.resize((settings.resize_width, settings.resize_height))
        image.save(f"output-evolve-{file_path.split('/')[-1]}-{i}.png")
        images.append(image)

    del pipe
    cache[file_path] = images
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

def claude_vlm_judge(criteria, prompts, b64_images_a, b64_images_b):
    client = anthropic.Anthropic()
    media_type = "image/png"
    prompts_list = "\n".join(["{0}. {1}".format(i, prompt) for i, prompt in enumerate(prompts)])
#    begin_text = f"""
#Here are the prompts for the generations:
#```
#{prompts_list}
#```
#
#Each candidate will be given these prompts to generate images. First you will receive candidate 1 generations based on these prompts, then candidate 2.
#
    begin_text = f"""You will first see both candidates images then judge which did better based on the following criteria:
```
Criteria: {criteria}
```

Candidate 1 generations:
""".strip()
    end_text = """
Which candidate won based on the criteria? If candidate 1, output '1'. If candidate 2, output '2'. This is automated, the first 1 or 2 you output will be the winner.
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
        system="You are diffusion evolver AI, a judge for an image generation contest. You will be presented images from two models with the same prompt and seed. At the end you will give your judgement based on a specified criteria.",
        messages=messages,
    )
    text = message.content[0].text
    for i, ch in enumerate(text):
        if ch == "1" or ch == "2":
            return int(ch)
    logging.info("wtf  bad output", text)
    raise "error"

def claude_vlm_judge_with_retry(*args, max_retries=3, initial_wait=1, max_wait=10):
    for attempt in range(max_retries):
        try:
            return claude_vlm_judge(*args)
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

def combine_pil(a, b):
    total_width = a.width + b.width
    max_height = max(a.height, b.height)

    combined_image = Image.new('RGB', (total_width, max_height))

    combined_image.paste(a, (0, 0))
    combined_image.paste(b, (a.width, 0))

    return combined_image

def llava_vlm_decide(prompt, img, device):
    import llava_util
    model = "liuhaotian/llava-v1.6-mistral-7b"
    text = llava_util.run_llava(model, None, prompt, [img], device=device, max_new_tokens=128)
    for i, ch in enumerate(text):
        if ch == "1" or ch == "2":
            return int(ch)
    logging.info("wtf  bad output", text)
    raise "error"

def llava_vlm_judge(criteria, prompts, b64_images_a, b64_images_b, device):
    for (prompt, img_a, img_b) in zip(prompts, b64_images_a, b64_images_b):
        img_combine = combine_pil(img_a, img_b)
        prompt = f"You are a judge in an image generation contest. {criteria} '1' for the image on the left, '2' for the image on the right. Answer only '1'(left) or '2'(right). This is automated and the first number in your answer will be chosen."
        return llava_vlm_decide(prompt, img_combine, device)

def llava_vlm_judge_with_retry(*args, max_retries=3):
    for i in range(max_retries):
        try:
            return llava_vlm_judge(*args)
        except Exception as e:
            if i < max_retries:
                logging.exception("Llava did not give output. Retrying...")
            else:
                logging.exception("Llava failed!")
                raise

def compare(cache, criteria, device, evals, metrics, vlm, settings):
    async def vlm_compare(a: evolve.Candidate, b:evolve.Candidate):
        cache_key = 'compare:'+a.file_path+'.'+b.file_path
        if cache_key in cache:
            return cache[cache_key]
        reverse = random.random() > 0.5
        prompts = [evl["prompt"] for evl in evals]
        if reverse:
            a, b = b, a

        if vlm == 'claude':
            b64_images_a = generate_b64_images(a.file_path, evals, device, cache, settings)
            b64_images_b = generate_b64_images(b.file_path, evals, device, cache, settings)
            judgement = claude_vlm_judge_with_retry(criteria, prompts, b64_images_a, b64_images_b)
        elif vlm == 'llava':
            images_a = generate_images(a.file_path, evals, device, cache, settings)
            images_b = generate_images(b.file_path, evals, device, cache, settings)
            judgement = llava_vlm_judge_with_retry(criteria, prompts, images_a, images_b, device)
        else:
            raise "vlm not supported:" + vlm

        if reverse:
            judgement = (1 if judgement == 2 else 2)
        metrics.total += 1

        if judgement == 1:
            metrics.yays += 1
        else:
            metrics.nays += 1
        logging.info(f"Number of comparisons Total: {metrics.total} Yay: {metrics.yays} Nay: {metrics.nays}")


        if judgement == 1:
            cache[cache_key] = 1
            return 1
        cache[cache_key] = -1
        return -1
    return vlm_compare

@dataclass
class Metrics:
    total: int = 0
    yays: int = 0
    nays: int = 0

@dataclass
class DiffusionSettings:
    guidance_scale: int
    negative_prompt: str
    append_prompt: str
    diffusion_steps: int
    width: int
    height: int
    resize_width: int
    resize_height: int
    scheduler: str
    vae: str

async def main():
    # Parse command-line arguments
    args = parse_arguments()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)
    metrics = Metrics()
    cache = {}
    evals = load_random_evals(args.eval_file, args.eval_samples)
    settings = DiffusionSettings(
            append_prompt = args.append_prompt,
            diffusion_steps = args.diffusion_steps,
            guidance_scale = args.guidance_scale,
            height = args.height,
            negative_prompt = args.negative_prompt,
            resize_height = args.resize_height,
            resize_width = args.resize_width,
            width = args.width,
            vae = args.vae,
            scheduler = args.scheduler
    )
    initial_population = evolve.load_candidates(args.model_list)
    initial_populiation_count = len(initial_population)
    while(len(initial_population) < args.perturb_seed_population):
        parent = random.choice(initial_population[:initial_populiation_count])
        file_path = str(Path(args.output_path) / (str(uuid.uuid4())+".safetensors"))
        offspring = evolve.Candidate(file_path, parent.p, parent.lambda_val, initial_population=True)
        offspring.generation = parent.generation + 1
        print("perturbing from(clone)", parent.file_path)
        tensor_map = evolve.perturb(parent)
        print("saving", offspring.file_path)
        evolve.save_file(tensor_map, offspring.file_path)
        del tensor_map
        initial_population.append(offspring)
    print("--", initial_population)

    population = list(initial_population)
    evolve.write_yaml(population, Path(args.output_path) / "initial.yaml")
    logging.info("Beginning evolution")

    async for i in tqdm(range(args.cycles), desc='Evolving'):
        if args.diffusion_prompt_change == "every_cycle":
            evals = load_random_evals(args.eval_file, args.eval_samples)
            cache = {}
        comparator = compare(cache, args.criteria, args.device, evals, metrics, args.vlm, settings)
        population = await evolve.run_evolution(population, args.elite, args.parents, args.population, args.mutation, args.output_path, comparator)
        evolve.write_yaml(population, Path(args.output_path) / f"step-{i}.yaml")
        if random.random() < args.reintroduction_threshold:
            population.append(random.choice(initial_population))

    logging.info("Resulting population:")
    evolve.log_candidates(population)
if __name__ == "__main__":
    asyncio.run(main())
