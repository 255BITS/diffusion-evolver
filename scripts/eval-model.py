import asyncio
import base64
import io
import json
import random
import re
import requests
import sys
import time
import torch
import os

from PIL import Image, PngImagePlugin

NUM_IMAGES=128
random.seed(238)

def load_random_evals(file_path, count):
    evals = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    count = min(count, len(lines))
    selected_lines = random.sample(lines, count)

    for line in selected_lines:
        evals.append({"prompt": line.strip(), "seed": random.randint(0, 2**32)})

    return evals

def load_file_and_return_random_line(file_path):
    with open(file_path, 'r') as file:
         file_map = file.read().split('\n')

    line = random.choice(file_map)
    return line

def wildcard_replace(s, directory):
    if directory is None:
        return s
    wildcards = re.findall(r'__(.*?)__', s)
    replaced = [load_file_and_return_random_line(directory+"/"+w+".txt") for w in wildcards]
    replacements = dict(zip(wildcards, replaced))
    for wildcard, replacement in replacements.items():
        s = s.replace('__{}__'.format(wildcard), replacement)

    return s

def generate_image(prompt, negative_prompt, config_file=None, fname=None):
    seed = random.randint(0, 2**32-1)
    if config_file is None:
        with open("txt2img/sdwebui_config.json", 'r') as file:
            config_file = json.load(file)
    headers = {"Content-Type": "application/json"}
    data = dict(config_file)

    prompt = wildcard_replace(prompt, "wildcards")
    data["prompt"]=prompt
    data["negative_prompt"]=negative_prompt

    data["seed"]=seed
    url = data["sd_webui_url"]
    del data["sd_webui_url"]

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        r = response.json()
        image = Image.open(io.BytesIO(base64.b64decode(r['images'][0].split(",",1)[0])))
        jsoninfo = json.loads(r['info'])
        #print(jsoninfo["infotexts"][0])
        png_payload = {
            "image": "data:image/png;base64," + r['images'][0]
        }
        response2 = requests.post(url=url.replace("txt2img", "png-info"), json=png_payload)

        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", response2.json().get("info"))

        if fname is None:
            fname= random_fname()
        image.save(fname, pnginfo=pnginfo)
        return fname

    else:
        print(f"Request failed with status code {response.status_code}")
        return generate_image(prompt, negative_prompt, config_file, fname)

def run_eval(working_dir):
    for i in range(NUM_IMAGES):
        config = {
          "sd_webui_url": "http://localhost:3000/sdapi/v1/txt2img",
          "height": 1152,
          "width": 896,
          "sampler_name": "Euler",
          "scheduler": "SGM Uniform",
          "cfg_scale": 1,
          "steps": 8
        }
        prompt = f"__person__, __sdprompt__, __bg__"
        img = generate_image(prompt, "nsfw", config, working_dir+"/"+str(i)+".png")

async def main(model, working_dir="."):
    subdir = os.path.join(working_dir, "evals", model.split("/")[-1].split(".")[0], "imgs")
    os.makedirs(subdir, exist_ok=True)
    run_eval(subdir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        print("Usage: script_name.py <model>+")
        sys.exit(1)
    for model in models:
        asyncio.run(main(model))

