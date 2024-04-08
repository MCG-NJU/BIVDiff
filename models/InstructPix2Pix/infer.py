from diffusers import StableDiffusionInstructPix2PixPipeline

import numpy as np
import torch
import torchvision
from einops import rearrange
import os
import imageio


import os
os.environ['CURL_CA_BUNDLE'] = ''

sd_path = "./checkpoints/stable-diffusion-v1-5"
instruct_pix2pix_path = "./checkpoints/InstructPix2Pix/instruct-pix2pix"

POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"




def infer_instructpix2pix(video, prompt, generator, output_path="./", height=512, width=512):
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        instruct_pix2pix_path, safety_checker=None, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    print("load idm done")

    t2i_transform = torchvision.transforms.ToPILImage()
    edit_images= []
    for frame in video:
        frame = (frame + 1) / 2  # from [-1, 1] to [0, 1]
        frame = t2i_transform(frame)
        edit_images.append(frame)

    # Reduce memory (optional)
    torch.cuda.empty_cache()


    pos_prompt = prompt + POS_PROMPT
    neg_prompt = NEG_PROMPT
    frames = [pipe(
        prompt=pos_prompt, negative_prompt=neg_prompt, num_inference_steps=50, generator=generator, image=image
    ).images[0] for image in edit_images]

    return frames, POS_PROMPT, NEG_PROMPT