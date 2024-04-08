from diffusers import StableDiffusionInpaintPipeline

import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from einops import rearrange
import os
import imageio


import os
os.environ['CURL_CA_BUNDLE'] = ''

sd_inpainting_path = "./checkpoints/StableDiffusionInpainting/stable-diffusion-inpainting"

POS_PROMPT = ""
NEG_PROMPT = ""


def prepare_outpainting_mask(type, video_length, height, width):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    mask = np.zeros((height, width), dtype=np.uint8)
    if type == "vertical":
        mask[:, : width // 4] = 1
        mask[:, width // 4 * 3:] = 1
    elif type == "horizontal":
        mask[:height // 4, :] = 1
        mask[height // 4 * 3:, :] = 1
    mask = mask[:, :, None].repeat(3, axis=-1) * 255
    mask = Image.fromarray(np.uint8(mask))
    mask = transform(mask).unsqueeze(0)
    mask = mask.repeat(video_length, 1, 1, 1)
    mask = mask * 2 - 1
    return mask



def infer_sd_inpatint(video, masks, prompt, generator, output_path="./", height=512, width=512):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_inpainting_path,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()
    print("load idm done")

    t2i_transform = torchvision.transforms.ToPILImage()
    video = (video + 1) / 2
    masks = (masks + 1) / 2
    inpaint_images= []
    inpaint_masks = []
    for i in range(len(video)):
        inpaint_images.append(t2i_transform(video[i]))
        inpaint_masks.append(t2i_transform(masks[i]))

    # Reduce memory (optional)
    torch.cuda.empty_cache()


    pos_prompt = prompt + POS_PROMPT
    neg_prompt = NEG_PROMPT
    frames = [pipe(
        prompt=pos_prompt, negative_prompt=neg_prompt, image=inpaint_images[i], mask_image=inpaint_masks[i], num_inference_steps=50, generator=generator
    ).images[0] for i in range(len(inpaint_images))]

    return frames, POS_PROMPT, NEG_PROMPT