import cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from einops import rearrange
import os
import imageio


from .pipeline_prompt2prompt import Prompt2PromptPipeline



import os
os.environ['CURL_CA_BUNDLE'] = ''

POS_PROMPT = ""
NEG_PROMPT = ""

sd_path = "./checkpoints/stable-diffusion-v1-5"
def infer_prompt2prompt(prompts, local_blend_words, controller_type, latents, generator, output_path="./", height=512, width=512):
    # current Prompt2Prompt may not support fp 16
    pipe = Prompt2PromptPipeline.from_pretrained(sd_path, safety_checker=None, torch_dtype=torch.float)
    pipe.to(latents.device)
    print("load idm done")

    # Reduce memory (optional)
    torch.cuda.empty_cache()

    if controller_type == "AttentionReplace":
        cross_attention_kwargs = {
            "edit_type": "replace",
            "cross_replace_steps": 0.4,
            "self_replace_steps": 0.4
        }
        if isinstance(local_blend_words, list) and len(local_blend_words) > 0:
            cross_attention_kwargs["local_blend_words"] = local_blend_words
    elif controller_type == "AttentionRefine":
        cross_attention_kwargs = {
            "edit_type": "refine",
            "cross_replace_steps": 0.4,
            "self_replace_steps": 0.4
        }
        if isinstance(local_blend_words, list) and len(local_blend_words) > 0:
            cross_attention_kwargs["local_blend_words"] = local_blend_words

    latents = rearrange(latents, "b c f h w -> f b c h w")
    latents = latents.to(dtype=torch.float)

    outputs = [pipe(prompt=prompts, height=height, width=width, num_inference_steps=50,
                   cross_attention_kwargs=cross_attention_kwargs, latents=latent).images for latent in latents]
    orignial_results = [output[0] for output in outputs]
    edited_results = [output[1] for output in outputs]

    return orignial_results, edited_results, POS_PROMPT, NEG_PROMPT
