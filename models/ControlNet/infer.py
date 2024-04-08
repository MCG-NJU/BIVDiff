from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux.processor import Processor

import numpy as np
import torch
import torchvision
from einops import rearrange
import os
import imageio


import os
os.environ['CURL_CA_BUNDLE'] = ''

sd_path = "./checkpoints/stable-diffusion-v1-5"
controlnet_dict = {
    "depth_midas": "./checkpoints/ControlNet/control_v11f1p_sd15_depth",
    "canny": "./checkpoints/ControlNet/control_v11p_sd15_canny",
    "openpose": "./checkpoints/ControlNet/control_v11p_sd15_openpose",
}

POS_PROMPT = " ,best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth"
NEG_PROMPT = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"




def infer_controlnet(video, condition, prompt, generator, output_path="./", height=512, width=512):
    controlnet_path = controlnet_dict[condition]
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_path, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    print("load idm done")

    processor = Processor(condition)
    t2i_transform = torchvision.transforms.ToPILImage()
    condition_images = []
    for frame in video:
        frame = (frame + 1) / 2  # from [-1, 1] to [0, 1]
        frame = t2i_transform(frame)
        condition_images.append(processor(frame, to_pil=True))

    video_cond = [np.array(cond).astype(np.uint8) for cond in condition_images]
    imageio.mimsave(os.path.join(output_path, f"{condition}_condition.gif"), video_cond, fps=8)

    # Reduce memory (optional)
    del processor;
    torch.cuda.empty_cache()


    pos_prompt = prompt + POS_PROMPT
    neg_prompt = NEG_PROMPT
    frames = [pipe(
        prompt=pos_prompt, negative_prompt=neg_prompt, num_inference_steps=50, generator=generator, image=image
    ).images[0] for image in condition_images]

    return frames, POS_PROMPT, NEG_PROMPT