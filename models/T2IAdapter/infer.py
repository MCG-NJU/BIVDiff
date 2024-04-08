from ..lib.diffusers_v23 import T2IAdapter, StableDiffusionAdapterPipeline
from ..lib.diffusers_v23 import DDIMScheduler as DDIMScheduler_t2i_adapter
from ..lib.diffusers_v23 import  AutoencoderKL as AutoencoderKL_t2i_adapter
from ..lib.diffusers_v23 import UNet2DConditionModel as UNet2DConditionModel_t2i_adapter

from transformers import CLIPTextModel, CLIPTokenizer
from controlnet_aux.processor import Processor

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

sd_path = "./checkpoints/stable-diffusion-v1-5"
adapter_dict = {
    "depth_midas": "./checkpoints/T2IAdapter/t2iadapter_depth_sd15v2",
    "canny": "./checkpoints/T2IAdapter/t2iadapter_canny_sd15v2",
}

POS_PROMPT = ""
NEG_PROMPT = ""




def infer_t2iadapter(video, condition, prompt, generator, output_path="./", height=512, width=512):
    adapter_path = adapter_dict[condition]
    t2i_adapter = T2IAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16)
    t2i_tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    t2i_text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    t2i_vae = AutoencoderKL_t2i_adapter.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    t2i_unet = UNet2DConditionModel_t2i_adapter.from_pretrained(sd_path, subfolder="unet",
                                                                sample_size=512 // 8).to(dtype=torch.float16)
    t2i_scheduler = DDIMScheduler_t2i_adapter.from_pretrained(sd_path, subfolder="scheduler")

    pipe = StableDiffusionAdapterPipeline(
        vae=t2i_vae,
        text_encoder=t2i_text_encoder,
        tokenizer=t2i_tokenizer,
        unet=t2i_unet,
        adapter=t2i_adapter,
        scheduler=t2i_scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("load idm done")

    processor = Processor(condition) # output 3 channels
    t2i_transform = torchvision.transforms.ToPILImage()
    condition_images = []
    for frame in video:
        # frame = (frame + 1) / 2  # from [-1, 1] to [0, 1]
        frame = t2i_transform(frame)
        if condition == "canny": # T2IAdapter requires 1 channel for canny edge maps
            frame = processor(frame, to_pil=True)
            frame = np.array(frame)[:, :, 0]
            condition_images.append(Image.fromarray(frame))
        else:
            condition_images.append(processor(frame, to_pil=True)) # default 3 channels

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

# 使用其他版本的diffusers，不要pipeline.from_pretrained(path, torch_dtype=torch.float16)
# 手动load各个module，然后给pipeline初始化