import logging
import hydra
from hydra.utils import instantiate
import imageio
import os
import torch
from einops import rearrange
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from models.scheduling_ddim import DDIMScheduler  # add reverse step
from models.StableDiffusion3D.unet import UNet3DConditionModel
from models.lib.diffusers_v23 import AutoencoderKL as AutoencoderKL_zeroscope
from models.lib.diffusers_v23.models.unet_3d_condition import UNet3DConditionModel as UNet3DConditionModel_zeroscope

from models.mixed_inversion import MixedInversionPipeline
from models.video_smoothing import VideoSmoothingPipeline
from models.util import save_videos_grid, read_video

from models.ReuseAndDiffuse.model import SDVideoModel
from models.ControlNet.infer import infer_controlnet
from models.T2IAdapter.infer import infer_t2iadapter
from models.InstructPix2Pix.infer import infer_instructpix2pix
from models.Prompt2Prompt.infer import infer_prompt2prompt
from models.StableDiffusionInpainting.infer import infer_sd_inpatint, prepare_outpainting_mask

device = "cuda"


def infer(video, generator, config, latents=None):
    model_name = config.Model.idm
    prompt = config.Model.idm_prompt
    output_path = config.Model.output_path
    height = config.Model.height
    width = config.Model.width
    if model_name == "ControlNet":
        condition = config.IDM.ControlNet.condition
        return infer_controlnet(video, condition, prompt, generator, output_path, height, width)
    elif model_name == "T2IAdapter":
        condition = config.IDM.T2IAdapter.condition
        return infer_t2iadapter(video, condition, prompt, generator, output_path, height, width)
    elif model_name == "InstructPix2Pix":
        return infer_instructpix2pix(video, prompt, generator, output_path, height, width)
    elif model_name == "Prompt2Prompt":
        controller_type = config.IDM.Prompt2Prompt.controller_type
        prompts = OmegaConf.to_object(config.IDM.Prompt2Prompt.prompts)
        local_blend_words = OmegaConf.to_object(config.IDM.Prompt2Prompt.local_blend_words)
        return infer_prompt2prompt(prompts, local_blend_words, controller_type, latents, generator, output_path, height,
                                   width)
    elif model_name == "StableDiffusionInpainting":
        masks = read_video(config.IDM.StableDiffusionInpainting.video_mask_path, video_length=config.Model.video_length,
                           width=config.Model.width,
                           height=config.Model.height, frame_rate=config.Model.frame_rate)
        original_pixels = rearrange(masks, "(b f) c h w -> b c f h w", b=1)
        save_videos_grid(original_pixels, os.path.join(output_path, "source_mask.gif"), rescale=True)

        return infer_sd_inpatint(video, masks, prompt, generator, output_path, height, width)
    elif model_name == "StableDiffusionOutpainting":
        masks = prepare_outpainting_mask(type=config.IDM.StableDiffusionOutpainting.mask_type,
                                         video_length=config.Model.video_length, width=config.Model.width,
                                         height=config.Model.height)

        video_masks = masks.clone()
        video_masks = (video_masks + 1) / 2  # [0, 1]
        video_masks[video_masks < 0.5] = 0
        video_masks[video_masks >= 0.5] = 1
        masked_video = (video + 1) / 2 * (video_masks < 0.5)
        masked_video = rearrange(masked_video, "(b f) c h w -> b c f h w", b=1)
        save_videos_grid(masked_video, os.path.join(output_path, "masked_video.gif"), rescale=False)

        return infer_sd_inpatint(video, masks, prompt, generator, output_path, height, width)


@hydra.main(config_path="configs", config_name="example", version_base=None)
def main(config):
    os.makedirs(config.Model.output_path, exist_ok=True)
    config.Model.height = (config.Model.height // 32) * 32
    config.Model.width = (config.Model.width // 32) * 32

    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.Model.seed)

    # 1. load models (note: image diffusion models are loaded in step 3. frame-wise video generation)
    # 1.1 video diffusion models for mixed inversion and video temporal smoothing
    video_unet = None
    video_text_encoder = None
    video_text_tokenizer = None
    if config.Model.vdm == "VidRD":
        vdm_model = instantiate(config.VDM.VidRD)
        vdm_model.setup(stage="test")
        video_unet = vdm_model.unet.to(dtype=torch.float16)
        video_text_tokenizer = vdm_model.tokenizer
        video_text_encoder = vdm_model.text_encoder.to(dtype=torch.float16)
    elif config.Model.vdm == "ZeroScope":
        zeroscope_path = config.VDM.ZeroScope.ckpt_path
        video_text_tokenizer = CLIPTokenizer.from_pretrained(zeroscope_path, subfolder="tokenizer")
        video_text_encoder = CLIPTextModel.from_pretrained(zeroscope_path, subfolder="text_encoder").to(
            dtype=torch.float16)
        video_unet = UNet3DConditionModel_zeroscope.from_pretrained(zeroscope_path, subfolder="unet",
                                                                    sample_size=config.Model.height // 8).to(
            dtype=torch.float16)
    print("load vdm done")

    # 1.2 image diffusion models for mixed inversion, video diffusion models are the same as Step 1.
    if config.MixedInversion.idm == "SD-v15":
        sd_path = config.MixedInversion.pretrained_model_path
        tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
        vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
        # extend stable diffusion from 2D to 3D for processing multiple frames in parallel
        unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet",
                                                       sample_size=config.Model.height // 8).to(
            dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    print("load mixed inversion models done")

    # 2. construct pipelines
    mixed_inversion_pipeline = MixedInversionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                                      unet=unet,
                                                      scheduler=scheduler,
                                                      video_unet=video_unet, video_text_tokenizer=video_text_tokenizer,
                                                      video_text_encoder=video_text_encoder, )
    video_smoothing_pipeline = VideoSmoothingPipeline(vae=vae, text_encoder=video_text_encoder,
                                                      tokenizer=video_text_tokenizer, unet=video_unet,
                                                      scheduler=scheduler, )
    mixed_inversion_pipeline.enable_vae_slicing()
    mixed_inversion_pipeline.enable_xformers_memory_efficient_attention()
    mixed_inversion_pipeline.to(device)

    video_smoothing_pipeline.enable_vae_slicing()
    video_smoothing_pipeline.enable_xformers_memory_efficient_attention()
    video_smoothing_pipeline.to(device)

    print("construct pipelines done")

    # 3. read video
    output_path = config.Model.output_path
    frames = read_video(config.Model.video_path, video_length=config.Model.video_length, width=config.Model.width,
                        height=config.Model.height, frame_rate=config.Model.frame_rate)
    original_pixels = rearrange(frames, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(output_path, "source_video.gif"), rescale=True)

    print("read and sample video done")

    # 4. frame-wise video generation
    '''
    Prompt2Prompt in the original paper generates two images (one is the original image, and the other is the edited image.)
    To use Prompt2Prompt for editing, we should invert the frames into initial noised latents by inversion
    '''
    latents = None
    if config.Model.idm == "Prompt2Prompt":
        with torch.no_grad():
            latents = vae.encode(frames.to(device=device, dtype=torch.float16)).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)
            latents = latents * 0.18215
        mixing_ratio = 1.0  # only use image model for inversion, 应该用prompt而不是""做ddim inversion的，不能直接调用mixed_inversion，但是实际结果看着还不错
        latents = mixed_inversion_pipeline(
            prompt=config.Model.idm_prompt,
            video_length=config.Model.video_length,
            num_inference_steps=config.Model.num_inference_steps,
            generator=generator,
            guidance_scale=config.Model.guidance_scale,
            width=config.Model.width, height=config.Model.height,
            latents=latents,
            mixing_ratio=mixing_ratio,
        )
        latents = torch.cat([latents] * 2)  # copy video latents twice for two prompts
        print("frames inversion done")

        original_video, frames_by_idm, POS_PROMPT, NEG_PROMPT = infer(frames, generator, config, latents)
        imageio.mimsave(os.path.join(output_path, "video_inverted_and_reconstructed.gif"), original_video, fps=8)
        imageio.mimsave(os.path.join(output_path, "video_by_idm.gif"), frames_by_idm, fps=8)
    else:
        frames_by_idm, POS_PROMPT, NEG_PROMPT = infer(frames, generator, config, latents)
        imageio.mimsave(os.path.join(output_path, "video_by_idm.gif"), frames_by_idm, fps=8)

    print("frame-wise video generation done")

    '''
    # For the case that your idm is difficult to integrate our framework
    # You can also use the result of your idm as the input of the following pipeline

    frames_by_idm = []
    from PIL import Image
    from PIL import ImageSequence
    gif = Image.open("./data/your_case.gif")
    i = 0
    for frame in ImageSequence.Iterator(gif):
        frame.save("frame%d.png" % i)
        frames_by_idm.append(Image.open("frame%d.png" % i).convert("RGB"))
        i += 1

    '''

    # 5. mixed inversion
    with torch.no_grad():
        video = torch.cat([mixed_inversion_pipeline.image_processor.preprocess(image=frame, width=config.Model.width,
                                                                               height=config.Model.height) for frame in
                           frames_by_idm])
        latents = vae.encode(video.to(device=device, dtype=torch.float16)).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=1)
        latents = latents * 0.18215
    mixing_ratio = config["IDM"][config.Model.idm]["mixing_ratio"]
    latents = mixed_inversion_pipeline(
        prompt=config.Model.vdm_prompt + POS_PROMPT,
        video_length=config.Model.video_length,
        num_inference_steps=config.Model.num_inference_steps,
        generator=generator,
        guidance_scale=config.Model.guidance_scale,
        negative_prompt=NEG_PROMPT,
        width=config.Model.width, height=config.Model.height,
        latents=latents,
        mixing_ratio=mixing_ratio,
    )

    print("mixed inversion done")

    # 6. video temporal smoothing
    video = video_smoothing_pipeline(
        prompt=config.Model.vdm_prompt + POS_PROMPT,
        video_length=config.Model.video_length,
        num_inference_steps=config.Model.num_inference_steps,
        generator=generator,
        guidance_scale=config.Model.guidance_scale,
        negative_prompt=NEG_PROMPT,
        width=config.Model.width, height=config.Model.height,
        latents=latents,
    ).videos
    print("video temporal smoothing done")

    save_videos_grid(video, f"{config.Model.output_path}/video_by_bivdiff.gif")


if __name__ == "__main__":
    main()

# For using diffusers of other versions, you should modify the "register_modules" and "load_sub_model" function in "pipeline_utils.py" to make the path available
'''
    if (__name__.split(".")[0] == 'models' and __name__.split(".")[1] == 'lib'):
        diffusers_module = importlib.import_module(__name__.split(".pipelines")[0])
    else:
        diffusers_module = importlib.import_module(__name__.split(".")[0])

'''