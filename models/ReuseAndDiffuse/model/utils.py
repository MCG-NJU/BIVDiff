import logging
from math import sqrt
from pathlib import Path
from typing import Union, Tuple, List, Optional

import imageio
import torch
import torchvision
from einops import rearrange

logger = logging.getLogger(__name__)


def randn_base(
    shape: Union[Tuple, List],
    mean: float = 0.0,
    std: float = 1.0,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
):
    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        tensor = [
            torch.normal(
                mean=mean,
                std=std,
                size=shape,
                generator=generator[i],
                device=device,
                dtype=dtype,
            )
            for i in range(len(generator))
        ]
        tensor = torch.cat(tensor, dim=0).to(device)
    else:
        tensor = torch.normal(
            mean=mean,
            std=std,
            size=shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    return tensor


def randn_mixed(
    shape: Union[Tuple, List],
    dim: int,
    alpha: float = 0.0,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
):
    """Refer to Section 4 of Preserve Your Own Correlation:
    [A Noise Prior for Video Diffusion Models](https://arxiv.org/abs/2305.10474)
    """
    shape_shared = shape[:dim] + (1,) + shape[dim + 1 :]

    # shared random tensor
    shared_std = alpha**2 / (1.0 + alpha**2)
    shared_tensor = randn_base(
        shape=shape_shared,
        mean=0.0,
        std=shared_std,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # individual random tensor
    indv_std = 1.0 / (1.0 + alpha**2)
    indv_tensor = randn_base(
        shape=shape,
        mean=0.0,
        std=indv_std,
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return shared_tensor + indv_tensor


def randn_progressive(
    shape: Union[Tuple, List],
    dim: int,
    alpha: float = 0.0,
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
):
    """Refer to Section 4 of Preserve Your Own Correlation:
    [A Noise Prior for Video Diffusion Models](https://arxiv.org/abs/2305.10474)
    """
    num_prog = shape[dim]
    shape_slice = shape[:dim] + (1,) + shape[dim + 1 :]
    tensors = [
        randn_base(
            shape=shape_slice,
            mean=0.0,
            std=1.0,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    ]
    beta = alpha / sqrt(1.0 + alpha**2)
    std = 1.0 / (1.0 + alpha**2)
    for i in range(1, num_prog):
        tensor_i = beta * tensors[-1] + randn_base(
            shape=shape_slice,
            mean=0.0,
            std=std,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        tensors.append(tensor_i)
    tensors = torch.cat(tensors, dim=dim)
    return tensors


def save_videos_grid(videos, path, rescale=False, n_rows=4, fps=8):
    if videos.dim() == 4:
        videos = videos.unsqueeze(0)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # [-1, 1) -> [0, 1)
        x = (x * 255).to(dtype=torch.uint8, device="cpu")
        outputs.append(x)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

@torch.no_grad()
def compute_clip_score(
    model, model_processor, images, texts, local_bs=32, rescale=False
):
    if rescale:
        images = (images + 1.0) / 2.0  # -1,1 -> 0,1
    images = (images * 255).to(torch.uint8)
    clip_scores = []
    for start_idx in range(0, images.shape[0], local_bs):
        img_batch = images[start_idx : start_idx + local_bs]
        batch_size = img_batch.shape[0]  # shape: [b c t h w]
        img_batch = rearrange(img_batch, "b c t h w -> (b t) c h w")
        outputs = []
        for i in range(len(img_batch)):
            images_part = img_batch[i : i + 1]
            model_inputs = model_processor(
                text=texts, images=list(images_part), return_tensors="pt", padding=True
            )
            model_inputs = {
                k: v.to(device=model.device, dtype=model.dtype)
                if k in ["pixel_values"]
                else v.to(device=model.device)
                for k, v in model_inputs.items()
            }
            logits = model(**model_inputs)["logits_per_image"]
            # For consistency with `torchmetrics.functional.multimodal.clip_score`.
            logits = logits / model.logit_scale.exp()
            outputs.append(logits)
        logits = torch.cat(outputs)
        logits = rearrange(logits, "(b t) p -> t b p", b=batch_size)
        frame_sims = []
        for logit in logits:
            frame_sims.append(logit.diagonal())
        frame_sims = torch.stack(frame_sims)  # [t, b]
        clip_scores.append(frame_sims.mean(dim=0))
    return torch.cat(clip_scores)
