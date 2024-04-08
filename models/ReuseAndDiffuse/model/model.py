import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

from ..vae import TemporalAutoencoderKL
from .modules.unet import UNet3DConditionModel
from .pipeline import SDVideoPipeline
from .utils import save_videos_grid, compute_clip_score

logger = logging.getLogger(__name__)


class SDVideoModel(pl.LightningModule):
    def __init__(self, pretrained_model_path, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model_path"], logger=False)
        # main training module
        self.unet: Union[str, UNet3DConditionModel] = Path(
            pretrained_model_path, "unet"
        ).as_posix()
        # components for training
        self.noise_scheduler_dir = Path(pretrained_model_path, "scheduler").as_posix()
        self.vae = Path(pretrained_model_path, "vae").as_posix()
        self.text_encoder = Path(pretrained_model_path, "text_encoder").as_posix()
        self.tokenizer: Union[str, CLIPTokenizer] = Path(
            pretrained_model_path, "tokenizer"
        ).as_posix()
        # clip model for metric
        self.clip = Path(pretrained_model_path, "clip").as_posix()
        self.clip_processor = Path(pretrained_model_path, "clip").as_posix()
        # define pipeline for inference
        self.val_pipeline = None
        # video frame resolution
        self.resolution = kwargs.get("resolution", 512)
        # use temporal_vae
        self.temporal_vae_path = kwargs.get("temporal_vae_path", None)

    def setup(self, stage: str) -> None:
        # build modules
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.noise_scheduler_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer)

        if self.temporal_vae_path:
            self.vae = TemporalAutoencoderKL.from_pretrained(self.temporal_vae_path)
        else:
            self.vae = AutoencoderKL.from_pretrained(self.vae)
        self.text_encoder = CLIPTextModel.from_pretrained(self.text_encoder)
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            self.unet,
            sample_size=self.resolution
            // (2 ** (len(self.vae.config.block_out_channels) - 1)),
            add_temp_transformer=self.hparams.get("add_temp_transformer", False),
            add_temp_attn_only_on_upblocks=self.hparams.get(
                "add_temp_attn_only_on_upblocks", False
            ),
            prepend_first_frame=self.hparams.get("prepend_first_frame", False),
            add_temp_embed=self.hparams.get("add_temp_embed", False),
            add_temp_conv=self.hparams.get("add_temp_conv", False),
        )

        # load previously trained components for resumed training
        ckpt_path = self.hparams.get("ckpt_path", None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
            mod_list = (
                ["unet", "text_encoder"]
                if self.temporal_vae_path
                else ["unet", "text_encoder", "vae"]
            )
            for mod in mod_list:
                if any(filter(lambda x: x.startswith(mod), state_dict.keys())):
                    mod_instance = getattr(self, mod)
                    mod_instance.load_state_dict(
                        {
                            k[len(mod) + 1 :]: v
                            for k, v in state_dict.items()
                            if k.startswith(mod)
                        }
                    )

        # null text for classifier-free guidance
        self.null_text_token_ids = self.tokenizer(  # noqa
            "",
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        # load clip modules for evaluation
        self.clip = CLIPModel.from_pretrained(self.clip)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_processor)
        # prepare modules
        for component in [self.vae, self.text_encoder, self.clip]:
            if not isinstance(component, CLIPTextModel) or self.hparams.get(
                "freeze_text_encoder", False
            ):
                component.requires_grad_(False).eval()
            if stage != "test" and self.trainer.precision.startswith("16"):
                component.to(dtype=torch.float16)
        # use gradient checkpointing
        if self.hparams.get("enable_gradient_checkpointing", True):
            if not self.hparams.get("freeze_text_encoder", False):
                self.text_encoder.gradient_checkpointing_enable()
            self.unet.enable_gradient_checkpointing()

        # construct pipeline for inference
        self.val_pipeline = SDVideoPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DDIMScheduler.from_pretrained(self.noise_scheduler_dir),
        )


class SDVideoModelEvaluator:
    def __init__(self, **kwargs):
        torch.multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")

        self.seed = kwargs.pop("seed", 42)
        self.prompts = kwargs.pop("prompts", None)
        if self.prompts is None:
            raise ValueError(f"No prompts provided.")
        elif isinstance(self.prompts, str) and not Path(self.prompts).exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompts}")
        elif isinstance(self.prompts, str):
            if self.prompts.endswith(".txt"):
                with open(self.prompts, "r", encoding="utf-8") as f:
                    self.prompts = [x.strip() for x in f.readlines() if x.strip()]
            elif self.prompts.endswith(".json"):
                with open(self.prompts, "r", encoding="utf-8") as f:
                    self.prompts = sorted(
                        [
                            random.choice(x) if isinstance(x, list) else x
                            for x in json.load(f).values()
                        ]
                    )
        self.add_file_logger(logger, kwargs.pop("log_file", None))
        self.output_file = kwargs.pop("output_file", "results.csv")
        self.batch_size = kwargs.pop("batch_size", 4)
        self.val_params = kwargs

    @staticmethod
    def add_file_logger(logger, log_file=None, log_level=logging.INFO):
        if log_file is not None:
            log_handler = logging.FileHandler(log_file, "w")
            log_handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
            )
            log_handler.setLevel(log_level)
            logger.addHandler(log_handler)

    @staticmethod
    def infer(rank, model, model_params, q_input, q_output, seed=42):
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + rank)
        output_video_dir = Path("output_videos")
        output_video_dir.mkdir(parents=True, exist_ok=True)
        while True:
            inputs = q_input.get()
            if inputs is None:  # check for sentinel value
                print(f"[{datetime.now()}] Process #{rank} ended.")
                break
            start_idx, prompts = inputs
            videos = model.val_pipeline(
                prompts,
                generator=generator,
                negative_prompt=["watermark"] * len(prompts),
                **model_params,
            ).videos
            for idx, prompt in enumerate(prompts):
                gif_file = output_video_dir.joinpath(f"{start_idx + idx}_{prompt}.gif")
                save_videos_grid(videos[idx : idx + 1, ...], gif_file)
                print(
                    f'[{datetime.now()}] Sample is saved #{start_idx + idx}: "{prompt}"'
                )
            clip_scores = compute_clip_score(
                model=model.clip,
                model_processor=model.clip_processor,
                images=videos,
                texts=prompts,
                rescale=False,
            )
            q_output.put((prompts, clip_scores.cpu().tolist()))
        return None

    def __call__(self, model):
        model.eval()

        if not torch.cuda.is_available():
            raise NotImplementedError(f"No GPU found.")

        self.val_params.setdefault(
            "num_inference_steps", model.hparams.get("num_inference_steps", 50)
        )
        self.val_params.setdefault(
            "guidance_scale", model.hparams.get("guidance_scale", 7.5)
        )
        self.val_params.setdefault("noise_alpha", model.hparams.get("noise_alpha", 0.0))
        logger.info(f"val_params: {self.val_params}")

        q_input = torch.multiprocessing.Queue()
        q_output = torch.multiprocessing.Queue()
        processes = []
        for rank in range(torch.cuda.device_count()):
            p = torch.multiprocessing.Process(
                target=self.infer,
                args=(rank, model, self.val_params, q_input, q_output, self.seed),
            )
            p.start()
            processes.append(p)
        # send model inputs to queue
        result_num = 0
        for start_idx in range(0, len(self.prompts), self.batch_size):
            result_num += 1
            q_input.put(
                (start_idx, self.prompts[start_idx : start_idx + self.batch_size])
            )
        for _ in processes:
            q_input.put(None)  # sentinel value to signal subprocesses to exit
        # The result queue has to be processed before joining the processes.
        results = [q_output.get() for _ in range(result_num)]
        # joining the processes
        for p in processes:
            p.join()  # wait for all subprocesses to finish
        all_prompts, all_clip_scores = [], []
        for prompts, clip_scores in results:
            all_prompts.extend(prompts)
            all_clip_scores.extend(clip_scores)
        output_df = pd.DataFrame({"prompt": all_prompts, "clip_score": all_clip_scores})
        output_df.to_csv(self.output_file, index=False)
        logger.info(f"--- Metrics ---")
        logger.info(f"Mean CLIP_SCORE: {sum(all_clip_scores) / len(all_clip_scores)}")
        logger.info(f"Test results saved in: {self.output_file}")
