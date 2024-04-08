# Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation

[Website](https://anonymous0x233.github.io/ReuseAndDiffuse) • [Paper](https://arxiv.org/abs/2309.03549) • [Code](https://github.com/anonymous0x233/ReuseAndDiffuse)

## Model preparation

1. **VidRD LDM model**: [GoogleDrive](https://drive.google.com/file/d/1rdT9cnMjjoggFBsu3LKFFJBl3b_gXa-N/view?usp=drive_link)
2. **VidRD Fine-tuned VAE**: [GoogleDrive](https://drive.google.com/file/d/1HfhpI4zy4kBmRSy0G600UDnDgJh6bAQp/view?usp=drive_link)
3. **StableDiffusion 2.1**: [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

Below is an example structure of these model files.

```
assets/
├── ModelT2V.pth
├── vae_finetuned/
│   ├── diffusion_pytorch_model.bin
│   └── config.json
└── stable-diffusion-2-1-base/
    ├── scheduler/...
    ├── text_encoder/...
    ├── tokenizer/...
    ├── unet/...
    ├── vae/...
    ├── ...
    └── README.md
```

## Environment setup

Python version needs to be >=3.10.

```bash
pip install -r requirements.txt
```

## Model inference

Configurations for model inferences are put in `configs/examples.yaml` including text prompts for video generation.

```bash
python main.py --config-name="example" \
  ++model.ckpt_path="assets/ModelT2V.pth" \
  ++model.temporal_vae_path="assets/vae_finetuned/" \
  ++model.pretrained_model_path="assets/stable-diffusion-2-1-base/"
```

## BibTex

```
@article{reuse2023,
  title     = {Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation},
  journal   = {arXiv preprint arXiv:2309.03549},
  year      = {2023}
}
```
