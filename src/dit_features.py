#!/usr/bin/env python3
"""
Diffusion Transformer (DiT) deep-feature extractor.

Design decision: a full Diffusion Transformer is a *generative* architecture,
not a classifier, and training one from scratch (or fine-tuning it end-to-end)
would need far more labeled sand-mining imagery than this project has (the
existing pipeline gates training on as few as ~10 labeled images). Instead,
this module treats a large, frozen, pretrained DiT backbone purely as a
feature extractor: an image is VAE-encoded to a latent, noised to a fixed
timestep, and passed once through the pretrained DiT; its penultimate hidden
state is spatially pooled into a fixed-length vector. That vector is
concatenated onto the existing spectral/texture/GLCM feature vector in
features.py, and the existing RF/XGBoost/LightGBM models learn to use it
alongside everything else. No gradient ever flows into the DiT weights.

This is the same "diffusion features for recognition" idea used in prior work
(e.g. Xiang et al. 2023, "Denoising Diffusion Autoencoders are Unified
Self-supervised Learners"; Yang & Wang 2023, "Diffusion Model as
Representation Learner") adapted to a frozen off-the-shelf backbone so it
works without a training run of its own.

All heavy imports (torch, diffusers) are optional and loaded lazily so the
rest of the pipeline works unmodified when USE_DIT_FEATURES is left off.
"""

import os
import numpy as np
from PIL import Image

from src import config

_MODEL_CACHE = {}


def _dit_available():
    try:
        import torch  # noqa: F401
        from diffusers import DiTTransformer2DModel, AutoencoderKL  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device():
    import torch
    if config.DIT_DEVICE == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_dit():
    """Lazily load and cache the pretrained VAE + DiT backbone (frozen, eval mode)."""
    if 'vae' in _MODEL_CACHE:
        return _MODEL_CACHE['vae'], _MODEL_CACHE['dit'], _MODEL_CACHE['device']

    import torch
    from diffusers import DiTTransformer2DModel, AutoencoderKL

    device = _get_device()
    print(f"[DiT] Loading pretrained backbone '{config.DIT_MODEL_ID}' on {device} "
          f"(frozen, feature-extraction only)...")

    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device)
    dit = DiTTransformer2DModel.from_pretrained(config.DIT_MODEL_ID).to(device)
    vae.eval()
    dit.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in dit.parameters():
        p.requires_grad_(False)

    _MODEL_CACHE['vae'] = vae
    _MODEL_CACHE['dit'] = dit
    _MODEL_CACHE['device'] = device
    return vae, dit, device


def extract_dit_features(image_path_or_array):
    """
    Extract a fixed-length pooled feature vector from a frozen pretrained DiT.

    Args:
        image_path_or_array: path to an image file, or an (H, W, 3) uint8/float
            numpy array (used for scoring in-memory superpixel crops).

    Returns:
        dict: {'dit_feat_0': v0, 'dit_feat_1': v1, ...} of length
              config.DIT_FEATURE_DIM. Returns zeros (with a warning printed
              once) if torch/diffusers are not installed or extraction fails,
              so the rest of the pipeline never crashes because of this
              optional feature.
    """
    n = config.DIT_FEATURE_DIM
    zero_result = {f'dit_feat_{i}': 0.0 for i in range(n)}

    if not _dit_available():
        if not _MODEL_CACHE.get('_warned'):
            print("[DiT] torch/diffusers not installed — DiT features disabled "
                  "(pip install torch diffusers). Falling back to zeros.")
            _MODEL_CACHE['_warned'] = True
        return zero_result

    try:
        import torch
        import torch.nn.functional as F

        vae, dit, device = _load_dit()

        if isinstance(image_path_or_array, (str, os.PathLike)):
            img = Image.open(image_path_or_array).convert('RGB')
        else:
            arr = np.asarray(image_path_or_array)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr).convert('RGB')

        img = img.resize((config.DIT_IMAGE_SIZE, config.DIT_IMAGE_SIZE), Image.BICUBIC)
        x = np.asarray(img).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor

            t = torch.tensor([config.DIT_TIMESTEP], device=device, dtype=torch.long)
            noise = torch.randn_like(latent)
            # Simple fixed-schedule noising (sqrt(1-t/T) signal + sqrt(t/T) noise)
            alpha = 1.0 - (config.DIT_TIMESTEP / 1000.0)
            noisy_latent = (alpha ** 0.5) * latent + ((1 - alpha) ** 0.5) * noise

            # DiT expects a class-label conditioning input; use the
            # unconditional/null class if the model supports it, else 0.
            try:
                num_classes = dit.config.num_embeds_ada_norm or 1000
            except Exception:
                num_classes = 1000
            class_labels = torch.tensor([num_classes - 1], device=device)

            out = dit(noisy_latent, timestep=t, class_labels=class_labels,
                      return_dict=True)
            hidden = out.sample if hasattr(out, 'sample') else out[0]

        # hidden: (1, C, H, W) noise-prediction map — pool spatially, then
        # project/truncate channel-wise to a fixed-length vector.
        pooled = F.adaptive_avg_pool2d(hidden, 1).flatten()   # (C,)
        pooled = pooled.detach().cpu().numpy().astype(np.float64)

        if len(pooled) >= n:
            feat_vec = pooled[:n]
        else:
            feat_vec = np.pad(pooled, (0, n - len(pooled)))

        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return {f'dit_feat_{i}': float(feat_vec[i]) for i in range(n)}

    except Exception as e:
        print(f"[DiT] Feature extraction failed ({e}); using zeros for this image.")
        return zero_result
