"""Vendored sRGB↔raw camera pipeline from Brooks et al., CVPR 2019.

Source: https://github.com/aasharma90/UnprocessDenoising_PyTorch
        (PyTorch port of google-research/google-research/unprocessing)

Original Apache-2.0 © Google Research Authors. Combined here from
``dataloader/unprocess.py`` and ``dataloader/process.py`` so we can call the
forward (sRGB→raw) and inverse (raw→sRGB) pipelines back-to-back as a single
domain-randomization augmentation. Random sampling helpers are exposed so that
callers can sample params once and apply them deterministically across frames
of the same clip.

Brooks et al., "Unprocessing Images for Learned Raw Denoising":
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributions as tdist


# ─── Random parameter sampling (factored out of `unprocess`) ────────────────


def random_ccm() -> torch.Tensor:
    """RGB → camera CCM, sampled as a convex combination of four XYZ→camera matrices."""
    xyz2cams = [
        [[1.0234, -0.2969, -0.2266], [-0.5625, 1.6328, -0.0469], [-0.0703, 0.2188, 0.6406]],
        [[0.4913, -0.0541, -0.0202], [-0.6130, 1.3513, 0.2906], [-0.1564, 0.2151, 0.7183]],
        [[0.8380, -0.2630, -0.0639], [-0.2887, 1.0725, 0.2496], [-0.0627, 0.1427, 0.5438]],
        [[0.6596, -0.2079, -0.0562], [-0.4782, 1.3016, 0.1933], [-0.0970, 0.1581, 0.5181]],
    ]
    xyz2cams = torch.tensor(xyz2cams, dtype=torch.float32)
    weights = torch.empty(xyz2cams.shape[0], 1, 1).uniform_(1e-8, 1e8)
    xyz2cam = (xyz2cams * weights).sum(dim=0) / weights.sum(dim=0)
    rgb2xyz = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=torch.float32,
    )
    rgb2cam = xyz2cam @ rgb2xyz
    rgb2cam = rgb2cam / rgb2cam.sum(dim=-1, keepdim=True)
    return rgb2cam


def random_gains() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB (brightness) gain + red/blue (white balance) gains."""
    n = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1]))
    rgb_gain = 1.0 / n.sample()
    red_gain = torch.empty(1).uniform_(1.9, 2.4)
    blue_gain = torch.empty(1).uniform_(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def random_noise_levels() -> tuple[torch.Tensor, torch.Tensor]:
    """Log-log linear shot/read noise sampling."""
    log_min = float(np.log(0.0001))
    log_max = float(np.log(0.012))
    log_shot = torch.empty(1).uniform_(log_min, log_max)
    shot = log_shot.exp()
    line = lambda x: 2.18 * x + 1.20
    n = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
    log_read = line(log_shot) + n.sample()
    read = log_read.exp()
    return shot, read


# ─── Forward: sRGB → simulated RAW Bayer ────────────────────────────────────


def inverse_smoothstep(image: torch.Tensor) -> torch.Tensor:
    """Inverts the smoothstep tone-mapping curve."""
    image = image.permute(1, 2, 0).clamp(0.0, 1.0)
    out = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
    return out.permute(2, 0, 1)


def gamma_expansion(image: torch.Tensor) -> torch.Tensor:
    """sRGB gamma → linear (x^2.2)."""
    image = image.permute(1, 2, 0)
    out = image.clamp(min=1e-8) ** 2.2
    return out.permute(2, 0, 1)


def apply_ccm(image: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    image = image.permute(1, 2, 0)
    shape = image.shape
    flat = image.reshape(-1, 3)
    flat = torch.tensordot(flat, ccm, dims=[[-1], [-1]])
    return flat.reshape(shape).permute(2, 0, 1)


def safe_invert_gains(
    image: torch.Tensor,
    rgb_gain: torch.Tensor,
    red_gain: torch.Tensor,
    blue_gain: torch.Tensor,
) -> torch.Tensor:
    image = image.permute(1, 2, 0)
    gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
    gains = gains.squeeze()[None, None, :]
    gray = image.mean(dim=-1, keepdim=True)
    inflection = 0.9
    mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    out = image * safe_gains
    return out.permute(2, 0, 1)


def mosaic(image: torch.Tensor) -> torch.Tensor:
    """RGB (3,H,W) → RGGB Bayer (4, H/2, W/2)."""
    image = image.permute(1, 2, 0)
    h, w, _ = image.shape
    red = image[0::2, 0::2, 0]
    green_red = image[0::2, 1::2, 1]
    green_blue = image[1::2, 0::2, 1]
    blue = image[1::2, 1::2, 2]
    out = torch.stack((red, green_red, green_blue, blue), dim=-1)  # (h/2, w/2, 4)
    return out.permute(2, 0, 1)


def unprocess_with_params(
    image: torch.Tensor,
    rgb2cam: torch.Tensor,
    rgb_gain: torch.Tensor,
    red_gain: torch.Tensor,
    blue_gain: torch.Tensor,
) -> torch.Tensor:
    """sRGB (3,H,W) → Bayer (4, H/2, W/2). Deterministic given params."""
    image = inverse_smoothstep(image)
    image = gamma_expansion(image)
    image = apply_ccm(image, rgb2cam)
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    image = image.clamp(0.0, 1.0)
    return mosaic(image)


def add_noise(bayer: torch.Tensor, shot_noise: torch.Tensor, read_noise: torch.Tensor) -> torch.Tensor:
    """Signal-dependent shot + read Gaussian noise on (4, H, W) Bayer."""
    bayer = bayer.permute(1, 2, 0)
    variance = bayer * shot_noise + read_noise
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=variance.clamp(min=1e-12).sqrt())
    out = bayer + n.sample()
    return out.permute(2, 0, 1)


# ─── Inverse: Bayer → sRGB ──────────────────────────────────────────────────


def apply_gains(bayer_images: torch.Tensor, red_gains: torch.Tensor, blue_gains: torch.Tensor) -> torch.Tensor:
    """White-balance gains on a batch (B,4,h,w) Bayer."""
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
    gains = gains[:, None, None, :]
    out = bayer_images * gains
    return out.permute(0, 3, 1, 2)


def demosaic(bayer_images: torch.Tensor) -> torch.Tensor:
    """Bilinear-style demosaic: (B, 4, h, w) → (B, 3, 2h, 2w)."""
    bayer_images = bayer_images.permute(0, 2, 3, 1)
    shape = bayer_images.shape
    red = bayer_images[..., 0:1]
    green_red = bayer_images[..., 1:2]
    green_blue = bayer_images[..., 2:3]
    blue = bayer_images[..., 3:4]
    red = torch.nn.functional.interpolate(
        red.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    green = torch.nn.functional.interpolate(
        ((green_red + green_blue) / 2.0).permute(0, 3, 1, 2),
        scale_factor=2,
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    blue = torch.nn.functional.interpolate(
        blue.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear", align_corners=False
    ).permute(0, 2, 3, 1)
    rgb = torch.cat([red, green, blue], dim=-1)
    return rgb.permute(0, 3, 1, 2)


def apply_ccms(images: torch.Tensor, ccms: torch.Tensor) -> torch.Tensor:
    images = images.permute(0, 2, 3, 1)
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    return (images * ccms).sum(dim=-1).permute(0, 3, 1, 2)


def gamma_compression(images: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    images = images.permute(0, 2, 3, 1).clamp(min=1e-8)
    return (images ** (1.0 / gamma)).permute(0, 3, 1, 2)


def process(
    bayer_images: torch.Tensor,
    red_gains: torch.Tensor,
    blue_gains: torch.Tensor,
    cam2rgbs: torch.Tensor,
) -> torch.Tensor:
    """Bayer (B,4,h,w) → sRGB (B,3,2h,2w)."""
    images = apply_gains(bayer_images, red_gains, blue_gains)
    images = images.clamp(0.0, 1.0)
    images = demosaic(images)
    images = apply_ccms(images, cam2rgbs)
    images = images.clamp(0.0, 1.0)
    return gamma_compression(images)
