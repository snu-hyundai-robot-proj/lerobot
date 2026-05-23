#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


class RandomLighting(Transform):
    """Per-channel multiplicative RGB gain — simulates lighting / white-balance shift.

    Input is assumed to be a float image in [0, 1] with shape (..., 3, H, W).
    """

    def __init__(self, gain_range: Sequence[float] = (0.7, 1.3)) -> None:
        super().__init__()
        if not (isinstance(gain_range, collections.abc.Sequence) and len(gain_range) == 2):
            raise TypeError(f"gain_range must be a length-2 sequence, got {gain_range}")
        lo, hi = float(gain_range[0]), float(gain_range[1])
        if not (0.0 <= lo <= hi):
            raise ValueError(f"gain_range must satisfy 0 <= lo <= hi, got {(lo, hi)}")
        self.gain_range = (lo, hi)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        lo, hi = self.gain_range
        return {"gain": torch.empty(3).uniform_(lo, hi)}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        gain = params["gain"].to(inpt.device, inpt.dtype)
        gain = gain.view(*([1] * (inpt.dim() - 3)), 3, 1, 1)
        return (inpt * gain).clamp(0.0, 1.0)


class RandomGamma(Transform):
    """Random gamma correction: out = in^gamma, gamma sampled uniformly per call."""

    def __init__(self, gamma_range: Sequence[float] = (0.7, 1.5)) -> None:
        super().__init__()
        if not (isinstance(gamma_range, collections.abc.Sequence) and len(gamma_range) == 2):
            raise TypeError(f"gamma_range must be a length-2 sequence, got {gamma_range}")
        lo, hi = float(gamma_range[0]), float(gamma_range[1])
        if not (0.0 < lo <= hi):
            raise ValueError(f"gamma_range must satisfy 0 < lo <= hi, got {(lo, hi)}")
        self.gamma_range = (lo, hi)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        lo, hi = self.gamma_range
        return {"gamma": float(torch.empty(1).uniform_(lo, hi).item())}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return inpt.clamp(min=1e-6).pow(params["gamma"]).clamp(0.0, 1.0)


class RandomIntrinsics(Transform):
    """Focal-length scaling (zoom) + radial distortion via ``cv2.remap``.

    Mirrors the ``online_intrinsics`` aug from the reference. ``k1 < 0`` produces a
    barrel/wide-angle look; ``focal_scale != 1`` zooms in/out. Output keeps the
    original H×W (border-replicate fill).
    """

    def __init__(
        self,
        focal_scale_range: Sequence[float] = (0.8, 1.2),
        distortion_range: Sequence[float] = (-0.3, 0.0),
    ) -> None:
        super().__init__()
        for name, rng in (
            ("focal_scale_range", focal_scale_range),
            ("distortion_range", distortion_range),
        ):
            if not (isinstance(rng, collections.abc.Sequence) and len(rng) == 2):
                raise TypeError(f"{name} must be a length-2 sequence, got {rng}")
        if not (0.0 < focal_scale_range[0] <= focal_scale_range[1]):
            raise ValueError(f"focal_scale_range must satisfy 0 < lo <= hi, got {focal_scale_range}")
        self.focal_scale_range = (float(focal_scale_range[0]), float(focal_scale_range[1]))
        self.distortion_range = (float(distortion_range[0]), float(distortion_range[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "focal_scale": float(torch.empty(1).uniform_(*self.focal_scale_range).item()),
            "k1": float(torch.empty(1).uniform_(*self.distortion_range).item()),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        import cv2
        import numpy as np

        if not torch.is_tensor(inpt):
            return inpt
        # Operate on (..., C, H, W) by collapsing leading dims into batch.
        orig_shape = inpt.shape
        x = inpt.reshape(-1, *orig_shape[-3:])  # (N, C, H, W)
        n, c, h, w = x.shape
        scale = params["focal_scale"]
        k1 = params["k1"]
        cx, cy = w / 2.0, h / 2.0
        fx = fy = max(h, w) * scale
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        new_K = K.copy()
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
        out = torch.empty_like(x)
        # cv2.remap wants HxWxC uint8 or float; do per-frame.
        x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
        for i in range(n):
            warped = cv2.remap(
                x_np[i], map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            if warped.ndim == 2:  # 1-channel collapse safety
                warped = warped[..., None]
            out[i] = torch.from_numpy(warped).permute(2, 0, 1).to(inpt.device, inpt.dtype)
        return out.reshape(orig_shape).clamp(0.0, 1.0)


class RandomUnprocessor(Transform):
    """sRGB → simulated RAW Bayer → noise → re-process to sRGB.

    Thin wrapper over the Brooks et al. reference pipeline (vendored in
    ``lerobot.datasets._unprocess_brooks``). Per-call sampling of CCM, RGB/WB
    gains, and shot/read noise — applied identically across all frames in a
    single ``forward`` call so frames within the same clip stay consistent.

    Note: requires even H and W (Bayer mosaic halves resolution then demosaic
    upsamples back).
    """

    def __init__(
        self,
        # Kept for backward compat; current vendored impl ignores these. Override the
        # noise levels via `shot_range` / `read_range` if you want to override the
        # paper's log-log linear sampling.
        gamma_range: Sequence[float] | None = None,
        ccm_noise: float | None = None,
        shot_range: Sequence[float] | None = None,
        read_range: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        # We expose `shot_range` / `read_range` for callers that want explicit noise
        # bounds (rather than the paper's log-log linear sampler). When provided,
        # they replace `random_noise_levels()`.
        for name, rng in (("shot_range", shot_range), ("read_range", read_range)):
            if rng is not None:
                if not (isinstance(rng, collections.abc.Sequence) and len(rng) == 2):
                    raise TypeError(f"{name} must be a length-2 sequence, got {rng}")
                if not (0.0 <= rng[0] <= rng[1]):
                    raise ValueError(f"{name} must satisfy 0 <= lo <= hi, got {rng}")
        self.shot_range = tuple(float(v) for v in shot_range) if shot_range is not None else None
        self.read_range = tuple(float(v) for v in read_range) if read_range is not None else None
        # Unused but accepted for backward-compat with prior config defaults.
        self._unused_gamma_range = gamma_range
        self._unused_ccm_noise = ccm_noise

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        from lerobot.datasets import _unprocess_brooks as ub

        rgb2cam = ub.random_ccm()
        cam2rgb = torch.inverse(rgb2cam)
        rgb_gain, red_gain, blue_gain = ub.random_gains()
        if self.shot_range is not None and self.read_range is not None:
            shot = torch.empty(1).uniform_(*self.shot_range)
            read = torch.empty(1).uniform_(*self.read_range)
        else:
            shot, read = ub.random_noise_levels()
        return {
            "rgb2cam": rgb2cam,
            "cam2rgb": cam2rgb,
            "rgb_gain": rgb_gain,
            "red_gain": red_gain,
            "blue_gain": blue_gain,
            "shot": shot,
            "read": read,
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        from lerobot.datasets import _unprocess_brooks as ub

        if not torch.is_tensor(inpt):
            return inpt
        x = inpt.clamp(0.0, 1.0)
        if x.dim() < 3 or x.shape[-3] != 3:
            return inpt
        # Mosaic requires even spatial dims; fall back to passthrough if odd.
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            return inpt
        device, dtype = x.device, x.dtype
        # Move all params to the right device/dtype once.
        rgb2cam = params["rgb2cam"].to(device, dtype)
        cam2rgb = params["cam2rgb"].to(device, dtype)
        rgb_gain = params["rgb_gain"].to(device, dtype)
        red_gain = params["red_gain"].to(device, dtype)
        blue_gain = params["blue_gain"].to(device, dtype)
        shot = params["shot"].to(device, dtype)
        read = params["read"].to(device, dtype)
        # Flatten leading dims so we can per-frame unprocess, then batch-process.
        leading = x.shape[:-3]
        flat = x.reshape(-1, *x.shape[-3:])  # (N, 3, H, W)
        bayers = []
        for i in range(flat.shape[0]):
            bayer = ub.unprocess_with_params(flat[i], rgb2cam, rgb_gain, red_gain, blue_gain)
            bayer = ub.add_noise(bayer, shot, read)
            bayers.append(bayer)
        bayer_batch = torch.stack(bayers, dim=0)  # (N, 4, H/2, W/2)
        n = bayer_batch.shape[0]
        rgb = ub.process(
            bayer_batch,
            red_gains=red_gain.expand(n),
            blue_gains=blue_gain.expand(n),
            cam2rgbs=cam2rgb.unsqueeze(0).expand(n, -1, -1),
        )
        return rgb.reshape(*leading, 3, x.shape[-2], x.shape[-1]).clamp(0.0, 1.0)


class RandomCorruption(Transform):
    """Hendrycks 15-corruption wrapper (gaussian/shot/impulse noise, blur, weather, jpeg, ...).

    Backed by the ``imagecorruptions`` package. Severities 1–5; robotics typical 1–3.
    Slow on weather kinds (snow/frost/fog). ``corruption_types`` filters by category.
    """

    _CATEGORY_TO_NAMES: dict[str, tuple[str, ...]] = {
        "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
        "blur": ("defocus_blur", "motion_blur", "zoom_blur"),
        "weather": ("snow", "frost", "fog"),
        "digital": ("jpeg_compression", "pixelate"),
    }

    def __init__(
        self,
        severity_range: Sequence[int] = (1, 3),
        corruption_types: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        lo, hi = int(severity_range[0]), int(severity_range[1])
        if not (1 <= lo <= hi <= 5):
            raise ValueError(f"severity_range must satisfy 1 <= lo <= hi <= 5, got {severity_range}")
        self.severity_range = (lo, hi)
        if corruption_types is None:
            self.corruption_names: tuple[str, ...] = tuple(
                n for ns in self._CATEGORY_TO_NAMES.values() for n in ns
            )
        else:
            names: list[str] = []
            for t in corruption_types:
                if t in self._CATEGORY_TO_NAMES:
                    names.extend(self._CATEGORY_TO_NAMES[t])
                else:
                    names.append(t)
            self.corruption_names = tuple(names)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        idx = int(torch.randint(0, len(self.corruption_names), (1,)).item())
        sev = int(torch.randint(self.severity_range[0], self.severity_range[1] + 1, (1,)).item())
        return {"corruption_name": self.corruption_names[idx], "severity": sev}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        from imagecorruptions import corrupt
        import numpy as np

        if not torch.is_tensor(inpt):
            return inpt
        orig_shape = inpt.shape
        x = inpt.reshape(-1, *orig_shape[-3:])  # (N, C, H, W)
        out = torch.empty_like(x)
        x_np = (x.detach().cpu().permute(0, 2, 3, 1).clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        name = params["corruption_name"]
        sev = params["severity"]
        for i in range(x.shape[0]):
            arr = x_np[i]
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            corrupted = np.ascontiguousarray(corrupt(arr, corruption_name=name, severity=sev))
            t = torch.from_numpy(corrupted).to(inpt.device, inpt.dtype) / 255.0
            out[i] = t.permute(2, 0, 1)[: orig_shape[-3]]
        return out.reshape(orig_shape).clamp(0.0, 1.0)


class RandomCornerCrop(Transform):
    """Crop a random corner region (per the ``online_crop`` reference) and resize back."""

    def __init__(self, crop_ratios: Sequence[float] = (0.05, 0.10, 0.20)) -> None:
        super().__init__()
        if not crop_ratios or any(r <= 0.0 or r >= 1.0 for r in crop_ratios):
            raise ValueError(f"crop_ratios entries must lie in (0, 1), got {crop_ratios}")
        self.crop_ratios = tuple(float(r) for r in crop_ratios)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        corner = int(torch.randint(0, 4, (1,)).item())  # 0=TL, 1=TR, 2=BL, 3=BR
        rh = self.crop_ratios[int(torch.randint(0, len(self.crop_ratios), (1,)).item())]
        rw = self.crop_ratios[int(torch.randint(0, len(self.crop_ratios), (1,)).item())]
        return {"corner": corner, "rh": float(rh), "rw": float(rw)}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not torch.is_tensor(inpt):
            return inpt
        h, w = inpt.shape[-2], inpt.shape[-1]
        ch = max(1, int(h * (1.0 - params["rh"])))
        cw = max(1, int(w * (1.0 - params["rw"])))
        corner = params["corner"]
        top = 0 if corner in (0, 1) else h - ch
        left = 0 if corner in (0, 2) else w - cw
        cropped = inpt[..., top : top + ch, left : left + cw]
        return F.resize(cropped, [h, w], antialias=True)


class RandomSensorNoise(Transform):
    """Signal-dependent shot + read noise (simplified Poisson-Gaussian camera model).

    Adds Gaussian noise with variance ``shot * I + read^2`` per pixel, where ``I`` is
    the input intensity. Skips the full sRGB↔linear / CCM pipeline of the reference
    implementation for speed; effective on float images in [0, 1].
    """

    def __init__(
        self,
        shot_range: Sequence[float] = (0.0, 0.04),
        read_range: Sequence[float] = (0.0, 0.01),
    ) -> None:
        super().__init__()
        for name, rng in (("shot_range", shot_range), ("read_range", read_range)):
            if not (isinstance(rng, collections.abc.Sequence) and len(rng) == 2):
                raise TypeError(f"{name} must be a length-2 sequence, got {rng}")
            lo, hi = float(rng[0]), float(rng[1])
            if not (0.0 <= lo <= hi):
                raise ValueError(f"{name} must satisfy 0 <= lo <= hi, got {(lo, hi)}")
        self.shot_range = (float(shot_range[0]), float(shot_range[1]))
        self.read_range = (float(read_range[0]), float(read_range[1]))

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        return {
            "shot": float(torch.empty(1).uniform_(*self.shot_range).item()),
            "read": float(torch.empty(1).uniform_(*self.read_range).item()),
        }

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        shot = params["shot"]
        read = params["read"]
        var = shot * inpt.clamp(min=0.0) + read * read
        noise = torch.randn_like(inpt) * var.clamp(min=0.0).sqrt()
        return (inpt + noise).clamp(0.0, 1.0)


class RandomWarp(Transform):
    """Random perspective warp via a sampled 4-corner homography.

    Each call samples a ``distortion_scale`` from ``distortion_scale_range`` and four
    independent corner displacements bounded by that scale. Goes beyond
    ``RandomAffine`` (rigid: rotate / translate / scale) and ``RandomIntrinsics``
    (radial distortion only) by introducing genuine perspective warp — useful for
    simulating small camera-pose errors and scene-depth variation.

    Uses torchvision's perspective kernel under the hood for batched (..., C, H, W).
    """

    def __init__(self, distortion_scale_range: Sequence[float] = (0.1, 0.4)) -> None:
        super().__init__()
        if not (isinstance(distortion_scale_range, collections.abc.Sequence) and len(distortion_scale_range) == 2):
            raise TypeError(
                f"distortion_scale_range must be a length-2 sequence, got {distortion_scale_range}"
            )
        lo, hi = float(distortion_scale_range[0]), float(distortion_scale_range[1])
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(
                f"distortion_scale_range must satisfy 0 <= lo <= hi <= 1, got {(lo, hi)}"
            )
        self.distortion_scale_range = (lo, hi)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        # Find first image-like tensor to read H,W from.
        h = w = None
        for inpt in flat_inputs:
            if torch.is_tensor(inpt) and inpt.dim() >= 3:
                h, w = int(inpt.shape[-2]), int(inpt.shape[-1])
                break
        if h is None or w is None:
            return {"startpoints": None, "endpoints": None}
        scale = float(torch.empty(1).uniform_(*self.distortion_scale_range).item())
        half_w = max(int(scale * w / 2), 1)
        half_h = max(int(scale * h / 2), 1)
        # Corners (TL, TR, BR, BL); each endpoint is the original corner perturbed
        # toward the image centre by a uniform amount up to (half_w, half_h).
        def _rand(maxval: int) -> int:
            return int(torch.randint(0, maxval + 1, (1,)).item())

        startpoints = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        endpoints = [
            [_rand(half_w),             _rand(half_h)],
            [w - 1 - _rand(half_w),     _rand(half_h)],
            [w - 1 - _rand(half_w),     h - 1 - _rand(half_h)],
            [_rand(half_w),             h - 1 - _rand(half_h)],
        ]
        return {"startpoints": startpoints, "endpoints": endpoints, "scale": scale}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not torch.is_tensor(inpt) or params.get("startpoints") is None:
            return inpt
        return F.perspective(
            inpt,
            startpoints=params["startpoints"],
            endpoints=params["endpoints"],
            interpolation=v2.InterpolationMode.BILINEAR,
            fill=0,
        )


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    # Master probability: each call applies augmentations with this probability, otherwise passes through.
    # Use 0.5 for "domain randomization on half the samples" (typical sim-to-real recipe).
    p_apply: float = 1.0
    # Convenience preset: when True, bump weights of the sensor-style DR augs (lighting,
    # gamma, sensor_noise, gaussian_blur, corner_crop, intrinsics, unprocessor, corruption)
    # so they are actually sampled. Default tfs leaves them at weight=0 because draccus
    # CLI cannot override dict entries individually.
    domain_randomization: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
            # Domain-randomization sensor augs. Disabled by default (weight=0).
            # Bump weight via CLI to enable (e.g. --dataset.image_transforms.tfs.lighting.weight=2.0).
            "lighting": ImageTransformConfig(
                weight=0.0,
                type="RandomLighting",
                kwargs={"gain_range": (0.7, 1.3)},
            ),
            "gamma": ImageTransformConfig(
                weight=0.0,
                type="RandomGamma",
                kwargs={"gamma_range": (0.7, 1.5)},
            ),
            "sensor_noise": ImageTransformConfig(
                weight=0.0,
                type="RandomSensorNoise",
                kwargs={"shot_range": (0.0, 0.04), "read_range": (0.0, 0.01)},
            ),
            "gaussian_blur": ImageTransformConfig(
                weight=0.0,
                type="GaussianBlur",
                kwargs={"kernel_size": 5, "sigma": (0.1, 2.0)},
            ),
            "intrinsics": ImageTransformConfig(
                weight=0.0,
                type="RandomIntrinsics",
                kwargs={"focal_scale_range": (0.8, 1.2), "distortion_range": (-0.3, 0.0)},
            ),
            "unprocessor": ImageTransformConfig(
                weight=0.0,
                type="RandomUnprocessor",
                kwargs={
                    "gamma_range": (1.8, 2.6),
                    "ccm_noise": 0.05,
                    "shot_range": (0.0, 5e-4),
                    "read_range": (0.0, 1e-4),
                },
            ),
            "corruption": ImageTransformConfig(
                weight=0.0,
                type="RandomCorruption",
                kwargs={"severity_range": (1, 3), "corruption_types": ("noise", "blur", "digital")},
            ),
            "corner_crop": ImageTransformConfig(
                weight=0.0,
                type="RandomCornerCrop",
                kwargs={"crop_ratios": (0.05, 0.10, 0.20)},
            ),
            "warping": ImageTransformConfig(
                weight=0.0,
                type="RandomWarp",
                kwargs={"distortion_scale_range": (0.1, 0.4)},
            ),
        }
    )

    # Cost-weighted DR preset applied when `domain_randomization=True`. Cheap augs
    # get higher weight; the slow Hendrycks corruption is rare on purpose.
    _DR_WEIGHTS = {
        "lighting": 3.0,
        "gamma": 2.0,
        "sensor_noise": 2.0,
        "gaussian_blur": 1.0,
        "corner_crop": 2.0,
        "intrinsics": 1.0,
        "unprocessor": 1.0,
        "corruption": 0.5,
        "warping": 1.5,
    }

    def __post_init__(self) -> None:
        if self.domain_randomization:
            for key, w in self._DR_WEIGHTS.items():
                if key in self.tfs:
                    self.tfs[key].weight = w


_CUSTOM_TRANSFORMS: dict[str, type[Transform]] = {
    "SharpnessJitter": SharpnessJitter,
    "RandomLighting": RandomLighting,
    "RandomGamma": RandomGamma,
    "RandomSensorNoise": RandomSensorNoise,
    "RandomIntrinsics": RandomIntrinsics,
    "RandomUnprocessor": RandomUnprocessor,
    "RandomCorruption": RandomCorruption,
    "RandomCornerCrop": RandomCornerCrop,
    "RandomWarp": RandomWarp,
}


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type in _CUSTOM_TRANSFORMS:
        return _CUSTOM_TRANSFORMS[cfg.type](**cfg.kwargs)

    transform_cls = getattr(v2, cfg.type, None)
    if isinstance(transform_cls, type) and issubclass(transform_cls, Transform):
        return transform_cls(**cfg.kwargs)

    raise ValueError(
        f"Transform '{cfg.type}' is not valid. It must be a class in "
        f"torchvision.transforms.v2 or one of {sorted(_CUSTOM_TRANSFORMS)}."
    )


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._p_apply = float(cfg.p_apply)
        if not 0.0 <= self._p_apply <= 1.0:
            raise ValueError(f"p_apply must be in [0, 1], got {self._p_apply}")

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        if self._p_apply < 1.0 and torch.rand(()).item() >= self._p_apply:
            return inputs[0] if len(inputs) == 1 else inputs
        return self.tf(*inputs)
