from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import CFG


MODEL_NAMES = ("unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net")
INPUT_MODES = ("auto", "s2", "s2_proxy", "methane_features", "s2_plus_methane", "l1c_pair_methane")


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    out_dir: Path = CFG.data_dir / "models" / "segmentation"
    model: str = "unet"
    epochs: int = 30
    batch_size: int = 2
    patch_size: int = 256
    patches_per_epoch: int = 4096
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    positive_patch_probability: float = 0.75
    min_patch_positive_fraction: float = 0.0
    max_patch_attempts: int = 25
    target_mode: str = "soft"
    target_dilation_pixels: int = 2
    input_mode: str = "auto"
    carbon_mapper_only: bool = False
    train_source: str | None = None
    min_train_positive_pixels: int = 50
    max_train_positive_fraction: float = 0.05
    min_eval_positive_pixels: int = 50
    max_eval_positive_fraction: float = 0.05
    seed: int = 7
    device: str = "auto"
    limit_train: int | None = None
    limit_val: int | None = None
    init_checkpoint: Path | None = None


@dataclass(frozen=True)
class SyntheticTrainConfig:
    manifest: Path = CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"
    out_dir: Path = CFG.data_dir / "models" / "l1c_synthetic_plumes"
    model: str = "phys_tau_net"
    epochs: int = 40
    batch_size: int = 8
    patch_size: int = 128
    patches_per_epoch: int = 8192
    validation_patches: int = 1024
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    input_mode: str = "l1c_pair_methane"
    min_plume_fraction: float = 0.01
    max_plume_fraction: float = 0.20
    min_tau: float = 0.004
    max_tau: float = 0.055
    target_mode: str = "soft"
    target_dilation_pixels: int = 2
    carbon_mapper_backgrounds_only: bool = False
    seed: int = 7
    device: str = "auto"
    limit_train: int | None = None
    limit_val: int | None = None


@dataclass(frozen=True)
class EvalConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    checkpoint: Path = CFG.data_dir / "models" / "segmentation" / "unet" / "best.pt"
    out_dir: Path = CFG.data_dir / "models" / "segmentation" / "unet"
    threshold: float = 0.5
    tolerance_pixels: int = 2
    split: str | None = None
    source: str | None = None
    min_positive_pixels: int | None = None
    max_positive_fraction: float | None = None
    limit: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class InputAuditConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    out: Path = CFG.data_dir / "training" / "segmentation_input_range_audit.csv"
    limit: int | None = None


@dataclass(frozen=True)
class ThresholdSweepConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    checkpoint: Path = CFG.data_dir / "models" / "segmentation" / "unet" / "best.pt"
    out: Path = CFG.data_dir / "models" / "segmentation" / "unet" / "threshold_sweep.csv"
    thresholds: tuple[float, ...] = (
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    )
    split: str = "VAL"
    tolerance_pixels: int = 2
    source: str | None = None
    min_positive_pixels: int | None = None
    max_positive_fraction: float | None = None
    limit: int | None = None
    device: str = "auto"


@dataclass(frozen=True)
class LabelSizeAuditConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    out: Path = CFG.data_dir / "training" / "label_size_feasibility_audit.csv"
    pixel_cutoffs: tuple[int, ...] = (5, 10, 25, 50, 100, 250, 500, 1000)


@dataclass(frozen=True)
class CropSizingAuditConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    out: Path = CFG.data_dir / "training" / "crop_sizing_audit.csv"
    crop_sizes: tuple[int, ...] = (32, 48, 64, 96, 128, 192, 256, 512)
    desired_fraction: float = 0.30


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install a CUDA build on VSC before running this command."
        ) from exc
    return torch, nn, F, DataLoader, Dataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, *_ = _require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str):
    torch, *_ = _require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _scale_band(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32")
    finite = np.isfinite(arr) & (arr != 0)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(arr[finite]))
        lo = float(np.nanmin(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1).astype("float32")
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")


def _read_s2(path: str, include_physics_proxy: bool = False) -> np.ndarray:
    with rasterio.open(path) as src:
        raw = src.read(out_dtype="float32")
    # Export band order: B2, B3, B4, B8, B11, B12, SCL. SCL is not used as a continuous input.
    bands = [_scale_band(raw[i]) for i in range(min(6, raw.shape[0]))]
    x = np.stack(bands, axis=0).astype("float32")
    if include_physics_proxy:
        b11 = raw[4].astype("float32")
        b12 = raw[5].astype("float32")
        proxy = (b12 - b11) / np.maximum(np.abs(b11), 1.0)
        proxy = np.clip(proxy, -1.5, 1.5)
        proxy = np.nan_to_num(proxy, nan=0.0, posinf=1.5, neginf=-1.5)
        x = np.concatenate([x, proxy[None].astype("float32")], axis=0)
    return np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")


def _robust_standardize(arr: np.ndarray, clip: float = 4.0) -> np.ndarray:
    arr = arr.astype("float32")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype="float32")
    med = float(np.nanmedian(arr[finite]))
    mad = float(np.nanmedian(np.abs(arr[finite] - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.nanstd(arr[finite]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return np.zeros_like(arr, dtype="float32")
    z = np.clip((arr - med) / scale, -clip, clip)
    z = (z + clip) / (2.0 * clip)
    return np.nan_to_num(z, nan=0.5, posinf=1.0, neginf=0.0).astype("float32")


def _swir_linear_residual(b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    finite = np.isfinite(b11) & np.isfinite(b12) & (b11 > 0) & (b12 > 0)
    residual = np.zeros_like(b12, dtype="float32")
    if int(finite.sum()) <= 100:
        return residual

    x = b11[finite].reshape(-1)
    y = b12[finite].reshape(-1)
    lo_x, hi_x = np.percentile(x, [2, 98])
    lo_y, hi_y = np.percentile(y, [2, 98])
    keep = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    if int(keep.sum()) <= 100:
        return residual

    x_keep = x[keep]
    y_keep = y[keep]
    if float(np.nanstd(x_keep)) <= 1e-6 or float(np.nanstd(y_keep)) <= 1e-6:
        return residual

    x_mean = float(np.nanmean(x_keep))
    y_mean = float(np.nanmean(y_keep))
    denom = float(np.nansum((x_keep - x_mean) ** 2))
    if not np.isfinite(denom) or denom <= 1e-6:
        return residual

    slope = float(np.nansum((x_keep - x_mean) * (y_keep - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return residual
    return (b12 - (slope * b11 + intercept)).astype("float32")


def _methane_feature_stack(raw: np.ndarray) -> np.ndarray:
    b2 = raw[0].astype("float32")
    b3 = raw[1].astype("float32")
    b4 = raw[2].astype("float32")
    b8 = raw[3].astype("float32")
    b11 = raw[4].astype("float32")
    b12 = raw[5].astype("float32")
    eps = 1.0

    ratio = np.log1p(np.maximum(b12, 0.0)) - np.log1p(np.maximum(b11, 0.0))
    norm_diff = (b12 - b11) / np.maximum(b12 + b11, eps)
    varon_like = (b12 - b11) / np.maximum(np.abs(b11), eps)

    residual = _swir_linear_residual(b11, b12)

    ndvi = (b8 - b4) / np.maximum(b8 + b4, eps)
    brightness = (b2 + b3 + b4 + b8 + b11 + b12) / 6.0

    features = [
        _scale_band(b11),
        _scale_band(b12),
        _robust_standardize(ratio),
        _robust_standardize(norm_diff),
        _robust_standardize(varon_like),
        _robust_standardize(residual),
        _robust_standardize(ndvi),
        _scale_band(brightness),
    ]
    return np.stack(features, axis=0).astype("float32")


def _methane_pair_feature_stack(raw: np.ndarray) -> np.ndarray:
    if raw.shape[0] < 12:
        raise ValueError("l1c_pair_methane input requires 12 bands: 6 event bands + 6 reference bands")
    event = raw[:6].astype("float32")
    reference = raw[6:12].astype("float32")
    eps = 1.0

    event_features = _methane_feature_stack(event)
    ref_features = _methane_feature_stack(reference)
    event_b11 = event[4]
    event_b12 = event[5]
    ref_b11 = reference[4]
    ref_b12 = reference[5]

    event_ratio = np.log1p(np.maximum(event_b12, 0.0)) - np.log1p(np.maximum(event_b11, 0.0))
    ref_ratio = np.log1p(np.maximum(ref_b12, 0.0)) - np.log1p(np.maximum(ref_b11, 0.0))
    ratio_delta = event_ratio - ref_ratio

    event_norm = (event_b12 - event_b11) / np.maximum(event_b12 + event_b11, eps)
    ref_norm = (ref_b12 - ref_b11) / np.maximum(ref_b12 + ref_b11, eps)
    norm_delta = event_norm - ref_norm

    event_varon = (event_b12 - event_b11) / np.maximum(np.abs(event_b11), eps)
    ref_varon = (ref_b12 - ref_b11) / np.maximum(np.abs(ref_b11), eps)
    varon_delta = event_varon - ref_varon

    b11_delta = event_b11 - ref_b11
    b12_delta = event_b12 - ref_b12
    brightness_delta = event.mean(axis=0) - reference.mean(axis=0)

    features = [
        event_features[0],
        event_features[1],
        event_features[2],
        event_features[3],
        event_features[4],
        event_features[5],
        _robust_standardize(ratio_delta),
        _robust_standardize(norm_delta),
        _robust_standardize(varon_delta),
        _robust_standardize(b11_delta),
        _robust_standardize(b12_delta),
        _robust_standardize(brightness_delta),
        ref_features[2],
        ref_features[3],
    ]
    return np.stack(features, axis=0).astype("float32")


def _resolve_input_mode(input_mode: str, model_name: str | None = None) -> str:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unknown input mode {input_mode!r}; expected one of {INPUT_MODES}")
    if input_mode == "auto":
        return "s2_proxy" if model_name == "phys_tau_net" else "s2"
    return input_mode


def _input_channels(input_mode: str) -> int:
    if input_mode == "s2":
        return 6
    if input_mode == "s2_proxy":
        return 7
    if input_mode == "methane_features":
        return 8
    if input_mode == "s2_plus_methane":
        return 14
    if input_mode == "l1c_pair_methane":
        return 14
    raise ValueError(f"Unknown resolved input mode {input_mode!r}")


def _read_model_input(path: str, input_mode: str) -> np.ndarray:
    if input_mode == "s2":
        return _read_s2(path, include_physics_proxy=False)
    if input_mode == "s2_proxy":
        return _read_s2(path, include_physics_proxy=True)
    with rasterio.open(path) as src:
        raw = src.read(out_dtype="float32")
    if input_mode == "l1c_pair_methane":
        return _methane_pair_feature_stack(raw)
    methane = _methane_feature_stack(raw)
    if input_mode == "methane_features":
        return methane
    if input_mode == "s2_plus_methane":
        return np.concatenate([_read_s2(path, include_physics_proxy=False), methane], axis=0).astype("float32")
    raise ValueError(f"Unknown resolved input mode {input_mode!r}")


def _read_raw_l1c_pair(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        raw = src.read(out_dtype="float32")
    if raw.shape[0] < 12:
        raise ValueError("Synthetic plume training requires paired L1C chips with 12 bands")
    return np.nan_to_num(raw[:12], nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _synthetic_plume_field(
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    min_fraction: float,
    max_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype("float32")
    best_field = None
    best_mask = None
    best_distance = float("inf")

    def _smooth_noise(scale: int) -> np.ndarray:
        coarse_h = max(3, int(np.ceil(height / scale)) + 2)
        coarse_w = max(3, int(np.ceil(width / scale)) + 2)
        coarse = rng.normal(0.0, 1.0, size=(coarse_h, coarse_w)).astype("float32")
        up = np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)[:height, :width]
        for _ in range(3):
            padded = np.pad(up, 1, mode="edge")
            up = (
                padded[:-2, :-2]
                + padded[:-2, 1:-1]
                + padded[:-2, 2:]
                + padded[1:-1, :-2]
                + 2.0 * padded[1:-1, 1:-1]
                + padded[1:-1, 2:]
                + padded[2:, :-2]
                + padded[2:, 1:-1]
                + padded[2:, 2:]
            ) / 10.0
        std = float(np.nanstd(up))
        if std <= 1e-6:
            return np.zeros((height, width), dtype="float32")
        return ((up - float(np.nanmean(up))) / std).astype("float32")

    def _plume_from_meandering_path() -> np.ndarray:
        source_x = float(rng.uniform(0.10 * width, 0.90 * width))
        source_y = float(rng.uniform(0.12 * height, 0.88 * height))
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        length = float(rng.uniform(0.28 * width, 0.95 * width))
        base_sigma = float(rng.uniform(0.018 * width, 0.075 * width))
        n_nodes = int(rng.integers(5, 10))

        nodes = []
        cross_offset = 0.0
        for i in range(n_nodes):
            t = i / max(n_nodes - 1, 1)
            step = t * length
            cross_offset += float(rng.normal(0.0, 0.09 * width))
            cross_offset = float(np.clip(cross_offset, -0.28 * width, 0.28 * width))
            taper_noise = float(rng.normal(0.0, 0.035 * width))
            px = source_x + step * math.cos(angle) - cross_offset * math.sin(angle)
            py = source_y + step * math.sin(angle) + cross_offset * math.cos(angle)
            sigma = base_sigma * float(rng.uniform(0.7, 1.8)) + taper_noise * (0.25 + t)
            amp = float((1.0 - 0.25 * t) * rng.uniform(0.55, 1.25))
            nodes.append((px, py, max(1.5, abs(sigma)), amp))

        field = np.zeros((height, width), dtype="float32")
        for idx, (px, py, sigma, amp) in enumerate(nodes):
            along_sigma = sigma * float(rng.uniform(1.0, 2.6))
            dx = xx - px
            dy = yy - py
            local_angle = angle + float(rng.normal(0.0, 0.35))
            along = dx * math.cos(local_angle) + dy * math.sin(local_angle)
            cross = -dx * math.sin(local_angle) + dy * math.cos(local_angle)
            blob = np.exp(-0.5 * (cross / sigma) ** 2) * np.exp(-0.5 * (along / along_sigma) ** 2)
            field += amp * blob.astype("float32")

            if idx > 0 and rng.random() < 0.45:
                px2 = px + float(rng.normal(0.0, 0.10 * width))
                py2 = py + float(rng.normal(0.0, 0.10 * height))
                sigma2 = sigma * float(rng.uniform(0.35, 0.85))
                satellite = np.exp(-0.5 * (((xx - px2) ** 2 + (yy - py2) ** 2) / max(sigma2**2, 1.0)))
                field += float(rng.uniform(0.10, 0.35)) * satellite.astype("float32")

        source_sigma = base_sigma * float(rng.uniform(0.45, 0.95))
        source = np.exp(-0.5 * (((xx - source_x) ** 2 + (yy - source_y) ** 2) / max(source_sigma**2, 1.0)))
        field += float(rng.uniform(0.45, 1.20)) * source.astype("float32")

        turbulence = 1.0 + float(rng.uniform(0.18, 0.55)) * _smooth_noise(int(rng.integers(5, 14)))
        field *= np.clip(turbulence, 0.05, 2.25)

        if rng.random() < 0.70:
            hole_noise = _smooth_noise(int(rng.integers(8, 20)))
            field[hole_noise > float(rng.uniform(1.0, 1.8))] *= float(rng.uniform(0.25, 0.65))

        if rng.random() < 0.55:
            n_wisps = int(rng.integers(1, 5))
            for _ in range(n_wisps):
                px = float(rng.uniform(0.0, width))
                py = float(rng.uniform(0.0, height))
                sx = float(rng.uniform(0.010 * width, 0.045 * width))
                sy = float(rng.uniform(0.025 * height, 0.110 * height))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                dx = xx - px
                dy = yy - py
                along = dx * math.cos(theta) + dy * math.sin(theta)
                cross = -dx * math.sin(theta) + dy * math.cos(theta)
                wisp = np.exp(-0.5 * ((along / max(sy, 1.0)) ** 2 + (cross / max(sx, 1.0)) ** 2))
                field += float(rng.uniform(0.04, 0.18)) * wisp.astype("float32")

        return np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    for _ in range(160):
        field = _plume_from_meandering_path()
        if float(field.max()) <= 0:
            continue
        field /= float(field.max())
        threshold = float(rng.uniform(0.16, 0.34))
        edge_noise = 0.06 * _smooth_noise(int(rng.integers(4, 10)))
        mask = field >= np.clip(threshold + edge_noise, 0.08, 0.50)

        if rng.random() < 0.45 and mask.any():
            # Drop a few weak edge pixels to avoid perfectly smooth contours.
            ragged = _smooth_noise(int(rng.integers(2, 6)))
            edge = (field > threshold * 0.65) & (field < threshold * 1.35)
            mask[edge & (ragged > float(rng.uniform(0.65, 1.15)))] = False

        fraction = float(mask.mean())
        if min_fraction <= fraction <= max_fraction:
            return field.astype("float32"), mask.astype("float32")
        distance = min(abs(fraction - min_fraction), abs(fraction - max_fraction))
        if distance < best_distance:
            best_distance = distance
            best_field = field.astype("float32")
            best_mask = mask.astype("float32")

    # Fallback keeps the original simple generator logic if the irregular
    # generator cannot satisfy the requested area bounds.
    for _ in range(80):
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        center_x = float(rng.uniform(0.15 * width, 0.85 * width))
        center_y = float(rng.uniform(0.15 * height, 0.85 * height))
        length = float(rng.uniform(0.20 * width, 0.75 * width))
        width_sigma = float(rng.uniform(0.035 * width, 0.14 * width))

        dx = xx - center_x
        dy = yy - center_y
        along = dx * math.cos(angle) + dy * math.sin(angle)
        cross = -dx * math.sin(angle) + dy * math.cos(angle)

        plume = np.exp(-0.5 * (cross / max(width_sigma, 1.0)) ** 2)
        downwind = np.exp(-0.5 * ((along - 0.25 * length) / max(0.40 * length, 1.0)) ** 2)
        source_core = np.exp(-0.5 * ((along + 0.08 * length) / max(0.13 * length, 1.0)) ** 2)
        field = plume * np.maximum(downwind, 0.45 * source_core)
        field *= (along > -0.25 * length) & (along < 0.95 * length)

        bend = float(rng.uniform(-0.35, 0.35))
        if abs(bend) > 0.05:
            curved_cross = cross + bend * (along / max(length, 1.0)) ** 2 * width
            field *= np.exp(-0.5 * (curved_cross / max(1.6 * width_sigma, 1.0)) ** 2)

        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        if float(field.max()) <= 0:
            continue
        field /= float(field.max())
        mask = field >= 0.18
        fraction = float(mask.mean())
        if min_fraction <= fraction <= max_fraction:
            return field.astype("float32"), mask.astype("float32")
        distance = min(abs(fraction - min_fraction), abs(fraction - max_fraction))
        if distance < best_distance:
            best_distance = distance
            best_field = field.astype("float32")
            best_mask = mask.astype("float32")

    if best_field is None or best_mask is None:
        field = np.zeros((height, width), dtype="float32")
        mask = np.zeros((height, width), dtype="float32")
        return field, mask
    return best_field, best_mask


def _inject_synthetic_methane(raw: np.ndarray, field: np.ndarray, tau: float) -> np.ndarray:
    injected = raw.copy()
    event_b12 = np.maximum(injected[5], 0.0)
    absorption = np.exp(-float(tau) * field.astype("float32"))
    injected[5] = event_b12 * absorption
    return injected.astype("float32")


def _dilate_binary(mask: np.ndarray, pixels: int) -> np.ndarray:
    mask = mask.astype(bool)
    if pixels <= 0 or not mask.any():
        return mask
    out = mask.copy()
    h, w = out.shape
    for _ in range(pixels):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy : dy + h, dx : dx + w]
        out = grown
    return out


def _read_mask(path: str, *, mode: str = "binary", dilation_pixels: int = 0) -> np.ndarray:
    with rasterio.open(path) as src:
        y = src.read(1, out_dtype="float32")
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    hard = y > 0
    if mode == "binary":
        return hard.astype("float32")
    if mode == "soft":
        dilated = _dilate_binary(hard, dilation_pixels)
        soft = np.zeros_like(y, dtype="float32")
        soft[dilated] = 0.35
        soft[hard] = 1.0
        return soft
    raise ValueError(f"Unknown target mode {mode!r}; expected 'binary' or 'soft'")


class _SegmentationPatchDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        patch_size: int,
        length: int,
        positive_patch_probability: float,
        min_patch_positive_fraction: float,
        max_patch_attempts: int,
        include_physics_proxy: bool,
        target_mode: str,
        target_dilation_pixels: int,
        seed: int,
        input_mode: str | None = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.patch_size = patch_size
        self.length = length
        self.positive_patch_probability = positive_patch_probability
        self.min_patch_positive_fraction = min_patch_positive_fraction
        self.max_patch_attempts = max_patch_attempts
        self.input_mode = input_mode or ("s2_proxy" if include_physics_proxy else "s2")
        self.target_mode = target_mode
        self.target_dilation_pixels = target_dilation_pixels
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _row(self) -> pd.Series:
        weights = self.frame["sample_weight"].astype("float64").to_numpy()
        probs = weights / weights.sum()
        return self.frame.iloc[int(self.rng.choice(len(self.frame), p=probs))]

    def _crop_window(self, mask: np.ndarray) -> tuple[int, int]:
        h, w = mask.shape
        ps = min(self.patch_size, h, w)
        positives = np.argwhere(mask > 0)
        if len(positives) and self.rng.random() < self.positive_patch_probability:
            cy, cx = positives[int(self.rng.integers(0, len(positives)))]
            y0 = int(np.clip(cy - self.rng.integers(0, ps), 0, h - ps))
            x0 = int(np.clip(cx - self.rng.integers(0, ps), 0, w - ps))
            return y0, x0
        y0 = int(self.rng.integers(0, max(1, h - ps + 1)))
        x0 = int(self.rng.integers(0, max(1, w - ps + 1)))
        return y0, x0

    def __getitem__(self, _: int):
        torch, *_ = _require_torch()
        row = None
        x_patch = None
        y_patch = None
        for _attempt in range(max(1, self.max_patch_attempts)):
            row = self._row()
            x = _read_model_input(row["s2_path"], self.input_mode)
            y = _read_mask(
                row["aligned_mask_path"],
                mode=self.target_mode,
                dilation_pixels=self.target_dilation_pixels,
            )
            y0, x0 = self._crop_window(y)
            ps = min(self.patch_size, y.shape[0], y.shape[1])
            candidate_y = y[None, y0 : y0 + ps, x0 : x0 + ps]
            if float((candidate_y > 0).mean()) >= self.min_patch_positive_fraction:
                x_patch = x[:, y0 : y0 + ps, x0 : x0 + ps]
                y_patch = candidate_y
                break
        if x_patch is None or y_patch is None:
            x_patch = x[:, y0 : y0 + ps, x0 : x0 + ps]
            y_patch = candidate_y
        return {
            "x": torch.from_numpy(x_patch),
            "y": torch.from_numpy(y_patch),
            "weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
        }


class _SyntheticPlumePatchDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        patch_size: int,
        length: int,
        input_mode: str,
        min_plume_fraction: float,
        max_plume_fraction: float,
        min_tau: float,
        max_tau: float,
        target_mode: str,
        target_dilation_pixels: int,
        seed: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.patch_size = patch_size
        self.length = length
        self.input_mode = input_mode
        self.min_plume_fraction = min_plume_fraction
        self.max_plume_fraction = max_plume_fraction
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.target_mode = target_mode
        self.target_dilation_pixels = target_dilation_pixels
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _row(self) -> pd.Series:
        weights = self.frame.get("sample_weight", pd.Series(np.ones(len(self.frame)))).astype("float64").to_numpy()
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones(len(self.frame), dtype="float64")
        probs = weights / weights.sum()
        return self.frame.iloc[int(self.rng.choice(len(self.frame), p=probs))]

    def _crop_raw(self, raw: np.ndarray) -> np.ndarray:
        _, h, w = raw.shape
        ps = min(self.patch_size, h, w)
        y0 = int(self.rng.integers(0, max(1, h - ps + 1)))
        x0 = int(self.rng.integers(0, max(1, w - ps + 1)))
        return raw[:, y0 : y0 + ps, x0 : x0 + ps]

    def __getitem__(self, _: int):
        torch, *_ = _require_torch()
        row = self._row()
        raw = self._crop_raw(_read_raw_l1c_pair(row["s2_path"]))
        _, h, w = raw.shape
        field, hard_mask = _synthetic_plume_field(
            h,
            w,
            self.rng,
            min_fraction=self.min_plume_fraction,
            max_fraction=self.max_plume_fraction,
        )
        tau = float(self.rng.uniform(self.min_tau, self.max_tau))
        injected = _inject_synthetic_methane(raw, field, tau)
        if self.input_mode == "l1c_pair_methane":
            x_patch = _methane_pair_feature_stack(injected)
        else:
            raise ValueError("Synthetic plume training currently supports input_mode='l1c_pair_methane'")

        if self.target_mode == "soft":
            dilated = _dilate_binary(hard_mask > 0, self.target_dilation_pixels)
            y_patch = np.zeros_like(hard_mask, dtype="float32")
            y_patch[dilated] = 0.35
            y_patch[hard_mask > 0] = 1.0
        elif self.target_mode == "binary":
            y_patch = hard_mask.astype("float32")
        else:
            raise ValueError("target_mode must be 'binary' or 'soft'")

        return {
            "x": torch.from_numpy(x_patch.astype("float32")),
            "y": torch.from_numpy(y_patch[None].astype("float32")),
            "weight": torch.tensor(float(row.get("sample_weight", 1.0)), dtype=torch.float32),
        }


class _FullTileDataset:
    def __init__(self, frame: pd.DataFrame, *, include_physics_proxy: bool, input_mode: str | None = None):
        self.frame = frame.reset_index(drop=True)
        self.input_mode = input_mode or ("s2_proxy" if include_physics_proxy else "s2")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        torch, *_ = _require_torch()
        row = self.frame.iloc[idx]
        return {
            "x": torch.from_numpy(_read_model_input(row["s2_path"], self.input_mode)),
            "y": torch.from_numpy(_read_mask(row["aligned_mask_path"])[None]),
            "weight": torch.tensor(float(row["sample_weight"]), dtype=torch.float32),
            "split": row["split"],
            "source": row["source"],
            "file_prefix": row["file_prefix"],
        }


def _conv_block(nn, in_ch: int, out_ch: int):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_block(nn, in_ch: int, out_ch: int):
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)


def build_model(model_name: str, in_channels: int):
    torch, nn, F, *_ = _require_torch()

    class UNet(nn.Module):
        def __init__(self, attention: bool = False):
            super().__init__()
            widths = [32, 64, 128, 256]
            self.e1 = _conv_block(nn, in_channels, widths[0])
            self.e2 = _conv_block(nn, widths[0], widths[1])
            self.e3 = _conv_block(nn, widths[1], widths[2])
            self.b = _conv_block(nn, widths[2], widths[3])
            self.pool = nn.MaxPool2d(2)
            self.u3 = _up_block(nn, widths[3], widths[2])
            self.d3 = _conv_block(nn, widths[2] * 2, widths[2])
            self.u2 = _up_block(nn, widths[2], widths[1])
            self.d2 = _conv_block(nn, widths[1] * 2, widths[1])
            self.u1 = _up_block(nn, widths[1], widths[0])
            self.d1 = _conv_block(nn, widths[0] * 2, widths[0])
            self.out = nn.Conv2d(widths[0], 1, 1)
            self.attention = attention
            if attention:
                self.g3 = nn.Sequential(nn.Conv2d(widths[2] * 2, widths[2], 1), nn.Sigmoid())
                self.g2 = nn.Sequential(nn.Conv2d(widths[1] * 2, widths[1], 1), nn.Sigmoid())
                self.g1 = nn.Sequential(nn.Conv2d(widths[0] * 2, widths[0], 1), nn.Sigmoid())

        def _gate(self, gate, skip, up):
            if not self.attention:
                return skip
            return skip * gate(torch.cat([skip, up], dim=1))

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            b = self.b(self.pool(e3))
            u3 = self.u3(b)
            d3 = self.d3(torch.cat([u3, self._gate(self.g3 if self.attention else None, e3, u3)], dim=1))
            u2 = self.u2(d3)
            d2 = self.d2(torch.cat([u2, self._gate(self.g2 if self.attention else None, e2, u2)], dim=1))
            u1 = self.u1(d2)
            d1 = self.d1(torch.cat([u1, self._gate(self.g1 if self.attention else None, e1, u1)], dim=1))
            return self.out(d1)

    class UNetPP(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = UNet(attention=False)
            self.refine = nn.Sequential(_conv_block(nn, 1 + in_channels, 16), nn.Conv2d(16, 1, 1))

        def forward(self, x):
            coarse = self.base(x)
            return coarse + self.refine(torch.cat([x, torch.sigmoid(coarse)], dim=1))

    class DeepLabV3P(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = _conv_block(nn, in_channels, 32)
            self.low = _conv_block(nn, 32, 48)
            self.enc = nn.Sequential(nn.MaxPool2d(2), _conv_block(nn, 48, 96), nn.MaxPool2d(2), _conv_block(nn, 96, 192))
            self.aspp = nn.ModuleList([nn.Conv2d(192, 64, 3, padding=d, dilation=d) for d in (1, 2, 4, 8)])
            self.dec = _conv_block(nn, 64 * 4 + 48, 64)
            self.out = nn.Conv2d(64, 1, 1)

        def forward(self, x):
            s = self.stem(x)
            low = self.low(s)
            enc = self.enc(low)
            z = torch.cat([F.relu(layer(enc), inplace=True) for layer in self.aspp], dim=1)
            z = F.interpolate(z, size=low.shape[-2:], mode="bilinear", align_corners=False)
            z = self.dec(torch.cat([z, low], dim=1))
            z = F.interpolate(z, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.out(z)

    class PhysTAUNet(nn.Module):
        def __init__(self):
            super().__init__()
            if in_channels < 8:
                raise ValueError("phys_tau_net requires methane feature inputs; use input_mode='l1c_pair_methane'")

            self.surface_channels = min(6, in_channels)
            self.physics_start = 2 if in_channels > 8 else 0
            physics_channels = in_channels - self.physics_start
            widths = [32, 64, 128, 256]
            phys_widths = [16, 32, 64, 128]

            self.surface_e1 = _conv_block(nn, self.surface_channels, widths[0])
            self.surface_e2 = _conv_block(nn, widths[0], widths[1])
            self.surface_e3 = _conv_block(nn, widths[1], widths[2])
            self.surface_b = _conv_block(nn, widths[2], widths[3])

            self.physics_e1 = _conv_block(nn, physics_channels, phys_widths[0])
            self.physics_e2 = _conv_block(nn, phys_widths[0], phys_widths[1])
            self.physics_e3 = _conv_block(nn, phys_widths[1], phys_widths[2])
            self.physics_b = _conv_block(nn, phys_widths[2], phys_widths[3])

            self.pool = nn.MaxPool2d(2)
            self.g1 = nn.Sequential(nn.Conv2d(phys_widths[0], widths[0], 1), nn.Sigmoid())
            self.g2 = nn.Sequential(nn.Conv2d(phys_widths[1], widths[1], 1), nn.Sigmoid())
            self.g3 = nn.Sequential(nn.Conv2d(phys_widths[2], widths[2], 1), nn.Sigmoid())
            self.gb = nn.Sequential(nn.Conv2d(phys_widths[3], widths[3], 1), nn.Sigmoid())

            self.b_fuse = _conv_block(nn, widths[3] + phys_widths[3], widths[3])
            self.u3 = _up_block(nn, widths[3], widths[2])
            self.d3 = _conv_block(nn, widths[2] * 2, widths[2])
            self.u2 = _up_block(nn, widths[2], widths[1])
            self.d2 = _conv_block(nn, widths[1] * 2, widths[1])
            self.u1 = _up_block(nn, widths[1], widths[0])
            self.d1 = _conv_block(nn, widths[0] * 2, widths[0])
            self.tau_head = nn.Sequential(
                nn.Conv2d(phys_widths[0], 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )
            self.out = nn.Conv2d(widths[0], 1, 1)

        @staticmethod
        def _physics_gate(skip, gate):
            return skip * (0.5 + gate)

        def forward(self, x):
            surface = x[:, : self.surface_channels]
            physics = x[:, self.physics_start :]

            s1 = self.surface_e1(surface)
            p1 = self.physics_e1(physics)
            s2 = self.surface_e2(self.pool(s1))
            p2 = self.physics_e2(self.pool(p1))
            s3 = self.surface_e3(self.pool(s2))
            p3 = self.physics_e3(self.pool(p2))
            sb = self.surface_b(self.pool(s3))
            pb = self.physics_b(self.pool(p3))

            b = self.b_fuse(torch.cat([self._physics_gate(sb, self.gb(pb)), pb], dim=1))
            u3 = self.u3(b)
            d3 = self.d3(torch.cat([u3, self._physics_gate(s3, self.g3(p3))], dim=1))
            u2 = self.u2(d3)
            d2 = self.d2(torch.cat([u2, self._physics_gate(s2, self.g2(p2))], dim=1))
            u1 = self.u1(d2)
            d1 = self.d1(torch.cat([u1, self._physics_gate(s1, self.g1(p1))], dim=1))

            # The auxiliary tau-like head exposes the methane branch to the final logit
            # without adding a second supervised target.
            return self.out(d1) + 0.25 * self.tau_head(p1)

    if model_name == "unet":
        return UNet(attention=False)
    if model_name == "attn_unet":
        return UNet(attention=True)
    if model_name == "unet_pp":
        return UNetPP()
    if model_name == "deeplabv3p":
        return DeepLabV3P()
    if model_name == "phys_tau_net":
        return PhysTAUNet()
    raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")


def _loss(logits, target, sample_weight):
    torch, *_ = _require_torch()
    positive_pixels = target.sum(dim=(1, 2, 3)).clamp_min(1.0)
    negative_pixels = (1.0 - target).sum(dim=(1, 2, 3)).clamp_min(1.0)
    pos_weight = torch.sqrt(negative_pixels / positive_pixels).clamp(1.0, 35.0).view(-1, 1, 1, 1)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    bce = bce.mean(dim=(1, 2, 3))
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return ((bce + dice) * sample_weight).mean()


def _metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": accuracy}


def _tolerance_counts(pred: np.ndarray, target: np.ndarray, tolerance_pixels: int) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    target_tol = _dilate_binary(target, tolerance_pixels)
    pred_tol = _dilate_binary(pred, tolerance_pixels)
    tp = float(np.logical_and(pred, target_tol).sum())
    fp = float(np.logical_and(pred, ~target_tol).sum())
    fn = float(np.logical_and(target, ~pred_tol).sum())
    tn = float(np.logical_and(~pred, ~target).sum())
    return tp, fp, fn, tn


def _evaluate_frame(
    model,
    frame: pd.DataFrame,
    *,
    device,
    threshold: float,
    include_physics_proxy: bool,
    input_mode: str | None = None,
    tolerance_pixels: int = 0,
) -> pd.DataFrame:
    torch, *_ = _require_torch()
    model.eval()
    rows = []
    resolved_input_mode = input_mode or ("s2_proxy" if include_physics_proxy else "s2")
    with torch.no_grad():
        for _, row in frame.iterrows():
            x = torch.from_numpy(_read_model_input(row["s2_path"], resolved_input_mode))[None].to(device)
            y_np = _read_mask(row["aligned_mask_path"], mode="binary").astype(bool)
            prob = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            pred = prob >= threshold
            tp = float(np.logical_and(pred, y_np).sum())
            fp = float(np.logical_and(pred, ~y_np).sum())
            fn = float(np.logical_and(~pred, y_np).sum())
            tn = float(np.logical_and(~pred, ~y_np).sum())
            tol_tp, tol_fp, tol_fn, tol_tn = _tolerance_counts(pred, y_np, tolerance_pixels)
            rec = {
                "split": row["split"],
                "source": row["source"],
                "file_prefix": row["file_prefix"],
                "threshold": threshold,
                "positive_fraction": float(y_np.mean()),
                "predicted_positive_fraction": float(pred.mean()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tol_tp": tol_tp,
                "tol_fp": tol_fp,
                "tol_fn": tol_fn,
                "tol_tn": tol_tn,
            }
            rec.update(_metrics_from_counts(tp, fp, fn, tn))
            tolerant = _metrics_from_counts(tol_tp, tol_fp, tol_fn, tol_tn)
            rec.update({f"tolerant_{key}": value for key, value in tolerant.items()})
            rows.append(rec)
    return pd.DataFrame(rows)


def _summarize_eval(file_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def summarize(group: pd.DataFrame) -> pd.Series:
        counts = group[["tp", "fp", "fn", "tn"]].sum()
        tolerant_counts = group[["tol_tp", "tol_fp", "tol_fn", "tol_tn"]].sum()
        metrics = _metrics_from_counts(float(counts.tp), float(counts.fp), float(counts.fn), float(counts.tn))
        tolerant = _metrics_from_counts(
            float(tolerant_counts.tol_tp),
            float(tolerant_counts.tol_fp),
            float(tolerant_counts.tol_fn),
            float(tolerant_counts.tol_tn),
        )
        return pd.Series(
            {
                "files": len(group),
                "positive_fraction_mean": group["positive_fraction"].mean(),
                "predicted_positive_fraction_mean": group["predicted_positive_fraction"].mean(),
                **metrics,
                **{f"tolerant_{key}": value for key, value in tolerant.items()},
            }
        )

    by_split = file_df.groupby(["split", "source"], dropna=False).apply(summarize, include_groups=False).reset_index()
    by_source = file_df.groupby(["source"], dropna=False).apply(summarize, include_groups=False).reset_index()
    return by_split, by_source


def _evaluate_synthetic_loader(model, loader, *, device, threshold: float, tolerance_pixels: int) -> dict[str, float]:
    torch, *_ = _require_torch()
    model.eval()
    totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0, "tol_tp": 0.0, "tol_fp": 0.0, "tol_fn": 0.0, "tol_tn": 0.0}
    positive_fractions = []
    predicted_fractions = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            prob = torch.sigmoid(model(x)).detach().cpu().numpy()[:, 0]
            target = (y.detach().cpu().numpy()[:, 0] > 0.5)
            pred = prob >= threshold
            for pred_i, target_i in zip(pred, target):
                totals["tp"] += float(np.logical_and(pred_i, target_i).sum())
                totals["fp"] += float(np.logical_and(pred_i, ~target_i).sum())
                totals["fn"] += float(np.logical_and(~pred_i, target_i).sum())
                totals["tn"] += float(np.logical_and(~pred_i, ~target_i).sum())
                tol_tp, tol_fp, tol_fn, tol_tn = _tolerance_counts(pred_i, target_i, tolerance_pixels)
                totals["tol_tp"] += tol_tp
                totals["tol_fp"] += tol_fp
                totals["tol_fn"] += tol_fn
                totals["tol_tn"] += tol_tn
                positive_fractions.append(float(target_i.mean()))
                predicted_fractions.append(float(pred_i.mean()))

    metrics = _metrics_from_counts(totals["tp"], totals["fp"], totals["fn"], totals["tn"])
    tolerant = _metrics_from_counts(totals["tol_tp"], totals["tol_fp"], totals["tol_fn"], totals["tol_tn"])
    return {
        **metrics,
        **{f"tolerant_{key}": value for key, value in tolerant.items()},
        "positive_fraction_mean": float(np.mean(positive_fractions)) if positive_fractions else 0.0,
        "predicted_positive_fraction_mean": float(np.mean(predicted_fractions)) if predicted_fractions else 0.0,
    }


def _load_manifest(
    path: Path,
    carbon_mapper_only: bool = False,
    source: str | None = None,
    min_positive_pixels: int | None = None,
    max_positive_fraction: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "usable_for_training" in df:
        df = df[df["usable_for_training"].astype(bool)]
    if carbon_mapper_only:
        df = df[df["source"].astype(str) == "carbon_mapper"]
    if source:
        df = df[df["source"].astype(str) == source]
    if min_positive_pixels is not None:
        df = df[df["mask_positive_pixels"] >= min_positive_pixels]
    if max_positive_fraction is not None:
        df = df[df["mask_positive_fraction"] <= max_positive_fraction]
    return df.reset_index(drop=True)


def audit_training_input_ranges(config: InputAuditConfig = InputAuditConfig()) -> Path:
    df = _load_manifest(config.manifest, carbon_mapper_only=False)
    if config.limit is not None:
        df = df.head(config.limit)

    rows = []
    for _, row in df.iterrows():
        try:
            x = _read_model_input(row["s2_path"], "s2_plus_methane")
            y = _read_mask(row["aligned_mask_path"])
            rec = {
                "split": row["split"],
                "source": row["source"],
                "file_prefix": row["file_prefix"],
                "s2_path": row["s2_path"],
                "aligned_mask_path": row["aligned_mask_path"],
                "x_finite": bool(np.isfinite(x).all()),
                "y_finite": bool(np.isfinite(y).all()),
                "x_min": float(np.min(x)),
                "x_max": float(np.max(x)),
                "y_min": float(np.min(y)),
                "y_max": float(np.max(y)),
                "error": "",
            }
        except Exception as exc:
            rec = {
                "split": row.get("split"),
                "source": row.get("source"),
                "file_prefix": row.get("file_prefix"),
                "s2_path": row.get("s2_path"),
                "aligned_mask_path": row.get("aligned_mask_path"),
                "x_finite": False,
                "y_finite": False,
                "x_min": np.nan,
                "x_max": np.nan,
                "y_min": np.nan,
                "y_max": np.nan,
                "error": repr(exc),
            }
        rows.append(rec)

    out = pd.DataFrame(rows)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out, index=False)
    print("\nFinite status:")
    print(out[["x_finite", "y_finite"]].value_counts(dropna=False).to_string())
    if "error" in out:
        errors = out.loc[out["error"].fillna("") != "", "error"].value_counts().head(10)
        print("\nTop errors:")
        print(errors.to_string() if len(errors) else "none")
    print("\nInput range summary:")
    print(out[["x_min", "x_max", "y_min", "y_max"]].describe().to_string())
    return config.out


def audit_label_size_feasibility(config: LabelSizeAuditConfig = LabelSizeAuditConfig()) -> Path:
    df = pd.read_csv(config.manifest)
    if "usable_for_training" in df:
        df = df[df["usable_for_training"].astype(bool)].copy()

    rows = []
    for split in sorted(df["split"].astype(str).unique()):
        for source in sorted(df["source"].astype(str).unique()):
            group = df[(df["split"].astype(str) == split) & (df["source"].astype(str) == source)]
            if group.empty:
                continue
            base = {
                "split": split,
                "source": source,
                "rows": int(len(group)),
                "median_positive_pixels": float(group["mask_positive_pixels"].median()),
                "mean_positive_pixels": float(group["mask_positive_pixels"].mean()),
                "median_positive_fraction": float(group["mask_positive_fraction"].median()),
                "mean_positive_fraction": float(group["mask_positive_fraction"].mean()),
            }
            for cutoff in config.pixel_cutoffs:
                kept = int((group["mask_positive_pixels"] >= cutoff).sum())
                base[f"rows_ge_{cutoff}px"] = kept
                base[f"fraction_ge_{cutoff}px"] = kept / max(len(group), 1)
            rows.append(base)

    out = pd.DataFrame(rows)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out, index=False)
    print(out.to_string(index=False))
    return config.out


def audit_crop_sizing(config: CropSizingAuditConfig = CropSizingAuditConfig()) -> Path:
    df = pd.read_csv(config.manifest)
    if "usable_for_training" in df:
        df = df[df["usable_for_training"].astype(bool)].copy()

    rows = []
    for _, row in df.iterrows():
        positive_pixels = float(row["mask_positive_pixels"])
        required_crop = math.sqrt(positive_pixels / config.desired_fraction) if positive_pixels > 0 else math.nan
        rec = {
            "split": row["split"],
            "source": row["source"],
            "file_prefix": row["file_prefix"],
            "mask_positive_pixels": positive_pixels,
            "positive_fraction_512": float(row["mask_positive_fraction"]),
            "crop_px_for_desired_fraction": required_crop,
        }
        for crop in config.crop_sizes:
            rec[f"max_fraction_if_{crop}px_crop"] = positive_pixels / float(crop * crop)
        rows.append(rec)

    out = pd.DataFrame(rows)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out, index=False)

    print("\nPlume pixels by source:")
    print(
        out.groupby("source")
        .agg(
            rows=("mask_positive_pixels", "size"),
            min_pixels=("mask_positive_pixels", "min"),
            p25_pixels=("mask_positive_pixels", lambda s: float(np.percentile(s, 25))),
            median_pixels=("mask_positive_pixels", "median"),
            p75_pixels=("mask_positive_pixels", lambda s: float(np.percentile(s, 75))),
            max_pixels=("mask_positive_pixels", "max"),
            median_crop_for_30pct=("crop_px_for_desired_fraction", "median"),
        )
        .to_string()
    )
    print("\nMaximum possible positive fraction if crop contains the full plume:")
    keep_cols = ["source"] + [f"max_fraction_if_{crop}px_crop" for crop in config.crop_sizes]
    print(out[keep_cols].groupby("source").median().to_string())
    return config.out


def train_segmentation_model(config: TrainConfig = TrainConfig()) -> Path:
    torch, nn, F, DataLoader, Dataset = _require_torch()
    if config.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {config.model!r}; expected one of {MODEL_NAMES}")
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    input_mode = _resolve_input_mode(config.input_mode, config.model)
    include_proxy = input_mode == "s2_proxy"

    df = _load_manifest(
        config.manifest,
        carbon_mapper_only=config.carbon_mapper_only,
        source=config.train_source,
        min_positive_pixels=config.min_train_positive_pixels,
        max_positive_fraction=config.max_train_positive_fraction,
    )
    train_df = df[df["split"].astype(str) == "TRAIN"].copy()
    val_df = df[
        (df["split"].astype(str) == "VAL")
        & (df["mask_positive_pixels"] >= config.min_eval_positive_pixels)
        & (df["mask_positive_fraction"] <= config.max_eval_positive_fraction)
    ].copy()
    if config.limit_train is not None:
        train_df = train_df.head(config.limit_train)
    if config.limit_val is not None:
        val_df = val_df.head(config.limit_val)
    if train_df.empty:
        raise ValueError("No training rows available")
    if val_df.empty:
        raise ValueError("No validation rows available")

    out_dir = config.out_dir / config.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")

    train_ds = _SegmentationPatchDataset(
        train_df,
        patch_size=config.patch_size,
        length=config.patches_per_epoch,
        positive_patch_probability=config.positive_patch_probability,
        min_patch_positive_fraction=config.min_patch_positive_fraction,
        max_patch_attempts=config.max_patch_attempts,
        include_physics_proxy=include_proxy,
        target_mode=config.target_mode,
        target_dilation_pixels=config.target_dilation_pixels,
        seed=config.seed,
        input_mode=input_mode,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config.model, in_channels=_input_channels(input_mode)).to(device)
    if config.init_checkpoint is not None:
        init_checkpoint = torch.load(config.init_checkpoint, map_location="cpu")
        init_config = init_checkpoint.get("config", {})
        init_model = init_config.get("model", config.model)
        init_input_mode = init_config.get("resolved_input_mode") or _resolve_input_mode(
            init_config.get("input_mode", input_mode),
            init_model,
        )
        if init_model != config.model:
            raise ValueError(f"Init checkpoint model {init_model!r} does not match requested model {config.model!r}")
        if init_input_mode != input_mode:
            raise ValueError(
                f"Init checkpoint input mode {init_input_mode!r} does not match requested input mode {input_mode!r}"
            )
        model.load_state_dict(init_checkpoint["model"])
        print(f"Initialized {config.model} from {config.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_f1 = -1.0
    history = []
    best_path = out_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            w = batch["weight"].to(device=device, dtype=torch.float32)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(w).all():
                raise ValueError("Non-finite input batch detected after preprocessing")
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x), y, w)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss detected; stop and inspect input ranges")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        eval_df = _evaluate_frame(
            model,
            val_df,
            device=device,
            threshold=0.5,
            include_physics_proxy=include_proxy,
            input_mode=input_mode,
            tolerance_pixels=config.target_dilation_pixels,
        )
        counts = eval_df[["tp", "fp", "fn", "tn"]].sum()
        tolerant_counts = eval_df[["tol_tp", "tol_fp", "tol_fn", "tol_tn"]].sum()
        metrics = _metrics_from_counts(float(counts.tp), float(counts.fp), float(counts.fn), float(counts.tn))
        tolerant = _metrics_from_counts(
            float(tolerant_counts.tol_tp),
            float(tolerant_counts.tol_fp),
            float(tolerant_counts.tol_fn),
            float(tolerant_counts.tol_tn),
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
            **{f"tolerant_{key}": value for key, value in tolerant.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_f1={row['f1']:.4f} val_iou={row['iou']:.4f} "
            f"val_tol_f1={row['tolerant_f1']:.4f} "
            f"val_precision={row['precision']:.4f} val_recall={row['recall']:.4f}"
        )
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            ckpt_config = asdict(config)
            ckpt_config["resolved_input_mode"] = input_mode
            torch.save({"model": model.state_dict(), "config": ckpt_config, "best_f1": best_f1}, best_path)

    print(f"Wrote best checkpoint {best_path}")
    return best_path


def train_synthetic_segmentation_model(config: SyntheticTrainConfig = SyntheticTrainConfig()) -> Path:
    torch, nn, F, DataLoader, Dataset = _require_torch()
    if config.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {config.model!r}; expected one of {MODEL_NAMES}")
    if config.input_mode != "l1c_pair_methane":
        raise ValueError("Synthetic L1C plume training currently requires --input-mode l1c_pair_methane")
    if not 0 < config.min_plume_fraction <= config.max_plume_fraction < 1:
        raise ValueError("Expected 0 < min_plume_fraction <= max_plume_fraction < 1")
    if not 0 < config.min_tau <= config.max_tau:
        raise ValueError("Expected 0 < min_tau <= max_tau")

    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    input_mode = _resolve_input_mode(config.input_mode, config.model)

    df = _load_manifest(config.manifest, carbon_mapper_only=config.carbon_mapper_backgrounds_only)
    if "pair_ok" in df:
        df = df[df["pair_ok"].astype(bool)].copy()
    df = df[df["s2_path"].astype(str).str.len() > 0].copy()
    train_df = df[df["split"].astype(str) == "TRAIN"].copy()
    val_df = df[df["split"].astype(str) == "VAL"].copy()
    if config.limit_train is not None:
        train_df = train_df.head(config.limit_train)
    if config.limit_val is not None:
        val_df = val_df.head(config.limit_val)
    if train_df.empty:
        raise ValueError("No training background rows available")
    if val_df.empty:
        raise ValueError("No validation background rows available")

    out_dir = config.out_dir / config.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(config), indent=2, default=str), encoding="utf-8")

    train_ds = _SyntheticPlumePatchDataset(
        train_df,
        patch_size=config.patch_size,
        length=config.patches_per_epoch,
        input_mode=input_mode,
        min_plume_fraction=config.min_plume_fraction,
        max_plume_fraction=config.max_plume_fraction,
        min_tau=config.min_tau,
        max_tau=config.max_tau,
        target_mode=config.target_mode,
        target_dilation_pixels=config.target_dilation_pixels,
        seed=config.seed,
    )
    val_ds = _SyntheticPlumePatchDataset(
        val_df,
        patch_size=config.patch_size,
        length=config.validation_patches,
        input_mode=input_mode,
        min_plume_fraction=config.min_plume_fraction,
        max_plume_fraction=config.max_plume_fraction,
        min_tau=config.min_tau,
        max_tau=config.max_tau,
        target_mode="binary",
        target_dilation_pixels=0,
        seed=config.seed + 10_000,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.model, in_channels=_input_channels(input_mode)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            y = batch["y"].to(device=device, dtype=torch.float32)
            w = batch["weight"].to(device=device, dtype=torch.float32)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.isfinite(w).all():
                raise ValueError("Non-finite synthetic input batch detected")
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(x), y, w)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite synthetic training loss detected")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        metrics = _evaluate_synthetic_loader(
            model,
            val_loader,
            device=device,
            threshold=0.5,
            tolerance_pixels=config.target_dilation_pixels,
        )
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"synthetic_val_f1={row['f1']:.4f} synthetic_val_iou={row['iou']:.4f} "
            f"synthetic_val_tol_f1={row['tolerant_f1']:.4f} "
            f"pred_pos={row['predicted_positive_fraction_mean']:.4f}"
        )
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            ckpt_config = asdict(config)
            ckpt_config["resolved_input_mode"] = input_mode
            ckpt_config["synthetic_training"] = True
            torch.save({"model": model.state_dict(), "config": ckpt_config, "best_f1": best_f1}, best_path)

    print(f"Wrote best checkpoint {best_path}")
    return best_path


def evaluate_segmentation_model(config: EvalConfig = EvalConfig()) -> tuple[Path, Path, Path]:
    torch, *_ = _require_torch()
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = checkpoint.get("config", {})
    model_name = train_config.get("model", config.checkpoint.parent.name)
    input_mode = train_config.get("resolved_input_mode") or _resolve_input_mode(train_config.get("input_mode", "auto"), model_name)
    include_proxy = input_mode == "s2_proxy"
    device = _resolve_device(config.device)
    model = build_model(model_name, in_channels=_input_channels(input_mode)).to(device)
    model.load_state_dict(checkpoint["model"])

    df = _load_manifest(
        config.manifest,
        carbon_mapper_only=False,
        min_positive_pixels=config.min_positive_pixels,
        max_positive_fraction=config.max_positive_fraction,
    )
    if config.split:
        df = df[df["split"].astype(str) == config.split]
    if config.source:
        df = df[df["source"].astype(str) == config.source]
    if config.limit is not None:
        df = df.head(config.limit)
    if df.empty:
        raise ValueError("No evaluation rows available")

    config.out_dir.mkdir(parents=True, exist_ok=True)
    file_df = _evaluate_frame(
        model,
        df,
        device=device,
        threshold=config.threshold,
        include_physics_proxy=include_proxy,
        input_mode=input_mode,
        tolerance_pixels=config.tolerance_pixels,
    )
    by_split, by_source = _summarize_eval(file_df)
    file_path = config.out_dir / "eval_by_file.csv"
    split_path = config.out_dir / "eval_by_split_source.csv"
    source_path = config.out_dir / "eval_by_source.csv"
    file_df.to_csv(file_path, index=False)
    by_split.to_csv(split_path, index=False)
    by_source.to_csv(source_path, index=False)
    print("\nBy split/source:")
    print(by_split.to_string(index=False))
    print("\nBy source:")
    print(by_source.to_string(index=False))
    return file_path, split_path, source_path


def sweep_segmentation_thresholds(config: ThresholdSweepConfig = ThresholdSweepConfig()) -> Path:
    torch, *_ = _require_torch()
    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    train_config = checkpoint.get("config", {})
    model_name = train_config.get("model", config.checkpoint.parent.name)
    input_mode = train_config.get("resolved_input_mode") or _resolve_input_mode(train_config.get("input_mode", "auto"), model_name)
    include_proxy = input_mode == "s2_proxy"
    device = _resolve_device(config.device)
    model = build_model(model_name, in_channels=_input_channels(input_mode)).to(device)
    model.load_state_dict(checkpoint["model"])

    df = _load_manifest(
        config.manifest,
        carbon_mapper_only=False,
        min_positive_pixels=config.min_positive_pixels,
        max_positive_fraction=config.max_positive_fraction,
    )
    df = df[df["split"].astype(str) == config.split]
    if config.source:
        df = df[df["source"].astype(str) == config.source]
    if config.limit is not None:
        df = df.head(config.limit)
    if df.empty:
        raise ValueError("No threshold-sweep rows available")

    rows = []
    for threshold in config.thresholds:
        file_df = _evaluate_frame(
            model,
            df,
            device=device,
            threshold=float(threshold),
            include_physics_proxy=include_proxy,
            input_mode=input_mode,
            tolerance_pixels=config.tolerance_pixels,
        )
        counts = file_df[["tp", "fp", "fn", "tn"]].sum()
        tolerant_counts = file_df[["tol_tp", "tol_fp", "tol_fn", "tol_tn"]].sum()
        metrics = _metrics_from_counts(float(counts.tp), float(counts.fp), float(counts.fn), float(counts.tn))
        tolerant = _metrics_from_counts(
            float(tolerant_counts.tol_tp),
            float(tolerant_counts.tol_fp),
            float(tolerant_counts.tol_fn),
            float(tolerant_counts.tol_tn),
        )
        rows.append(
            {
                "model": model_name,
                "split": config.split,
                "source": config.source or "all",
                "threshold": float(threshold),
                "files": int(len(file_df)),
                "positive_fraction_mean": float(file_df["positive_fraction"].mean()),
                "predicted_positive_fraction_mean": float(file_df["predicted_positive_fraction"].mean()),
                **metrics,
                **{f"tolerant_{key}": value for key, value in tolerant.items()},
            }
        )

    out = pd.DataFrame(rows).sort_values(["tolerant_f1", "f1", "iou"], ascending=False)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out, index=False)
    print(out.to_string(index=False))
    best = out.iloc[0]
    print(
        f"\nBest threshold={best['threshold']:.3f} "
        f"f1={best['f1']:.4f} tolerant_f1={best['tolerant_f1']:.4f} "
        f"iou={best['iou']:.4f} "
        f"precision={best['precision']:.4f} recall={best['recall']:.4f}"
    )
    return config.out
