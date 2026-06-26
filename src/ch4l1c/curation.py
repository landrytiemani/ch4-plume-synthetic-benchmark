from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG


@dataclass(frozen=True)
class TrainingCurationConfig:
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_manifest.csv"
    out: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv"
    min_positive_pixels: int = 5
    tiny_positive_pixels: int = 50
    very_large_fraction: float = 0.05
    extreme_fraction: float = 0.10
    carbon_mapper_weight: float = 1.0
    emit_weight: float = 0.45
    tiny_weight_multiplier: float = 0.35
    very_large_emit_multiplier: float = 0.45


def _mask_size_class(positive_pixels: float, positive_fraction: float, cfg: TrainingCurationConfig) -> str:
    if positive_pixels <= 0:
        return "zero"
    if positive_pixels < cfg.tiny_positive_pixels:
        return "tiny"
    if positive_fraction >= cfg.extreme_fraction:
        return "extreme_large"
    if positive_fraction >= cfg.very_large_fraction:
        return "very_large"
    if positive_fraction >= 0.01:
        return "large"
    if positive_fraction >= 0.001:
        return "medium"
    return "small"


def _label_quality(source: str, size_class: str) -> str:
    if size_class == "zero":
        return "reject"
    if source == "carbon_mapper":
        return "review" if size_class == "extreme_large" else "high"
    if source == "emit":
        return "review" if size_class in {"very_large", "extreme_large"} else "medium"
    return "review"


def curate_training_manifest(config: TrainingCurationConfig = TrainingCurationConfig()) -> Path:
    df = pd.read_csv(config.manifest)
    if "pair_ok" not in df:
        raise ValueError(f"{config.manifest} is missing required column 'pair_ok'")

    curated = df.copy()
    curated["mask_size_class"] = [
        _mask_size_class(px, frac, config)
        for px, frac in zip(curated["mask_positive_pixels"], curated["mask_positive_fraction"])
    ]
    curated["label_quality"] = [
        _label_quality(str(source), str(size_class))
        for source, size_class in zip(curated["source"], curated["mask_size_class"])
    ]

    source_weight = np.where(
        curated["source"].astype(str) == "carbon_mapper",
        config.carbon_mapper_weight,
        config.emit_weight,
    ).astype("float32")
    source_weight = np.where(
        curated["mask_size_class"].astype(str) == "tiny",
        source_weight * config.tiny_weight_multiplier,
        source_weight,
    )
    source_weight = np.where(
        (curated["source"].astype(str) == "emit")
        & curated["mask_size_class"].astype(str).isin(["very_large", "extreme_large"]),
        source_weight * config.very_large_emit_multiplier,
        source_weight,
    )

    curated["sample_weight"] = source_weight.round(6)
    curated["usable_for_training"] = (
        curated["pair_ok"].astype(bool)
        & (curated["mask_positive_pixels"] >= config.min_positive_pixels)
        & (curated["label_quality"] != "reject")
    )
    curated["recommended_role"] = np.where(
        curated["source"].astype(str) == "carbon_mapper",
        "primary_supervision",
        "auxiliary_supervision",
    )
    curated["review_reason"] = ""
    curated.loc[curated["mask_size_class"] == "tiny", "review_reason"] = "tiny plume mask"
    curated.loc[
        (curated["source"].astype(str) == "emit")
        & curated["mask_size_class"].astype(str).isin(["very_large", "extreme_large"]),
        "review_reason",
    ] = "large EMIT mask; use lower weight"
    curated.loc[~curated["pair_ok"].astype(bool), "review_reason"] = "invalid S2/mask pair"

    config.out.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(config.out, index=False)
    return config.out


def audit_curated_training_manifest(
    manifest: Path = CFG.data_dir / "training" / "segmentation_training_curated.csv",
) -> Path:
    df = pd.read_csv(manifest)
    out_path = manifest.parent / "segmentation_training_curated_audit.csv"

    print("\nUsable for training:")
    print(df["usable_for_training"].value_counts(dropna=False).to_string())
    print("\nBy split/source:")
    print(
        df.groupby(["split", "source"], dropna=False)
        .agg(
            rows=("usable_for_training", "size"),
            usable=("usable_for_training", "sum"),
            median_positive_fraction=("mask_positive_fraction", "median"),
            mean_positive_fraction=("mask_positive_fraction", "mean"),
            median_sample_weight=("sample_weight", "median"),
            mean_sample_weight=("sample_weight", "mean"),
        )
        .to_string()
    )
    print("\nMask size classes:")
    print(pd.crosstab(df["source"], df["mask_size_class"]).to_string())
    print("\nLabel quality:")
    print(pd.crosstab(df["source"], df["label_quality"]).to_string())

    summary = (
        df.groupby(["split", "source", "mask_size_class", "label_quality"], dropna=False)
        .agg(
            rows=("usable_for_training", "size"),
            usable=("usable_for_training", "sum"),
            median_positive_fraction=("mask_positive_fraction", "median"),
            mean_positive_fraction=("mask_positive_fraction", "mean"),
            mean_sample_weight=("sample_weight", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_path, index=False)
    return out_path
