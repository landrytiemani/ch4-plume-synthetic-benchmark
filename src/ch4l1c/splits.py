"""Build a spatial-block TRAIN/VAL/TEST split catalog from the S2 match catalog.

Spatial blocking prevents leakage: all plumes within the same 0.25-degree
geographic block are assigned to the same split.  Blocks are assigned greedily
to hit the 70/15/15 target fractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CFG


@dataclass(frozen=True)
class SplitConfig:
    # Input: the S2 L1C reference match catalog produced by sentinel2_l1c.py
    match_catalog: Path = CFG.data_dir / "raw" / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"
    # Output: the split catalog consumed by sentinel2_l1c.py chip export and dataset.py
    out_path: Path = CFG.splits_dir / "segmentation_split_catalog.csv"
    block_degrees: float = 0.25
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: str = "ch4syn-v1"


def _stable_score(text: str) -> int:
    return int(md5(text.encode("utf-8")).hexdigest()[:12], 16)


def _spatial_block(lon: float, lat: float, block_degrees: float) -> str:
    lon_bin = int(np.floor(lon / block_degrees))
    lat_bin = int(np.floor(lat / block_degrees))
    return f"lon{lon_bin}_lat{lat_bin}"


def _assign_groups(groups: pd.DataFrame, targets: dict[str, float], seed: str) -> dict[str, str]:
    total = int(groups["n"].sum())
    target_counts = {split: total * fraction for split, fraction in targets.items()}
    current = {split: 0 for split in targets}
    assignments: dict[str, str] = {}

    ordered = groups.copy()
    ordered["stable_score"] = ordered["spatial_block"].map(lambda v: _stable_score(f"{seed}:{v}"))
    ordered = ordered.sort_values(["n", "stable_score"], ascending=[False, True])

    for _, group in ordered.iterrows():
        block = str(group["spatial_block"])
        n = int(group["n"])

        def deficit(split: str) -> float:
            return target_counts[split] - current[split]

        split = max(targets, key=deficit)
        assignments[block] = split
        current[split] += n
    return assignments


def build_split_catalog(config: SplitConfig = SplitConfig()) -> Path:
    """Assign each plume event to TRAIN, VAL, or TEST using spatial blocking.

    Reads the S2 L1C reference match catalog and writes the split catalog.
    """
    df = pd.read_csv(config.match_catalog)
    valid = df[df["l1c_pair_ok"].fillna(False)].copy()
    valid = valid[valid["roi_min_lon"].notna() & valid["roi_min_lat"].notna()]
    valid["roi_center_lon"] = (valid["roi_min_lon"].astype(float) + valid["roi_max_lon"].astype(float)) / 2.0
    valid["roi_center_lat"] = (valid["roi_min_lat"].astype(float) + valid["roi_max_lat"].astype(float)) / 2.0
    valid["spatial_block"] = [
        _spatial_block(lon, lat, config.block_degrees)
        for lon, lat in zip(valid["roi_center_lon"], valid["roi_center_lat"])
    ]

    targets = {
        "TRAIN": config.train_fraction,
        "VAL": config.val_fraction,
        "TEST": config.test_fraction,
    }
    target_sum = sum(targets.values())
    if not np.isclose(target_sum, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {target_sum}")

    groups = valid.groupby("spatial_block", as_index=False).size().rename(columns={"size": "n"})
    assignments = _assign_groups(groups, targets, config.seed)
    valid["split"] = valid["spatial_block"].map(assignments)
    valid["split_policy"] = f"spatial_block_{config.block_degrees:g}_deg_greedy_{config.seed}"

    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(config.out_path, index=False)
    print(f"  → split catalog: {len(valid)} rows → {config.out_path}")
    for split, grp in valid.groupby("split"):
        print(f"     {split}: {len(grp)} ({100*len(grp)/len(valid):.1f}%)")
    return config.out_path


def audit_split_catalog(split_catalog: Path = CFG.splits_dir / "segmentation_split_catalog.csv") -> Path:
    """Print per-split statistics and write an audit CSV."""
    df = pd.read_csv(split_catalog)
    rows = []
    for split, group in df.groupby("split", dropna=False):
        rows.append({
            "split": split,
            "rows": int(len(group)),
            "fraction": float(len(group) / len(df)) if len(df) else 0.0,
            "sources": ",".join(sorted(group["source"].dropna().astype(str).unique())),
            "spatial_blocks": int(group["spatial_block"].nunique()),
        })
    summary = pd.DataFrame(rows).sort_values("split")
    out_path = Path(split_catalog).parent / "segmentation_split_audit.csv"
    summary.to_csv(out_path, index=False)
    print("\nBy split:")
    print(summary.to_string(index=False))
    return out_path
