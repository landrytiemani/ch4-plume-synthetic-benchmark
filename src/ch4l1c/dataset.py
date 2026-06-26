from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from .config import CFG
from .sentinel2 import _safe_task_name


@dataclass(frozen=True)
class AlignedDatasetConfig:
    split_catalog: Path = CFG.data_dir / "splits" / "segmentation_split_catalog.csv"
    s2_dir: Path = CFG.data_dir / "sentinel2" / "exports"
    out_dir: Path = CFG.data_dir / "training" / "aligned_masks"
    manifest_path: Path = CFG.data_dir / "training" / "segmentation_training_manifest.csv"
    limit: int | None = None
    overwrite: bool = False
    file_prefix_kind: str = "s2"


def _file_prefix(row: pd.Series, kind: str = "s2") -> str:
    split = row.get("split") if "split" in row else None
    split_prefix = f"{split}_" if split is not None and str(split) != "nan" else ""
    if kind == "s2":
        return _safe_task_name(f"{split_prefix}s2_{row.get('source')}_{row.get('plume_id')}")
    if kind == "s2l1c_pair":
        return _safe_task_name(f"{split_prefix}s2l1c_pair_{row.get('source')}_{row.get('plume_id')}")
    raise ValueError("file_prefix_kind must be 's2' or 's2l1c_pair'")


def _find_s2_path(s2_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(s2_dir.glob(f"{prefix}*.tif"))
    return candidates[0] if candidates else None


def _read_source_mask(mask_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with rasterio.open(mask_path) as src:
        binary = src.read(1, out_dtype="float32")
        intensity = src.read(2, out_dtype="float32") if src.count >= 2 else binary.copy()
        if src.nodata is not None:
            binary = np.where(binary == src.nodata, np.nan, binary)
            intensity = np.where(intensity == src.nodata, np.nan, intensity)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }
    binary = np.where(np.isfinite(binary) & (binary > 0), 1.0, 0.0).astype("float32")
    intensity = np.where(np.isfinite(intensity), intensity, 0.0).astype("float32")
    return binary, intensity, meta


def _reproject_to_s2_grid(
    source: np.ndarray,
    source_meta: dict,
    s2: rasterio.DatasetReader,
    *,
    dtype: str,
    resampling: Resampling,
) -> np.ndarray:
    destination = np.zeros((s2.height, s2.width), dtype=dtype)
    reproject(
        source=source,
        destination=destination,
        src_transform=source_meta["transform"],
        src_crs=source_meta["crs"],
        dst_transform=s2.transform,
        dst_crs=s2.crs,
        dst_width=s2.width,
        dst_height=s2.height,
        src_nodata=0,
        dst_nodata=0,
        resampling=resampling,
    )
    return destination


def _write_aligned_mask(path: Path, s2: rasterio.DatasetReader, mask: np.ndarray, intensity: np.ndarray) -> None:
    profile = s2.profile.copy()
    profile.update(
        driver="GTiff",
        count=2,
        dtype="float32",
        nodata=0.0,
        compress="deflate",
        predictor=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("float32"), 1)
        dst.write(intensity.astype("float32"), 2)
        dst.set_band_description(1, "aligned_binary_plume_mask")
        dst.set_band_description(2, "aligned_source_intensity")


def build_aligned_training_dataset(config: AlignedDatasetConfig = AlignedDatasetConfig()) -> Path:
    catalog = pd.read_csv(config.split_catalog)
    if config.limit is not None:
        catalog = catalog.head(config.limit)

    rows = []
    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="align plume masks to S2"):
        prefix = _file_prefix(row, config.file_prefix_kind)
        s2_path = _find_s2_path(config.s2_dir, prefix)
        mask_path = Path(str(row.get("mask_path", "")))
        out_path = config.out_dir / str(row.get("split", "UNKNOWN")) / f"{prefix}_mask.tif"
        record = {
            "source": row.get("source"),
            "split": row.get("split"),
            "spatial_block": row.get("spatial_block") if "spatial_block" in row else None,
            "plume_id": row.get("plume_id"),
            "timestamp": row.get("timestamp"),
            "year": row.get("year"),
            "month": row.get("month"),
            "file_prefix": prefix,
            "s2_path": str(s2_path) if s2_path else "",
            "source_mask_path": str(mask_path),
            "aligned_mask_path": str(out_path),
            "pair_ok": False,
            "s2_crs": None,
            "s2_width": None,
            "s2_height": None,
            "s2_band_count": None,
            "mask_positive_pixels": 0,
            "mask_positive_fraction": 0.0,
            "intensity_max": 0.0,
            "error": "",
        }
        if s2_path is None:
            record["error"] = "missing Sentinel-2 export"
            rows.append(record)
            continue
        if not mask_path.exists():
            record["error"] = "missing source mask"
            rows.append(record)
            continue
        if out_path.exists() and not config.overwrite:
            try:
                with rasterio.open(s2_path) as s2, rasterio.open(out_path) as mask_src:
                    aligned_mask = mask_src.read(1)
                    aligned_intensity = mask_src.read(2) if mask_src.count >= 2 else aligned_mask
                    record.update(
                        {
                            "pair_ok": True,
                            "s2_crs": str(s2.crs) if s2.crs else None,
                            "s2_width": int(s2.width),
                            "s2_height": int(s2.height),
                            "s2_band_count": int(s2.count),
                            "mask_positive_pixels": int((aligned_mask > 0).sum()),
                            "mask_positive_fraction": float((aligned_mask > 0).mean()),
                            "intensity_max": float(np.nanmax(aligned_intensity)) if aligned_intensity.size else 0.0,
                        }
                    )
            except Exception as exc:
                record["error"] = repr(exc)
            rows.append(record)
            continue
        try:
            source_binary, source_intensity, source_meta = _read_source_mask(mask_path)
            with rasterio.open(s2_path) as s2:
                aligned_mask = _reproject_to_s2_grid(
                    source_binary,
                    source_meta,
                    s2,
                    dtype="float32",
                    resampling=Resampling.nearest,
                )
                aligned_mask = (aligned_mask > 0.5).astype("float32")
                aligned_intensity = _reproject_to_s2_grid(
                    source_intensity,
                    source_meta,
                    s2,
                    dtype="float32",
                    resampling=Resampling.nearest,
                )
                aligned_intensity = np.where(aligned_mask > 0, aligned_intensity, 0.0).astype("float32")
                _write_aligned_mask(out_path, s2, aligned_mask, aligned_intensity)
                record.update(
                    {
                        "pair_ok": True,
                        "s2_crs": str(s2.crs) if s2.crs else None,
                        "s2_width": int(s2.width),
                        "s2_height": int(s2.height),
                        "s2_band_count": int(s2.count),
                        "mask_positive_pixels": int(aligned_mask.sum()),
                        "mask_positive_fraction": float(aligned_mask.mean()),
                        "intensity_max": float(np.nanmax(aligned_intensity)) if aligned_intensity.size else 0.0,
                    }
                )
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    manifest = pd.DataFrame(rows)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.manifest_path, index=False)
    return config.manifest_path


def audit_aligned_training_dataset(
    manifest_path: Path = CFG.data_dir / "training" / "segmentation_training_manifest.csv",
) -> Path:
    df = pd.read_csv(manifest_path)
    out_path = manifest_path.parent / "segmentation_training_audit.csv"

    print("\nPair status:")
    print(df["pair_ok"].value_counts(dropna=False).to_string())
    print("\nBy split:")
    print(
        df.groupby("split", dropna=False)
        .agg(
            rows=("pair_ok", "size"),
            pair_ok=("pair_ok", "sum"),
            median_positive_fraction=("mask_positive_fraction", "median"),
            mean_positive_fraction=("mask_positive_fraction", "mean"),
            zero_mask_rows=("mask_positive_pixels", lambda s: int((s == 0).sum())),
        )
        .to_string()
    )
    print("\nBy source:")
    print(
        df.groupby("source", dropna=False)
        .agg(
            rows=("pair_ok", "size"),
            pair_ok=("pair_ok", "sum"),
            median_positive_fraction=("mask_positive_fraction", "median"),
            mean_positive_fraction=("mask_positive_fraction", "mean"),
            zero_mask_rows=("mask_positive_pixels", lambda s: int((s == 0).sum())),
        )
        .to_string()
    )
    print("\nCRS:")
    print(df["s2_crs"].value_counts(dropna=False).to_string())
    if "error" in df:
        errors = df.loc[df["error"].fillna("") != "", "error"].value_counts().head(20)
        print("\nTop errors:")
        print(errors.to_string() if len(errors) else "none")

    summary = []
    for keys, group in df.groupby(["split", "source"], dropna=False):
        split, source = keys
        summary.append(
            {
                "split": split,
                "source": source,
                "rows": int(len(group)),
                "pair_ok": int(group["pair_ok"].sum()),
                "median_positive_fraction": float(group["mask_positive_fraction"].median()),
                "mean_positive_fraction": float(group["mask_positive_fraction"].mean()),
                "zero_mask_rows": int((group["mask_positive_pixels"] == 0).sum()),
            }
        )
    pd.DataFrame(summary).to_csv(out_path, index=False)
    return out_path
