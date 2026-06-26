"""Convert raw Carbon Mapper and EMIT plume rasters into standardised binary masks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

from .config import CFG


@dataclass(frozen=True)
class SourceMaskConfig:
    catalog: Path = CFG.label_dir / "plume_label_catalog.csv"
    out_dir: Path = CFG.label_dir / "source_masks"
    manifest_path: Path = CFG.label_dir / "source_mask_manifest.csv"
    emit_core_threshold_ppb: float = 100.0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _read_with_nodata(src: rasterio.DatasetReader, band: int) -> np.ndarray:
    arr = src.read(band, out_dtype="float32")
    if src.nodata is not None:
        arr = np.where(arr == src.nodata, np.nan, arr)
    return arr


def _carbon_mapper_mask(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray, str]:
    if src.count >= 4:
        alpha = _read_with_nodata(src, 4)
        mask = np.isfinite(alpha) & (alpha > 0)
        return mask, alpha, "alpha_band_gt0"
    arr = _read_with_nodata(src, 1)
    mask = np.isfinite(arr) & (arr > 0)
    return mask, arr, "band1_gt0_fallback"


def _emit_mask(src: rasterio.DatasetReader, core_threshold_ppb: float) -> tuple[np.ndarray, np.ndarray, str]:
    arr = _read_with_nodata(src, 1)
    finite = np.isfinite(arr)
    mask = finite & (arr > core_threshold_ppb)
    return mask, arr, f"finite_band1_gt_{core_threshold_ppb:g}_ppb"


def _write_mask(path: Path, src: rasterio.DatasetReader, mask: np.ndarray, intensity: np.ndarray) -> None:
    profile = src.profile.copy()
    profile.update(driver="GTiff", count=2, dtype="float32", nodata=np.nan, compress="deflate", predictor=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("float32"), 1)
        dst.write(intensity.astype("float32"), 2)
        dst.set_band_description(1, "binary_plume_mask")
        dst.set_band_description(2, "source_intensity")


def _output_name(row: pd.Series) -> str:
    source = str(row.get("source", "source"))
    plume_id = str(row.get("plume_id", Path(str(row.get("raster_path", "label"))).stem))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in plume_id)
    return f"{source}_{safe}_mask.tif"


def build_source_masks(config: SourceMaskConfig = SourceMaskConfig()) -> Path:
    """Convert raw plume rasters to georeferenced binary mask TIFs.

    Mask band 1 = binary (0/1), band 2 = source intensity.
    Carbon Mapper: alpha-channel threshold.
    EMIT: ppb threshold (default 100 ppb).
    """
    catalog = pd.read_csv(config.catalog)
    rows = []
    if catalog.empty:
        config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(config.manifest_path, index=False)
        return config.manifest_path

    valid = catalog[catalog["download_ok"].fillna(False)].copy()
    valid = valid[valid["raster_path"].notna()]
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="build source masks"):
        raster_path = Path(str(row["raster_path"]))
        out_path = config.out_dir / _output_name(row)
        record = {
            "source": row.get("source"),
            "plume_id": row.get("plume_id"),
            "timestamp": row.get("timestamp"),
            "year": row.get("year"),
            "month": row.get("month"),
            "source_raster_path": str(raster_path),
            "mask_path": str(out_path),
            "mask_ok": False,
            "width": None, "height": None, "crs": None,
            "threshold_rule": "",
            "positive_pixels": 0,
            "total_pixels": 0,
            "positive_fraction": 0.0,
            "intensity_min": None, "intensity_max": None,
            "error": "",
        }
        if not raster_path.exists() or raster_path.stat().st_size == 0:
            record["error"] = "missing or empty raster"
            rows.append(record)
            continue
        try:
            with rasterio.open(raster_path) as src:
                source = str(row.get("source"))
                if source == "carbon_mapper":
                    mask, intensity, rule = _carbon_mapper_mask(src)
                elif source == "emit":
                    mask, intensity, rule = _emit_mask(src, config.emit_core_threshold_ppb)
                else:
                    arr = _read_with_nodata(src, 1)
                    mask = np.isfinite(arr) & (arr > 0)
                    intensity = arr
                    rule = "band1_gt0_unknown_source"

                finite_intensity = intensity[np.isfinite(intensity)]
                _write_mask(out_path, src, mask, intensity)
                record.update({
                    "mask_ok": True,
                    "width": int(src.width),
                    "height": int(src.height),
                    "crs": str(src.crs) if src.crs else None,
                    "threshold_rule": rule,
                    "positive_pixels": int(mask.sum()),
                    "total_pixels": int(mask.size),
                    "positive_fraction": float(mask.mean()),
                    "intensity_min": _safe_float(np.nanmin(finite_intensity)) if finite_intensity.size else None,
                    "intensity_max": _safe_float(np.nanmax(finite_intensity)) if finite_intensity.size else None,
                })
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    manifest = pd.DataFrame(rows)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(config.manifest_path, index=False)
    ok_count = int(manifest["mask_ok"].sum()) if not manifest.empty else 0
    print(f"  → {ok_count}/{len(manifest)} source masks written to {config.out_dir}")
    return config.manifest_path
