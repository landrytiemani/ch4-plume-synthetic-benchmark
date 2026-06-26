"""Build a unified plume label catalog from Carbon Mapper and EMIT manifests."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio

from .config import CFG


EMIT_NAME_RE = re.compile(r"EMIT_L2B_CH4PLM_\d+_(?P<stamp>\d{8}T\d{6})_(?P<orbit>\d+)\.tif$")


def _empty_label_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source", "plume_id", "timestamp", "year", "month",
            "lon", "lat", "emission_auto", "raster_path", "metadata_path",
            "download_ok", "size_bytes", "crs", "width", "height",
            "min_lon", "min_lat", "max_lon", "max_lat",
            "positive_fraction", "raster_min", "raster_max", "source_manifest",
        ]
    )


def _raster_stats(path: object) -> dict:
    raster_path = Path(str(path))
    empty = {
        "crs": None, "width": None, "height": None,
        "min_lon": None, "min_lat": None, "max_lon": None, "max_lat": None,
        "positive_fraction": None, "raster_min": None, "raster_max": None,
    }
    if not raster_path.exists() or raster_path.stat().st_size == 0:
        return empty
    try:
        with rasterio.open(raster_path) as src:
            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            finite = np.isfinite(arr)
            valid = arr[finite]
            positive = valid > 0
            try:
                from rasterio.warp import transform_bounds
                min_lon, min_lat, max_lon, max_lat = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
            except Exception:
                min_lon, min_lat, max_lon, max_lat = None, None, None, None
            return {
                "crs": str(src.crs) if src.crs else None,
                "width": int(src.width),
                "height": int(src.height),
                "min_lon": min_lon, "min_lat": min_lat,
                "max_lon": max_lon, "max_lat": max_lat,
                "positive_fraction": float(positive.mean()) if valid.size else 0.0,
                "raster_min": float(np.nanmin(valid)) if valid.size else None,
                "raster_max": float(np.nanmax(valid)) if valid.size else None,
            }
    except Exception:
        return empty


def _emit_timestamp_from_name(filename: object) -> pd.Timestamp | None:
    match = EMIT_NAME_RE.match(str(Path(str(filename)).name))
    if not match:
        return None
    return pd.to_datetime(match.group("stamp"), format="%Y%m%dT%H%M%S", utc=True)


def _emit_metadata_path(tif_path: object) -> str | None:
    path = Path(str(tif_path))
    meta_name = path.name.replace("CH4PLM_", "CH4PLMMETA_").replace(".tif", ".json")
    meta_path = path.with_name(meta_name)
    return str(meta_path) if meta_path.exists() else None


def _carbon_mapper_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_label_catalog()
    df = pd.read_csv(path)
    if df.empty:
        return _empty_label_catalog()
    rows = []
    for _, row in df.iterrows():
        stats = _raster_stats(row.get("local_path"))
        rows.append({
            "source": "carbon_mapper",
            "plume_id": row.get("plume_id"),
            "timestamp": row.get("timestamp"),
            "year": row.get("year"),
            "month": row.get("month"),
            "lon": row.get("lon"),
            "lat": row.get("lat"),
            "emission_auto": row.get("emission_auto"),
            "raster_path": row.get("local_path"),
            "metadata_path": None,
            "download_ok": row.get("download_ok"),
            "size_bytes": row.get("size_bytes"),
            **stats,
            "source_manifest": str(path),
        })
    return pd.DataFrame(rows)


def _emit_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_label_catalog()
    df = pd.read_csv(path)
    if df.empty:
        return _empty_label_catalog()
    tif_df = df[df["filename"].astype(str).str.endswith(".tif")].copy()
    rows = []
    for _, row in tif_df.iterrows():
        timestamp = _emit_timestamp_from_name(row.get("filename"))
        stats = _raster_stats(row.get("local_path"))
        rows.append({
            "source": "emit",
            "plume_id": Path(str(row.get("filename"))).stem,
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "year": int(timestamp.year) if timestamp is not None else None,
            "month": int(timestamp.month) if timestamp is not None else None,
            "lon": None, "lat": None,
            "emission_auto": None,
            "raster_path": row.get("local_path"),
            "metadata_path": _emit_metadata_path(row.get("local_path")),
            "download_ok": row.get("download_ok"),
            "size_bytes": row.get("size_bytes"),
            **stats,
            "source_manifest": str(path),
        })
    return pd.DataFrame(rows)


def build_unified_label_catalog(
    carbon_mapper_manifest: Path | None = None,
    emit_manifest: Path | None = None,
    out_path: Path = CFG.label_dir / "plume_label_catalog.csv",
) -> Path:
    """Merge Carbon Mapper and EMIT download manifests into one label catalog."""
    frames = []
    if carbon_mapper_manifest is not None and Path(carbon_mapper_manifest).exists():
        frames.append(_carbon_mapper_rows(Path(carbon_mapper_manifest)))
    if emit_manifest is not None and Path(emit_manifest).exists():
        frames.append(_emit_rows(Path(emit_manifest)))
    catalog = pd.concat(frames, ignore_index=True) if frames else _empty_label_catalog()
    if not catalog.empty:
        catalog["download_ok"] = catalog["download_ok"].fillna(False).astype(bool)
        catalog = catalog.sort_values(["source", "timestamp", "plume_id"], na_position="last")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)
    print(f"  → label catalog: {len(catalog)} rows → {out_path}")
    return out_path
