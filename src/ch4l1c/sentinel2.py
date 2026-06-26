from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm

from .config import CFG


S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


@dataclass(frozen=True)
class Sentinel2MatchConfig:
    mask_manifest: Path = CFG.label_dir / "source_mask_manifest.csv"
    out_path: Path = CFG.data_dir / "sentinel2" / "sentinel2_match_catalog.csv"
    project: str | None = None
    collection: str = S2_COLLECTION
    days_before: int = 3
    days_after: int = 3
    max_cloud_pct: float = 20.0
    limit: int | None = None


@dataclass(frozen=True)
class Sentinel2ExportConfig:
    match_catalog: Path = CFG.data_dir / "splits" / "segmentation_split_catalog.csv"
    out_manifest: Path = CFG.data_dir / "sentinel2" / "sentinel2_export_manifest.csv"
    project: str | None = None
    drive_folder: str = "CH4_Plume_Segmentation_S2"
    bands: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12", "SCL")
    scale_m: int = 20
    chip_size_px: int = 512
    limit: int | None = None
    source: str | None = None
    split: str | None = None
    start_index: int = 0


def _require_ee():
    try:
        import ee
    except ModuleNotFoundError as exc:
        raise SystemExit("earthengine-api is required. Install with: python -m pip install earthengine-api") from exc
    return ee


def _init_ee(project: str | None):
    ee = _require_ee()
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:
        ee.Authenticate(auth_mode="notebook")
        ee.Initialize(project=project) if project else ee.Initialize()
    return ee


def _date_window(timestamp: object, days_before: int, days_after: int) -> tuple[pd.Timestamp | None, str | None, str | None]:
    ts = pd.to_datetime(timestamp, errors="coerce", utc=True)
    if pd.isna(ts):
        return None, None, None
    start = ts - pd.Timedelta(days=days_before)
    end = ts + pd.Timedelta(days=days_after + 1)
    return ts, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _mask_bounds_lonlat(path: Path) -> tuple[float, float, float, float] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        return tuple(float(v) for v in transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))


def _safe_task_name(value: object, max_len: int = 90) -> str:
    text = str(value)
    safe = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in text)
    return safe[:max_len].strip("_") or "s2_chip"


def _pick_best_feature(features: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not features:
        return None

    def key(feature: dict[str, Any]) -> tuple[float, float]:
        props = feature.get("properties", {})
        delta = props.get("date_delta_days")
        cloud = props.get("CLOUDY_PIXEL_PERCENTAGE")
        return (
            float(delta) if delta is not None and np.isfinite(float(delta)) else 9999.0,
            float(cloud) if cloud is not None and np.isfinite(float(cloud)) else 9999.0,
        )

    return sorted(features, key=key)[0]


def _query_s2_match(
    ee,
    bounds: tuple[float, float, float, float],
    event_ts: pd.Timestamp,
    start_date: str,
    end_date: str,
    max_cloud_pct: float,
    collection_id: str,
) -> dict[str, Any] | None:
    min_lon, min_lat, max_lon, max_lat = bounds
    geom = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], proj="EPSG:4326", geodesic=False)
    event_millis = int(event_ts.timestamp() * 1000)

    def add_delta(image):
        delta = image.date().difference(ee.Date(event_millis), "day").abs()
        return image.set("date_delta_days", delta)

    base = (
        ee.ImageCollection(collection_id)
        .filterBounds(geom)
        .filterDate(start_date, end_date)
        .map(add_delta)
    )
    base_count = int(base.size().getInfo())
    filtered = (
        base.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("date_delta_days")
    )
    filtered_count = int(filtered.size().getInfo())
    features = filtered.limit(10).getInfo().get("features", [])
    best = _pick_best_feature(features)
    if best is None:
        return {
            "s2_image_id": None,
            "s2_product_id": None,
            "s2_date": None,
            "s2_cloud_pct": None,
            "s2_mgrs_tile": None,
            "s2_orbit": None,
            "date_delta_days": None,
            "s2_candidate_count": base_count,
            "s2_candidate_count_cloud_filtered": filtered_count,
        }
    props = best.get("properties", {})
    image_id = best.get("id")
    return {
        "s2_image_id": image_id,
        "s2_product_id": props.get("PRODUCT_ID"),
        "s2_date": pd.to_datetime(props.get("system:time_start"), unit="ms", utc=True).isoformat()
        if props.get("system:time_start") is not None
        else None,
        "s2_cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
        "s2_mgrs_tile": props.get("MGRS_TILE"),
        "s2_orbit": props.get("SENSING_ORBIT_NUMBER"),
        "date_delta_days": props.get("date_delta_days"),
        "s2_candidate_count": base_count,
        "s2_candidate_count_cloud_filtered": filtered_count,
    }


def build_sentinel2_match_catalog(config: Sentinel2MatchConfig = Sentinel2MatchConfig()) -> Path:
    ee = _init_ee(config.project)
    masks = pd.read_csv(config.mask_manifest)
    valid = masks[masks["mask_ok"].fillna(False)].copy()
    valid = valid[valid["mask_path"].notna()]
    if config.limit is not None:
        valid = valid.head(config.limit)

    rows = []
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="match Sentinel-2"):
        mask_path = Path(str(row["mask_path"]))
        record = {
            "source": row.get("source"),
            "plume_id": row.get("plume_id"),
            "timestamp": row.get("timestamp"),
            "year": row.get("year"),
            "month": row.get("month"),
            "mask_path": str(mask_path),
            "positive_fraction": row.get("positive_fraction"),
            "s2_match_ok": False,
            "s2_image_id": None,
            "s2_product_id": None,
            "s2_date": None,
            "date_delta_days": None,
            "s2_cloud_pct": None,
            "s2_mgrs_tile": None,
            "s2_orbit": None,
            "s2_collection": config.collection,
            "s2_candidate_count": None,
            "s2_candidate_count_cloud_filtered": None,
            "roi_min_lon": None,
            "roi_min_lat": None,
            "roi_max_lon": None,
            "roi_max_lat": None,
            "date_window_start": None,
            "date_window_end": None,
            "error": "",
        }
        try:
            event_ts, start_date, end_date = _date_window(row.get("timestamp"), config.days_before, config.days_after)
            bounds = _mask_bounds_lonlat(mask_path)
            if event_ts is None or start_date is None or end_date is None:
                record["error"] = "missing timestamp"
                rows.append(record)
                continue
            if bounds is None:
                record["error"] = "missing mask bounds"
                rows.append(record)
                continue
            record.update(
                {
                    "roi_min_lon": bounds[0],
                    "roi_min_lat": bounds[1],
                    "roi_max_lon": bounds[2],
                    "roi_max_lat": bounds[3],
                    "date_window_start": start_date,
                    "date_window_end": end_date,
                }
            )
            match = _query_s2_match(ee, bounds, event_ts, start_date, end_date, config.max_cloud_pct, config.collection)
            record.update(match)
            if not match.get("s2_image_id"):
                record["error"] = "no S2 scene under filters"
            else:
                record["s2_match_ok"] = True
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    out = pd.DataFrame(rows)
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out_path, index=False)
    return config.out_path


def queue_sentinel2_exports(config: Sentinel2ExportConfig = Sentinel2ExportConfig()) -> Path:
    ee = _init_ee(config.project)
    catalog = pd.read_csv(config.match_catalog)
    valid = catalog[catalog["s2_match_ok"].fillna(False)].copy()
    valid = valid[valid["s2_image_id"].notna()]
    if config.source is not None:
        valid = valid[valid["source"] == config.source]
    if config.split is not None and "split" in valid:
        valid = valid[valid["split"] == config.split]
    if config.start_index:
        valid = valid.iloc[config.start_index :]
    if config.limit is not None:
        valid = valid.head(config.limit)

    rows = []
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="queue Sentinel-2 exports"):
        plume_id = str(row.get("plume_id"))
        split = row.get("split") if "split" in row else None
        split_prefix = f"{split}_" if split is not None and str(split) != "nan" else ""
        file_prefix = _safe_task_name(f"{split_prefix}s2_{row.get('source')}_{plume_id}")
        description = _safe_task_name(f"export_{file_prefix}")
        record = {
            "source": row.get("source"),
            "split": split,
            "spatial_block": row.get("spatial_block") if "spatial_block" in row else None,
            "plume_id": plume_id,
            "timestamp": row.get("timestamp"),
            "year": row.get("year"),
            "month": row.get("month"),
            "mask_path": row.get("mask_path"),
            "s2_image_id": row.get("s2_image_id"),
            "s2_date": row.get("s2_date"),
            "date_delta_days": row.get("date_delta_days"),
            "s2_cloud_pct": row.get("s2_cloud_pct"),
            "drive_folder": config.drive_folder,
            "file_prefix": file_prefix,
            "scale_m": config.scale_m,
            "chip_size_px": config.chip_size_px,
            "chip_size_m": int(config.scale_m * config.chip_size_px),
            "bands": ",".join(config.bands),
            "task_id": None,
            "task_state": None,
            "queued_ok": False,
            "error": "",
        }
        try:
            center_lon = (float(row["roi_min_lon"]) + float(row["roi_max_lon"])) / 2.0
            center_lat = (float(row["roi_min_lat"]) + float(row["roi_max_lat"])) / 2.0
            half_size_m = (config.scale_m * config.chip_size_px) / 2.0
            geom = ee.Geometry.Point([center_lon, center_lat]).buffer(half_size_m).bounds(maxError=1)
            image = ee.Image(str(row["s2_image_id"])).select(list(config.bands)).toFloat()
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=config.drive_folder,
                fileNamePrefix=file_prefix,
                region=geom,
                dimensions=f"{config.chip_size_px}x{config.chip_size_px}",
                maxPixels=1e9,
                fileFormat="GeoTIFF",
            )
            task.start()
            status = task.status()
            record.update(
                {
                    "task_id": status.get("id"),
                    "task_state": status.get("state"),
                    "queued_ok": True,
                }
            )
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    out = pd.DataFrame(rows)
    config.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out_manifest, index=False)
    return config.out_manifest


def audit_sentinel2_exports(
    export_manifest: Path = CFG.data_dir / "sentinel2" / "sentinel2_export_manifest.csv",
    download_dir: Path = CFG.data_dir / "sentinel2" / "exports",
) -> Path:
    try:
        manifest = pd.read_csv(export_manifest)
    except pd.errors.EmptyDataError:
        out_path = export_manifest.parent / "sentinel2_export_audit.csv"
        pd.DataFrame(
            columns=[
                "source",
                "split",
                "plume_id",
                "file_prefix",
                "exists",
                "read_ok",
                "expected_shape",
                "expected_band_count",
                "error",
            ]
        ).to_csv(out_path, index=False)
        print("\nExport audit:")
        print("Manifest is empty; no files were queued for audit.")
        return out_path
    rows = []
    for _, row in manifest.iterrows():
        prefix = str(row.get("file_prefix"))
        candidates = sorted(download_dir.glob(f"{prefix}*.tif"))
        record = {
            "source": row.get("source"),
            "split": row.get("split") if "split" in row else None,
            "plume_id": row.get("plume_id"),
            "file_prefix": prefix,
            "expected_bands": len(str(row.get("bands", "")).split(",")) if row.get("bands") else None,
            "expected_width": row.get("chip_size_px") if "chip_size_px" in row else None,
            "expected_height": row.get("chip_size_px") if "chip_size_px" in row else None,
            "local_path": str(candidates[0]) if candidates else None,
            "exists": bool(candidates),
            "read_ok": False,
            "band_count": None,
            "width": None,
            "height": None,
            "crs": None,
            "dtype": None,
            "finite_fraction": None,
            "expected_shape": False,
            "expected_band_count": False,
            "error": "",
        }
        if not candidates:
            record["error"] = "missing local tif"
            rows.append(record)
            continue
        try:
            with rasterio.open(candidates[0]) as src:
                sample = src.read(1, masked=True)
                finite = np.isfinite(np.asarray(sample.filled(np.nan), dtype="float32"))
                record.update(
                    {
                        "read_ok": True,
                        "band_count": int(src.count),
                        "width": int(src.width),
                        "height": int(src.height),
                        "crs": str(src.crs) if src.crs else None,
                        "dtype": ",".join(src.dtypes),
                        "finite_fraction": float(finite.mean()),
                        "expected_shape": int(src.width) == int(record["expected_width"])
                        and int(src.height) == int(record["expected_height"]),
                        "expected_band_count": int(src.count) == int(record["expected_bands"]),
                    }
                )
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    out = pd.DataFrame(rows)
    out_path = export_manifest.parent / "sentinel2_export_audit.csv"
    out.to_csv(out_path, index=False)

    print("\nExport audit:")
    if not out.empty:
        print(out[["exists", "read_ok", "expected_shape", "expected_band_count"]].value_counts(dropna=False).to_string())
        print("\nDimensions:")
        print(out.groupby(["width", "height", "band_count"], dropna=False).size().to_string())
        print("\nCRS:")
        print(out["crs"].value_counts(dropna=False).to_string())
    return out_path


def audit_sentinel2_matches(match_catalog: Path = CFG.data_dir / "sentinel2" / "sentinel2_match_catalog.csv") -> Path:
    df = pd.read_csv(match_catalog)
    rows = []
    for source, group in df.groupby("source", dropna=False):
        ok = group[group["s2_match_ok"].fillna(False)]
        rows.append(
            {
                "source": source,
                "rows": int(len(group)),
                "s2_match_ok": int(len(ok)),
                "match_fraction": float(len(ok) / len(group)) if len(group) else 0.0,
                "median_date_delta_days": float(ok["date_delta_days"].median()) if len(ok) else None,
                "max_date_delta_days": float(ok["date_delta_days"].max()) if len(ok) else None,
                "median_cloud_pct": float(ok["s2_cloud_pct"].median()) if len(ok) else None,
                "max_cloud_pct": float(ok["s2_cloud_pct"].max()) if len(ok) else None,
                "unique_s2_images": int(ok["s2_image_id"].nunique()) if len(ok) else 0,
                "median_candidate_count": float(group["s2_candidate_count"].dropna().median())
                if "s2_candidate_count" in group and group["s2_candidate_count"].notna().any()
                else None,
                "median_cloud_filtered_count": float(group["s2_candidate_count_cloud_filtered"].dropna().median())
                if "s2_candidate_count_cloud_filtered" in group
                and group["s2_candidate_count_cloud_filtered"].notna().any()
                else None,
            }
        )
    summary = pd.DataFrame(rows)
    out_path = match_catalog.parent / "sentinel2_match_audit.csv"
    summary.to_csv(out_path, index=False)

    print("\nBy source:")
    print(summary.to_string(index=False))
    if not df.empty and "year" in df:
        print("\nBy source/year:")
        print(pd.crosstab(df["source"], df["year"], values=df["s2_match_ok"], aggfunc="sum").fillna(0).to_string())
    if not df.empty and "error" in df:
        errors = df.loc[~df["s2_match_ok"].fillna(False), "error"].value_counts(dropna=False).head(20)
        print("\nTop unmatched reasons:")
        print(errors.to_string())
    return out_path
