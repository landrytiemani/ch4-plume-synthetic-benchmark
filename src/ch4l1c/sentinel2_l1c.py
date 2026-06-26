from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CFG
from .sentinel2 import _date_window, _init_ee, _mask_bounds_lonlat, _safe_task_name


L1C_COLLECTION = "COPERNICUS/S2_HARMONIZED"


@dataclass(frozen=True)
class L1CReferenceMatchConfig:
    split_catalog: Path = CFG.data_dir / "splits" / "segmentation_split_catalog.csv"
    out: Path = CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"
    project: str | None = None
    collection: str = L1C_COLLECTION
    event_days_before: int = 3
    event_days_after: int = 3
    reference_days_before: int = 180
    reference_days_after: int = 180
    exclude_reference_days: int = 14
    event_max_cloud_pct: float = 80.0
    reference_max_cloud_pct: float = 30.0
    source: str | None = None
    split: str | None = None
    start_index: int = 0
    limit: int | None = None


@dataclass(frozen=True)
class L1CPairExportConfig:
    match_catalog: Path = CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"
    out_manifest: Path = CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_pair_export_manifest.csv"
    project: str | None = None
    drive_folder: str = "CH4_Plume_L1C_S2_pairs"
    bands: tuple[str, ...] = ("B2", "B3", "B4", "B8", "B11", "B12")
    scale_m: int = 20
    chip_size_px: int = 512
    source: str | None = None
    split: str | None = None
    start_index: int = 0
    limit: int | None = None


def _geom_from_bounds(ee, bounds: tuple[float, float, float, float]):
    min_lon, min_lat, max_lon, max_lat = bounds
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], proj="EPSG:4326", geodesic=False)


def _feature_record(feature: dict[str, Any] | None, prefix: str) -> dict[str, Any]:
    if feature is None:
        return {
            f"{prefix}_image_id": None,
            f"{prefix}_product_id": None,
            f"{prefix}_date": None,
            f"{prefix}_cloud_pct": None,
            f"{prefix}_mgrs_tile": None,
            f"{prefix}_orbit": None,
            f"{prefix}_date_delta_days": None,
        }
    props = feature.get("properties", {})
    millis = props.get("system:time_start")
    return {
        f"{prefix}_image_id": feature.get("id"),
        f"{prefix}_product_id": props.get("PRODUCT_ID"),
        f"{prefix}_date": pd.to_datetime(millis, unit="ms", utc=True).isoformat() if millis is not None else None,
        f"{prefix}_cloud_pct": props.get("CLOUDY_PIXEL_PERCENTAGE"),
        f"{prefix}_mgrs_tile": props.get("MGRS_TILE"),
        f"{prefix}_orbit": props.get("SENSING_ORBIT_NUMBER"),
        f"{prefix}_date_delta_days": props.get("date_delta_days"),
    }


def _best_feature(features: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not features:
        return None

    def key(feature: dict[str, Any]) -> tuple[float, float]:
        props = feature.get("properties", {})
        cloud = props.get("CLOUDY_PIXEL_PERCENTAGE")
        delta = props.get("date_delta_days")
        return (
            float(cloud) if cloud is not None and np.isfinite(float(cloud)) else 9999.0,
            float(delta) if delta is not None and np.isfinite(float(delta)) else 9999.0,
        )

    return sorted(features, key=key)[0]


def _query_pair(
    ee,
    *,
    bounds: tuple[float, float, float, float],
    event_ts: pd.Timestamp,
    collection_id: str,
    event_start: str,
    event_end: str,
    reference_start: str,
    reference_end: str,
    exclude_reference_days: int,
    event_max_cloud_pct: float,
    reference_max_cloud_pct: float,
) -> dict[str, Any]:
    geom = _geom_from_bounds(ee, bounds)
    event_millis = int(event_ts.timestamp() * 1000)

    def add_delta(image):
        delta = image.date().difference(ee.Date(event_millis), "day").abs()
        return image.set("date_delta_days", delta)

    event_collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(geom)
        .filterDate(event_start, event_end)
        .map(add_delta)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", event_max_cloud_pct))
        .sort("date_delta_days")
    )
    event_features = event_collection.limit(10).getInfo().get("features", [])
    event = sorted(
        event_features,
        key=lambda f: (
            float(f.get("properties", {}).get("date_delta_days", 9999)),
            float(f.get("properties", {}).get("CLOUDY_PIXEL_PERCENTAGE", 9999)),
        ),
    )[0] if event_features else None

    event_mgrs = event.get("properties", {}).get("MGRS_TILE") if event else None
    ref_collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(geom)
        .filterDate(reference_start, reference_end)
        .map(add_delta)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", reference_max_cloud_pct))
        .filter(ee.Filter.gt("date_delta_days", exclude_reference_days))
    )
    if event_mgrs:
        ref_collection = ref_collection.filter(ee.Filter.eq("MGRS_TILE", event_mgrs))
    ref_features = ref_collection.limit(50).getInfo().get("features", [])
    reference = _best_feature(ref_features)

    return {
        **_feature_record(event, "event"),
        **_feature_record(reference, "reference"),
        "event_candidate_count": int(event_collection.size().getInfo()),
        "reference_candidate_count": int(ref_collection.size().getInfo()),
    }


def build_l1c_reference_match_catalog(config: L1CReferenceMatchConfig = L1CReferenceMatchConfig()) -> Path:
    ee = _init_ee(config.project)
    if not config.split_catalog.exists():
        raise FileNotFoundError(
            f"Missing split catalog: {config.split_catalog}. "
            "Run Step 0 first to build it from scratch: GEE_PROJECT=your-project bash scripts/build_plume_catalog.sh"
        )
    catalog = pd.read_csv(config.split_catalog)
    if config.source is not None:
        catalog = catalog[catalog["source"].astype(str) == config.source].copy()
    if config.split is not None and "split" in catalog:
        catalog = catalog[catalog["split"].astype(str) == config.split].copy()
    if config.start_index:
        catalog = catalog.iloc[config.start_index :].copy()
    if config.limit is not None:
        catalog = catalog.head(config.limit).copy()

    rows = []
    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="match S2 L1C event/reference"):
        record = {
            **row.to_dict(),
            "l1c_pair_ok": False,
            "l1c_collection": config.collection,
            "error": "",
        }
        try:
            event_ts, event_start, event_end = _date_window(row.get("timestamp"), config.event_days_before, config.event_days_after)
            if event_ts is None or event_start is None or event_end is None:
                record["error"] = "missing timestamp"
                rows.append(record)
                continue
            bounds = _mask_bounds_lonlat(Path(str(row.get("mask_path", ""))))
            if bounds is None:
                record["error"] = "missing mask bounds"
                rows.append(record)
                continue
            reference_start = (event_ts - pd.Timedelta(days=config.reference_days_before)).strftime("%Y-%m-%d")
            reference_end = (event_ts + pd.Timedelta(days=config.reference_days_after + 1)).strftime("%Y-%m-%d")
            record.update(
                {
                    "roi_min_lon": bounds[0],
                    "roi_min_lat": bounds[1],
                    "roi_max_lon": bounds[2],
                    "roi_max_lat": bounds[3],
                    "event_window_start": event_start,
                    "event_window_end": event_end,
                    "reference_window_start": reference_start,
                    "reference_window_end": reference_end,
                }
            )
            match = _query_pair(
                ee,
                bounds=bounds,
                event_ts=event_ts,
                collection_id=config.collection,
                event_start=event_start,
                event_end=event_end,
                reference_start=reference_start,
                reference_end=reference_end,
                exclude_reference_days=config.exclude_reference_days,
                event_max_cloud_pct=config.event_max_cloud_pct,
                reference_max_cloud_pct=config.reference_max_cloud_pct,
            )
            record.update(match)
            if not record.get("event_image_id"):
                record["error"] = "no L1C event scene under filters"
            elif not record.get("reference_image_id"):
                record["error"] = "no L1C reference scene under filters"
            else:
                record["l1c_pair_ok"] = True
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    if rows:
        out = pd.DataFrame(rows)
    else:
        out = pd.DataFrame(
            columns=[
                "source",
                "split",
                "spatial_block",
                "plume_id",
                "timestamp",
                "year",
                "month",
                "mask_path",
                "event_image_id",
                "reference_image_id",
                "event_date",
                "reference_date",
                "event_cloud_pct",
                "reference_cloud_pct",
                "reference_date_delta_days",
                "drive_folder",
                "file_prefix",
                "scale_m",
                "chip_size_px",
                "bands",
                "task_id",
                "task_state",
                "queued_ok",
                "error",
            ]
        )
    config.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out, index=False)
    print("\nL1C pair status:")
    print(out["l1c_pair_ok"].value_counts(dropna=False).to_string())
    if "split" in out:
        print("\nBy split:")
        print(pd.crosstab(out["split"], out["l1c_pair_ok"]).to_string())
    if "error" in out:
        print("\nTop errors:")
        print(out.loc[~out["l1c_pair_ok"].fillna(False), "error"].value_counts().head(10).to_string())
    return config.out


def queue_l1c_pair_exports(config: L1CPairExportConfig = L1CPairExportConfig()) -> Path:
    ee = _init_ee(config.project)
    catalog = pd.read_csv(config.match_catalog)
    valid = catalog[catalog["l1c_pair_ok"].fillna(False)].copy()
    valid = valid[valid["event_image_id"].notna() & valid["reference_image_id"].notna()]
    if config.source is not None:
        valid = valid[valid["source"].astype(str) == config.source]
    if config.split is not None and "split" in valid:
        valid = valid[valid["split"].astype(str) == config.split]
    if config.start_index:
        valid = valid.iloc[config.start_index :]
    if config.limit is not None:
        valid = valid.head(config.limit)

    rows = []
    for _, row in tqdm(valid.iterrows(), total=len(valid), desc="queue S2 L1C pair exports"):
        plume_id = str(row.get("plume_id"))
        split = row.get("split") if "split" in row else None
        split_prefix = f"{split}_" if split is not None and str(split) != "nan" else ""
        file_prefix = _safe_task_name(f"{split_prefix}s2l1c_pair_{row.get('source')}_{plume_id}")
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
            "event_image_id": row.get("event_image_id"),
            "reference_image_id": row.get("reference_image_id"),
            "event_date": row.get("event_date"),
            "reference_date": row.get("reference_date"),
            "event_cloud_pct": row.get("event_cloud_pct"),
            "reference_cloud_pct": row.get("reference_cloud_pct"),
            "reference_date_delta_days": row.get("reference_date_delta_days"),
            "drive_folder": config.drive_folder,
            "file_prefix": file_prefix,
            "scale_m": config.scale_m,
            "chip_size_px": config.chip_size_px,
            "bands": ",".join([f"event_{b}" for b in config.bands] + [f"reference_{b}" for b in config.bands]),
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
            event = ee.Image(str(row["event_image_id"])).select(list(config.bands)).toFloat()
            reference = ee.Image(str(row["reference_image_id"])).select(list(config.bands)).toFloat()
            event = event.rename([f"event_{band}" for band in config.bands])
            reference = reference.rename([f"reference_{band}" for band in config.bands])
            image = event.addBands(reference).toFloat()
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
            record.update({"task_id": status.get("id"), "task_state": status.get("state"), "queued_ok": True})
        except Exception as exc:
            record["error"] = repr(exc)
        rows.append(record)

    out = pd.DataFrame(rows)
    config.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.out_manifest, index=False)
    print("\nQueued:")
    print(out["queued_ok"].value_counts(dropna=False).to_string() if len(out) else "none")
    return config.out_manifest
