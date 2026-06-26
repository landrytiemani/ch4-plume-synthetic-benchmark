"""Download Carbon Mapper methane plume catalog and raster files.

The Carbon Mapper API is publicly accessible without authentication.
Plume raster URLs are embedded in the catalog response.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .config import CFG, PERMIAN_BBOX


API_URL = "https://api.carbonmapper.org/api/v1/catalog/plumes/annotated"


@dataclass(frozen=True)
class CarbonMapperCatalogConfig:
    out_dir: Path = CFG.carbon_mapper_dir
    bbox: tuple[float, float, float, float] = PERMIAN_BBOX
    gas: str = "CH4"
    limit: int = 1000
    max_pages: int | None = None
    start_year: int = 2019
    end_year: int = 2025


@dataclass(frozen=True)
class CarbonMapperRasterConfig:
    catalog: Path = CFG.carbon_mapper_dir / "carbon_mapper_ch4_permian_2019_2025.csv"
    out_dir: Path = CFG.carbon_mapper_dir / "plume_tifs"
    manifest_path: Path | None = None
    limit: int | None = None
    start_year: int | None = None
    end_year: int | None = None


def _require_requests():
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise SystemExit("requests is required. Install with: pip install requests") from exc
    return requests


def _safe_name(value: object) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in text)


def _request_params(config: CarbonMapperCatalogConfig, offset: int) -> list[tuple[str, str]]:
    minx, miny, maxx, maxy = config.bbox
    return [
        ("plume_gas", config.gas),
        ("limit", str(config.limit)),
        ("offset", str(offset)),
        ("bbox", str(minx)),
        ("bbox", str(miny)),
        ("bbox", str(maxx)),
        ("bbox", str(maxy)),
    ]


def _flatten_item(item: dict[str, Any]) -> dict[str, Any]:
    geom = item.get("geometry_json") or {}
    coords = geom.get("coordinates") or [None, None]
    timestamp = pd.to_datetime(item.get("scene_timestamp"), errors="coerce", utc=True)
    return {
        "source": "carbon_mapper",
        "id": item.get("id"),
        "plume_id": item.get("plume_id"),
        "gas": item.get("gas"),
        "lon": coords[0],
        "lat": coords[1],
        "scene_id": item.get("scene_id"),
        "scene_timestamp": item.get("scene_timestamp"),
        "year": int(timestamp.year) if not pd.isna(timestamp) else None,
        "month": int(timestamp.month) if not pd.isna(timestamp) else None,
        "instrument": item.get("instrument"),
        "platform": item.get("platform"),
        "mission_phase": item.get("mission_phase"),
        "emission_auto": item.get("emission_auto"),
        "emission_uncertainty_auto": item.get("emission_uncertainty_auto"),
        "emission_cmf_type": item.get("emission_cmf_type"),
        "gsd": item.get("gsd"),
        "sensitivity_mode": item.get("sensitivity_mode"),
        "off_nadir": item.get("off_nadir"),
        "plume_png": item.get("plume_png"),
        "plume_rgb_png": item.get("plume_rgb_png"),
        "plume_tif": item.get("plume_tif"),
    }


def download_carbon_mapper_catalog(config: CarbonMapperCatalogConfig = CarbonMapperCatalogConfig()) -> Path:
    """Download Carbon Mapper plume catalog from the public API.

    No authentication required.  Returns path to the saved CSV.
    """
    requests = _require_requests()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 0
    total = None

    while True:
        page += 1
        response = requests.get(API_URL, params=_request_params(config, offset), timeout=60)
        response.raise_for_status()
        payload = response.json()
        items = payload.get("items", [])
        total = payload.get("bbox_count", total)
        rows.extend(_flatten_item(item) for item in items)
        print(f"page={page} offset={offset} items={len(items)} total_bbox={total}")
        if not items:
            break
        offset += len(items)
        if total is not None and offset >= int(total):
            break
        if config.max_pages is not None and page >= config.max_pages:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[(df["year"] >= config.start_year) & (df["year"] <= config.end_year)].copy()
        df = df.sort_values(["scene_timestamp", "plume_id"], na_position="last")
    out_path = config.out_dir / f"carbon_mapper_{config.gas.lower()}_permian_{config.start_year}_{config.end_year}.csv"
    df.to_csv(out_path, index=False)
    print(f"  → wrote {len(df)} rows to {out_path}")
    return out_path


def download_carbon_mapper_rasters(config: CarbonMapperRasterConfig = CarbonMapperRasterConfig()) -> Path:
    """Download Carbon Mapper plume raster TIFs (URLs embedded in catalog).

    Returns path to the download manifest CSV.
    """
    requests = _require_requests()
    catalog = pd.read_csv(config.catalog)
    required = {"plume_id", "plume_tif"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"{config.catalog} is missing required columns: {sorted(missing)}")

    unique = catalog.dropna(subset=["plume_id", "plume_tif"]).drop_duplicates("plume_id").sort_values("plume_id")
    if config.start_year is not None:
        unique = unique[unique["year"] >= config.start_year]
    if config.end_year is not None:
        unique = unique[unique["year"] <= config.end_year]
    if config.limit is not None:
        unique = unique.head(config.limit)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in tqdm(unique.iterrows(), total=len(unique), desc="download Carbon Mapper rasters"):
        plume_id = row["plume_id"]
        path = config.out_dir / f"{_safe_name(plume_id)}.tif"
        ok = False
        error = ""
        if not path.exists() or path.stat().st_size == 0:
            try:
                response = requests.get(row["plume_tif"], timeout=120)
                response.raise_for_status()
                path.write_bytes(response.content)
            except Exception as exc:
                error = repr(exc)
        if path.exists() and path.stat().st_size > 0:
            ok = True
        rows.append(
            {
                "source": "carbon_mapper",
                "plume_id": plume_id,
                "year": row.get("year"),
                "month": row.get("month"),
                "timestamp": row.get("scene_timestamp"),
                "lon": row.get("lon"),
                "lat": row.get("lat"),
                "emission_auto": row.get("emission_auto"),
                "remote_url": row.get("plume_tif"),
                "local_path": str(path),
                "download_ok": ok,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "error": error,
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = config.manifest_path or (config.out_dir.parent / "carbon_mapper_plume_raster_manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    ok_count = int(manifest["download_ok"].sum()) if not manifest.empty else 0
    print(f"  → {ok_count}/{len(manifest)} rasters downloaded to {config.out_dir}")
    return manifest_path
