"""Download NASA EMIT Level-2B methane plume products.

Requires a free NASA Earthdata account.  Authenticate once with:
    earthaccess.login(strategy="interactive", persist=True)

Product: EMITL2BCH4PLM — per-plume methane enhancement rasters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import CFG, PERMIAN_BBOX


EMIT_CH4_SHORT_NAME = "EMITL2BCH4PLM"


@dataclass(frozen=True)
class EmitDownloadConfig:
    out_dir: Path = CFG.emit_dir
    bbox: tuple[float, float, float, float] = PERMIAN_BBOX
    start_date: str = "2022-08-01"   # EMIT launched August 2022
    end_date: str = "2026-01-01"
    short_name: str = EMIT_CH4_SHORT_NAME
    limit: int | None = None
    download: bool = True


def _require_earthaccess():
    try:
        import earthaccess
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "earthaccess is required for EMIT downloads. "
            "Install with: pip install earthaccess\n"
            "Then authenticate once: python -c \"import earthaccess; earthaccess.login(strategy='interactive', persist=True)\""
        ) from exc
    return earthaccess


def _granule_row(granule) -> dict:
    meta = getattr(granule, "meta", {}) or {}
    umm = meta.get("umm", {}) if isinstance(meta, dict) else {}
    temporal = umm.get("TemporalExtent", {}) or {}
    range_date = temporal.get("RangeDateTime", {}) if isinstance(temporal, dict) else {}
    try:
        links = list(granule.data_links(access="external"))
    except Exception:
        try:
            links = list(granule.data_links())
        except Exception:
            links = []
    return {
        "source": "emit",
        "granule_ur": umm.get("GranuleUR") or meta.get("native-id") or meta.get("concept-id"),
        "concept_id": meta.get("concept-id") if isinstance(meta, dict) else None,
        "begin_time": range_date.get("BeginningDateTime"),
        "end_time": range_date.get("EndingDateTime"),
        "download_url_count": len(links),
        "download_urls": "|".join(str(v) for v in links),
    }


def download_emit_products(config: EmitDownloadConfig = EmitDownloadConfig()) -> tuple[Path, Path]:
    """Search and download EMIT CH4 plume products via NASA earthaccess.

    Returns (catalog_path, manifest_path).
    """
    earthaccess = _require_earthaccess()
    config.out_dir.mkdir(parents=True, exist_ok=True)

    earthaccess.login(strategy="interactive", persist=True)
    search_kwargs = {
        "short_name": config.short_name,
        "bounding_box": config.bbox,
        "temporal": (config.start_date, config.end_date),
    }
    if config.limit is not None:
        search_kwargs["count"] = config.limit
    granules = earthaccess.search_data(**search_kwargs)
    print(f"  found {len(granules)} EMIT granules")

    catalog = pd.DataFrame([_granule_row(g) for g in granules])
    catalog_path = config.out_dir / f"emit_{config.short_name.lower()}_permian_catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    manifest_path = config.out_dir / f"emit_{config.short_name.lower()}_download_manifest.csv"
    if config.download and granules:
        downloaded = earthaccess.download(granules, local_path=str(config.out_dir / "products"))
        manifest = pd.DataFrame(
            [
                {
                    "source": "emit",
                    "local_path": str(path),
                    "filename": Path(path).name,
                    "download_ok": Path(path).exists() and Path(path).stat().st_size > 0,
                    "size_bytes": Path(path).stat().st_size if Path(path).exists() else 0,
                }
                for path in downloaded
            ]
        )
    else:
        manifest = pd.DataFrame(columns=["source", "local_path", "filename", "download_ok", "size_bytes"])
    manifest.to_csv(manifest_path, index=False)
    ok_count = int(manifest["download_ok"].sum()) if not manifest.empty else 0
    print(f"  → {ok_count}/{len(manifest)} EMIT products downloaded to {config.out_dir}/products")
    return catalog_path, manifest_path
