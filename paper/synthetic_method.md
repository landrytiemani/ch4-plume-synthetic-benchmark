# Synthetic Method Notes

This project is a controlled synthetic Sentinel-2 methane plume segmentation benchmark. It does not use Carbon Mapper or EMIT masks as pixel-perfect training labels.

## Inputs

- Raw Sentinel-2 L1C event/reference chip pairs acquired via Google Earth Engine, exported to Google Drive, and downloaded to `data/raw/sentinel2_l1c/exports/` (see `scripts/acquire_s2_data.sh`).
- Training manifests built from scratch by `scripts/build_training_manifest.sh` into `data/training_l1c/`.

Each raw chip contains event and reference L1C bands. The model input uses real background texture plus methane-sensitive features from the event/reference SWIR pair.

## Synthetic Injection

For each training patch, the code generates a plume optical-depth field and attenuates the event-scene methane-sensitive SWIR band. The target mask is generated from the same optical-depth field, so the segmentation label is controlled and pixel-aligned.

The generator is intentionally irregular:

- meandering plume centerline
- source/core region
- width variation downwind
- detached wisps and lobes
- textured turbulence
- holes and ragged edges

This avoids the smooth, repeated ellipse shape that made earlier examples visually unrealistic.

## Published-Method Basis

The route follows the published pattern of using Sentinel-2 SWIR methane sensitivity and controlled/synthetic plume data for model development:

- Sentinel-2 methane work commonly exploits B11/B12 SWIR behavior, event/reference differencing, ratios, or matched-filter-style residuals.
- Synthetic plume benchmarks are acceptable only when the claim is explicitly synthetic or controlled; real operational plume segmentation needs separately validated real labels.
- The generated masks are not claimed as observed real plumes.

References used to frame the method:

- Climate Change AI / NeurIPS 2023, methane plume detection with U-Net on Sentinel-2: https://www.climatechange.ai/papers/neurips2023/78
- Ruzicka et al. 2023, semantic segmentation of methane plumes with hyperspectral ML models: https://www.nature.com/articles/s41598-023-44918-6
- Wang et al. 2024, matched filter for Sentinel-2 methane plume detection: https://www.mdpi.com/2072-4292/16/6/1023
- Sentinel-2 methane monitoring physics and B11/B12 transmittance context: https://amt.copernicus.org/articles/16/89/2023/
