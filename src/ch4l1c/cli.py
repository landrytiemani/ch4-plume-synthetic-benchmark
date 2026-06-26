from __future__ import annotations

import argparse
from pathlib import Path

from .config import CFG


def main() -> None:
    parser = argparse.ArgumentParser(description="CH4 plume L1C event/reference methane-feature pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    l1c_match = sub.add_parser("s2-l1c-reference-match", help="Match L1C event scenes with low-cloud reference scenes")
    l1c_match.add_argument("--split-catalog", default=str(CFG.data_dir / "splits" / "segmentation_split_catalog.csv"))
    l1c_match.add_argument("--out", default=str(CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"))
    l1c_match.add_argument("--project", default=None)
    l1c_match.add_argument("--collection", default="COPERNICUS/S2_HARMONIZED")
    l1c_match.add_argument("--event-days-before", type=int, default=3)
    l1c_match.add_argument("--event-days-after", type=int, default=3)
    l1c_match.add_argument("--reference-days-before", type=int, default=180)
    l1c_match.add_argument("--reference-days-after", type=int, default=180)
    l1c_match.add_argument("--exclude-reference-days", type=int, default=14)
    l1c_match.add_argument("--event-max-cloud-pct", type=float, default=80.0)
    l1c_match.add_argument("--reference-max-cloud-pct", type=float, default=30.0)
    l1c_match.add_argument("--max-cloud-pct", type=float, default=None, help="Legacy option: sets both event and reference cloud filters")
    l1c_match.add_argument("--source", choices=["carbon_mapper", "emit"], default=None)
    l1c_match.add_argument("--split", choices=["TRAIN", "VAL", "TEST"], default=None)
    l1c_match.add_argument("--start-index", type=int, default=0)
    l1c_match.add_argument("--limit", type=int, default=None)

    l1c_export = sub.add_parser("s2-l1c-export-pairs", help="Queue paired L1C event/reference chip exports to Drive")
    l1c_export.add_argument("--match-catalog", default=str(CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"))
    l1c_export.add_argument("--out-manifest", default=str(CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_pair_export_manifest.csv"))
    l1c_export.add_argument("--project", default=None)
    l1c_export.add_argument("--drive-folder", default="CH4_Plume_L1C_S2_pairs")
    l1c_export.add_argument("--bands", nargs="+", default=["B2", "B3", "B4", "B8", "B11", "B12"])
    l1c_export.add_argument("--scale-m", type=int, default=20)
    l1c_export.add_argument("--chip-size-px", type=int, default=512)
    l1c_export.add_argument("--source", choices=["carbon_mapper", "emit"], default=None)
    l1c_export.add_argument("--split", choices=["TRAIN", "VAL", "TEST"], default=None)
    l1c_export.add_argument("--start-index", type=int, default=0)
    l1c_export.add_argument("--limit", type=int, default=None)

    s2_audit = sub.add_parser("audit-s2-exports", help="Audit downloaded L1C pair GeoTIFFs")
    s2_audit.add_argument("--manifest", default=str(CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_pair_export_manifest.csv"))
    s2_audit.add_argument("--download-dir", default=str(CFG.data_dir / "sentinel2_l1c" / "exports"))

    train_dataset = sub.add_parser("build-training-dataset", help="Align plume masks to L1C pair chip grids")
    train_dataset.add_argument("--split-catalog", default=str(CFG.data_dir / "sentinel2_l1c" / "sentinel2_l1c_reference_match_catalog.csv"))
    train_dataset.add_argument("--s2-dir", default=str(CFG.data_dir / "sentinel2_l1c" / "exports"))
    train_dataset.add_argument("--out-dir", default=str(CFG.data_dir / "training_l1c" / "aligned_masks"))
    train_dataset.add_argument("--manifest-path", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_manifest.csv"))
    train_dataset.add_argument("--limit", type=int, default=None)
    train_dataset.add_argument("--overwrite", action="store_true")
    train_dataset.add_argument("--file-prefix-kind", choices=["s2", "s2l1c_pair"], default="s2l1c_pair")

    train_audit = sub.add_parser("audit-training-dataset", help="Audit aligned L1C/mask pairs")
    train_audit.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_manifest.csv"))

    curate = sub.add_parser("curate-training-manifest", help="Add training quality flags and weights")
    curate.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_manifest.csv"))
    curate.add_argument("--out", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"))
    curate.add_argument("--min-positive-pixels", type=int, default=5)
    curate.add_argument("--tiny-positive-pixels", type=int, default=50)
    curate.add_argument("--very-large-fraction", type=float, default=0.05)
    curate.add_argument("--extreme-fraction", type=float, default=0.10)
    curate.add_argument("--carbon-mapper-weight", type=float, default=1.0)
    curate.add_argument("--emit-weight", type=float, default=0.45)

    curate_audit = sub.add_parser("audit-curated-training-manifest", help="Audit curated L1C training manifest")
    curate_audit.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"))

    seg_train = sub.add_parser("train-segmentation", help="Train L1C pair methane-feature segmentation model")
    seg_train.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"))
    seg_train.add_argument("--out-dir", default=str(CFG.data_dir / "models" / "segmentation_l1c_pair_methane"))
    seg_train.add_argument("--model", choices=["unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net"], default="phys_tau_net")
    seg_train.add_argument("--epochs", type=int, default=40)
    seg_train.add_argument("--batch-size", type=int, default=4)
    seg_train.add_argument("--patch-size", type=int, default=128)
    seg_train.add_argument("--patches-per-epoch", type=int, default=4096)
    seg_train.add_argument("--lr", type=float, default=1e-4)
    seg_train.add_argument("--weight-decay", type=float, default=1e-4)
    seg_train.add_argument("--num-workers", type=int, default=4)
    seg_train.add_argument("--positive-patch-probability", type=float, default=0.95)
    seg_train.add_argument("--min-patch-positive-fraction", type=float, default=0.005)
    seg_train.add_argument("--max-patch-attempts", type=int, default=25)
    seg_train.add_argument("--target-mode", choices=["binary", "soft"], default="soft")
    seg_train.add_argument("--target-dilation-pixels", type=int, default=4)
    seg_train.add_argument("--input-mode", choices=["l1c_pair_methane"], default="l1c_pair_methane")
    seg_train.add_argument("--carbon-mapper-only", action="store_true")
    seg_train.add_argument("--train-source", choices=["carbon_mapper", "emit"], default=None)
    seg_train.add_argument("--min-train-positive-pixels", type=int, default=250)
    seg_train.add_argument("--max-train-positive-fraction", type=float, default=0.05)
    seg_train.add_argument("--min-eval-positive-pixels", type=int, default=250)
    seg_train.add_argument("--max-eval-positive-fraction", type=float, default=0.05)
    seg_train.add_argument("--seed", type=int, default=7)
    seg_train.add_argument("--device", default="auto")
    seg_train.add_argument("--limit-train", type=int, default=None)
    seg_train.add_argument("--limit-val", type=int, default=None)
    seg_train.add_argument("--init-checkpoint", default=None, help="Optional checkpoint used to initialize weights before real-label fine-tuning")

    synth_train = sub.add_parser("train-synthetic-segmentation", help="Train on physics-injected synthetic methane plumes")
    synth_train.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"))
    synth_train.add_argument("--out-dir", default=str(CFG.data_dir / "models" / "l1c_synthetic_plumes"))
    synth_train.add_argument("--model", choices=["unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net"], default="phys_tau_net")
    synth_train.add_argument("--epochs", type=int, default=40)
    synth_train.add_argument("--batch-size", type=int, default=8)
    synth_train.add_argument("--patch-size", type=int, default=128)
    synth_train.add_argument("--patches-per-epoch", type=int, default=8192)
    synth_train.add_argument("--validation-patches", type=int, default=1024)
    synth_train.add_argument("--lr", type=float, default=1e-4)
    synth_train.add_argument("--weight-decay", type=float, default=1e-4)
    synth_train.add_argument("--num-workers", type=int, default=4)
    synth_train.add_argument("--input-mode", choices=["l1c_pair_methane"], default="l1c_pair_methane")
    synth_train.add_argument("--min-plume-fraction", type=float, default=0.01)
    synth_train.add_argument("--max-plume-fraction", type=float, default=0.20)
    synth_train.add_argument("--min-tau", type=float, default=0.004)
    synth_train.add_argument("--max-tau", type=float, default=0.055)
    synth_train.add_argument("--target-mode", choices=["binary", "soft"], default="soft")
    synth_train.add_argument("--target-dilation-pixels", type=int, default=2)
    synth_train.add_argument("--carbon-mapper-backgrounds-only", action="store_true")
    synth_train.add_argument("--seed", type=int, default=7)
    synth_train.add_argument("--device", default="auto")
    synth_train.add_argument("--limit-train", type=int, default=None)
    synth_train.add_argument("--limit-val", type=int, default=None)

    sweep = sub.add_parser("sweep-segmentation-thresholds", help="Evaluate L1C model across thresholds")
    sweep.add_argument("--manifest", default=str(CFG.data_dir / "training_l1c" / "segmentation_training_curated.csv"))
    sweep.add_argument("--checkpoint", required=True)
    sweep.add_argument("--out", default=None)
    sweep.add_argument("--thresholds", nargs="+", type=float, default=None)
    sweep.add_argument("--split", choices=["TRAIN", "VAL", "TEST"], default="VAL")
    sweep.add_argument("--tolerance-pixels", type=int, default=4)
    sweep.add_argument("--source", choices=["carbon_mapper", "emit"], default=None)
    sweep.add_argument("--min-positive-pixels", type=int, default=None)
    sweep.add_argument("--max-positive-fraction", type=float, default=None)
    sweep.add_argument("--limit", type=int, default=None)
    sweep.add_argument("--device", default="auto")

    # ── Plume catalog acquisition (Step 0) ──────────────────────────────────
    cm_catalog = sub.add_parser("download-cm-catalog", help="Download Carbon Mapper CH4 plume catalog from public API (no auth required)")
    cm_catalog.add_argument("--start-year", type=int, default=2019)
    cm_catalog.add_argument("--end-year", type=int, default=2025)
    cm_catalog.add_argument("--max-pages", type=int, default=None)

    cm_rasters = sub.add_parser("download-cm-rasters", help="Download Carbon Mapper plume raster TIFs")
    cm_rasters.add_argument("--catalog", default=None, help="Path to Carbon Mapper catalog CSV (default: auto-detect latest)")
    cm_rasters.add_argument("--out-dir", default=str(CFG.carbon_mapper_dir / "plume_tifs"))
    cm_rasters.add_argument("--manifest-path", default=None)
    cm_rasters.add_argument("--limit", type=int, default=None)
    cm_rasters.add_argument("--start-year", type=int, default=None)
    cm_rasters.add_argument("--end-year", type=int, default=None)

    emit_dl = sub.add_parser("download-emit", help="Download NASA EMIT CH4 plume products (free NASA Earthdata account required)")
    emit_dl.add_argument("--start-date", default="2022-08-01")
    emit_dl.add_argument("--end-date", default="2026-01-01")
    emit_dl.add_argument("--limit", type=int, default=None)
    emit_dl.add_argument("--no-download", action="store_true", help="Search only - do not download files")

    label_cat = sub.add_parser("build-label-catalog", help="Build unified plume label catalog from CM and EMIT manifests")
    label_cat.add_argument("--carbon-mapper-manifest", default=str(CFG.carbon_mapper_dir / "carbon_mapper_plume_raster_manifest.csv"))
    label_cat.add_argument("--emit-manifest", default=str(CFG.emit_dir / "emit_emitl2bch4plm_download_manifest.csv"))
    label_cat.add_argument("--out", default=str(CFG.label_dir / "plume_label_catalog.csv"))

    src_masks = sub.add_parser("build-source-masks", help="Convert raw plume rasters to binary mask GeoTIFs")
    src_masks.add_argument("--catalog", default=str(CFG.label_dir / "plume_label_catalog.csv"))
    src_masks.add_argument("--out-dir", default=str(CFG.label_dir / "source_masks"))
    src_masks.add_argument("--manifest-path", default=str(CFG.label_dir / "source_mask_manifest.csv"))
    src_masks.add_argument("--emit-threshold-ppb", type=float, default=100.0)

    split_cat = sub.add_parser("build-split-catalog", help="Assign TRAIN/VAL/TEST splits via spatial blocking (70/15/15)")
    split_cat.add_argument("--match-catalog", default=None)
    split_cat.add_argument("--out", default=None)
    split_cat.add_argument("--block-degrees", type=float, default=0.25)
    split_cat.add_argument("--train-fraction", type=float, default=0.70)
    split_cat.add_argument("--val-fraction", type=float, default=0.15)
    split_cat.add_argument("--test-fraction", type=float, default=0.15)
    split_cat.add_argument("--seed", default="ch4syn-v1")

    audit_split = sub.add_parser("audit-split-catalog", help="Print per-split row counts and spatial-block statistics")
    audit_split.add_argument("--catalog", default=None)

    args = parser.parse_args()

    if args.command == "s2-l1c-reference-match":
        from .sentinel2_l1c import L1CReferenceMatchConfig, build_l1c_reference_match_catalog

        out_path = build_l1c_reference_match_catalog(
            L1CReferenceMatchConfig(
                split_catalog=Path(args.split_catalog),
                out=Path(args.out),
                project=args.project,
                collection=args.collection,
                event_days_before=args.event_days_before,
                event_days_after=args.event_days_after,
                reference_days_before=args.reference_days_before,
                reference_days_after=args.reference_days_after,
                exclude_reference_days=args.exclude_reference_days,
                event_max_cloud_pct=args.max_cloud_pct if args.max_cloud_pct is not None else args.event_max_cloud_pct,
                reference_max_cloud_pct=args.max_cloud_pct if args.max_cloud_pct is not None else args.reference_max_cloud_pct,
                source=args.source,
                split=args.split,
                start_index=args.start_index,
                limit=args.limit,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "s2-l1c-export-pairs":
        from .sentinel2_l1c import L1CPairExportConfig, queue_l1c_pair_exports

        out_path = queue_l1c_pair_exports(
            L1CPairExportConfig(
                match_catalog=Path(args.match_catalog),
                out_manifest=Path(args.out_manifest),
                project=args.project,
                drive_folder=args.drive_folder,
                bands=tuple(args.bands),
                scale_m=args.scale_m,
                chip_size_px=args.chip_size_px,
                source=args.source,
                split=args.split,
                start_index=args.start_index,
                limit=args.limit,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "audit-s2-exports":
        from .sentinel2 import audit_sentinel2_exports

        out_path = audit_sentinel2_exports(Path(args.manifest), Path(args.download_dir))
        print(f"Wrote {out_path}")
        return

    if args.command == "build-training-dataset":
        from .dataset import AlignedDatasetConfig, build_aligned_training_dataset

        out_path = build_aligned_training_dataset(
            AlignedDatasetConfig(
                split_catalog=Path(args.split_catalog),
                s2_dir=Path(args.s2_dir),
                out_dir=Path(args.out_dir),
                manifest_path=Path(args.manifest_path),
                limit=args.limit,
                overwrite=args.overwrite,
                file_prefix_kind=args.file_prefix_kind,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "audit-training-dataset":
        from .dataset import audit_aligned_training_dataset

        out_path = audit_aligned_training_dataset(Path(args.manifest))
        print(f"Wrote {out_path}")
        return

    if args.command == "curate-training-manifest":
        from .curation import TrainingCurationConfig, curate_training_manifest

        out_path = curate_training_manifest(
            TrainingCurationConfig(
                manifest=Path(args.manifest),
                out=Path(args.out),
                min_positive_pixels=args.min_positive_pixels,
                tiny_positive_pixels=args.tiny_positive_pixels,
                very_large_fraction=args.very_large_fraction,
                extreme_fraction=args.extreme_fraction,
                carbon_mapper_weight=args.carbon_mapper_weight,
                emit_weight=args.emit_weight,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "audit-curated-training-manifest":
        from .curation import audit_curated_training_manifest

        out_path = audit_curated_training_manifest(Path(args.manifest))
        print(f"Wrote {out_path}")
        return

    if args.command == "train-segmentation":
        from .train import TrainConfig, train_segmentation_model

        best_path = train_segmentation_model(
            TrainConfig(
                manifest=Path(args.manifest),
                out_dir=Path(args.out_dir),
                model=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                patches_per_epoch=args.patches_per_epoch,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                positive_patch_probability=args.positive_patch_probability,
                min_patch_positive_fraction=args.min_patch_positive_fraction,
                max_patch_attempts=args.max_patch_attempts,
                target_mode=args.target_mode,
                target_dilation_pixels=args.target_dilation_pixels,
                input_mode=args.input_mode,
                carbon_mapper_only=args.carbon_mapper_only,
                train_source=args.train_source,
                min_train_positive_pixels=args.min_train_positive_pixels,
                max_train_positive_fraction=args.max_train_positive_fraction,
                min_eval_positive_pixels=args.min_eval_positive_pixels,
                max_eval_positive_fraction=args.max_eval_positive_fraction,
                seed=args.seed,
                device=args.device,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
                init_checkpoint=Path(args.init_checkpoint) if args.init_checkpoint else None,
            )
        )
        print(f"Wrote {best_path}")
        return

    if args.command == "train-synthetic-segmentation":
        from .train import SyntheticTrainConfig, train_synthetic_segmentation_model

        best_path = train_synthetic_segmentation_model(
            SyntheticTrainConfig(
                manifest=Path(args.manifest),
                out_dir=Path(args.out_dir),
                model=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                patches_per_epoch=args.patches_per_epoch,
                validation_patches=args.validation_patches,
                lr=args.lr,
                weight_decay=args.weight_decay,
                num_workers=args.num_workers,
                input_mode=args.input_mode,
                min_plume_fraction=args.min_plume_fraction,
                max_plume_fraction=args.max_plume_fraction,
                min_tau=args.min_tau,
                max_tau=args.max_tau,
                target_mode=args.target_mode,
                target_dilation_pixels=args.target_dilation_pixels,
                carbon_mapper_backgrounds_only=args.carbon_mapper_backgrounds_only,
                seed=args.seed,
                device=args.device,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
            )
        )
        print(f"Wrote {best_path}")
        return

    if args.command == "sweep-segmentation-thresholds":
        from .train import ThresholdSweepConfig, sweep_segmentation_thresholds

        out = Path(args.out) if args.out else Path(args.checkpoint).parent / "threshold_sweep.csv"
        out_path = sweep_segmentation_thresholds(
            ThresholdSweepConfig(
                manifest=Path(args.manifest),
                checkpoint=Path(args.checkpoint),
                out=out,
                thresholds=tuple(args.thresholds) if args.thresholds else ThresholdSweepConfig.thresholds,
                split=args.split,
                tolerance_pixels=args.tolerance_pixels,
                source=args.source,
                min_positive_pixels=args.min_positive_pixels,
                max_positive_fraction=args.max_positive_fraction,
                limit=args.limit,
                device=args.device,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "download-cm-catalog":
        from .carbon_mapper import CarbonMapperCatalogConfig, download_carbon_mapper_catalog

        out_path = download_carbon_mapper_catalog(
            CarbonMapperCatalogConfig(
                start_year=args.start_year,
                end_year=args.end_year,
                max_pages=args.max_pages,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "download-cm-rasters":
        from .carbon_mapper import CarbonMapperRasterConfig, download_carbon_mapper_rasters
        import glob as _glob

        catalog = args.catalog
        if catalog is None:
            pattern = str(CFG.carbon_mapper_dir / "carbon_mapper_ch4_permian_*.csv")
            matches = sorted(_glob.glob(pattern))
            if not matches:
                raise SystemExit(f"No Carbon Mapper catalog found at {CFG.carbon_mapper_dir}. Run download-cm-catalog first.")
            catalog = matches[-1]
        manifest_path = Path(args.manifest_path) if args.manifest_path else None
        out_path = download_carbon_mapper_rasters(
            CarbonMapperRasterConfig(
                catalog=Path(catalog),
                out_dir=Path(args.out_dir),
                manifest_path=manifest_path,
                limit=args.limit,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "download-emit":
        from .emit import EmitDownloadConfig, download_emit_products

        _, manifest_path = download_emit_products(
            EmitDownloadConfig(
                start_date=args.start_date,
                end_date=args.end_date,
                limit=args.limit,
                download=not args.no_download,
            )
        )
        print(f"Wrote {manifest_path}")
        return

    if args.command == "build-label-catalog":
        from .labels import build_unified_label_catalog

        out_path = build_unified_label_catalog(
            carbon_mapper_manifest=Path(args.carbon_mapper_manifest),
            emit_manifest=Path(args.emit_manifest),
            out_path=Path(args.out),
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "build-source-masks":
        from .masks import SourceMaskConfig, build_source_masks

        out_path = build_source_masks(
            SourceMaskConfig(
                catalog=Path(args.catalog),
                out_dir=Path(args.out_dir),
                manifest_path=Path(args.manifest_path),
                emit_core_threshold_ppb=args.emit_threshold_ppb,
            )
        )
        print(f"Wrote {out_path}")
        return

    if args.command == "build-split-catalog":
        from .splits import SplitConfig, build_split_catalog

        kwargs: dict = {
            "block_degrees": args.block_degrees,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "seed": args.seed,
        }
        if args.match_catalog is not None:
            kwargs["match_catalog"] = Path(args.match_catalog)
        if args.out is not None:
            kwargs["out_path"] = Path(args.out)
        out_path = build_split_catalog(SplitConfig(**kwargs))
        print(f"Wrote {out_path}")
        return

    if args.command == "audit-split-catalog":
        from .splits import audit_split_catalog

        catalog = Path(args.catalog) if args.catalog else None
        out_path = audit_split_catalog() if catalog is None else audit_split_catalog(catalog)
        print(f"Wrote {out_path}")
        return


if __name__ == "__main__":
    main()
