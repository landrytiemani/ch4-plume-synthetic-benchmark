from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ch4l1c.train import SyntheticTrainConfig, train_synthetic_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model on synthetic Sentinel-2 L1C methane plumes.")
    parser.add_argument("--manifest", default="data/training_l1c/segmentation_training_curated.csv")
    parser.add_argument("--out-dir", default="data/models/synthetic_publication_benchmark")
    parser.add_argument("--model", required=True, choices=["unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-epoch", type=int, default=8192)
    parser.add_argument("--validation-patches", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-plume-fraction", type=float, default=0.01)
    parser.add_argument("--max-plume-fraction", type=float, default=0.20)
    parser.add_argument("--min-tau", type=float, default=0.004)
    parser.add_argument("--max-tau", type=float, default=0.055)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_synthetic_segmentation_model(
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
            input_mode="l1c_pair_methane",
            min_plume_fraction=args.min_plume_fraction,
            max_plume_fraction=args.max_plume_fraction,
            min_tau=args.min_tau,
            max_tau=args.max_tau,
            target_mode="soft",
            target_dilation_pixels=2,
            carbon_mapper_backgrounds_only=True,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
