from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ch4l1c.train import (
    _SyntheticPlumePatchDataset,
    _evaluate_synthetic_loader,
    _input_channels,
    _load_manifest,
    _require_torch,
    _resolve_device,
    _resolve_input_mode,
    build_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one trained model on held-out synthetic plume patches.")
    parser.add_argument("--manifest", default="data/training_l1c/segmentation_training_curated.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--evaluation-patches", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-plume-fraction", type=float, default=0.01)
    parser.add_argument("--max-plume-fraction", type=float, default=0.20)
    parser.add_argument("--min-tau", type=float, default=0.004)
    parser.add_argument("--max-tau", type=float, default=0.055)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch, _, _, DataLoader, _ = _require_torch()
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint.get("config", {})
    model_name = train_config.get("model", checkpoint_path.parent.name)
    input_mode = train_config.get("resolved_input_mode") or _resolve_input_mode(
        train_config.get("input_mode", "l1c_pair_methane"),
        model_name,
    )

    df = _load_manifest(Path(args.manifest), carbon_mapper_only=True)
    if "pair_ok" in df.columns:
        df = df[df["pair_ok"].astype(bool)].copy()
    df = df[df["split"].astype(str) == args.split].copy()
    if df.empty:
        raise ValueError(f"No {args.split} background rows available in {args.manifest}")

    dataset = _SyntheticPlumePatchDataset(
        df,
        patch_size=args.patch_size,
        length=args.evaluation_patches,
        input_mode=input_mode,
        min_plume_fraction=args.min_plume_fraction,
        max_plume_fraction=args.max_plume_fraction,
        min_tau=args.min_tau,
        max_tau=args.max_tau,
        target_mode="binary",
        target_dilation_pixels=0,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(model_name, in_channels=_input_channels(input_mode)).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = _evaluate_synthetic_loader(
        model,
        loader,
        device=device,
        threshold=args.threshold,
        tolerance_pixels=int(train_config.get("target_dilation_pixels", 2)),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model_name,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "background_rows": int(len(df)),
        "evaluation_patches": int(args.evaluation_patches),
        "threshold": float(args.threshold),
        **metrics,
    }
    out_path = out_dir / f"synthetic_eval_{args.split.lower()}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    (out_dir / f"synthetic_eval_{args.split.lower()}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
