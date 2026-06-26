from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ch4l1c.train import (
    _inject_synthetic_methane,
    _input_channels,
    _load_manifest,
    _methane_pair_feature_stack,
    _read_raw_l1c_pair,
    _require_torch,
    _resolve_device,
    _resolve_input_mode,
    _robust_standardize,
    _scale_band,
    _synthetic_plume_field,
    build_model,
)


MODEL_LABELS = {
    "unet": "U-Net",
    "attn_unet": "Attention U-Net",
    "unet_pp": "U-Net++",
    "deeplabv3p": "DeepLabV3+",
    "phys_tau_net": "PhysTAUNet",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic-only publication metric and example figures.")
    parser.add_argument("--manifest", default="data/training_l1c/segmentation_training_curated.csv")
    parser.add_argument("--benchmark-dir", default="data/models/synthetic_publication_benchmark")
    parser.add_argument("--out-dir", default="data/outputs/publication_figures")
    parser.add_argument("--models", nargs="+", default=["unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net"])
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--min-plume-fraction", type=float, default=0.01)
    parser.add_argument("--max-plume-fraction", type=float, default=0.20)
    parser.add_argument("--min-tau", type=float, default=0.004)
    parser.add_argument("--max-tau", type=float, default=0.055)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:180]


def _read_history_summary(benchmark_dir: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        path = benchmark_dir / model / "history.csv"
        if not path.exists():
            print(f"Skipping metrics for {model}: missing {path}")
            continue
        df = pd.read_csv(path)
        if df.empty or "f1" not in df.columns:
            print(f"Skipping metrics for {model}: incompatible history columns")
            continue
        row = df.sort_values(["f1", "tolerant_f1"], ascending=False).iloc[0]
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "best_epoch": int(row["epoch"]),
                "f1": float(row["f1"]),
                "iou": float(row["iou"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "tolerant_f1": float(row["tolerant_f1"]),
                "tolerant_iou": float(row["tolerant_iou"]),
                "predicted_positive_fraction_mean": float(row["predicted_positive_fraction_mean"]),
            }
        )
        test_path = benchmark_dir / model / "synthetic_eval_test.csv"
        if test_path.exists():
            test = pd.read_csv(test_path).iloc[0]
            rows[-1].update(
                {
                    "test_f1": float(test.get("f1", np.nan)),
                    "test_iou": float(test.get("iou", np.nan)),
                    "test_precision": float(test.get("precision", np.nan)),
                    "test_recall": float(test.get("recall", np.nan)),
                    "test_tolerant_f1": float(test.get("tolerant_f1", np.nan)),
                    "test_tolerant_iou": float(test.get("tolerant_iou", np.nan)),
                }
            )
    if not rows:
        raise ValueError(f"No usable history.csv files found under {benchmark_dir}")
    return pd.DataFrame(rows)


def _plot_grouped_bars(df: pd.DataFrame, metrics: list[str], title: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    for i, metric in enumerate(metrics):
        vals = df[metric].astype(float).to_numpy()
        xpos = x - 0.4 + width / 2 + i * width
        bars = ax.bar(xpos, vals, width=width, label=metric.replace("_", " "))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _event_rgb(raw: np.ndarray) -> np.ndarray:
    event = raw[:6]
    return np.dstack([_scale_band(event[5]), _scale_band(event[4]), _scale_band(event[2])])


def _ratio_delta(raw: np.ndarray) -> np.ndarray:
    event = raw[:6]
    ref = raw[6:12]
    event_ratio = np.log1p(np.maximum(event[5], 0.0)) - np.log1p(np.maximum(event[4], 0.0))
    ref_ratio = np.log1p(np.maximum(ref[5], 0.0)) - np.log1p(np.maximum(ref[4], 0.0))
    return _robust_standardize(event_ratio - ref_ratio)


def _crop_raw(raw: np.ndarray, patch_size: int, rng: np.random.Generator) -> np.ndarray:
    _, h, w = raw.shape
    ps = min(patch_size, h, w)
    y0 = int(rng.integers(0, max(1, h - ps + 1)))
    x0 = int(rng.integers(0, max(1, w - ps + 1)))
    return raw[:, y0 : y0 + ps, x0 : x0 + ps]


def _load_model(checkpoint_path: Path, device):
    torch, *_ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint.get("config", {})
    model_name = train_config.get("model", checkpoint_path.parent.name)
    input_mode = train_config.get("resolved_input_mode") or _resolve_input_mode(
        train_config.get("input_mode", "l1c_pair_methane"),
        model_name,
    )
    model = build_model(model_name, in_channels=_input_channels(input_mode)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model_name, input_mode, model


def _plot_synthetic_examples(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    torch, *_ = _require_torch()
    device = _resolve_device(args.device)
    benchmark_dir = Path(args.benchmark_dir)
    rng = np.random.default_rng(args.seed)
    frame = _load_manifest(Path(args.manifest), carbon_mapper_only=True)
    frame = frame[frame["split"].astype(str).isin(["TRAIN", "VAL"])].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No background rows available for synthetic examples")

    loaded = {}
    for model_key in args.models:
        ckpt = benchmark_dir / model_key / "best.pt"
        if not ckpt.exists():
            print(f"Skipping {model_key}: missing {ckpt}")
            continue
        _, input_mode, model = _load_model(ckpt, device)
        loaded[model_key] = {"input_mode": input_mode, "model": model}
    if not loaded:
        raise ValueError("No checkpoints available for synthetic examples")

    rows = []
    synth_dir = out_dir / "synthetic_segmentation_examples"
    synth_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in range(args.examples):
            row = frame.iloc[int(rng.integers(0, len(frame)))]
            raw = _crop_raw(_read_raw_l1c_pair(row["s2_path"]), args.patch_size, rng)
            _, h, w = raw.shape
            field, hard_mask = _synthetic_plume_field(
                h,
                w,
                rng,
                min_fraction=args.min_plume_fraction,
                max_fraction=args.max_plume_fraction,
            )
            tau = float(rng.uniform(args.min_tau, args.max_tau))
            injected = _inject_synthetic_methane(raw, field, tau)
            x = _methane_pair_feature_stack(injected)
            rgb = _event_rgb(injected)
            delta = _ratio_delta(injected)

            ncols = 3 + len(loaded)
            fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.4), constrained_layout=True)
            fig.suptitle(f"Synthetic L1C plume example {idx + 1} | tau={tau:.4f}", fontsize=14)

            axes[0].imshow(rgb)
            axes[0].contour(hard_mask, levels=[0.5], colors="red", linewidths=1.0)
            axes[0].set_title("Injected L1C false colour\nred=synthetic mask")
            axes[0].axis("off")

            im = axes[1].imshow(delta, cmap="viridis", vmin=0, vmax=1)
            axes[1].contour(hard_mask, levels=[0.5], colors="red", linewidths=1.0)
            axes[1].set_title("Methane-sensitive\nratio delta")
            axes[1].axis("off")
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            im = axes[2].imshow(hard_mask, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title(f"Synthetic target\npositive={hard_mask.mean():.3f}")
            axes[2].axis("off")
            fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

            for j, (model_key, item) in enumerate(loaded.items(), start=3):
                xt = torch.from_numpy(x)[None].to(device=device, dtype=torch.float32)
                prob = torch.sigmoid(item["model"](xt))[0, 0].detach().cpu().numpy()
                pred = prob >= 0.5
                tp = float(np.logical_and(pred, hard_mask > 0).sum())
                fp = float(np.logical_and(pred, hard_mask <= 0).sum())
                fn = float(np.logical_and(~pred, hard_mask > 0).sum())
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                im = axes[j].imshow(prob, cmap="viridis", vmin=0, vmax=1)
                axes[j].contour(hard_mask, levels=[0.5], colors="red", linewidths=1.0)
                if pred.any():
                    axes[j].contour(pred.astype("float32"), levels=[0.5], colors="yellow", linewidths=0.8)
                axes[j].set_title(f"{MODEL_LABELS.get(model_key, model_key)}\nF1={f1:.3f}")
                axes[j].axis("off")
                fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
                rows.append(
                    {
                        "example": idx + 1,
                        "model": model_key,
                        "tau": tau,
                        "target_positive_fraction": float(hard_mask.mean()),
                        "predicted_positive_fraction": float(pred.mean()),
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )

            out_path = synth_dir / f"synthetic_example_{idx + 1:02d}_model_comparison.png"
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            for rec in rows[-len(loaded) :]:
                rec["figure_path"] = str(out_path)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(synth_dir / "synthetic_example_manifest.csv", index=False)
    return manifest


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dir = Path(args.benchmark_dir)
    synthetic = _read_history_summary(benchmark_dir, args.models)
    synthetic.to_csv(out_dir / "synthetic_metric_summary.csv", index=False)
    _plot_grouped_bars(
        synthetic,
        ["f1", "tolerant_f1", "iou"],
        "Controlled Synthetic L1C Segmentation Metrics",
        "score",
        out_dir / "metrics_synthetic_segmentation.png",
    )
    if {"test_f1", "test_tolerant_f1", "test_iou"}.issubset(synthetic.columns):
        _plot_grouped_bars(
            synthetic,
            ["test_f1", "test_tolerant_f1", "test_iou"],
            "Held-Out Synthetic TEST Segmentation Metrics",
            "score",
            out_dir / "metrics_synthetic_test_segmentation.png",
        )
    _plot_grouped_bars(
        synthetic,
        ["precision", "recall"],
        "Controlled Synthetic L1C Precision and Recall",
        "score",
        out_dir / "metrics_synthetic_precision_recall.png",
    )
    manifest = _plot_synthetic_examples(args, out_dir)

    print(f"Wrote {out_dir / 'metrics_synthetic_segmentation.png'}")
    if (out_dir / "metrics_synthetic_test_segmentation.png").exists():
        print(f"Wrote {out_dir / 'metrics_synthetic_test_segmentation.png'}")
    print(f"Wrote {out_dir / 'metrics_synthetic_precision_recall.png'}")
    print(f"Wrote {out_dir / 'synthetic_metric_summary.csv'}")
    print(f"Wrote {out_dir / 'synthetic_segmentation_examples' / 'synthetic_example_manifest.csv'}")
    print(manifest.groupby("model")[["f1", "predicted_positive_fraction"]].mean().to_string())


if __name__ == "__main__":
    main()
