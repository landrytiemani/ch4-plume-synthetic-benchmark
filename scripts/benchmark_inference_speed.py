from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from ch4l1c.train import (
    _SyntheticPlumePatchDataset,
    _input_channels,
    _load_manifest,
    _require_torch,
    _resolve_device,
    _resolve_input_mode,
    build_model,
)


MODEL_NAMES = ("phys_tau_net", "unet", "attn_unet", "unet_pp", "deeplabv3p")
DISPLAY_NAMES = {
    "phys_tau_net": "PhysTAUNet",
    "unet": "U-Net",
    "attn_unet": "Attention U-Net",
    "unet_pp": "U-Net++",
    "deeplabv3p": "DeepLabV3+",
}


def _cuda_sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def _checkpoint_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024.0 * 1024.0)


def _estimate_conv_flops_per_image(torch, model, in_channels: int, patch_size: int, device) -> tuple[float, float]:
    """Return approximate GFLOPs and GMACs per one image.

    Convention: one multiply-add is counted as 2 FLOPs. The estimate includes
    Conv2d, ConvTranspose2d, and Linear modules. It intentionally ignores
    BatchNorm, activations, pooling, concatenation, and interpolation, so use it
    as an architecture-comparison estimate rather than an exact profiler trace.
    """

    macs = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal macs
        out = output
        batch = int(out.shape[0])
        out_channels = int(out.shape[1])
        out_h = int(out.shape[2])
        out_w = int(out.shape[3])
        kernel_h, kernel_w = module.kernel_size
        in_per_group = int(module.in_channels // module.groups)
        macs += batch * out_channels * out_h * out_w * in_per_group * kernel_h * kernel_w

    def linear_hook(module, inputs, output):
        nonlocal macs
        batch = int(output.shape[0]) if output.ndim > 1 else 1
        macs += batch * int(module.in_features) * int(module.out_features)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model_was_training = model.training
    model.eval()
    with torch.inference_mode():
        dummy = torch.zeros(1, in_channels, patch_size, patch_size, dtype=torch.float32, device=device)
        _cuda_sync(torch, device)
        _ = model(dummy)
        _cuda_sync(torch, device)

    for hook in hooks:
        hook.remove()
    if model_was_training:
        model.train()

    gmacs = macs / 1e9
    gflops = (2.0 * macs) / 1e9
    return float(gflops), float(gmacs)


def _load_model(torch, checkpoint_path: Path, requested_device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint.get("config", {})
    model_name = train_config.get("model", checkpoint_path.parent.name)
    input_mode = train_config.get("resolved_input_mode") or _resolve_input_mode(
        train_config.get("input_mode", "l1c_pair_methane"),
        model_name,
    )
    device = _resolve_device(requested_device)
    model = build_model(model_name, in_channels=_input_channels(input_mode)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_name, input_mode, train_config, device


def _prepare_batches(args, torch, DataLoader, input_mode: str, device):
    df = _load_manifest(Path(args.manifest), carbon_mapper_only=args.carbon_mapper_only)
    if "pair_ok" in df.columns:
        df = df[df["pair_ok"].astype(bool)].copy()
    df = df[df["split"].astype(str) == args.split].copy()
    if args.limit_rows is not None:
        df = df.head(args.limit_rows)
    if df.empty:
        raise ValueError(f"No {args.split} rows available in {args.manifest}")

    needed_images = args.batch_size * (args.warmup_batches + args.timed_batches)
    dataset = _SyntheticPlumePatchDataset(
        df,
        patch_size=args.patch_size,
        length=needed_images,
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
    batches = []
    for batch in loader:
        batches.append(batch["x"].to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda"))
        if len(batches) >= args.warmup_batches + args.timed_batches:
            break
    if len(batches) < args.warmup_batches + 1:
        raise ValueError("Not enough batches prepared for timing")
    _cuda_sync(torch, device)
    return batches, int(len(df))


def _benchmark_one(args, checkpoint_path: Path) -> dict[str, object]:
    torch, _, _, DataLoader, _ = _require_torch()
    model, model_name, input_mode, train_config, device = _load_model(torch, checkpoint_path, args.device)
    in_channels = _input_channels(input_mode)
    params_total, params_trainable = _count_parameters(model)
    gflops_per_image, gmacs_per_image = _estimate_conv_flops_per_image(
        torch, model, in_channels, args.patch_size, device
    )
    batches, background_rows = _prepare_batches(args, torch, DataLoader, input_mode, device)

    with torch.inference_mode():
        for x in batches[: args.warmup_batches]:
            _ = model(x)
        _cuda_sync(torch, device)

        batch_seconds = []
        image_seconds = []
        timed_images = 0
        for x in batches[args.warmup_batches : args.warmup_batches + args.timed_batches]:
            _cuda_sync(torch, device)
            start = time.perf_counter()
            _ = model(x)
            _cuda_sync(torch, device)
            elapsed = time.perf_counter() - start
            batch_seconds.append(elapsed)
            this_batch = int(x.shape[0])
            timed_images += this_batch
            image_seconds.append(elapsed / max(1, this_batch))

    total_forward_seconds = float(sum(batch_seconds))
    images_per_second = float(timed_images / total_forward_seconds) if total_forward_seconds > 0 else 0.0
    p50_ms = float(statistics.median(image_seconds) * 1000.0)
    mean_ms = float(statistics.mean(image_seconds) * 1000.0)
    p95_ms = float(pd.Series(image_seconds).quantile(0.95) * 1000.0)
    effective_tflops = float((gflops_per_image * images_per_second) / 1000.0)

    return {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "device": str(device),
        "input_mode": input_mode,
        "patch_size": int(args.patch_size),
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "timed_batches": int(args.timed_batches),
        "timed_images": int(timed_images),
        "background_rows": int(background_rows),
        "parameters": params_total,
        "trainable_parameters": params_trainable,
        "parameter_millions": params_total / 1e6,
        "checkpoint_mb": _checkpoint_size_mb(checkpoint_path),
        "gmacs_per_image": gmacs_per_image,
        "gflops_per_image": gflops_per_image,
        "forward_ms_per_image_mean": mean_ms,
        "forward_ms_per_image_p50": p50_ms,
        "forward_ms_per_image_p95": p95_ms,
        "forward_images_per_second": images_per_second,
        "effective_forward_tflops": effective_tflops,
        "synthetic_training": bool(train_config.get("synthetic_training", False)),
    }


def _write_figure(frame: pd.DataFrame, figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = frame.copy()
    plot_df["display_model"] = plot_df["model"].map(DISPLAY_NAMES).fillna(plot_df["model"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = ["#2F5597", "#F28E2B", "#59A14F"]

    axes[0].bar(plot_df["display_model"], plot_df["forward_ms_per_image_p50"], color=colors[0])
    axes[0].set_title("Forward latency")
    axes[0].set_ylabel("p50 ms / image")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(plot_df["display_model"], plot_df["forward_images_per_second"], color=colors[1])
    axes[1].set_title("Forward throughput")
    axes[1].set_ylabel("images / second")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(plot_df["display_model"], plot_df["gflops_per_image"], color=colors[2])
    axes[2].set_title("Model compute estimate")
    axes[2].set_ylabel("GFLOPs / image")
    axes[2].tick_params(axis="x", rotation=25)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig.suptitle("Held-Out Synthetic TEST Inference Speed and Compute", fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark model-only inference speed and approximate GFLOPs on held-out synthetic TEST patches."
    )
    parser.add_argument("--manifest", default="data/training_l1c/segmentation_training_curated.csv")
    parser.add_argument("--benchmark-dir", default="data/models/synthetic_publication_benchmark")
    parser.add_argument("--out", default="data/outputs/tables/synthetic_inference_speed_test.csv")
    parser.add_argument("--figure", default="data/outputs/publication_figures/inference_speed_gflops_test.png")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--timed-batches", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Reserved for reporting parity; not used for speed.")
    parser.add_argument("--min-plume-fraction", type=float, default=0.01)
    parser.add_argument("--max-plume-fraction", type=float, default=0.20)
    parser.add_argument("--min-tau", type=float, default=0.004)
    parser.add_argument("--max-tau", type=float, default=0.055)
    parser.add_argument("--carbon-mapper-only", action="store_true", default=True)
    parser.add_argument("--include-all-sources", action="store_true", help="Use all manifest sources instead of Carbon Mapper only.")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.device = str(args.device).strip().rstrip(".")
    if args.include_all_sources:
        args.carbon_mapper_only = False

    benchmark_dir = Path(args.benchmark_dir)
    rows = []
    for model_name in args.models:
        checkpoint_path = benchmark_dir / model_name / "best.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: missing checkpoint {checkpoint_path}")
            continue
        print(f"Benchmarking {model_name} on held-out {args.split} synthetic patches...")
        rows.append(_benchmark_one(args, checkpoint_path))

    if not rows:
        raise ValueError(f"No checkpoints found under {benchmark_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("forward_ms_per_image_p50")
    frame.to_csv(out_path, index=False)
    out_json = out_path.with_suffix(".json")
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_figure(frame, Path(args.figure))
    print(frame.to_string(index=False))
    print(f"Wrote {out_path}")
    print(f"Wrote {out_json}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
