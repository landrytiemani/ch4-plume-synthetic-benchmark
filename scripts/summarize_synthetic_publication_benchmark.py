from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "phys_tau_net": "PhysTAUNet",
    "unet": "U-Net",
    "attn_unet": "Attention U-Net",
    "unet_pp": "U-Net++",
    "deeplabv3p": "DeepLabV3+",
}


def _fmt(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "nan"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(_fmt(value) if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _best_history_row(path: Path, model: str) -> dict[str, object] | None:
    if not path.exists():
        print(f"Skipping {model}: missing {path}")
        return None
    df = pd.read_csv(path)
    if df.empty or "f1" not in df.columns:
        print(f"Skipping {model}: incompatible history columns")
        return None
    row = df.sort_values(["f1", "tolerant_f1"], ascending=False).iloc[0]
    return {
        "model": model,
        "label": MODEL_LABELS.get(model, model),
        "best_epoch": int(row["epoch"]),
        "f1": float(row["f1"]),
        "iou": float(row["iou"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "tolerant_f1": float(row["tolerant_f1"]),
        "tolerant_iou": float(row["tolerant_iou"]),
        "accuracy": float(row["accuracy"]),
        "positive_fraction_mean": float(row["positive_fraction_mean"]),
        "predicted_positive_fraction_mean": float(row["predicted_positive_fraction_mean"]),
    }


def _test_row(path: Path, model: str) -> dict[str, object]:
    prefix = "test_"
    empty = {
        f"{prefix}f1": float("nan"),
        f"{prefix}iou": float("nan"),
        f"{prefix}precision": float("nan"),
        f"{prefix}recall": float("nan"),
        f"{prefix}tolerant_f1": float("nan"),
        f"{prefix}tolerant_iou": float("nan"),
        f"{prefix}positive_fraction_mean": float("nan"),
        f"{prefix}predicted_positive_fraction_mean": float("nan"),
    }
    if not path.exists():
        print(f"Skipping test metrics for {model}: missing {path}")
        return empty
    df = pd.read_csv(path)
    if df.empty:
        print(f"Skipping test metrics for {model}: empty {path}")
        return empty
    row = df.iloc[0]
    out = {}
    for key in ["f1", "iou", "precision", "recall", "tolerant_f1", "tolerant_iou", "positive_fraction_mean", "predicted_positive_fraction_mean"]:
        out[f"{prefix}{key}"] = float(row.get(key, float("nan")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize synthetic-only publication benchmark histories.")
    parser.add_argument("--benchmark-dir", default="data/models/synthetic_publication_benchmark")
    parser.add_argument("--out", default="data/outputs/tables/synthetic_publication_benchmark_summary.csv")
    parser.add_argument("--report", default="reports/synthetic_publication_benchmark_report.md")
    parser.add_argument("--models", nargs="+", default=["unet", "attn_unet", "unet_pp", "deeplabv3p", "phys_tau_net"])
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    rows = []
    for model in args.models:
        rec = _best_history_row(benchmark_dir / model / "history.csv", model)
        if rec is not None:
            rec.update(_test_row(benchmark_dir / model / "synthetic_eval_test.csv", model))
            rows.append(rec)
    if not rows:
        raise ValueError(f"No usable synthetic histories found under {benchmark_dir}")

    summary = pd.DataFrame(rows).sort_values(["test_f1", "test_tolerant_f1", "f1"], ascending=False)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    best = summary.iloc[0]
    report = f"""# Synthetic Sentinel-2 L1C Methane Plume Segmentation Benchmark

## Benchmark Scope

This project evaluates controlled, physics-injected synthetic methane plumes on Sentinel-2 L1C event/reference chips. No Carbon Mapper or EMIT labels are used for the main claim.

## Main Result

Best model: **{best['label']}**

- F1: `{_fmt(best['f1'])}`
- IoU: `{_fmt(best['iou'])}`
- Tolerant F1: `{_fmt(best['tolerant_f1'])}`
- Precision: `{_fmt(best['precision'])}`
- Recall: `{_fmt(best['recall'])}`
- Held-out synthetic TEST F1: `{_fmt(best['test_f1'])}`
- Held-out synthetic TEST IoU: `{_fmt(best['test_iou'])}`
- Held-out synthetic TEST tolerant F1: `{_fmt(best['test_tolerant_f1'])}`

## Benchmark Table

{_markdown_table(summary)}

## Publication Claim

PhysTAUNet is benchmarked against U-Net, Attention U-Net, U-Net++, and DeepLabV3+ on a synthetic Sentinel-2 L1C methane plume segmentation task. The synthetic plumes are injected into real Sentinel-2 L1C backgrounds using methane-sensitive SWIR absorption behavior and irregular plume morphology.
"""
    report_path.write_text(report, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
