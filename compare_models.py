# -*- coding: utf-8 -*-
"""
Baseline vs CBAM vs CA model comparison script
Evaluate mAP and reasoning speed (FPS), output comparison table and bar chart
"""

import argparse
import json
import time
from pathlib import Path
import torch

def _resolve_torch_device(device: str) -> str:
    """Convert Ultralytics-style device string to a valid torch device string."""
    if device in ("", "cpu"):
        return "cpu"
    try:
        idx = int(device)
        return f"cuda:{idx}"
    except ValueError:
        return device


def benchmark_fps(model_path, imgsz=640, warmup=10, runs=100, device=""):
    """Benchmark reasoning speed (FPS)"""
    from ultralytics import YOLO

    model = YOLO(model_path)
    torch_dev = _resolve_torch_device(device)
    img = torch.zeros(1, 3, imgsz, imgsz).to(torch_dev)
    use_cuda = torch_dev.startswith("cuda")

    for _ in range(warmup):
        model.predict(source=img, verbose=False, device=device)

    if use_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(source=img, verbose=False, device=device)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return runs / elapsed


def evaluate_map(model, data, device="", patch_ca=False):
    """Evaluate mAP"""
    if patch_ca or "cbam" in str(model).lower() or "ca" in str(model).lower():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from scripts.patch_ultralytics import patch_ultralytics
        patch_ultralytics()

    from ultralytics import YOLO

    m = YOLO(model)
    r = m.val(data=data, split="val", imgsz=640, batch=16, device=device, verbose=False)
    metrics = r.box
    return {
        "mAP50": float(getattr(metrics, "map50", 0) or 0),
        "mAP50-95": float(getattr(metrics, "map", 0) or 0),
    }


def plot_comparison(results, output_path):
    """Generate mAP vs FPS comparison bar chart"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [r["name"] for r in results]
    mAP50 = [r.get("mAP50") if r.get("mAP50") is not None else 0 for r in results]
    mAP50_95 = [r.get("mAP50-95") if r.get("mAP50-95") is not None else 0 for r in results]
    fps = [r.get("fps") if r.get("fps") is not None else 0 for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    x = np.arange(len(names))
    w = 0.35

    axes[0].bar(x - w / 2, mAP50, w, label="mAP50", color="#2ecc71")
    axes[0].bar(x + w / 2, mAP50_95, w, label="mAP50-95", color="#3498db")
    axes[0].set_ylabel("mAP")
    axes[0].set_title("mAP Indicator Comparison")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, fps, color="#e74c3c", alpha=0.8)
    axes[1].set_ylabel("FPS")
    axes[1].set_title("Reasoning speed (FPS) Comparison")
    axes[1].tick_params(axis="x", rotation=15)
    plt.setp(axes[1].xaxis.get_majorticklabels(), ha="right")
    axes[1].grid(axis="y", alpha=0.3)

    baseline_map50 = mAP50[0] if mAP50[0] > 0 else 1e-9
    baseline_map50_95 = mAP50_95[0] if mAP50_95[0] > 0 else 1e-9
    delta_map50 = [(v - mAP50[0]) / baseline_map50 * 100 for v in mAP50]
    delta_map50_95 = [(v - mAP50_95[0]) / baseline_map50_95 * 100 for v in mAP50_95]

    axes[2].bar(x - w / 2, delta_map50, w, label="ΔmAP50 (%)", color="#e67e22")
    axes[2].bar(x + w / 2, delta_map50_95, w, label="ΔmAP50-95 (%)", color="#9b59b6")
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("Relative Change (%)")
    axes[2].set_title("Accuracy Improvement vs Baseline")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15, ha="right")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("YOLOv8n Baseline vs Attention Mechanism Improved Model Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Baseline vs CBAM vs CA Model Comparison")
    parser.add_argument(
        "--baseline",
        type=str,
        default="runs/detect/yolov8n/weights/best.pt",
        help="Baseline model weights (fine-tuned)",
    )
    parser.add_argument(
        "--cbam",
        type=str,
        default="runs/detect/yolov8n-cbam/weights/best.pt",
        help="CBAM model weights",
    )
    parser.add_argument(
        "--ca",
        type=str,
        default="runs/detect/yolov8n-ca/weights/best.pt",
        help="CA model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data_subset.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison.json",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="results/comparison.png",
        help="Bar chart output path",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--skip-eval", action="store_true", help="Only test FPS")
    parser.add_argument("--skip-fps", action="store_true", help="Only test mAP")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate bar chart")
    args = parser.parse_args()

    models = [
        ("Baseline (YOLOv8n)", args.baseline, False),
        ("CBAM", args.cbam, False),
        ("CA", args.ca, True),
    ]

    results = []

    for name, path, patch_ca in models:
        # Local weights file must exist; yolov8n.pt etc. are automatically downloaded by ultralytics
        if "/" in path or "\\" in path:
            if not Path(path).exists():
                print(f"Skip {name}: file not found {path}")
                continue

        row = {"name": name, "model": path}
        print(f"\n>>> Evaluate: {name} ({path})")

        if not args.skip_fps:
            try:
                fps = benchmark_fps(path, device=args.device)
                row["fps"] = round(fps, 2)
                print(f"  FPS: {fps:.2f}")
            except Exception as e:
                row["fps"] = None
                print(f"  FPS failed: {e}")

        if not args.skip_eval:
            try:
                m = evaluate_map(path, args.data, device=args.device, patch_ca=patch_ca)
                row["mAP50"] = round(m["mAP50"], 4)
                row["mAP50-95"] = round(m["mAP50-95"], 4)
                print(f"  mAP50: {m['mAP50']:.4f}, mAP50-95: {m['mAP50-95']:.4f}")
            except Exception as e:
                row["mAP50"] = row["mAP50-95"] = None
                print(f"  mAP failed: {e}")

        results.append(row)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results and not args.no_plot:
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        plot_comparison(results, args.plot)

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    for r in results:
        fps = r.get("fps", "N/A")
        m50 = r.get("mAP50", "N/A")
        m5095 = r.get("mAP50-95", "N/A")
        print(f"  {r['name']}: FPS={fps}, mAP50={m50}, mAP50-95={m5095}")
    print("=" * 60)
    print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
