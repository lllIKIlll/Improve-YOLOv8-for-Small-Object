# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model weights or yaml path")
    parser.add_argument("--data", type=str, default="data/coco_small.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--save-json", type=str, default="", help="Save mAP results to JSON")
    parser.add_argument("--patch-ca", action="store_true", help="If CA model, patch first")
    args = parser.parse_args()

    if args.patch_ca or "ca" in args.model.lower():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from scripts.patch_ultralytics import patch_ultralytics
        patch_ultralytics()

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    metrics = results.box
    if metrics is None:
        print("No box metrics obtained")
        return

    out = {
        "mAP50": float(getattr(metrics, "map50", 0) or 0),
        "mAP50-95": float(getattr(metrics, "map", 0) or 0),
        "precision": float(getattr(metrics, "mp", 0) or 0),
        "recall": float(getattr(metrics, "mr", 0) or 0),
    }
    print("=" * 50)
    print("Evaluation results (COCO small subset)")
    print("=" * 50)
    for k, v in out.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.save_json}")


if __name__ == "__main__":
    main()
