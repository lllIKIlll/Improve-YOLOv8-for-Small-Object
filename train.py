"""
YOLOv8n small object detection - attention mechanism improved model training script
Supports baseline / CBAM / CA three models
Uses COCO pre-trained weights + FP16 mixed precision acceleration
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n small object detection model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolov8n", "yolov8n-cbam", "yolov8n-ca"],
        default="yolov8n",
        help="Model type",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data_subset.yaml",
        help="Dataset configuration (default uses 15,000 subset)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.02, help="Initial learning rate")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation mode: use coco8 small dataset + 10 epochs",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--name", type=str, default="", help="Experiment name")
    parser.add_argument("--fraction", type=float, default=0.127,
                        help="Training set sampling ratio (0.127≈15000/118287)")
    parser.add_argument("--lmdb", action="store_true",
                        help="Enable LMDB image reading acceleration (run scripts/create_lmdb.py first)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.data = "data/coco8.yaml"
        args.epochs = 10
        args.fraction = 1.0
        print("Quick validation mode: data=coco8, epochs=10")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # CBAM / CA models need to patch ultralytics (some versions of tasks.py do not import CBAM/CA)
    if args.model in ("yolov8n-cbam", "yolov8n-ca"):
        from scripts.patch_ultralytics import patch_ultralytics
        patch_ultralytics()

    # LMDB acceleration: replace BaseDataset.load_image with memory mapping read
    if args.lmdb:
        from scripts.lmdb_patch import patch_lmdb_loader
        patch_lmdb_loader()

    from ultralytics import YOLO

    # All models load COCO pre-trained weights, skip initial feature learning stage
    if args.model == "yolov8n":
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(f"models/{args.model}.yaml")
        # Use smart weight transfer: handle layer index offset caused by CBAM/CA insertion
        from scripts.transfer_weights import transfer_pretrained_weights
        transfer_pretrained_weights(model, "yolov8n.pt")

    name = args.name or args.model
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=name,
        resume=args.resume,
        optimizer="SGD",
        lr0=args.lr0,
        mosaic=0.0,
        amp=True,
        plots=True,
        pretrained=True,
        fraction=args.fraction,
    )
    print(f"Training completed: {args.project}/{name}")


if __name__ == "__main__":
    main()
