# YOLOv8n Small Object Detection - Attention Mechanism Improvement

Integrating lightweight attention mechanisms (CBAM/CA) into the YOLOv8n backbone network, optimized for detecting small objects (area < 32×32 pixels) on the COCO dataset. Comparing baseline and improved models via mAP and inference speed (FPS) to analyze the effectiveness of attention mechanisms in enhancing small object detection performance.

## Environment Configuration

```bash
pip install -r requirements.txt
```

## Attention Module Registration

```bash
python scripts/patch_ultralytics.py
```

## Datasets

### Full Datasets (COCO train2017 / val2017)

Place the COCO datasets under `data/`:

```
data/
└──images
   ├── train2017/        # Training images + labels
   ├── val2017/          # Verification Image + Label
   └── data.yaml         # Configuration file
```

### COCO Small Object Subset (Optional)

Filter images containing small objects (area < 32² = 1024 pixels) from the full COCO annotations:

```bash
python scripts/create_coco_small_subset.py \
  --anno /path/to/instances_train2017.json \
  --output annotations/instances_train2017_small.json
```

## Training

### Quick Validation (COCO8 + 10 epochs)

```bash
python train.py --model yolov8n --quick
python train.py --model yolov8n-cbam --quick
python train.py --model yolov8n-ca --quick
```

### Full Training

```bash
#  YOLOv8n
python train.py --model yolov8n --epochs 100

# CBAM 
python train.py --model yolov8n-cbam --epochs 100

# CA 
python train.py --model yolov8n-ca --epochs 100
```

Training results are saved in `runs/detect/{model name}/`, including weights (`weights/best.pt`) and training curves.

## Evaluation

```bash
python evaluate.py \
  --model runs/detect/yolov8n-cbam/weights/best.pt \
  --data data/data.yaml \
  --save-json results/cbam_map.json
```

## Model Comparison

Compare the mAP (mAP50 / mAP50-95) and inference speed (FPS) of the baseline, CBAM, and CA models, and generate a bar chart:

```bash
python compare_models.py --data data/data.yaml
```

Default weight path:

| model    | path                                       |
|----------|--------------------------------------------|
| baseline | `yolov8n.pt`（自动下载）                    |
| CBAM     | `runs/detect/yolov8n-cbam/weights/best.pt` |
| CA       | `runs/detect/yolov8n-ca/weights/best.pt`   |

Custom Path：

```bash
python compare_models.py \
  --baseline yolov8n.pt \
  --cbam runs/detect/yolov8n-cbam/weights/best.pt \
  --ca runs/detect/yolov8n-ca/weights/best.pt \
  --data data/data.yaml \
  --plot results/comparison.png
```

Output:
- `results/comparison.json` — Numerical results
- `results/comparison.png` — Bar chart comparing mAP and FPS

## Attention Mechanism Explanation

| Module | Position | Principle |
| **CBAM** | Backbone P3/P4  | Channel Attention + Spatial Attention, enhancing key feature channels and spatial regions |
| **CA**   | Backbone P3/P4  | Coordinate attention encodes positional information into channel attention, making it more sensitive to small object locations |

## Important Notes

1. `patch_ultralytics.py` will run automatically before training CBAM or CA models; manual execution is unnecessary.
2. Full COCO training is time-consuming; we recommend using `--quick` to validate the workflow first.
3. `compare_models.py` will automatically skip models with missing weights, ensuring all three models are fully trained before comparison.
