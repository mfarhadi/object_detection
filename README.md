# YOLO Backbone Testbed

Experimental project for comparing YOLO detector backbones on COCO with a
dataset layer that can be reused from single-GPU runs through DDP/FSDP jobs.

The first implemented piece is a COCO image loader for Ultralytics-style YOLO
labels. It reads images with OpenCV, converts images and boxes with torchvision,
and returns both pixel `xyxy` boxes and normalized YOLO `cxcywh` boxes.

For usage examples, test commands, and the current feature list, see `docs/INSTRUCTIONS.md`.

## Loader smoke test

```bash
PYTHONPATH=/data/yolo/src python3 /data/yolo/scripts/check_coco_loader.py \
  --root /data/yolo/datasets/coco \
  --split train2017 \
  --batch-size 2 \
  --image-size 640
```

The dataloader helper automatically uses `torch.utils.data.DistributedSampler`
when `torch.distributed` is initialized, so the same loader can be used under
single-node or multi-node DDP/FSDP launchers.

## Package layout

```text
configs/data/coco_local.yaml       # Local dataset and loader defaults
src/yolo_tests/data/coco_yolo.py   # OpenCV + torchvision COCO/YOLO loader
scripts/check_coco_loader.py       # Loader smoke test
scripts/train_resnet_yolo.py       # Educational ResNet-101 DDP one-anchor trainer
src/yolo_tests/models/             # ResNet detector model
src/yolo_tests/losses/             # Clear one-anchor YOLO loss
tests/test_coco_yolo_dataset.py    # Synthetic dataset tests
```

## COCO 2017 (YOLO Format)

Full COCO 2017 object detection dataset in Ultralytics YOLO bbox format.

## Dataset location

```
/data/yolo/datasets/coco/
├── images/
│   ├── train2017/   # 118,287 images
│   └── val2017/     # 5,000 images
├── labels/
│   ├── train2017/   # YOLO .txt labels (class x y w h, normalized)
│   └── val2017/
├── train2017.txt
└── val2017.txt
```

## Train with YOLO

```bash
yolo detect train data=/data/yolo/coco.yaml model=yolo11n.pt epochs=100
```

## Re-download

```bash
python3 /data/yolo/download_coco.py
```

## Monitor in-progress download

```bash
tail -f /data/yolo/download_coco.log
```
