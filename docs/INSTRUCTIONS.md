# Project Instructions

This project is a YOLO experiment testbed. The first completed layer is the
COCO dataset loader for images stored in COCO folders with YOLO-format label
files.

## Environment

Use the source tree directly while developing:

```bash
cd /data/yolo
export PYTHONPATH=/data/yolo/src
```

The package metadata is in `pyproject.toml`. If you create an environment with
all dev tools installed, this is the intended editable install command:

```bash
python3 -m pip install -e '.[dev]'
```

## Dataset Layout

The local COCO dataset is expected at `/data/yolo/datasets/coco`:

```text
datasets/coco/
  images/train2017/*.jpg
  images/val2017/*.jpg
  labels/train2017/*.txt
  labels/val2017/*.txt
  train2017.txt
  val2017.txt
```

Each label row must use normalized YOLO detection format:

```text
class_id x_center y_center width height
```

The split files may contain relative paths such as
`./images/train2017/000000109622.jpg`. The loader resolves the matching label
path by replacing `images` with `labels` and changing the suffix to `.txt`.

## Load One Sample

```python
from yolo_tests.data import CocoYoloDetection

dataset = CocoYoloDetection(
    root='/data/yolo/datasets/coco',
    split='train2017',
    image_size=640,
    letterbox=True,
)

image, target = dataset[0]
print(image.shape)              # torch.Size([3, 640, 640])
print(target['labels'].shape)   # [num_objects]
print(target['boxes'].shape)    # [num_objects, 4], pixel xyxy
print(target['boxes_yolo'])     # [num_objects, 4], normalized cxcywh
```

The image tensor is RGB float32 in `[0, 1]` with shape `[3, H, W]`.

## Target Fields

Each sample returns `(image, target)`.

```text
target['boxes']                 pixel xyxy boxes after resize/letterbox
target['boxes_yolo']            normalized cxcywh boxes after resize/letterbox
target['boxes_yolo_original']   normalized cxcywh boxes from the label file
target['labels']                class ids as int64
target['image_id']              dataset index tensor
target['orig_size']             original [height, width]
target['size']                  current [height, width]
target['path']                  image path
target['label_path']            label path
```

Use `boxes_yolo` for YOLO losses. Use `boxes` for debugging, visualization, or
TorchVision-style utilities.

## Load Batches

```python
from yolo_tests.data import CocoYoloDetection, create_coco_yolo_dataloader

dataset = CocoYoloDetection(
    root='/data/yolo/datasets/coco',
    split='train2017',
    image_size=640,
    letterbox=True,
)

loader = create_coco_yolo_dataloader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

batch = next(iter(loader))
images = batch['images']              # [B, 3, 640, 640]
yolo_targets = batch['yolo_targets']  # [N, 6]
```

`yolo_targets` is a flat tensor with one row per object:

```text
batch_index class_id x_center y_center width height
```

The collate function also preserves `batch['targets']`, a list of the original
per-image target dictionaries.

## Distributed Runs

The dataloader helper checks `torch.distributed.is_initialized()`. If a process
group exists, it uses `torch.utils.data.DistributedSampler`; otherwise it uses a
normal shuffled dataloader. This works for both DDP and FSDP because the dataset
sharding lives in the sampler, not in the model wrapper.

Inside a future training loop, call `set_epoch` when a distributed sampler is
active:

```python
sampler = getattr(loader, 'sampler', None)
if hasattr(sampler, 'set_epoch'):
    sampler.set_epoch(epoch)
```

A future training entrypoint can be launched with the usual PyTorch launcher
shape:

```bash
torchrun --standalone --nproc-per-node=8 scripts/train.py --config configs/data/coco_local.yaml
```

For multi-node jobs, pass your cluster rendezvous settings to `torchrun`; the
loader code does not need to change.

## Test The Loader

Run the real COCO smoke test:

```bash
PYTHONPATH=/data/yolo/src python3 /data/yolo/scripts/check_coco_loader.py \
  --root /data/yolo/datasets/coco \
  --split train2017 \
  --batch-size 2 \
  --image-size 640
```

Expected output includes:

```text
samples: 118287
image_batch_shape: (2, 3, 640, 640)
flat_yolo_targets_shape: (..., 6)
```

Run syntax checks:

```bash
python3 -m compileall -q src scripts tests
```

Run unit tests if `pytest` is installed:

```bash
PYTHONPATH=/data/yolo/src python3 -m pytest -q
```

If `pytest` is missing, install the dev dependencies in your environment first.

## Educational ResNet Training

This project now includes a simple one-anchor YOLO-style trainer. The default
training backbone is pretrained torchvision ResNet-101. The model reads three
feature levels (`layer2`, `layer3`, `layer4`), upsamples the last two levels to
the first level, concatenates all three, and predicts `80 + 4` channels with a
1x1 convolution.

Run a tiny CPU smoke training pass without downloading pretrained weights:

```bash
PYTHONPATH=/data/yolo/src python3 /data/yolo/scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --backbone resnet18 \
  --weights none \
  --batch-size 2 \
  --workers 0 \
  --image-size 128 \
  --max-steps 1 \
  --device cpu
```


Run on this one node with the two local A40 GPUs:

```bash
PYTHONPATH=/data/yolo/src torchrun \
  --standalone \
  --nproc-per-node=2 \
  /data/yolo/scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --backbone resnet101 \
  --weights default \
  --wrap ddp \
  --batch-size 8 \
  --workers 8 \
  --image-size 640 \
  --progress auto \
   --nccl-safe-mode
```

`--progress auto` shows a rank-0 progress bar with loss, class loss, box loss, positives, and throughput. Use `--progress text` for periodic plain logs or `--progress none` for summary-only runs.

After every epoch the trainer runs validation on `--val-split val2017` by
default, prints `mAP@0.50`, and saves one overlay image to
`runs/resnet_one_anchor/val_examples/epoch_001.png`. Green boxes are validation
labels and red dashed boxes are predictions. The plotted image cycles by epoch
by default; use `--val-plot-sample-index 0` to pin a fixed validation image.
Plot overlays use `--val-plot-score-threshold`, which is separate from the
lower score threshold used for mAP. Use `--val-max-steps` for quick checks,
`--no-validation` to skip validation, and `--no-plot-val-example` to skip the
saved image.

Loss/progress metrics:

```text
loss      weighted total loss used for backprop
pos_cls   true-object class BCE plus a small penalty for other class channels
bg        background class BCE before negative weighting
box       IoU loss plus a small normalized L1 box penalty
iou       mean IoU at positive assigned grid cells
pcls      mean predicted probability of the target class at positive cells
pos       number of positive assigned grid cells in the local rank batch
```

If the run spins at 100% GPU before the first batch, it is usually hanging
during NCCL/DDP model synchronization. Use `--debug-stages` to see the last
completed stage and keep `--nccl-safe-mode` enabled on this local two-A40 node.
If NCCL reports `Call to socket failed: Address family not supported by
protocol`, force IPv4 sockets with `NCCL_SOCKET_FAMILY=AF_INET`. The trainer
sets this by default for CUDA distributed runs unless you override it.
If NCCL then tries a link-local address such as `169.254.x.x`, pin the NCCL
socket interface to the routable interface for the `--master-addr` network.
The interface name can differ by node, for example `eth_10g0` on one node and
`eth_1g0` on another.
If it stops at `wrapping model wrap=ddp`, add `--dist-smoke-test` for a tiny
CUDA all-reduce before model construction. If that hangs too, the issue is NCCL
connectivity rather than the training loop.

`--nproc-per-node` must match the number of GPUs you want to use on this node.
This machine has 2 A40 GPUs, so use `--nproc-per-node=2`. `--batch-size` is per
GPU; with two GPUs, `--batch-size 8` means a global batch size of 16. If you want
a global batch size of 64, use `--batch-size 32` with two GPUs, assuming memory
is sufficient.

Run the intended pretrained ResNet-101 version on one node:

```bash
PYTHONPATH=/data/yolo/src python3 /data/yolo/scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --backbone resnet101 \
  --weights default \
  --batch-size 8 \
  --image-size 640
```

Run pretrained ResNet-101 with DDP across two nodes. On node 0:

```bash
PYTHONPATH=/data/yolo/src NCCL_SOCKET_FAMILY=AF_INET torchrun \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=2 \
  --master-addr=10.206.150.8 \
  --master-port=29500 \
  /data/yolo/scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --backbone resnet101 \
  --weights default \
  --wrap ddp \
  --batch-size 16 \
  --workers 8 \
  --image-size 640 \
  --nccl-socket-ifname eth_10g0 \
  --nccl-safe-mode

PYTHONPATH=/data/yolo/src NCCL_SOCKET_FAMILY=AF_INET torchrun \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=2 \
  --master-addr=10.206.150.8 \
  --master-port=29500 \
  /data/yolo/scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --backbone resnet101 \
  --weights default \
  --wrap ddp \
  --batch-size 16 \
  --workers 8 \
  --image-size 640 \
  --nccl-socket-ifname eth_1g0 \
  --nccl-safe-mode
```

On node 1, use the same command but change `--node-rank=1`. Add more nodes by
raising `--nnodes` and assigning each machine a unique `--node-rank`. Set
`--nproc-per-node` to the GPU count on each node, for example `2` on a node with
two A40 GPUs.

For pretrained weights, make sure every node can read the torchvision weight
cache. The easiest options are to pre-cache the weights on each node, or point
`TORCH_HOME` to a shared filesystem before launching `torchrun`.

The model code lives in `src/yolo_tests/models/resnet_one_anchor.py`. The loss
code lives in `src/yolo_tests/losses/one_anchor_yolo_loss.py`. The loss is
written for clarity: it assigns every ground-truth object to one grid cell, uses
BCE over the 80 class logits, and applies Smooth L1 only to boxes at positive
grid cells.

## Current Features

Implemented now:

- OpenCV image loading with BGR to RGB conversion.
- TorchVision tensor conversion and box format conversion.
- YOLO txt label parsing for normalized `class cx cy w h` labels.
- Missing label files treated as images with no objects.
- COCO split-file loading and image-directory discovery fallback.
- Square or `(height, width)` image sizing.
- Letterbox resize with box updates.
- Direct resize mode with `letterbox=False`.
- Per-sample targets with both pixel `xyxy` and normalized YOLO boxes.
- Batch collate with flat YOLO target tensor.
- COCO class-name loading from `coco.yaml`.
- DistributedSampler activation when `torch.distributed` is initialized.
- Educational pretrained ResNet-101 DDP training entrypoint.
- Per-epoch validation with mAP at a configurable IoU threshold, default 0.50.
- Per-epoch validation example image with ground-truth and predicted boxes.

Not implemented yet:

- Large-scale backbone comparison registry and experiment sweeps.
- Mosaic, mixup, random affine, HSV, or advanced detection augmentation.
- Checkpointing and experiment logging.
- Production-grade DDP/FSDP checkpoint sharding and resume logic.

## Useful Files

```text
configs/data/coco_local.yaml       local dataset and loader defaults
src/yolo_tests/data/coco_yolo.py   dataset, collate, and dataloader code
scripts/check_coco_loader.py       real COCO loader smoke check
scripts/train_resnet_yolo.py       ResNet-101 DDP training entrypoint
src/yolo_tests/models/             educational detector model
src/yolo_tests/losses/             educational one-anchor loss
tests/test_coco_yolo_dataset.py    small synthetic tests
coco.yaml                          COCO class names and Ultralytics data config
```
