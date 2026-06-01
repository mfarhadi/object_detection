# Experiment Monitoring And Reproducibility

This project uses two complementary tools:

- W&B monitors live training behavior: losses, throughput, validation metrics,
  validation images, and optional model artifacts.
- DVC records the reproducible experiment contract: parameters, data/code
  dependencies, model outputs, metrics files, and artifact inventory metadata.

The trainer also writes local JSON metadata even when W&B and DVC are not
installed. That keeps every run auditable from the filesystem alone.

## Install

```bash
cd /data/yolo
python3 -m pip install -e '.[dev,experiment]'
export PYTHONPATH=/data/yolo/src
```

For online W&B logging, provide the key through the environment. Do not commit
API keys to Git, DVC, notebooks, or shell scripts.

```bash
export WANDB_API_KEY='replace-with-your-key'
```

W&B offline mode does not need a key:

```bash
PYTHONPATH=/data/yolo/src python3 scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --wandb-mode offline
```

## W&B Monitoring

Enable W&B explicitly:

```bash
PYTHONPATH=/data/yolo/src python3 scripts/train_resnet_yolo.py \
  --root /data/yolo/datasets/coco \
  --wandb-mode online \
  --wandb-project yolo-backbone-tests
```

Useful W&B switches:

```text
--wandb-mode disabled|offline|online
--wandb-project yolo-backbone-tests
--wandb-entity <team-or-user>
--wandb-name <run-name>
--wandb-tags baseline,resnet101,coco
--wandb-watch gradients|parameters|all
--wandb-log-checkpoints none|last|all
--wandb-log-every 10
```

Only rank 0 initializes W&B, so DDP/FSDP jobs do not create one run per GPU.
The trainer logs step-level training metrics every `--wandb-log-every` steps,
epoch summaries, validation loss, `mAP@IoU`, validation counts, and the saved
validation overlay image. Dataset and model inventory JSON files are logged as
small W&B artifacts by default. Checkpoint files are uploaded only when
`--wandb-log-checkpoints` is set.

## Local Run Metadata

Every rank-0 run writes:

```text
runs/resnet_anchor_free/run_manifest.json
runs/resnet_anchor_free/metrics.json
runs/resnet_anchor_free/metrics_history.jsonl
runs/resnet_anchor_free/inventory/dataset_inventory.json
runs/resnet_anchor_free/inventory/model_inventory.json
```

When the DVC pipeline is used, report files are written outside the checkpoint
directory:

```text
reports/resnet_anchor_free/metrics.json
reports/resnet_anchor_free/metrics_history.jsonl
reports/resnet_anchor_free/inventory/dataset_inventory.json
reports/resnet_anchor_free/inventory/model_inventory.json
```

`run_manifest.json` captures command-line args, `sys.argv`, Git commit/status,
Python/PyTorch/TorchVision/CUDA runtime metadata, and the dataset inventory.

`metrics.json` is the latest machine-readable summary for DVC metrics. The
history file is line-delimited JSON with one train or validation record per
epoch.

## Dataset Inventory

The dataset inventory records:

- dataset root, split names, sample counts, image size, and letterbox setting
- split file fingerprints
- annotation JSON and image archive metadata when present
- total image bytes, label bytes, missing image count, missing label count
- a small sample of resolved image and label paths
- class-name YAML fingerprint

Small control files are SHA-256 hashed automatically. Large image archives and
model checkpoints are not hashed by default because COCO and checkpoints are
large. Use `--inventory-hash-files` only when the extra runtime is acceptable.

## Model Inventory

The model inventory records:

- architecture class, backbone, pretrained weight setting, class count, head
  width, and freeze setting
- total/trainable parameters and parameter bytes
- checkpoint directory, checkpoint count, total checkpoint bytes, and latest
  checkpoint metadata
- per-checkpoint path, size, modification time, and epoch when parseable

This inventory is useful for model reviews because it answers "what model is
this?" without opening the checkpoint.

## DVC Pipeline

DVC files added here:

```text
.dvc/config
.dvcignore
dvc.yaml
params.yaml
```

`params.yaml` holds the editable experiment contract. `dvc.yaml` defines one
stage, `train_resnet_anchor_free`, with:

- code dependencies under `scripts/` and `src/yolo_tests/`
- data dependency on `datasets/coco`
- parameter dependencies on `params.yaml`
- model output at `runs/resnet_anchor_free`
- metrics and inventory reports under `reports/resnet_anchor_free`
- artifact declarations for the COCO dataset and ResNet YOLO model

Run the DVC stage:

```bash
dvc repro train_resnet_anchor_free
dvc metrics show
```

Change parameters in `params.yaml`, then run `dvc repro` again. DVC will mark
the stage outdated when code, data, or parameter dependencies change.

## DVC Data And Model Storage

The repository declares the dataset and model paths for DVC. To make them fully
portable across machines, configure a remote and push the DVC cache:

```bash
dvc remote add -d storage /mnt/dvc/yolo
dvc add datasets/coco
dvc repro train_resnet_anchor_free
dvc push
git add .dvc .dvcignore dvc.yaml params.yaml datasets/coco.dvc
git add reports/resnet_anchor_free
git commit -m "Track YOLO dataset and model pipeline with DVC"
```

`dvc add datasets/coco` will hash a large local COCO tree, so expect it to take
time and require cache storage. The training stage output under
`runs/resnet_anchor_free` is the model artifact that DVC will cache after
`dvc repro`. For S3, Azure, GCS, or SSH storage, install the matching DVC remote
extra and use that remote URL instead of the local path above.

On another machine:

```bash
git clone <repo>
cd yolo
python3 -m pip install -e '.[dev,experiment]'
dvc pull
dvc repro train_resnet_anchor_free
```

## Reproducibility Notes

The trainer sets Python, NumPy, and PyTorch seeds from `--seed`. Checkpoints
include Python, NumPy, PyTorch, and CUDA RNG state so a later resume workflow
can restore the stochastic state. Use `--deterministic` when bit-for-bit
repeatability matters more than throughput.

For distributed jobs, keep the same launcher shape, GPU count, batch size,
seed, dataset version, and Git commit when comparing experiments. W&B is the
live monitoring layer; DVC is the source of truth for which data, code,
parameters, reports, and model outputs produced the result.
