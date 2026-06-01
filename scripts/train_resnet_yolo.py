#!/usr/bin/env python3
"""Train the educational ResNet anchor-free YOLO model.

This script is intentionally small and explicit. It is useful for learning the
flow from dataloader -> model -> assignment/loss -> optimizer step. The model is
not intended to match modern YOLO accuracy yet.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import box_iou, nms

from yolo_tests.data import CocoYoloDetection, create_coco_yolo_dataloader, load_coco_class_names
from yolo_tests.losses import AnchorFreeYoloLoss
from yolo_tests.models import ResNetAnchorFreeYolo, ensure_resnet_weights_available

try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:  # pragma: no cover - depends on the local PyTorch build
    FullyShardedDataParallel = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - progress fallback handles this
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/yolo/datasets/coco"))
    parser.add_argument("--split", default="train2017")
    parser.add_argument("--val-split", default="val2017")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size. Defaults to --batch-size.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--val-workers", type=int, default=None, help="Validation workers. Defaults to the training worker count.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None, help="Stop early after this many steps per epoch.")
    parser.add_argument("--val-max-steps", type=int, default=None, help="Stop validation early after this many batches.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument(
        "--backbone",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
        default="resnet101",
        help="ResNet backbone. The default is pretrained ResNet-101.",
    )
    parser.add_argument("--weights", choices=["default", "pretrained", "imagenet", "none"], default="default")
    parser.add_argument("--anchor-size", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--head-channels", type=int, default=256, help="Channel width of the anchor-free detection head.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help="Accepted for torchrun compatibility.")
    parser.add_argument("--wrap", choices=["auto", "none", "ddp", "fsdp"], default="auto")
    parser.add_argument("--dist-backend", default="auto", help="auto, nccl, or gloo")
    parser.add_argument("--dist-timeout-minutes", type=int, default=10)
    parser.add_argument(
        "--nccl-socket-ifname",
        default=None,
        help="Override NCCL_SOCKET_IFNAME, for example eth_10g0 on the 10.206.x network.",
    )
    parser.add_argument(
        "--nccl-safe-mode",
        action="store_true",
        help="Force conservative NCCL settings by using IPv4 sockets and disabling P2P/IB/SHM paths.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet_anchor_free"))
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for Python, NumPy, and PyTorch.")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request deterministic PyTorch algorithms where possible. This can reduce throughput.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="JSON metrics summary path. Defaults to <output-dir>/metrics.json.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="Line-delimited JSON metrics history path. Defaults to <output-dir>/metrics_history.jsonl.",
    )
    parser.add_argument(
        "--inventory-dir",
        type=Path,
        default=None,
        help="Directory for dataset/model inventory JSON. Defaults to <output-dir>/inventory.",
    )
    parser.add_argument(
        "--write-inventory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write dataset and model inventory files for reproducibility and DVC metadata.",
    )
    parser.add_argument(
        "--inventory-hash-files",
        action="store_true",
        help="Include SHA-256 hashes for large dataset/model files. Off by default because COCO/checkpoints are large.",
    )
    parser.add_argument("--no-validation", action="store_true", help="Skip validation after each epoch.")
    parser.add_argument("--names-yaml", type=Path, default=Path("/data/yolo/coco.yaml"))
    parser.add_argument("--val-score-threshold", type=float, default=0.001)
    parser.add_argument("--val-iou-threshold", type=float, default=0.5)
    parser.add_argument("--val-nms-threshold", type=float, default=0.5)
    parser.add_argument("--val-pre-nms-topk", type=int, default=1000)
    parser.add_argument("--val-max-detections", type=int, default=100)
    parser.add_argument("--val-plot-max-boxes", type=int, default=20)
    parser.add_argument(
        "--val-plot-score-threshold",
        type=float,
        default=0.25,
        help="Score threshold used only for saved validation example overlays.",
    )
    parser.add_argument(
        "--val-plot-sample-index",
        type=int,
        default=None,
        help="Validation dataset index to plot every epoch. Defaults to cycling by epoch.",
    )
    parser.add_argument(
        "--plot-val-example",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one validation image with ground-truth and predicted boxes after each epoch.",
    )
    parser.add_argument("--log-every", type=int, default=10, help="Text log interval when progress bars are disabled.")
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default="online",
        help="Enable W&B logging on rank 0. Use online with WANDB_API_KEY set, or offline for local logs.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "yolo-backbone-tests"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument("--wandb-notes", default=os.environ.get("WANDB_NOTES"))
    parser.add_argument("--wandb-tags", default=os.environ.get("WANDB_TAGS", ""), help="Comma-separated W&B tags.")
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"), help="Local W&B run directory.")
    parser.add_argument(
        "--wandb-watch",
        choices=["none", "gradients", "parameters", "all"],
        default="none",
        help="Optional W&B parameter/gradient watching mode.",
    )
    parser.add_argument("--wandb-log-every", type=int, default=10, help="Step interval for W&B train metrics.")
    parser.add_argument(
        "--wandb-log-checkpoints",
        choices=["none", "last", "all"],
        default="none",
        help="Upload checkpoint files as W&B model artifacts. DVC remains the primary model versioning layer.",
    )
    parser.add_argument(
        "--wandb-log-inventory-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log dataset/model inventory JSON as small W&B artifacts when W&B is enabled.",
    )
    parser.add_argument(
        "--progress",
        choices=["auto", "tqdm", "text", "none"],
        default="auto",
        help="Progress visualization on rank 0. auto uses tqdm when installed.",
    )
    parser.add_argument("--debug-stages", action="store_true", help="Print flushed startup/training stages from every rank.")
    parser.add_argument("--debug-batches", type=int, default=0, help="Print the first N batch tensor shapes from every rank.")
    parser.add_argument("--dist-smoke-test", action="store_true", help="Run a tiny CUDA all-reduce after distributed setup.")
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--box-weight", type=float, default=5.0)
    parser.add_argument("--negative-class-weight", type=float, default=0.02)

    parser.add_argument(
        "--amp",
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
        help="Mixed precision mode. auto uses bf16 when supported, otherwise fp16 on CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile to improve speed on supported PyTorch/CUDA builds.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
        help="torch.compile mode.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_metrics_file(args: argparse.Namespace) -> Path:
    return args.metrics_file if args.metrics_file is not None else args.output_dir / "metrics.json"


def resolve_history_file(args: argparse.Namespace) -> Path:
    return args.history_file if args.history_file is not None else args.output_dir / "metrics_history.jsonl"


def resolve_inventory_dir(args: argparse.Namespace) -> Path:
    return args.inventory_dir if args.inventory_dir is not None else args.output_dir / "inventory"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_inventory(path: Path, include_hash: bool = False, small_hash_limit: int = 64 * 1024 * 1024) -> dict[str, Any]:
    resolved = path.expanduser()
    info: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
    }
    if not resolved.exists() or not resolved.is_file():
        return info

    stat = resolved.stat()
    info.update(
        {
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
        }
    )
    if include_hash or stat.st_size <= small_hash_limit:
        info["sha256"] = sha256_file(resolved)
    return info


def run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_inventory() -> dict[str, Any]:
    status = run_git_command(["status", "--short"])
    return {
        "commit": run_git_command(["rev-parse", "HEAD"]),
        "branch": run_git_command(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def runtime_inventory() -> dict[str, Any]:
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:  # pragma: no cover - runtime metadata only
        torchvision_version = None

    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
            )

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": cuda_devices,
    }


def configure_reproducibility(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def split_file_path(root: Path, split: str) -> Path:
    return root / f"{split}.txt"


def dataset_split_inventory(dataset: CocoYoloDetection, args: argparse.Namespace) -> dict[str, Any]:
    image_bytes = 0
    label_bytes = 0
    missing_images = 0
    missing_labels = 0
    label_file_count = 0
    sample_records: list[dict[str, str]] = []

    for index, record in enumerate(dataset.records):
        if index < 5:
            sample_records.append(
                {
                    "image_path": str(record.image_path),
                    "label_path": str(record.label_path),
                }
            )

        if record.image_path.exists():
            image_bytes += record.image_path.stat().st_size
        else:
            missing_images += 1

        if record.label_path.exists():
            label_bytes += record.label_path.stat().st_size
            label_file_count += 1
        else:
            missing_labels += 1

    root = dataset.root
    annotation_name = f"instances_{dataset.split}.json"
    return {
        "split": dataset.split,
        "samples": len(dataset),
        "image_size": dataset.image_size,
        "letterbox": dataset.letterbox,
        "split_file": file_inventory(split_file_path(root, dataset.split), include_hash=True),
        "annotation_file": file_inventory(root / "annotations" / annotation_name, include_hash=args.inventory_hash_files),
        "image_archive": file_inventory(root / "images" / f"{dataset.split}.zip", include_hash=args.inventory_hash_files),
        "image_bytes": image_bytes,
        "label_files": label_file_count,
        "label_bytes": label_bytes,
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "sample_records": sample_records,
    }


def build_dataset_inventory(
    train_dataset: CocoYoloDetection,
    val_dataset: CocoYoloDetection | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    datasets = [train_dataset]
    if val_dataset is not None and val_dataset.split != train_dataset.split:
        datasets.append(val_dataset)

    return {
        "generated_at_utc": utc_now_iso(),
        "root": str(train_dataset.root),
        "names_yaml": file_inventory(args.names_yaml, include_hash=True),
        "splits": {dataset.split: dataset_split_inventory(dataset, args) for dataset in datasets},
    }


def checkpoint_inventory(path: Path, include_hash: bool) -> dict[str, Any]:
    info = file_inventory(path, include_hash=include_hash)
    if path.name.startswith("epoch_") and path.suffix == ".pt":
        try:
            info["epoch"] = int(path.stem.removeprefix("epoch_"))
        except ValueError:
            pass
    return info


def scan_checkpoints(output_dir: Path, include_hash: bool) -> list[dict[str, Any]]:
    if not output_dir.exists():
        return []
    return [
        checkpoint_inventory(path, include_hash=include_hash)
        for path in sorted(output_dir.glob("*.pt"))
        if path.is_file()
    ]


def count_parameters(model: nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in parameters),
        "buffers": sum(buffer.numel() for buffer in buffers),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in buffers),
    }


def build_model_inventory(
    model: nn.Module,
    args: argparse.Namespace,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    checkpoints = scan_checkpoints(args.output_dir, include_hash=args.inventory_hash_files)
    latest_checkpoint = checkpoint_inventory(checkpoint_path, include_hash=args.inventory_hash_files) if checkpoint_path else None
    return {
        "generated_at_utc": utc_now_iso(),
        "architecture": type(unwrap_model(model)).__name__,
        "backbone": args.backbone,
        "weights": args.weights,
        "num_classes": args.num_classes,
        "head_channels": args.head_channels,
        "freeze_backbone": args.freeze_backbone,
        "output_dir": str(args.output_dir),
        "checkpoint_count": len(checkpoints),
        "checkpoint_bytes": sum(int(item.get("size_bytes", 0)) for item in checkpoints),
        "latest_checkpoint": latest_checkpoint,
        "checkpoints": checkpoints,
        "parameters": count_parameters(unwrap_model(model)),
    }


def write_run_manifest(args: argparse.Namespace, dataset_inventory: dict[str, Any] | None) -> Path:
    manifest_path = args.output_dir / "run_manifest.json"
    write_json(
        manifest_path,
        {
            "generated_at_utc": utc_now_iso(),
            "argv": sys.argv,
            "args": vars(args),
            "git": git_inventory(),
            "runtime": runtime_inventory(),
            "dataset_inventory": dataset_inventory,
        },
    )
    return manifest_path


def parse_wandb_tags(tags: str | None) -> list[str] | None:
    if not tags:
        return None
    parsed = [tag.strip() for tag in tags.split(",") if tag.strip()]
    return parsed or None


class ExperimentTracker:
    def __init__(self, wandb_module: Any | None = None, run: Any | None = None) -> None:
        self.wandb = wandb_module
        self.run = run

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @property
    def url(self) -> str | None:
        if self.run is None:
            return None
        return getattr(self.run, "url", None)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(json_safe(metrics), step=step)

    def watch(self, model: nn.Module, mode: str) -> None:
        if self.run is None or self.wandb is None or mode == "none":
            return
        self.wandb.watch(model, log=mode, log_freq=100)

    def log_image(self, key: str, path: Path | None, caption: str, step: int | None = None) -> None:
        if self.run is None or self.wandb is None or path is None or not path.exists():
            return
        self.run.log({key: self.wandb.Image(str(path), caption=caption)}, step=step)

    def log_file_artifact(
        self,
        name: str,
        artifact_type: str,
        path: Path,
        metadata: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        if self.run is None or self.wandb is None or not path.exists():
            return
        artifact = self.wandb.Artifact(name=name, type=artifact_type, metadata=json_safe(metadata or {}))
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def initialize_wandb(
    args: argparse.Namespace,
    dataset_inventory: dict[str, Any] | None,
    manifest_path: Path | None,
) -> ExperimentTracker:
    if not is_main_process() or args.wandb_mode == "disabled":
        return ExperimentTracker()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested. Install it with: python3 -m pip install -e '.[experiment]'") from exc

    args.wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        dir=str(args.wandb_dir),
        name=args.wandb_name,
        notes=args.wandb_notes,
        tags=parse_wandb_tags(args.wandb_tags),
        config={
            "args": json_safe(vars(args)),
            "git": git_inventory(),
            "runtime": runtime_inventory(),
            "dataset_inventory": dataset_inventory,
            "run_manifest": str(manifest_path) if manifest_path is not None else None,
        },
        mode=args.wandb_mode,
        job_type="train",
        save_code=True,
    )
    tracker = ExperimentTracker(wandb_module=wandb, run=run)
    if tracker.url:
        print(f"wandb_run: {tracker.url}", flush=True)
    else:
        print(f"wandb_run_id: {run.id}", flush=True)
    return tracker


def distributed_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def process_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return process_rank() == 0


def local_rank_from_env(args: argparse.Namespace) -> int:
    if args.local_rank is not None:
        return args.local_rank
    return int(os.environ.get("LOCAL_RANK", "0"))


def log_stage(args: argparse.Namespace, message: str, all_ranks: bool = False) -> None:
    if not args.debug_stages:
        return
    if all_ranks or is_main_process():
        local_rank = os.environ.get("LOCAL_RANK", "0")
        print(f"[rank={process_rank()} local_rank={local_rank}] {message}", flush=True)


def is_single_node_torchrun() -> bool:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    world_size = distributed_world_size()
    return world_size > 1 and local_world_size == world_size


def is_ignored_socket_interface(interface_name: str) -> bool:
    ignored_prefixes = ("br-", "cali", "docker", "flannel", "lo", "tunl", "veth")
    return interface_name.startswith(ignored_prefixes)


def resolve_ipv4_address(host: str) -> ipaddress.IPv4Address | None:
    if not host:
        return None
    try:
        return ipaddress.IPv4Address(host)
    except ValueError:
        pass
    try:
        return ipaddress.IPv4Address(socket.gethostbyname(host))
    except OSError:
        return None


def route_interface_for_ipv4(address: ipaddress.IPv4Address) -> str | None:
    route_path = Path("/proc/net/route")
    if not route_path.exists():
        return None

    best_interface = None
    best_prefix_length = -1
    best_metric = 2**31 - 1
    address_int = int(address)

    for line in route_path.read_text().splitlines()[1:]:
        fields = line.split()
        if len(fields) < 11:
            continue
        interface_name, destination_hex, _, flags_hex, _, _, metric_text, mask_hex = fields[:8]
        if is_ignored_socket_interface(interface_name):
            continue
        try:
            flags = int(flags_hex, 16)
            metric = int(metric_text)
            destination_int = int.from_bytes(bytes.fromhex(destination_hex), byteorder="little")
            mask_int = int.from_bytes(bytes.fromhex(mask_hex), byteorder="little")
        except ValueError:
            continue
        if flags & 0x1 == 0:
            continue
        if address_int & mask_int != destination_int & mask_int:
            continue

        prefix_length = mask_int.bit_count()
        if prefix_length > best_prefix_length or (prefix_length == best_prefix_length and metric < best_metric):
            best_interface = interface_name
            best_prefix_length = prefix_length
            best_metric = metric

    return best_interface


def configure_nccl_socket_interface(args: argparse.Namespace) -> None:
    if args.nccl_socket_ifname:
        os.environ["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname
        log_stage(args, f"NCCL_SOCKET_IFNAME={args.nccl_socket_ifname}", all_ranks=True)
        return
    if "NCCL_SOCKET_IFNAME" in os.environ:
        log_stage(args, f"NCCL_SOCKET_IFNAME={os.environ['NCCL_SOCKET_IFNAME']}", all_ranks=True)
        return

    master_address = resolve_ipv4_address(os.environ.get("MASTER_ADDR", ""))
    if master_address is None:
        return
    interface_name = route_interface_for_ipv4(master_address)
    if interface_name is None:
        return
    os.environ["NCCL_SOCKET_IFNAME"] = interface_name
    log_stage(args, f"NCCL_SOCKET_IFNAME={interface_name}", all_ranks=True)


def configure_distributed_environment(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        return
    if distributed_world_size() <= 1 and args.wrap not in {"ddp", "fsdp"}:
        return

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    log_stage(args, f"NCCL_SOCKET_FAMILY={os.environ['NCCL_SOCKET_FAMILY']}", all_ranks=True)
    configure_nccl_socket_interface(args)

    use_safe_mode = args.nccl_safe_mode or is_single_node_torchrun()
    if use_safe_mode:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
        log_stage(
            args,
            "NCCL safe mode active: NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1",
            all_ranks=True,
        )


def choose_device(args: argparse.Namespace) -> torch.device:
    if args.device != "auto":
        device = torch.device(args.device)
        if device.type == "cuda":
            validate_cuda_rank(device.index if device.index is not None else local_rank_from_env(args))
        return device
    if torch.cuda.is_available():
        local_rank = local_rank_from_env(args)
        validate_cuda_rank(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def validate_cuda_rank(local_rank: int) -> None:
    device_count = torch.cuda.device_count()
    if local_rank < device_count:
        return
    raise RuntimeError(
        f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) are visible. "
        f"Set torchrun --nproc-per-node to at most {device_count}, or set "
        "CUDA_VISIBLE_DEVICES to the GPUs you want to use."
    )


def setup_distributed(args: argparse.Namespace, device: torch.device) -> None:
    should_init = distributed_world_size() > 1 or (args.wrap in {"ddp", "fsdp"} and "RANK" in os.environ)
    if not should_init or dist.is_initialized():
        return
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backend = args.dist_backend
    if backend == "auto":
        backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cpu" and backend == "nccl":
        backend = "gloo"
    log_stage(args, f"init_process_group backend={backend}", all_ranks=True)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=args.dist_timeout_minutes))
    log_stage(args, "init_process_group done", all_ranks=True)


def run_distributed_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    if not args.dist_smoke_test or not dist.is_initialized():
        return
    log_stage(args, "dist smoke all_reduce start", all_ranks=True)
    value = torch.ones((), device=device)
    dist.all_reduce(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    log_stage(args, f"dist smoke all_reduce done value={float(value.item()):.1f}", all_ranks=True)


def synchronize_pretrained_weight_cache(args: argparse.Namespace) -> None:
    if args.weights == "none":
        return

    lock_name = f"yolo_{args.backbone}_{args.weights}_weights.lock"
    lock_path = Path(os.environ.get("TMPDIR", "/tmp")) / lock_name
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_stage(args, f"waiting for weight cache lock {lock_path}", all_ranks=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        log_stage(args, f"caching weights backbone={args.backbone} weights={args.weights}", all_ranks=True)
        ensure_resnet_weights_available(args.backbone, args.weights)
        log_stage(args, "caching weights done", all_ranks=True)
        fcntl.flock(lock_file, fcntl.LOCK_UN)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, args: argparse.Namespace, device: torch.device) -> nn.Module:
    requested = args.wrap
    if requested == "auto":
        requested = "ddp" if distributed_world_size() > 1 else "none"

    if requested == "none":
        return model
    if requested == "ddp":
        if not dist.is_initialized():
            raise RuntimeError("DDP requested but torch.distributed is not initialized.")
        if device.type == "cuda":
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            device_ids = [cuda_index]
        else:
            device_ids = None
        return DistributedDataParallel(model, device_ids=device_ids)
    if requested == "fsdp":
        if FullyShardedDataParallel is None:
            raise RuntimeError("FSDP is not available in this PyTorch build.")
        if not dist.is_initialized():
            raise RuntimeError("FSDP requested but torch.distributed is not initialized.")
        return FullyShardedDataParallel(model)
    raise ValueError(f"Unknown wrap mode: {requested}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"]
    if not isinstance(images, torch.Tensor):
        raise TypeError("Training requires fixed-size images so the batch can be a tensor. Set image_size.")
    return images.to(device, non_blocking=True), batch["yolo_targets"].to(device, non_blocking=True)


def epoch_step_count(loader: Any, max_steps: int | None) -> int:
    total_steps = len(loader)
    return min(total_steps, max_steps) if max_steps is not None else total_steps


def create_progress_bar(args: argparse.Namespace, epoch: int, total_steps: int) -> Any:
    if not is_main_process() or args.progress == "none":
        return None
    if args.progress in {"auto", "tqdm"} and tqdm is not None:
        return tqdm(
            total=total_steps,
            desc=f"epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    if args.progress == "tqdm" and tqdm is None:
        print("tqdm is not installed; falling back to text progress.", flush=True)
    return None


def create_validation_progress_bar(args: argparse.Namespace, epoch: int, total_steps: int) -> Any:
    if not is_main_process() or args.progress == "none":
        return None
    if args.progress in {"auto", "tqdm"} and tqdm is not None:
        return tqdm(
            total=total_steps,
            desc=f"val {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
    return None


def should_use_text_progress(args: argparse.Namespace) -> bool:
    if args.progress == "text":
        return True
    return args.progress in {"auto", "tqdm"} and tqdm is None


def maybe_print_text_progress(
    args: argparse.Namespace,
    step: int,
    total_steps: int,
    global_step: int,
    loss_value: float,
    class_loss_value: float,
    positive_class_loss_value: float,
    negative_class_loss_value: float,
    box_loss_value: float,
    mean_iou_value: float,
    target_class_prob_value: float,
    positive_count: int,
    images_per_second: float,
) -> None:
    if not is_main_process() or not should_use_text_progress(args):
        return
    if args.log_every <= 0 or step % args.log_every != 0:
        return
    print(
        f"step={step + 1}/{total_steps} global_step={global_step} "
        f"loss={loss_value:.4f} cls={class_loss_value:.4f} "
        f"pos_cls={positive_class_loss_value:.4f} bg_cls={negative_class_loss_value:.4f} "
        f"box={box_loss_value:.4f} iou={mean_iou_value:.3f} "
        f"p_cls={target_class_prob_value:.3f} positives={positive_count} "
        f"img/s={images_per_second:.1f}",
        flush=True,
    )


def normalized_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_empty((0, 4))

    boxes = boxes.reshape(-1, 4)
    centers = boxes[:, 0:2]
    sizes = boxes[:, 2:4].clamp_min(1e-6)
    top_left = centers - sizes / 2
    bottom_right = centers + sizes / 2
    return torch.cat([top_left, bottom_right], dim=1).clamp(0.0, 1.0)


def tensor_to_numpy_image(image: torch.Tensor) -> Any:
    image = image.detach().cpu()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.float()
    if image.max() > 1.5:
        image = image / 255.0
    return image.clamp(0, 1).numpy()


def target_image_id(target: dict[str, Any]) -> int:
    return int(target["image_id"].reshape(-1)[0].item())


def decode_batch_detections(predictions: dict[str, Any], args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    class_probs = predictions["class_probs"].detach().cpu().to(dtype=torch.float32)
    boxes_yolo = predictions["boxes_yolo"].detach().cpu().to(dtype=torch.float32)
    batch_size, num_classes, _, _ = class_probs.shape
    detections: list[dict[str, torch.Tensor]] = []

    for batch_index in range(batch_size):
        scores_by_cell, labels_by_cell = class_probs[batch_index].permute(1, 2, 0).reshape(-1, num_classes).max(dim=1)
        boxes = normalized_cxcywh_to_xyxy(boxes_yolo[batch_index].reshape(-1, 4))

        keep = scores_by_cell >= args.val_score_threshold
        scores = scores_by_cell[keep]
        labels = labels_by_cell[keep]
        boxes = boxes[keep]

        if scores.numel() == 0:
            detections.append(
                {
                    "boxes": boxes.new_empty((0, 4)),
                    "scores": scores.new_empty((0,)),
                    "labels": labels.new_empty((0,), dtype=torch.long),
                }
            )
            continue

        if args.val_pre_nms_topk > 0 and scores.numel() > args.val_pre_nms_topk:
            scores, top_indices = scores.topk(args.val_pre_nms_topk)
            labels = labels[top_indices]
            boxes = boxes[top_indices]

        kept_indices: list[torch.Tensor] = []
        for label in labels.unique(sorted=True):
            class_indices = torch.where(labels == label)[0]
            class_keep = nms(boxes[class_indices], scores[class_indices], args.val_nms_threshold)
            kept_indices.append(class_indices[class_keep])

        if kept_indices:
            keep_after_nms = torch.cat(kept_indices)
            keep_after_nms = keep_after_nms[scores[keep_after_nms].argsort(descending=True)]
            if args.val_max_detections > 0:
                keep_after_nms = keep_after_nms[: args.val_max_detections]
            boxes = boxes[keep_after_nms]
            scores = scores[keep_after_nms]
            labels = labels[keep_after_nms]

        detections.append({"boxes": boxes, "scores": scores, "labels": labels})

    return detections


def average_precision_from_pr(recall: torch.Tensor, precision: torch.Tensor) -> float:
    if recall.numel() == 0:
        return 0.0

    zero = recall.new_tensor([0.0])
    one = recall.new_tensor([1.0])
    mrec = torch.cat([zero, recall, one])
    mpre = torch.cat([zero, precision, zero])

    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])

    changed = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]).sum().item())


def compute_map_at_iou(
    detections: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    num_classes: int,
    iou_threshold: float,
) -> dict[str, Any]:
    detections_by_class: list[list[tuple[int, float, torch.Tensor]]] = [[] for _ in range(num_classes)]
    gt_boxes_by_class: list[dict[int, torch.Tensor]] = [dict() for _ in range(num_classes)]
    gt_count_by_class = [0 for _ in range(num_classes)]

    for target in targets:
        image_id = int(target["image_id"])
        labels = target["labels"].to(dtype=torch.long).cpu()
        boxes = normalized_cxcywh_to_xyxy(target["boxes_yolo"].cpu())
        for class_id_tensor in labels.unique(sorted=True):
            class_id = int(class_id_tensor.item())
            if not (0 <= class_id < num_classes):
                continue
            class_boxes = boxes[labels == class_id]
            gt_boxes_by_class[class_id][image_id] = class_boxes
            gt_count_by_class[class_id] += int(class_boxes.shape[0])

    prediction_count = 0
    for detection in detections:
        image_id = int(detection["image_id"])
        boxes = detection["boxes"].cpu()
        scores = detection["scores"].cpu()
        labels = detection["labels"].to(dtype=torch.long).cpu()
        prediction_count += int(scores.numel())
        for box, score, label in zip(boxes, scores, labels, strict=True):
            class_id = int(label.item())
            if 0 <= class_id < num_classes:
                detections_by_class[class_id].append((image_id, float(score.item()), box))

    average_precisions: list[float] = []
    per_class_ap: dict[int, float] = {}
    for class_id in range(num_classes):
        num_gt = gt_count_by_class[class_id]
        if num_gt == 0:
            continue

        class_detections = sorted(detections_by_class[class_id], key=lambda item: item[1], reverse=True)
        if not class_detections:
            average_precisions.append(0.0)
            per_class_ap[class_id] = 0.0
            continue

        matched_by_image = {
            image_id: torch.zeros(boxes.shape[0], dtype=torch.bool)
            for image_id, boxes in gt_boxes_by_class[class_id].items()
        }
        true_positive = torch.zeros(len(class_detections), dtype=torch.float32)
        false_positive = torch.zeros(len(class_detections), dtype=torch.float32)

        for index, (image_id, _, box) in enumerate(class_detections):
            gt_boxes = gt_boxes_by_class[class_id].get(image_id)
            if gt_boxes is None or gt_boxes.numel() == 0:
                false_positive[index] = 1.0
                continue

            ious = box_iou(box.view(1, 4), gt_boxes).squeeze(0)
            best_iou, best_index = ious.max(dim=0)
            matched = matched_by_image[image_id]
            if float(best_iou.item()) >= iou_threshold and not bool(matched[best_index].item()):
                true_positive[index] = 1.0
                matched[best_index] = True
            else:
                false_positive[index] = 1.0

        true_positive_cumsum = true_positive.cumsum(dim=0)
        false_positive_cumsum = false_positive.cumsum(dim=0)
        recall = true_positive_cumsum / max(num_gt, 1)
        precision = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum).clamp_min(1e-12)
        ap = average_precision_from_pr(recall, precision)
        average_precisions.append(ap)
        per_class_ap[class_id] = ap

    map_value = sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    return {
        "map": map_value,
        "per_class_ap": per_class_ap,
        "evaluated_classes": len(average_precisions),
        "targets": sum(gt_count_by_class),
        "predictions": prediction_count,
    }


def load_class_names(args: argparse.Namespace) -> dict[int, str]:
    if not args.names_yaml.exists():
        return {}
    try:
        return load_coco_class_names(args.names_yaml)
    except Exception as exc:  # pragma: no cover - best-effort labels for plots
        print(f"class_names_warning: could not read {args.names_yaml}: {exc}", flush=True)
        return {}


def class_label(class_id: int, class_names: dict[int, str]) -> str:
    return class_names.get(class_id, str(class_id))


def validation_plot_image_id(loader: Any, epoch: int, args: argparse.Namespace) -> int | None:
    if args.val_plot_sample_index is not None:
        return max(args.val_plot_sample_index, 0)
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None
    try:
        dataset_size = len(dataset)
    except TypeError:
        return None
    if dataset_size <= 0:
        return None
    return epoch % dataset_size


def filter_detection_for_plot(detection: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    boxes = detection["boxes"].detach().cpu().to(dtype=torch.float32)
    scores = detection["scores"].detach().cpu().to(dtype=torch.float32)
    labels = detection["labels"].detach().cpu().to(dtype=torch.long)

    keep = scores >= args.val_plot_score_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    max_predictions = max(args.val_plot_max_boxes, 0)
    if max_predictions == 0:
        return {
            "boxes": boxes.new_empty((0, 4)),
            "scores": scores.new_empty((0,)),
            "labels": labels.new_empty((0,), dtype=torch.long),
        }
    if max_predictions > 0 and scores.numel() > max_predictions:
        top_indices = scores.argsort(descending=True)[:max_predictions]
        boxes = boxes[top_indices]
        scores = scores[top_indices]
        labels = labels[top_indices]

    return {"boxes": boxes, "scores": scores, "labels": labels}


def choose_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or args.amp == "none":
        return None
    if args.amp == "bf16":
        return torch.bfloat16
    if args.amp == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def save_validation_example(
    example: dict[str, Any],
    epoch: int,
    args: argparse.Namespace,
    class_names: dict[int, str],
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("validation_example_skipped: matplotlib is not installed", flush=True)
        return None

    image = example["image"]
    target = example["target"]
    detection = filter_detection_for_plot(example["detection"], args)
    image_np = tensor_to_numpy_image(image)
    height, width = image_np.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)
    ax.set_title(
        f"validation epoch {epoch + 1}: {Path(target['path']).name} "
        f"preds>={args.val_plot_score_threshold:g}: {int(detection['scores'].numel())}"
    )
    ax.axis("off")

    gt_boxes = normalized_cxcywh_to_xyxy(target["boxes_yolo"].detach().cpu())
    gt_labels = target["labels"].detach().cpu().to(dtype=torch.long)
    for box, label in zip(gt_boxes[: args.val_plot_max_boxes], gt_labels[: args.val_plot_max_boxes], strict=False):
        x1, y1, x2, y2 = box.tolist()
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor="lime",
                linestyle="-",
            )
        )
        ax.text(
            x1,
            max(0.0, y1 - 3.0),
            f"gt {class_label(int(label.item()), class_names)}",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="lime", alpha=0.75, edgecolor="none", pad=1.5),
        )

    pred_boxes = detection["boxes"]
    pred_scores = detection["scores"]
    pred_labels = detection["labels"]
    for box, score, label in zip(
        pred_boxes,
        pred_scores,
        pred_labels,
        strict=False,
    ):
        x1, y1, x2, y2 = box.tolist()
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                fill=False,
                linewidth=2,
                edgecolor="red",
                linestyle="--",
            )
        )
        ax.text(
            x1,
            min(height - 1.0, y2 + 3.0),
            f"pred {class_label(int(label.item()), class_names)} {float(score.item()):.2f}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="red", alpha=0.75, edgecolor="none", pad=1.5),
        )

    output_path = args.output_dir / "val_examples" / f"epoch_{epoch + 1:03d}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def validate_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: AnchorFreeYoloLoss,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    class_names: dict[int, str],
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()

    total_steps = epoch_step_count(loader, args.val_max_steps)
    progress_bar = create_validation_progress_bar(args, epoch, total_steps)
    completed_steps = 0
    image_count = 0
    loss_sum = 0.0
    all_detections: list[dict[str, Any]] = []
    all_targets: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None
    fallback_example: dict[str, Any] | None = None
    plot_image_id = validation_plot_image_id(loader, epoch, args) if args.plot_val_example else None

    try:
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if args.val_max_steps is not None and step >= args.val_max_steps:
                    break

                images, yolo_targets = move_batch_to_device(batch, device)
                with autocast_context(device, amp_dtype):
                    predictions = model(images)
                    loss_dict = criterion(predictions, yolo_targets)
                batch_detections = decode_batch_detections(predictions, args)

                completed_steps += 1
                image_count += int(images.shape[0])
                loss_sum += float(loss_dict["loss"].item())

                for batch_index, target in enumerate(batch["targets"]):
                    image_id = target_image_id(target)
                    detection = batch_detections[batch_index]
                    all_detections.append(
                        {
                            "image_id": image_id,
                            "boxes": detection["boxes"],
                            "scores": detection["scores"],
                            "labels": detection["labels"],
                        }
                    )
                    all_targets.append(
                        {
                            "image_id": image_id,
                            "boxes_yolo": target["boxes_yolo"].detach().cpu(),
                            "labels": target["labels"].detach().cpu(),
                        }
                    )
                    if args.plot_val_example:
                        candidate_example = {
                            "image": images[batch_index].detach().cpu(),
                            "target": target,
                            "detection": detection,
                        }
                        if fallback_example is None:
                            fallback_example = candidate_example
                        if example is None and (plot_image_id is None or image_id == plot_image_id):
                            example = candidate_example

                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f"{loss_sum / completed_steps:.4f}")
                    progress_bar.update(1)
                elif should_use_text_progress(args) and args.log_every > 0 and step % args.log_every == 0:
                    print(
                        f"val_step={step + 1}/{total_steps} loss={loss_sum / completed_steps:.4f}",
                        flush=True,
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if was_training:
            model.train()

    metrics = compute_map_at_iou(
        detections=all_detections,
        targets=all_targets,
        num_classes=args.num_classes,
        iou_threshold=args.val_iou_threshold,
    )
    metrics["loss"] = loss_sum / completed_steps if completed_steps > 0 else 0.0
    metrics["steps"] = completed_steps
    metrics["images"] = image_count

    if example is None:
        example = fallback_example
    if example is not None:
        metrics["example_path"] = save_validation_example(example, epoch, args, class_names)
    else:
        metrics["example_path"] = None

    return metrics


def print_validation_summary(epoch: int, metrics: dict[str, Any], args: argparse.Namespace) -> None:
    example_path = metrics.get("example_path")
    example_text = f" example={example_path}" if example_path is not None else ""
    print(
        f"epoch={epoch} validation "
        f"steps={metrics['steps']} "
        f"images={metrics['images']} "
        f"val_loss={metrics['loss']:.4f} "
        f"mAP@{args.val_iou_threshold:.2f}={metrics['map']:.4f} "
        f"classes={metrics['evaluated_classes']} "
        f"targets={metrics['targets']} "
        f"predictions={metrics['predictions']}"
        f"{example_text}",
        flush=True,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
        "rng_state": capture_rng_state(),
        "git": git_inventory(),
    }
    path = args.output_dir / f"epoch_{epoch:03d}.pt"
    torch.save(checkpoint, path)
    print(f"saved_checkpoint: {path}", flush=True)
    return path


def main() -> None:
    args = parse_args()
    configure_reproducibility(args)
    log_stage(args, "parsed args", all_ranks=True)

    device = choose_device(args)
    log_stage(args, f"selected device={device}", all_ranks=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
        torch.backends.cudnn.allow_tf32 = not args.deterministic
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest" if args.deterministic else "high")

    configure_distributed_environment(args, device)
    setup_distributed(args, device)
    run_distributed_smoke_test(args, device)

    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        history_file = resolve_history_file(args)
        if history_file.exists():
            history_file.unlink()

    amp_dtype = choose_amp_dtype(args, device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    log_stage(args, "building dataset", all_ranks=True)
    dataset = CocoYoloDetection(
        root=args.root,
        split=args.split,
        image_size=args.image_size,
        letterbox=True,
    )
    log_stage(args, f"dataset ready samples={len(dataset)}", all_ranks=True)

    if device.type == "cpu" and distributed_world_size() > 1:
        if args.device == "auto":
            if is_main_process():
                print(
                    "No CUDA device available but distributed world size > 1."
                    " If you intended to use GPUs, update your NVIDIA driver or"
                    " install a PyTorch build compatible with your driver."
                    " To continue on CPU, run without torchrun or set --device cpu.",
                    flush=True,
                )
            raise SystemExit(1)

    loader_workers = args.workers if device.type == "cuda" else min(args.workers, 2)
    log_stage(args, f"building dataloader workers={loader_workers}", all_ranks=True)
    loader = create_coco_yolo_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=device.type == "cuda",
        distributed=dist.is_initialized(),
    )
    log_stage(args, f"dataloader ready steps_per_epoch={len(loader)}", all_ranks=True)

    val_loader = None
    val_dataset = None
    val_sample_count = 0
    class_names: dict[int, str] = {}
    if not args.no_validation and is_main_process():
        log_stage(args, "building validation dataset", all_ranks=False)
        val_dataset = CocoYoloDetection(
            root=args.root,
            split=args.val_split,
            image_size=args.image_size,
            letterbox=True,
        )
        val_batch_size = args.val_batch_size if args.val_batch_size is not None else args.batch_size
        val_workers = args.val_workers if args.val_workers is not None else loader_workers
        val_loader = create_coco_yolo_dataloader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=device.type == "cuda",
            distributed=False,
        )
        val_sample_count = len(val_dataset)
        class_names = load_class_names(args)
        log_stage(args, f"validation dataloader ready steps={len(val_loader)}", all_ranks=False)

    tracker = ExperimentTracker()
    dataset_inventory = None
    manifest_path = None
    dataset_inventory_path = resolve_inventory_dir(args) / "dataset_inventory.json"
    if is_main_process():
        if args.write_inventory:
            dataset_inventory = build_dataset_inventory(dataset, val_dataset, args)
            write_json(dataset_inventory_path, dataset_inventory)
            print(f"dataset_inventory: {dataset_inventory_path}", flush=True)
        manifest_path = write_run_manifest(args, dataset_inventory)
        print(f"run_manifest: {manifest_path}", flush=True)
        tracker = initialize_wandb(args, dataset_inventory, manifest_path)
        if args.write_inventory and args.wandb_log_inventory_artifacts:
            tracker.log_file_artifact(
                name="coco-yolo-dataset-inventory",
                artifact_type="dataset",
                path=dataset_inventory_path,
                metadata={"root": str(args.root), "split": args.split, "val_split": args.val_split},
                aliases=["latest"],
            )

    log_stage(args, "building model on CPU", all_ranks=True)
    synchronize_pretrained_weight_cache(args)
    model = ResNetAnchorFreeYolo(
        num_classes=args.num_classes,
        backbone_name=args.backbone,
        weights=args.weights,
        freeze_backbone=args.freeze_backbone,
        head_channels=args.head_channels,
    )
    log_stage(args, f"moving model to {device}", all_ranks=True)
    model = model.to(device)

    if args.compile and device.type == "cuda" and args.wrap != "fsdp":
        log_stage(args, f"compiling model mode={args.compile_mode}", all_ranks=True)
        model = torch.compile(model, mode=args.compile_mode)

    log_stage(args, f"wrapping model wrap={args.wrap}", all_ranks=True)
    model = wrap_model(model, args, device)
    log_stage(args, "model ready", all_ranks=True)
    if is_main_process():
        tracker.watch(unwrap_model(model), args.wandb_watch)
        if args.write_inventory:
            model_inventory_path = resolve_inventory_dir(args) / "model_inventory.json"
            model_inventory = build_model_inventory(model, args)
            write_json(model_inventory_path, model_inventory)
            print(f"model_inventory: {model_inventory_path}", flush=True)
            if args.wandb_log_inventory_artifacts:
                tracker.log_file_artifact(
                    name="resnet-anchor-free-yolo-model-inventory",
                    artifact_type="model",
                    path=model_inventory_path,
                    metadata={"backbone": args.backbone, "weights": args.weights},
                    aliases=["latest"],
                )

    criterion = AnchorFreeYoloLoss(
        num_classes=args.num_classes,
        class_weight=args.class_weight,
        box_weight=args.box_weight,
        negative_class_weight=args.negative_class_weight,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    log_stage(args, "criterion and optimizer ready", all_ranks=True)

    if is_main_process():
        print(f"device: {device}", flush=True)
        print(f"samples: {len(dataset)}", flush=True)
        if not args.no_validation:
            print(f"validation_samples: {val_sample_count}", flush=True)
        print(f"backbone: {args.backbone}, weights: {args.weights}", flush=True)
        print(f"distributed: {dist.is_initialized()}, wrap: {args.wrap}, world_size: {distributed_world_size()}", flush=True)
        print(f"head_channels: {args.head_channels}", flush=True)
        print(f"amp: {args.amp}, amp_dtype: {amp_dtype}", flush=True)
        print(f"compile: {args.compile}, compile_mode: {args.compile_mode}", flush=True)
        print(f"seed: {args.seed}, deterministic: {args.deterministic}", flush=True)
        print(f"wandb_mode: {args.wandb_mode}, wandb_project: {args.wandb_project}", flush=True)

    global_step = 0
    for epoch in range(args.epochs):
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        model.train()
        total_steps = epoch_step_count(loader, args.max_steps)
        progress_bar = create_progress_bar(args, epoch, total_steps)
        completed_steps = 0
        epoch_loss_sum = 0.0
        epoch_class_loss_sum = 0.0
        epoch_positive_class_loss_sum = 0.0
        epoch_negative_class_loss_sum = 0.0
        epoch_box_loss_sum = 0.0
        epoch_iou_sum = 0.0
        epoch_target_class_prob_sum = 0.0
        epoch_positive_sum = 0
        epoch_start = time.perf_counter()
        log_stage(args, f"starting epoch={epoch} total_steps={total_steps}", all_ranks=True)

        try:
            for step, batch in enumerate(loader):
                if args.max_steps is not None and step >= args.max_steps:
                    break

                step_start = time.perf_counter()
                images, yolo_targets = move_batch_to_device(batch, device)
                if args.debug_batches > 0 and step < args.debug_batches:
                    print(
                        f"[rank={process_rank()}] batch={step} images={tuple(images.shape)} yolo_targets={tuple(yolo_targets.shape)}",
                        flush=True,
                    )

                optimizer.zero_grad(set_to_none=True)

                with autocast_context(device, amp_dtype):
                    predictions = model(images)
                    loss_dict = criterion(predictions, yolo_targets)
                    loss = loss_dict["loss"]

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                local_batch_size = int(images.shape[0])
                global_batch_size = local_batch_size * distributed_world_size()
                step_time = max(time.perf_counter() - step_start, 1e-9)
                images_per_second = global_batch_size / step_time

                loss_value = float(loss.detach().item())
                class_loss_value = float(loss_dict["class_loss"].item())
                positive_class_loss_value = float(loss_dict["positive_class_loss"].item())
                negative_class_loss_value = float(loss_dict["negative_class_loss"].item())
                box_loss_value = float(loss_dict["box_loss"].item())
                mean_iou_value = float(loss_dict["mean_iou"].item())
                target_class_prob_value = float(loss_dict["target_class_prob"].item())
                positive_count = int(loss_dict["num_positive"].item())

                if (
                    is_main_process()
                    and tracker.enabled
                    and args.wandb_log_every > 0
                    and global_step % args.wandb_log_every == 0
                ):
                    tracker.log(
                        {
                            "train/loss": loss_value,
                            "train/class_loss": class_loss_value,
                            "train/positive_class_loss": positive_class_loss_value,
                            "train/negative_class_loss": negative_class_loss_value,
                            "train/box_loss": box_loss_value,
                            "train/mean_iou": mean_iou_value,
                            "train/target_class_prob": target_class_prob_value,
                            "train/positive_cells": positive_count,
                            "train/images_per_second": images_per_second,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )

                completed_steps += 1
                epoch_loss_sum += loss_value
                epoch_class_loss_sum += class_loss_value
                epoch_positive_class_loss_sum += positive_class_loss_value
                epoch_negative_class_loss_sum += negative_class_loss_value
                epoch_box_loss_sum += box_loss_value
                epoch_iou_sum += mean_iou_value
                epoch_target_class_prob_sum += target_class_prob_value
                epoch_positive_sum += positive_count

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        pos_cls=f"{positive_class_loss_value:.3f}",
                        bg=f"{negative_class_loss_value:.3f}",
                        box=f"{box_loss_value:.4f}",
                        iou=f"{mean_iou_value:.3f}",
                        pcls=f"{target_class_prob_value:.3f}",
                        pos=positive_count,
                        imgs=f"{images_per_second:.1f}/s",
                    )
                    progress_bar.update(1)
                else:
                    maybe_print_text_progress(
                        args=args,
                        step=step,
                        total_steps=total_steps,
                        global_step=global_step,
                        loss_value=loss_value,
                        class_loss_value=class_loss_value,
                        positive_class_loss_value=positive_class_loss_value,
                        negative_class_loss_value=negative_class_loss_value,
                        box_loss_value=box_loss_value,
                        mean_iou_value=mean_iou_value,
                        target_class_prob_value=target_class_prob_value,
                        positive_count=positive_count,
                        images_per_second=images_per_second,
                    )

                global_step += 1
        finally:
            if progress_bar is not None:
                progress_bar.close()

        train_epoch_metrics: dict[str, Any] | None = None
        if is_main_process() and completed_steps > 0:
            epoch_time = max(time.perf_counter() - epoch_start, 1e-9)
            global_examples = completed_steps * args.batch_size * distributed_world_size()
            train_epoch_metrics = {
                "epoch": epoch + 1,
                "steps": completed_steps,
                "loss": epoch_loss_sum / completed_steps,
                "class_loss": epoch_class_loss_sum / completed_steps,
                "positive_class_loss": epoch_positive_class_loss_sum / completed_steps,
                "negative_class_loss": epoch_negative_class_loss_sum / completed_steps,
                "box_loss": epoch_box_loss_sum / completed_steps,
                "mean_iou": epoch_iou_sum / completed_steps,
                "target_class_prob": epoch_target_class_prob_sum / completed_steps,
                "positive_cells": epoch_positive_sum,
                "avg_images_per_second": global_examples / epoch_time,
                "epoch_seconds": epoch_time,
                "global_step": global_step,
            }
            print(
                f"epoch={epoch} summary "
                f"steps={train_epoch_metrics['steps']} "
                f"loss={train_epoch_metrics['loss']:.4f} "
                f"class={train_epoch_metrics['class_loss']:.4f} "
                f"pos_cls={train_epoch_metrics['positive_class_loss']:.4f} "
                f"bg_cls={train_epoch_metrics['negative_class_loss']:.4f} "
                f"box={train_epoch_metrics['box_loss']:.4f} "
                f"iou={train_epoch_metrics['mean_iou']:.4f} "
                f"p_cls={train_epoch_metrics['target_class_prob']:.4f} "
                f"positives={train_epoch_metrics['positive_cells']} "
                f"avg_img/s={train_epoch_metrics['avg_images_per_second']:.1f}",
                flush=True,
            )
            append_jsonl(
                resolve_history_file(args),
                {
                    "phase": "train",
                    "time_utc": utc_now_iso(),
                    **train_epoch_metrics,
                },
            )
            tracker.log(
                {
                    "epoch/train_loss": train_epoch_metrics["loss"],
                    "epoch/train_class_loss": train_epoch_metrics["class_loss"],
                    "epoch/train_box_loss": train_epoch_metrics["box_loss"],
                    "epoch/train_mean_iou": train_epoch_metrics["mean_iou"],
                    "epoch/train_target_class_prob": train_epoch_metrics["target_class_prob"],
                    "epoch/train_positive_cells": train_epoch_metrics["positive_cells"],
                    "epoch/train_avg_images_per_second": train_epoch_metrics["avg_images_per_second"],
                    "epoch": epoch + 1,
                },
                step=global_step,
            )

        validation_metrics: dict[str, Any] | None = None
        if not args.no_validation:
            if is_main_process() and val_loader is not None:
                metrics = validate_one_epoch(
                    model=unwrap_model(model),
                    loader=val_loader,
                    criterion=criterion,
                    args=args,
                    device=device,
                    epoch=epoch,
                    class_names=class_names,
                    amp_dtype=amp_dtype,
                )
                print_validation_summary(epoch + 1, metrics, args)
                validation_metrics = {
                    "epoch": epoch + 1,
                    "loss": metrics["loss"],
                    "map": metrics["map"],
                    "iou_threshold": args.val_iou_threshold,
                    "evaluated_classes": metrics["evaluated_classes"],
                    "targets": metrics["targets"],
                    "predictions": metrics["predictions"],
                    "steps": metrics["steps"],
                    "images": metrics["images"],
                    "example_path": metrics.get("example_path"),
                    "global_step": global_step,
                }
                append_jsonl(
                    resolve_history_file(args),
                    {
                        "phase": "validation",
                        "time_utc": utc_now_iso(),
                        **validation_metrics,
                    },
                )
                tracker.log(
                    {
                        "val/loss": validation_metrics["loss"],
                        "val/map": validation_metrics["map"],
                        "val/evaluated_classes": validation_metrics["evaluated_classes"],
                        "val/targets": validation_metrics["targets"],
                        "val/predictions": validation_metrics["predictions"],
                        "val/images": validation_metrics["images"],
                        "epoch": epoch + 1,
                    },
                    step=global_step,
                )
                example_path = metrics.get("example_path")
                tracker.log_image(
                    "val/example",
                    Path(example_path) if example_path is not None else None,
                    caption=f"validation epoch {epoch + 1}",
                    step=global_step,
                )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if is_main_process():
            write_json(
                resolve_metrics_file(args),
                {
                    "time_utc": utc_now_iso(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "train": train_epoch_metrics,
                    "validation": validation_metrics,
                    "wandb_url": tracker.url,
                    "run_manifest": str(manifest_path) if manifest_path is not None else None,
                    "dataset_inventory": str(dataset_inventory_path) if args.write_inventory else None,
                    "model_inventory": str(resolve_inventory_dir(args) / "model_inventory.json") if args.write_inventory else None,
                },
            )

        if is_main_process() and args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, global_step, args)
            if args.write_inventory:
                model_inventory_path = resolve_inventory_dir(args) / "model_inventory.json"
                model_inventory = build_model_inventory(model, args, checkpoint_path=checkpoint_path)
                write_json(model_inventory_path, model_inventory)
                if args.wandb_log_inventory_artifacts:
                    tracker.log_file_artifact(
                        name="resnet-anchor-free-yolo-model-inventory",
                        artifact_type="model",
                        path=model_inventory_path,
                        metadata={"backbone": args.backbone, "weights": args.weights, "checkpoint_count": model_inventory["checkpoint_count"]},
                        aliases=["latest"],
                    )
            if args.wandb_log_checkpoints == "all" or (
                args.wandb_log_checkpoints == "last" and epoch + 1 == args.epochs
            ):
                tracker.log_file_artifact(
                    name="resnet-anchor-free-yolo-checkpoint",
                    artifact_type="model",
                    path=checkpoint_path,
                    metadata={"epoch": epoch + 1, "global_step": global_step, "backbone": args.backbone},
                    aliases=["latest", f"epoch-{epoch + 1:03d}"],
                )

    tracker.finish()
    cleanup_distributed()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_distributed()
