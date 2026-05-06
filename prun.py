from __future__ import annotations

import argparse
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch_pruning as tp
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("runs/detect/best.pt")
DEFAULT_DATA = Path("ultralytics/cfg/datasets/my_VisDrone.yaml")


# These modules are fragile for structural pruning in this custom model. Their
# output channels are either fixed by detection semantics, routing decisions, or
# branch fusion contracts.
DEFAULT_IGNORE_KEYWORDS = (
    # "model.24",  # Detect head
    "model.10",  # C2PSA block with fixed attention residual contracts
    "model.13",  # C2f/Bottleneck split-concat block
    "model.16",  # C2f/Bottleneck split-concat block
    "model.19",  # C2f/Bottleneck split-concat block
    # "attn",
    # "cross_att",
    # "shape_router",
    # "shape_h",
    # "shape_v",
    # "sfbs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Taylor expansion channel pruning for the custom YOLO11 VisDrone model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Input .pt checkpoint.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset yaml used for calibration labels.")
    parser.add_argument("--save", type=Path, default=None, help="Output .pt path.")
    parser.add_argument("--ratio", type=float, default=0.50, help="Global channel pruning ratio.")
    parser.add_argument("--imgsz", type=int, default=640, help="Calibration image size.")
    parser.add_argument("--batch", type=int, default=2, help="Calibration batch size.")
    parser.add_argument("--calib-batches", type=int, default=8, help="Number of calibration batches.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or cuda index like '0'.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for calibration image sampling.")
    parser.add_argument("--min-channels", type=int, default=8, help="Minimum channels retained per pruned conv.")
    parser.add_argument("--max-layer-ratio", type=float, default=0.60, help="Max fraction removed from one conv.")
    parser.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE_KEYWORDS), help="Name keywords to skip.")
    parser.add_argument("--dry-run", action="store_true", help="Score channels and print plan without pruning.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file.")
    return parser.parse_args()


def pick_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def resolve_path(path):
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_dataset_paths(data_yaml: Path) -> tuple[Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data["path"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    train_images = root / data["train"]
    train_labels = train_images.parent / "labels" if train_images.name == "images" else root / "labels"
    return train_images, train_labels


def image_to_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root).with_suffix(".txt")
    return label_root / rel


def load_image_tensor(image_path: Path, imgsz: int) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float().div_(255.0)


def load_labels(label_path: Path) -> torch.Tensor:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return torch.zeros((0, 5), dtype=torch.float32)

    rows: list[list[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(parts[0]), *[float(x) for x in parts[1:5]]])

    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def iter_calibration_batches(
    image_root: Path,
    label_root: Path,
    imgsz: int,
    batch_size: int,
    max_batches: int,
    seed: int,
):
    images = sorted(
        p for p in image_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise FileNotFoundError(f"No calibration images found under {image_root}")

    rng = random.Random(seed)
    rng.shuffle(images)
    images = images[: max_batches * batch_size]

    for start in range(0, len(images), batch_size):
        selected = images[start : start + batch_size]
        if not selected:
            break

        batch_images = []
        batch_cls = []
        batch_boxes = []
        batch_idx = []

        for i, image_path in enumerate(selected):
            batch_images.append(load_image_tensor(image_path, imgsz))
            labels = load_labels(image_to_label_path(image_path, image_root, label_root))
            if labels.numel():
                batch_cls.append(labels[:, 0])
                batch_boxes.append(labels[:, 1:5].clamp_(0, 1))
                batch_idx.append(torch.full((labels.shape[0],), i, dtype=torch.float32))

        if batch_cls:
            cls = torch.cat(batch_cls, dim=0)
            bboxes = torch.cat(batch_boxes, dim=0)
            idx = torch.cat(batch_idx, dim=0)
        else:
            cls = torch.zeros((0,), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            idx = torch.zeros((0,), dtype=torch.float32)

        yield {
            "img": torch.stack(batch_images, dim=0),
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": idx,
        }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def loss_scalar(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    loss, _loss_items = model.loss(batch)
    if isinstance(loss, torch.Tensor):
        return loss.sum()
    if isinstance(loss, (tuple, list)):
        return sum(x.sum() for x in loss if isinstance(x, torch.Tensor))
    raise TypeError(f"Unsupported loss type: {type(loss)!r}")


def ensure_loss_args(model: torch.nn.Module, ckpt: dict) -> None:
    train_args = dict(ckpt.get("train_args") or {})
    train_args.setdefault("task", "detect")
    train_args.setdefault("mode", "train")
    train_args.setdefault("imgsz", 640)
    train_args.setdefault("box", 7.5)
    train_args.setdefault("cls", 0.5)
    train_args.setdefault("dfl", 1.5)
    train_args.setdefault("box_iou", "CIoU")
    model.args = get_cfg(overrides=train_args)


def detect_terminal_conv_ids(model: torch.nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if module.__class__.__name__ != "Detect":
            continue
        for sub_name, sub in module.named_modules():
            if isinstance(sub, torch.nn.Conv2d) and (sub_name.endswith(".2") or sub_name == "dfl.conv"):
                ids.add(id(sub))
    return ids


def collect_taylor_scores(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    ignore_keywords: list[str],
) -> dict[int, tuple[str, torch.nn.Conv2d, torch.Tensor]]:
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch in batches:
        batch = move_batch(batch, device)
        loss = loss_scalar(model, batch) / max(1, len(batches))
        total_loss += float(loss.detach().cpu())
        loss.backward()

    terminal_ids = detect_terminal_conv_ids(model)
    scores: dict[int, tuple[str, torch.nn.Conv2d, torch.Tensor]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue
        if id(module) in terminal_ids:
            continue
        if any(keyword in name for keyword in ignore_keywords):
            continue
        if module.groups != 1:
            continue
        if module.weight.grad is None:
            continue

        score = (module.weight * module.weight.grad).detach().abs().flatten(1).sum(dim=1)
        if module.bias is not None and module.bias.grad is not None:
            score = score + (module.bias * module.bias.grad).detach().abs()
        scores[id(module)] = (name, module, score.cpu())

    print(f"Accumulated Taylor loss proxy: {total_loss:.6f}")
    print(f"Scored prunable Conv2d layers: {len(scores)}")
    return scores


def build_pruning_plan(
    scores: dict[int, tuple[str, torch.nn.Conv2d, torch.Tensor]],
    ratio: float,
    min_channels: int,
    max_layer_ratio: float,
) -> dict[int, list[int]]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("--ratio must be between 0 and 1")
    if not scores:
        raise RuntimeError("No Taylor scores were collected. Check ignore list and calibration loss.")

    all_scores = torch.cat([item[2] for item in scores.values()])
    threshold_index = min(int(all_scores.numel() * ratio), all_scores.numel() - 1)
    threshold = all_scores.sort().values[threshold_index]

    plan: dict[int, list[int]] = {}
    for conv_id, (_name, conv, score) in scores.items():
        idxs = torch.nonzero(score <= threshold, as_tuple=False).flatten().tolist()
        max_remove = int(conv.out_channels * max_layer_ratio)
        keep_limit = max(0, conv.out_channels - min_channels)
        limit = max(0, min(max_remove, keep_limit))
        if len(idxs) > limit:
            idxs = sorted(idxs, key=lambda i: float(score[i]))[:limit]
        if idxs:
            plan[conv_id] = idxs

    return plan


def fix_depthwise_groups(model: torch.nn.Module) -> int:
    fixed = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.groups != 1 and module.weight.shape[1] == 1:
            if module.groups != module.in_channels:
                module.groups = module.in_channels
                fixed += 1
    return fixed


def save_checkpoint(ckpt: dict, model: torch.nn.Module, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pruned_model = deepcopy(model).half().cpu()
    ckpt["model"] = pruned_model
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["scaler"] = None
    torch.save(ckpt, save_path)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    save_path = resolve_path(args.save) or model_path.with_name(f"{model_path.stem}_taylor_pruned_r{args.ratio:.2f}.pt")

    if save_path.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{save_path} already exists. Use --overwrite or choose --save.")

    device = pick_device(args.device)
    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    yolo = YOLO(str(model_path))
    model = yolo.model.to(device)
    ensure_loss_args(model, ckpt)

    image_root, label_root = resolve_dataset_paths(data_path)
    batches = list(
        iter_calibration_batches(
            image_root=image_root,
            label_root=label_root,
            imgsz=args.imgsz,
            batch_size=args.batch,
            max_batches=args.calib_batches,
            seed=args.seed,
        )
    )
    print(f"Calibration batches: {len(batches)} x batch={args.batch}")

    scores = collect_taylor_scores(model, batches, device, args.ignore)
    plan = build_pruning_plan(scores, args.ratio, args.min_channels, args.max_layer_ratio)
    planned_channels = sum(len(v) for v in plan.values())

    print("\nTaylor pruning plan:")
    for conv_id, idxs in sorted(plan.items(), key=lambda item: scores[item[0]][0]):
        name, conv, _score = scores[conv_id]
        print(f"  {name}: remove {len(idxs)}/{conv.out_channels}")
    print(f"Total planned output channels: {planned_channels}")

    if args.dry_run:
        return

    model.eval()
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    dependency_graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    pruned_layers = 0
    pruned_channels = 0
    for conv_id, idxs in sorted(plan.items(), key=lambda item: scores[item[0]][0]):
        name, conv, _score = scores[conv_id]
        try:
            group = dependency_graph.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            if dependency_graph.check_pruning_group(group):
                group.prune()
                pruned_layers += 1
                pruned_channels += len(idxs)
                print(f"Pruned {name}: {len(idxs)} channels")
        except Exception as exc:
            print(f"Skipped {name}: {exc}")

    fixed_dw = fix_depthwise_groups(model)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(1, 3, args.imgsz, args.imgsz, device=device))

    save_checkpoint(ckpt, model, save_path)
    print(f"\nPruned layers: {pruned_layers}")
    print(f"Pruned output channels: {pruned_channels}")
    print(f"Fixed depthwise groups: {fixed_dw}")
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
