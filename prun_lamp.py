from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.cfg import get_cfg

# ==========================================
# 1. 剪枝保护黑名单
# ==========================================
DEFAULT_IGNORE_KEYWORDS = (
    "model.24",            # 保护检测头
    "model.10.cv1_right",  # 保护 C2PSA 右路输入 (注意力源头)
    "model.10.m.",         # 保护 C2PSA 内部结构 (多头注意力、FFN等)
    "attn",
    "cross_att",
    "shape_router",
    "shape_h",
    "shape_v",
    "sfbs",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAMP channel pruning for YOLOv11.")
    parser.add_argument("--model", type=Path, default=Path("runs/detect/prun/ALL/weights/best.pt"))
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=0.50, help="全局通道剪枝率")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-channels", type=int, default=8, help="每层最少保留的通道数")
    parser.add_argument("--max-layer-ratio", type=float, default=0.80, help="单层最大允许剪裁比例")
    parser.add_argument("--ignore", nargs="*", default=list(DEFAULT_IGNORE_KEYWORDS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def pick_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)

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
        if module.__class__.__name__ == "Detect":
            for sub_name, sub in module.named_modules():
                if isinstance(sub, nn.Conv2d) and (sub_name.endswith(".2") or sub_name == "dfl.conv"):
                    ids.add(id(sub))
    return ids

# ==========================================
# 2. 核心算法：LAMP 结构化评分收集器
# ==========================================
@torch.no_grad()
def collect_lamp_scores(
    model: torch.nn.Module,
    ignore_keywords: list[str],
    p: int = 2
) -> dict[int, tuple[str, nn.Conv2d, torch.Tensor]]:
    
    terminal_ids = detect_terminal_conv_ids(model)
    scores: dict[int, tuple[str, nn.Conv2d, torch.Tensor]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d): continue
        if id(module) in terminal_ids: continue
        if any(keyword in name for keyword in ignore_keywords): continue
        if module.groups != 1: continue

        w = module.weight.data
        base_imp = torch.norm(w.reshape(w.shape[0], -1), dim=1, p=p)

        imp_sq = base_imp ** 2
        sorted_imp_sq, indices = torch.sort(imp_sq)
        surviving_sum = torch.cumsum(sorted_imp_sq.flip(0), dim=0).flip(0)
        lamp_score = sorted_imp_sq / (surviving_sum + 1e-8)

        final_scores = torch.zeros_like(lamp_score)
        final_scores[indices] = lamp_score
        scores[id(module)] = (name, module, final_scores.cpu())

    return scores

def build_pruning_plan(
    scores: dict[int, tuple[str, nn.Conv2d, torch.Tensor]],
    ratio: float,
    min_channels: int,
    max_layer_ratio: float,
) -> dict[int, list[int]]:
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
        if idxs: plan[conv_id] = idxs
    return plan

def fix_depthwise_groups(model: torch.nn.Module) -> int:
    fixed = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.groups != 1 and module.weight.shape[1] == 1:
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
    ckpt["updates"] = None
    torch.save(ckpt, save_path)

# ==========================================
# 3. 主流程
# ==========================================
def main() -> None:
    args = parse_args()
    model_path = args.model
    if not model_path.exists(): raise FileNotFoundError(f"未找到模型: {model_path}")
    save_path = args.save or model_path.with_name(f"{model_path.stem}_lamp_pruned_r{args.ratio:.2f}.pt")

    device = pick_device(args.device)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    yolo = YOLO(str(model_path))
    model = yolo.model.to(device)
    ensure_loss_args(model, ckpt)

    # =========================================================================
    # 🌟 终极致胜魔法：解除 YOLO 推理模式下的参数冻结，让追踪器睁开眼睛！
    # =========================================================================
    for p in model.parameters():
        p.requires_grad_(True)
    # =========================================================================

    model.eval()

    print("\nCollecting LAMP scores...")
    scores = collect_lamp_scores(model, args.ignore, p=2)
    plan = build_pruning_plan(scores, args.ratio, args.min_channels, args.max_layer_ratio)
    
    print("\nLAMP pruning plan:")
    for conv_id, idxs in sorted(plan.items(), key=lambda item: scores[item[0]][0]):
        name, conv, _score = scores[conv_id]
        print(f"  {name}: remove {len(idxs)}/{conv.out_channels}")

    if args.dry_run: return

    print("\nBuilding Dependency Graph...")
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

    fix_depthwise_groups(model)

    # 保存权重
    save_checkpoint(ckpt, model, save_path)
    print(f"\n✅ 真实剪枝执行完毕！成功修剪了 {pruned_channels} 个通道。模型已保存至: {save_path}")

    # 后置清算计算量 (隔绝 Hook 污染)
    print("\nCalculating Compression Ratio...")
    pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    yolo_base = YOLO(str(model_path))
    model_base = yolo_base.model.to(device)
    ensure_loss_args(model_base, ckpt)
    model_base.eval()
    base_ops, base_params = tp.utils.count_ops_and_params(model_base, example_inputs)
    
    print(f"[Before Pruning] FLOPs: {(base_ops*2) / 1e9:.2f} G, Params: {base_params / 1e6:.2f} M")
    print(f"[After  Pruning] FLOPs: {(pruned_ops*2) / 1e9:.2f} G, Params: {pruned_params / 1e6:.2f} M")
    print(f"Compression Ratio -> FLOPs reduced: {1 - pruned_ops/base_ops:.2%}, Params reduced: {1 - pruned_params/base_params:.2%}")

if __name__ == "__main__":
    main()