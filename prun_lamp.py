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
    # "model.24",            # 保护检测头
    # "model.20.out_convs",  # 保护 SFB 输出卷积 (直接喂入 Detect 分类分支 input)
    "model.10.cv1_right",  # 保护 C2PSA 右路输入 (注意力源头)
    "model.10.m.",         # 保护 C2PSA 内部结构 (多头注意力、FFN等)
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAMP channel pruning for YOLOv11.")
    parser.add_argument("--model", type=Path, default=Path(r"C:\workspace\python\yolov11-small-object-dection\runs\detect\train8\weights\last.pt"))
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--ratio", type=float, default=0.66, help="全局通道剪枝率")
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
                if isinstance (sub,nn.Conv2d ) and (sub_name.endswith( ".2" ) or sub_name == "dfl.conv" ):
                # 保护检测头内的全部 Conv2d，不仅是终端层
                # 这样依赖图不会把上游的输入通道剪裁自动传播到检测头中间层
                # if isinstance(sub, nn.Conv2d):
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


def deduplicate_shared_activations(model: torch.nn.Module) -> int:
    activation_types = (nn.ReLU, nn.SiLU, nn.LeakyReLU, nn.Hardswish, nn.Mish)
    seen: set[int] = set()
    fixed = 0
    for parent in model.modules():
        for name, child in list(parent._modules.items()):
            if isinstance(child, activation_types):
                if id(child) in seen:
                    parent._modules[name] = deepcopy(child)
                    fixed += 1
                else:
                    seen.add(id(child))
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
    shared_acts = deduplicate_shared_activations(model)
    if shared_acts:
        print(f"Deduplicated {shared_acts} shared activation modules.")
    ensure_loss_args(model, ckpt)

    for p in model.parameters():
        p.requires_grad_(True)

    model.eval()

    print("\nCollecting LAMP scores...")
    scores = collect_lamp_scores(model, args.ignore, p=2)
    plan = build_pruning_plan(scores, args.ratio, args.min_channels, args.max_layer_ratio)
    
    print("\nLAMP initial pruning plan (Pre-Alignment):")
    for conv_id, idxs in sorted(plan.items(), key=lambda item: scores[item[0]][0]):
        name, conv, _score = scores[conv_id]
        print(f"  {name}: remove {len(idxs)}/{conv.out_channels}")

    if args.dry_run: return

    print("\nBuilding Dependency Graph & Aligning Channels...")
    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    # 让 Detect 头在依赖图追踪时不 detach one2one 分支，
    # 否则 torch_pruning 追不到 neck→one2one_cv2 的依赖，导致端对端头输入通道不对齐。
    from ultralytics.nn.modules.head import Detect
    prev_pruning_flag = Detect.pruning
    Detect.pruning = True
    try:
        dependency_graph = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    finally:
        Detect.pruning = prev_pruning_flag

    original_out_channels = {id(m): m.out_channels for _, m in model.named_modules() if isinstance(m, nn.Conv2d)}

    pruned_layers = 0
    pruned_channels = 0
    
    for conv_id, idxs in sorted(plan.items(), key=lambda item: scores[item[0]][0]):
        name, conv, _score = scores[conv_id]
        
        if conv.out_channels != original_out_channels[id(conv)]:
            print(f"Skipped {name}: 已在耦合组中被连带剪枝，免疫二次伤害。")
            continue
            
        try:
# =====================================================================
            # 🌟 破除对齐魔咒：直接信任 torch_pruning 的依赖图
            # =====================================================================
            # 1. 预构建组
            group = dependency_graph.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            
            # 2. 探雷器：只拦截非 DWConv 的奇葩分组卷积
            align_multiple = 1
            for dep, _ in group:
                module = dep.target.module
                if isinstance(module, nn.Conv2d) and module.groups > 1:
                    # 只要是深度卷积 (DWConv)，直接放行，不需要任何对齐！
                    # 依赖图会自动切断对应通道，最后的 fix_depthwise_groups 会修复 groups 参数
                    if module.groups == module.in_channels or module.groups == module.out_channels:
                        continue
                    align_multiple = max(align_multiple, module.groups)
            
            # 3. 对齐计算：仅在真·分组卷积时触发
            if align_multiple > 1:
                orig_out = conv.out_channels
                target_retained = orig_out - len(idxs)
                aligned_retained = ((target_retained + align_multiple - 1) // align_multiple) * align_multiple
                aligned_retained = min(aligned_retained, orig_out)
                aligned_remove = orig_out - aligned_retained
                
                if aligned_remove != len(idxs):
                    if aligned_remove == 0:
                        continue
                    idxs = torch.argsort(_score).flatten().tolist()[:aligned_remove]
                    group = dependency_graph.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            # =====================================================================

            if dependency_graph.check_pruning_group(group):
                group.prune()
                pruned_layers += 1
                pruned_channels += len(idxs)
                print(f"Pruned {name}: {len(idxs)} channels")
                
        except Exception as exc:
            print(f"Skipped {name}: {exc}")

    fix_depthwise_groups(model)
    save_checkpoint(ckpt, model, save_path)
    print(f"\n✅ 真实剪枝执行完毕！成功修剪了 {pruned_channels} 个通道。模型已保存至: {save_path}")

    print("\nCalculating Compression Ratio...")
    pruned_ops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    
    yolo_base = YOLO(str(model_path))
    model_base = yolo_base.model.to(device)
    deduplicate_shared_activations(model_base)
    ensure_loss_args(model_base, ckpt)
    model_base.eval()
    base_ops, base_params = tp.utils.count_ops_and_params(model_base, example_inputs)
    
    print(f"[Before Pruning] FLOPs: {(base_ops*2) / 1e9:.2f} G, Params: {base_params / 1e6:.2f} M")
    print(f"[After  Pruning] FLOPs: {(pruned_ops*2) / 1e9:.2f} G, Params: {pruned_params / 1e6:.2f} M")
    print(f"Compression Ratio -> FLOPs reduced: {1 - pruned_ops/base_ops:.2%}, Params reduced: {1 - pruned_params/base_params:.2%}")

if __name__ == "__main__":
    main()