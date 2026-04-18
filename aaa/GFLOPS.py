import sys
import os
sys.path.insert(0, '/3240410021/ultralytics-main')
import torch
import torch.nn as nn
from ultralytics import YOLO
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 尝试导入 thop
try:
    from thop import profile
except ImportError:
    profile = None
    print("Error: 未检测到 thop 库，GFLOPs 将显示为 0。请先运行: pip install thop")


def get_layer_macs(module, input_meta):
    """
    在钩子外部单独计算一层的 MACs
    input_meta: 存储的输入结构信息，格式为列表，对应 forward 的参数列表
    """
    if profile is None or input_meta is None: return 0.0

    try:
        # 重建 Dummy Input (模拟 forward 的参数)
        dummy_args = []

        for item in input_meta:
            # 情况 1: 参数是一个 Tensor 形状 [C, H, W]
            if isinstance(item, tuple) and all(isinstance(i, int) for i in item):
                # 构造 [1, C, H, W] 的张量
                dummy_args.append(torch.randn(1, *item).to(next(module.parameters()).device))

            # 情况 2: 参数是一个 Tensor 列表 (如 Detect/SFPN 的输入) [[C,H,W], [C,H,W]]
            elif isinstance(item, list):
                tensor_list = []
                for shape in item:
                    if shape is not None:
                        tensor_list.append(torch.randn(1, *shape).to(next(module.parameters()).device))
                    else:
                        tensor_list.append(None)
                dummy_args.append(tensor_list)

            # 情况 3: 其他情况 (None 等)
            else:
                dummy_args.append(item)

        # 转换为 tuple 传入 thop (对应 *args)
        # verbose=False 关闭 thop 自身的打印
        macs, _ = profile(module, inputs=tuple(dummy_args), verbose=False)
        return macs / 1e9 * 2  # GFLOPs

    except Exception as e:
        # 如果你想看具体的报错，可以取消下面这行的注释
        # print(f"Calc Error in {module._get_name()}: {e}")
        return 0.0


def analyze_model(config_path, imgsz=640):
    print(f"Loading model: {config_path} ...")
    model = YOLO(config_path)
    net = model.model
    device = next(net.parameters()).device

    # 存储每一层的输入元数据
    # Key: layer_idx, Value: [arg1_shape, arg2_shape, ...]
    layer_inputs = {}

    # --- 步骤 1: 注册钩子捕获形状 (Probe Phase) ---
    def probe_hook(idx):
        def hook(module, input, output):
            # input 是一个 tuple，代表 forward(*args) 的所有参数
            args_meta = []
            for arg in input:
                if isinstance(arg, torch.Tensor):
                    # 记录 Tensor 形状 (去掉 Batch 维度)
                    args_meta.append(tuple(arg.shape[1:]))
                elif isinstance(arg, (list, tuple)) and len(arg) > 0 and isinstance(arg[0], torch.Tensor):
                    # 记录 Tensor 列表的形状 (如 Detect 的输入)
                    list_meta = [tuple(x.shape[1:]) for x in arg]
                    args_meta.append(list_meta)
                else:
                    args_meta.append(None)

            layer_inputs[idx] = args_meta

        return hook

    hooks = []
    for i, m in enumerate(net.model):
        hooks.append(m.register_forward_hook(probe_hook(i)))

    # --- 步骤 2: 运行一次前向传播 ---
    print("Running probe pass (capturing shapes)...")
    dummy_img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    with torch.no_grad():
        try:
            net(dummy_img)
        except Exception as e:
            # 这里的报错通常是 Detect 头后处理引起的，不影响形状捕获
            pass

    # 移除钩子 (防止重复打印和干扰后续计算)
    for h in hooks:
        h.remove()

    # --- 步骤 3: 离线计算 GFLOPs (Calc Phase) ---
    print(
        f"\n{'Idx':>3} | {'From':>4} | {'n':>3} | {'Params':>10} | {'Module':<25} | {'GFLOPs':>8} | {'Input Shape':>20}")
    print("-" * 110)

    total_flops = 0

    for i, m in enumerate(net.model):
        params = sum(p.numel() for p in m.parameters())
        module_name = m._get_name()

        # 获取刚才捕获的输入信息
        input_meta = layer_inputs.get(i, None)

        # 计算 GFLOPs
        flops = 0.0
        in_shape_str = "-"

        if input_meta:
            flops = get_layer_macs(m, input_meta)
            total_flops += flops

            # 格式化打印形状
            if len(input_meta) == 1:
                # 单参数
                data = input_meta[0]
                if isinstance(data, list):  # 列表输入
                    in_shape_str = str(data)
                elif isinstance(data, tuple):  # Tensor输入
                    in_shape_str = str(list(data))
            else:
                # 多参数
                in_shape_str = "Multi-Args"

        print(
            f"{i:>3} | {'-1':>4} | {'1':>3} | {params:>10,d} | {module_name:<25} | {flops:>8.3f} | {in_shape_str:>20}")

    print("-" * 110)
    print(f"Total GFLOPs: {total_flops:.3f}")


if __name__ == "__main__":
    # 请确保这里是你的 yaml 文件路径
    # CONFIG = "ultralytics/cfg/models/11/yolo11MSAL.yaml"
    CONFIG = r"runs/detect/train/weights/best.pt"
    analyze_model(CONFIG, imgsz=640)