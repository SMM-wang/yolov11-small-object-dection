import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from ultralytics import YOLO

def generate_channel_contrast_diagram(base_model_path: str, pruned_model_path: str, save_path: str = "channel_contrast.png"):
    print("正在加载模型...")
    
    # 1. 加载未剪枝的基础模型 (Base)
    base_yolo = YOLO(base_model_path)
    base_model = base_yolo.model

    # 2. 加载剪枝后的模型 (Prune)
    # 注意：剪枝后的模型通常保存为普通的 dict，提取 'model' 键即可
    pruned_ckpt = torch.load(pruned_model_path, map_location="cpu", weights_only=False)
    pruned_model = pruned_ckpt.get("model") or pruned_ckpt
    
    # 3. 提取所有 Conv2d 层的通道数
    base_channels = {}
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d):
            base_channels[name] = module.out_channels

    pruned_channels = {}
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            pruned_channels[name] = module.out_channels

    # 4. 对齐数据
    layer_names = list(base_channels.keys())
    base_counts = [base_channels[name] for name in layer_names]
    # 如果某一层在剪枝后完全消失，则设为0（通常不会发生，安全起见加个默认值）
    pruned_counts = [pruned_channels.get(name, 0) for name in layer_names]

    print(f"共提取到 {len(layer_names)} 个卷积层。正在绘图...")

    # ==========================================
    # 5. 导出文字版txt
    # ==========================================
    txt_path = save_path.replace(".png", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(layer_names):
            f.write(f"{name}: {base_counts[i]}, {pruned_counts[i]}\n")
    print(f"✅ 文字版已成功保存至: {txt_path}")

    # ==========================================
    # 6. 开始使用 Matplotlib 绘图
    # ==========================================
    # 设置画布大小（宽度设得很大，防止底部标签重叠）
    plt.figure(figsize=(24, 6))
    
    # 设置 X 轴位置
    x_positions = range(len(layer_names))
    
    # 核心逻辑：先画黄色的 Base，再画红色的 Prune
    plt.bar(x_positions, base_counts, color='orange', label='base')
    plt.bar(x_positions, pruned_counts, color='red', label='prune')

    # 设置标题和图例
    plt.title('Channel contrast diagram', fontsize=18)
    plt.legend(fontsize=14)

    # 设置 X 轴的刻度和标签，旋转 90 度防止文字挤在一起
    plt.xticks(x_positions, layer_names, rotation=90, fontsize=8)
    
    # 去除两边多余的空白
    plt.xlim(-1, len(layer_names))
    
    # 紧凑布局，防止底部的名字被截断
    plt.tight_layout()

    # 6. 保存高清图表
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已成功保存至: {save_path}")

if __name__ == "__main__":
    # ⚠️ 在这里替换成你真实的模型路径
    BASE_MODEL = "runs/detect/prun/ALL/weights/best.pt" 
    PRUNED_MODEL = "runs/detect/prun/ALL/weights/best_taylor_pruned_r0.50.pt" 
    
    generate_channel_contrast_diagram(BASE_MODEL, PRUNED_MODEL, "lamp_channel_contrast.png")
