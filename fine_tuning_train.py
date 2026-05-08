import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import torch.nn as nn

# ================= 终极破解补丁：适配剪枝模型的所有兼容性问题 =================
# 1. 备份原版的 get_model 方法
original_get_model = DetectionTrainer.get_model

# 2. 定义拦截方法
def custom_get_model(self, cfg=None, weights=None, verbose=True):
    if isinstance(weights, nn.Module):
        print("🚀 成功拦截！正在处理剪枝模型的设备与参数适配...")
        
        # A. 强制设备同步：将模型移动到 Trainer 指定的设备 (如 cuda:0)
        weights.to(self.device)
        
        # B. 核心修复：参数对象适配
        # 将 Trainer 已经封装好的 self.args 直接赋值给 weights.args
        # self.args 是一个支持 .box 访问的对象，能完美解决 AttributeError
        weights.args = self.args 
        
        # C. 重新初始化损失函数：确保 Loss 内部的张量和模型在同一设备
        if hasattr(weights, 'init_criterion'):
            weights.criterion = weights.init_criterion()
            
        print("✅ 适配完成！剪枝模型已就绪，开始训练进度条...")
        return weights
    
    # 如果不是剪枝模型，走原版逻辑
    return original_get_model(self, cfg, weights, verbose)

# 3. 注入补丁
DetectionTrainer.get_model = custom_get_model
# ===========================================================================


if __name__ == '__main__':
    # 使用你最初始的剪枝权重即可
    model1 = YOLO(r"runs/detect/best_taylor_pruned_r0.50.pt")
    
    model1.train(
        data="my_VisDrone.yaml",
        epochs=400,
        batch=8,
        imgsz=640,
        device=0,
        workers=0,
        amp=True,
        cos_lr=True,
        optimizer="SGD",
        lr0=0.005,
        lrf=0.01,
        box_iou="SNAIoU",
        warmup_epochs=0,
        cache=True,
    )