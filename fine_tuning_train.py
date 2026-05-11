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
        
        # ------------------ [新增核心修复：防 fuse 崩溃] ------------------
        # 在模型被真正用于训练和评估前，强制将底层属性与张量真实形状对齐
        print("🔧 正在同步剪枝模型的通道属性...")
        for m in weights.modules():
            if isinstance(m, nn.Conv2d):
                m.out_channels = m.weight.shape[0]
                m.in_channels = m.weight.shape[1] * m.groups
            elif isinstance(m, nn.BatchNorm2d):
                m.num_features = m.weight.shape[0]
        # ------------------------------------------------------------------
        
        # A. 强制设备同步：将模型移动到 Trainer 指定的设备 (如 cuda:0)
        weights.to(self.device)
        
        # B. 核心修复：参数对象适配
        # 将 Trainer 已经封装好的 self.args 直接赋值给 weights.args
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
    # 注意路径：检查一下之前路径里的多余字母 r"rruns/..."，改回正常路径
    model1 = YOLO(r"runs/detect/prun/ALL/weights/pruned.pt")
    
    model1.train(
        data="my_VisDrone.yaml",
        epochs=300,   # 建议先跑 1 个 epoch，确保最后的 fuse 和 val 顺利通过
        batch=8,
        patience=50,
        imgsz=640,
        device=0,
        workers=2,
        amp=True,
        cos_lr=True,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        box_iou="CIoU",
        warmup_epochs=0,
        cache=True,
        project='prun',
        name='prun_fine',
    )
    from distillation import DistillationTrainer_dis
    # 配置训练参数
    args = {
        'model': 'runs/detect/prun/ALL/weights/pruned.pt',   # 学生模型 (你剪枝后的模型)
        'data': 'ultralytics/cfg/datasets/my_VisDrone.yaml',      # 数据集配置文件
        'epochs': 300,             # 设大一点没关系，我们用 patience 兜底
        'batch': 8,               # 批次大小
        'patience': 50,            # 50 个 epoch mAP 不涨就自动停止
        'imgsz': 640,              # 图像尺寸
        'device': 0,               # 蒸馏过程中含有 Hook，强烈建议使用单卡(0)进行训练
        'workers': 2,
        'amp': True,
        'cos_lr': True,
        'optimizer': 'SGD',
        'lr0': 0.001,              # 剪枝后的微调学习率建议设小一点 (如 1e-3 或 1e-4)
        'lrf': 0.01,              # 学习率衰减因子
        'box_iou': 'CIoU',
        'warmup_epochs': 0,
        'cache': True,
        'project': 'prun',
        'name': 'prun_distill',    # 本次运行的文件名
        'conf': 0.05,        # 【过滤垃圾框】你上一个配置忘了加这个！把验证阈值从 0.001 提高到 0.05，大幅减轻 NMS 压力。
    }

    # 教师模型路径 (剪枝前的最优模型)
    TEACHER_WEIGHTS = 'runs/detect/prun/ALL/weights/best.pt'

    # 实例化自定义蒸馏训练器
    trainer = DistillationTrainer_dis(teacher_path=TEACHER_WEIGHTS, overrides=args)
    
    # 启动训练
    print("🚀 开始特征+逻辑双重蒸馏训练...")
    trainer.train()    