import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer

# ==========================================
# 自定义 EMA 自蒸馏训练器
# ==========================================
class EMASelfDistillTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks_initialized = False

    # ==========================================
    # 👇 完美集成的 get_model 拦截器 (保证加载剪枝结构)
    # ==========================================
    def get_model(self, cfg=None, weights=None, verbose=True):
        print(f"\n[Self-Distill] 🛑 拦截官方重建逻辑，强制加载剪枝物理结构...")
        
        # 1. 智能判断 weights 的类型
        if isinstance(weights, str) or hasattr(weights, 'resolve'):
            ckpt = torch.load(weights, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model']
            else:
                model = ckpt
        elif isinstance(weights, torch.nn.Module):
            model = weights
        elif isinstance(weights, dict) and 'model' in weights:
            model = weights['model']
        else:
            raise TypeError(f"无法识别的 weights 类型: {type(weights)}")

        # 2. 【关键】解除冻结并设置为训练模式
        model = model.float()
        for param in model.parameters():
            param.requires_grad = True # 强制要求计算梯度
        model.train()
        
        # 3. 挂载 args
        model.args = self.args 
        
        print(f"[Self-Distill] ✅ 成功加载剪枝模型！准备就绪。")
        return model 

    # ==========================================
    # EMA 蒸馏核心逻辑
    # ==========================================
    def setup_distill_hooks(self):
        """为在线模型(Student)和 EMA 模型(Teacher)挂载 Hook"""
        print("\n[Self-Distill] 🚀 初始化 EMA 自蒸馏逻辑...")
        
        self.s_feats = {}
        self.t_feats = {}

        def get_hook(name, storage):
            def hook(module, input, output):
                storage[name] = output
            return hook

        # 核心层索引：16-P3, 19-P4, 22-P5, 23-Detect Head
        target_layers = {'P3': 16, 'P4': 19, 'P5': 22, 'Head': 23}

        for name, idx in target_layers.items():
            # 挂载到在线模型 (Student)
            self.model.model[idx].register_forward_hook(get_hook(name, self.s_feats))
            # 挂载到 EMA 模型 (Teacher) - self.ema.ema 是真正的权重模型
            self.ema.ema.model[idx].register_forward_hook(get_hook(name, self.t_feats))

        # 蒸馏超参数
        self.temp = 3.0        # 温度参数
        self.alpha_feat = 2.0  # 特征蒸馏权重
        self.alpha_kl = 1.0    # 逻辑蒸馏权重
        self.mse_loss = nn.MSELoss()
        
        print(f"[Self-Distill] EMA 教师已就绪，通道天然完全匹配，无需 1x1 卷积对齐！\n")

    def train_step(self):
        """重写核心训练步"""
        if not self.hooks_initialized:
            self.setup_distill_hooks()
            self.hooks_initialized = True

        self.optimizer.zero_grad()

        # 使用 AMP 混合精度上下文
        with torch.cuda.amp.autocast(self.amp):
            # 1. 在线模型 (Student) 前向传播计算 Hard Loss
            hard_loss, loss_items = self.model(self.batch)

            # 2. EMA 模型 (Teacher) 前向传播
            with torch.no_grad():
                _ = self.ema.ema(self.batch['img'])

            # 3. 计算特征图自蒸馏损失 (MSE)
            # 因为是自蒸馏，通道完全一致，直接计算 MSE，无需对齐！
            loss_p3 = self.mse_loss(self.s_feats['P3'], self.t_feats['P3'].detach())
            loss_p4 = self.mse_loss(self.s_feats['P4'], self.t_feats['P4'].detach())
            loss_p5 = self.mse_loss(self.s_feats['P5'], self.t_feats['P5'].detach())
            
            # 针对小目标的特征权重分配
            feat_loss = 2.0 * loss_p3 + 1.0 * loss_p4 + 0.5 * loss_p5

            # 4. 计算逻辑层自蒸馏损失 (KL)
            kl_loss = 0.0
            nc = self.model.model[-1].nc
            
            for s_logits, t_logits in zip(self.s_feats['Head'], self.t_feats['Head']):
                s_cls = s_logits[:, :nc, :, :]
                t_cls = t_logits[:, :nc, :, :].detach()

                s_cls_flat = s_cls.reshape(s_cls.size(0), nc, -1).transpose(1, 2)
                t_cls_flat = t_cls.reshape(t_cls.size(0), nc, -1).transpose(1, 2)

                s_soft = F.log_softmax(s_cls_flat / self.temp, dim=-1)
                t_soft = F.softmax(t_cls_flat / self.temp, dim=-1)
                kl_loss += F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.temp ** 2)

            kl_loss = kl_loss / len(self.s_feats['Head'])

            # 5. 合并总损失
            total_loss = hard_loss + self.alpha_feat * feat_loss + self.alpha_kl * kl_loss

        # 6. 反向传播与梯度更新
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return total_loss, loss_items

# ==========================================
# 启动脚本
# ==========================================
if __name__ == "__main__":
    args = {
        'model': 'runs/detect/prun/yolov11n/weights/pruned.pt',   # 学生模型 (你剪枝后的模型)
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
        'name': 'ema_prun_distill',    # 本次运行的文件名
        'conf': 0.05,    
    }

    # 实例化 EMA 蒸馏训练器 (不需要传 teacher_path，因为老师就是模型本身)
    trainer = EMASelfDistillTrainer(overrides=args)
    
    print("🚀 开始 EMA 特征+逻辑自蒸馏训练...")
    trainer.train()
