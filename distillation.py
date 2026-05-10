import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect.train import DetectionTrainer


# ==========================================
# 1. 特征通道对齐模块 (Feature Adaptation)
# ==========================================
class FeatureAdaptation(nn.Module):
    """用于将学生模型较少的通道数，通过 1x1 卷积升维对齐到教师模型的通道数"""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        
    def forward(self, x):
        return self.bn(self.conv(x))


# ==========================================
# 2. 自定义蒸馏训练器 (继承官方 Trainer)
# ==========================================
class DistillationTrainer(DetectionTrainer):
    def __init__(self, teacher_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_path = teacher_path
        self.teacher_loaded = False  # 懒加载标记

    def get_model(self, cfg=None, weights=None, verbose=True):
        print(f"\n[Distillation] 🛑 拦截官方重建逻辑，强制加载剪枝物理结构...")
        
        # 1. 智能判断 weights 的类型
        if isinstance(weights, str) or hasattr(weights, 'resolve'):
            # 如果是字符串或 Path 对象 (文件路径)
            ckpt = torch.load(weights, map_location=self.device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model']
            else:
                model = ckpt
        elif isinstance(weights, torch.nn.Module):
            # 如果 Ultralytics 底层已经帮你把模型对象实例化并传进来了，直接接收！
            model = weights
        elif isinstance(weights, dict) and 'model' in weights:
            # 极端情况：传进来的是整个 ckpt 字典
            model = weights['model']
        else:
            raise TypeError(f"无法识别的 weights 类型: {type(weights)}")

        # 2. 【关键】解除冻结并设置为训练模式 (解决 0 gradients 问题)
        model = model.float()
        for param in model.parameters():
            param.requires_grad = True # 强制要求计算梯度
        model.train()
        
        # 3. 挂载 args (Ultralytics 训练代码强依赖此属性)
        model.args = self.args 
        
        print(f"[Distillation] ✅ 成功加载剪枝模型！准备就绪。")
        return model    

    def load_teacher_and_hooks(self):
        """懒加载教师模型，挂载 Hook，并初始化特征对齐模块"""
        print("\n[Distillation] 初始化教师模型与特征对齐模块...")
        
        # 1. 使用原生 PyTorch 加载教师模型 (最稳定的方式，完美避开 API 版本差异)
        ckpt = torch.load(self.teacher_path, map_location=self.device)
        self.teacher = ckpt['model'].float().eval() # 提取模型本体，转为单精度并设置为 eval 模式
        
        for param in self.teacher.parameters():
            param.requires_grad = False

        # 2. 初始化 Hook 字典
        self.t_feats = {}
        self.s_feats = {}

        def get_t_hook(name):
            def hook(module, input, output): self.t_feats[name] = output
            return hook

        def get_s_hook(name):
            def hook(module, input, output): self.s_feats[name] = output
            return hook

        # 3. 在网络关键节点挂载 Hook
        # YOLO11 中：16层为P3(小目标), 19层为P4(中目标), 22层为P5(大目标), 23层为Detect头
        target_layers = {'P3': 16, 'P4': 19, 'P5': 22, 'Head': 23}
        for name, idx in target_layers.items():
            self.teacher.model[idx].register_forward_hook(get_t_hook(name))
            self.model.model[idx].register_forward_hook(get_s_hook(name))

        # 4. 前向传播一次 Dummy Data，自动探测通道数并初始化 1x1 对齐卷积
        dummy_img = torch.zeros((1, 3, self.args.imgsz, self.args.imgsz), device=self.device)
        with torch.no_grad():
            self.teacher(dummy_img)
        self.model(dummy_img) # 学生模型前向

        self.adapt_modules = nn.ModuleDict({
            'P3': FeatureAdaptation(self.s_feats['P3'].shape[1], self.t_feats['P3'].shape[1]),
            'P4': FeatureAdaptation(self.s_feats['P4'].shape[1], self.t_feats['P4'].shape[1]),
            'P5': FeatureAdaptation(self.s_feats['P5'].shape[1], self.t_feats['P5'].shape[1])
        }).to(self.device)

        # 5. 为 1x1 对齐模块单独设置优化器
        self.adapt_optimizer = torch.optim.AdamW(self.adapt_modules.parameters(), lr=1e-3)
        self.mse_loss = nn.MSELoss()
        self.temp = 2.0 # 蒸馏温度参数
        
        print("[Distillation] 初始化完成！通道映射关系：")
        print(f"P3: {self.s_feats['P3'].shape[1]} -> {self.t_feats['P3'].shape[1]}")
        print(f"P4: {self.s_feats['P4'].shape[1]} -> {self.t_feats['P4'].shape[1]}")
        print(f"P5: {self.s_feats['P5'].shape[1]} -> {self.t_feats['P5'].shape[1]}\n")

    def train_step(self):
        """重写核心训练步，注入蒸馏逻辑"""
        # 在第一次 Step 时懒加载教师模型（确保此时环境和 Device 已完全准备好）
        if not self.teacher_loaded:
            self.load_teacher_and_hooks()
            self.teacher_loaded = True

        self.optimizer.zero_grad()
        self.adapt_optimizer.zero_grad()

        # 使用 AMP 混合精度上下文
        with torch.cuda.amp.autocast(self.amp):
            # 1. 学生模型前向传播（自动计算 Hard Loss）
            hard_loss, loss_items = self.model(self.batch)

            # 2. 教师模型前向传播（不计算梯度，仅触发 Hook 获取特征）
            with torch.no_grad():
                self.teacher(self.batch['img'])

            # ================= 计算特征图蒸馏损失 =================
            s_p3_aligned = self.adapt_modules['P3'](self.s_feats['P3'])
            s_p4_aligned = self.adapt_modules['P4'](self.s_feats['P4'])
            s_p5_aligned = self.adapt_modules['P5'](self.s_feats['P5'])

            loss_p3 = self.mse_loss(s_p3_aligned, self.t_feats['P3'].detach())
            loss_p4 = self.mse_loss(s_p4_aligned, self.t_feats['P4'].detach())
            loss_p5 = self.mse_loss(s_p5_aligned, self.t_feats['P5'].detach())

            # 针对 VisDrone 小目标场景，赋予 P3 更高的蒸馏权重
            feat_loss = 2.0 * loss_p3 + 1.0 * loss_p4 + 0.5 * loss_p5

            # ================= 计算逻辑层软标签蒸馏损失 (KL) =================
            kl_loss = 0.0
            nc = self.model.model[-1].nc # 自动获取类别数
            
            # self.s_feats['Head'] 包含 P3, P4, P5 尺度的预测列表
            for s_logits, t_logits in zip(self.s_feats['Head'], self.t_feats['Head']):
                # 截取前 nc 个通道 (分类通道)
                s_cls = s_logits[:, :nc, :, :]
                t_cls = t_logits[:, :nc, :, :].detach()

                # 展平以便计算 KL 散度: (B, C, H, W) -> (B, H*W, C)
                s_cls_flat = s_cls.reshape(s_cls.size(0), nc, -1).transpose(1, 2)
                t_cls_flat = t_cls.reshape(t_cls.size(0), nc, -1).transpose(1, 2)

                s_soft = F.log_softmax(s_cls_flat / self.temp, dim=-1)
                t_soft = F.softmax(t_cls_flat / self.temp, dim=-1)

                kl_loss += F.kl_div(s_soft, t_soft, reduction='batchmean') * (self.temp ** 2)

            kl_loss = kl_loss / len(self.s_feats['Head']) # 取多个尺度的均值

            # ================= 总损失合并 =================
            # 损失权重超参数 (可根据训练情况微调)
            alpha_hard = 1.0
            alpha_feat = 3.0  # 特征蒸馏权重
            alpha_kl = 1.0    # 软标签蒸馏权重
            
            total_loss = alpha_hard * hard_loss + alpha_feat * feat_loss + alpha_kl * kl_loss

        # 3. 反向传播与梯度更新 (支持 AMP scaler)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.step(self.adapt_optimizer) # 更新 1x1 卷积模块
        self.scaler.update()

        return total_loss, loss_items
    

# ==========================================
# 3. 启动训练入口
# ==========================================
if __name__ == "__main__":
    # 配置训练参数
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
        'name': 'prun_distill',    # 本次运行的文件名
        'conf': 0.05,        # 【过滤垃圾框】你上一个配置忘了加这个！把验证阈值从 0.001 提高到 0.05，大幅减轻 NMS 压力。
    }

    # 教师模型路径 (剪枝前的最优模型)
    TEACHER_WEIGHTS = 'runs/detect/prun/yolov11n/weights/best.pt'

    # 实例化自定义蒸馏训练器
    trainer = DistillationTrainer(teacher_path=TEACHER_WEIGHTS, overrides=args)
    
    # 启动训练
    print("🚀 开始特征+逻辑双重蒸馏训练...")
    trainer.train()