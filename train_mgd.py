import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

# ================= 1. 核心计算：掩码引导的特征蒸馏 Loss =================
def compute_mask_guided_kd_loss(s_feats, t_feats, batch, img_size=640):
    # (这部分代码保持不变，与上一版完全一致)
    total_kd_loss = 0.0
    for s_feat, t_feat in zip(s_feats, t_feats):
        B, C, H, W = s_feat.shape
        device = s_feat.device
        mask = torch.zeros((B, 1, H, W), device=device)
        bboxes = batch['bboxes']
        batch_indices = batch['batch_idx'].int()
        
        for i in range(len(bboxes)):
            b_idx = batch_indices[i]
            cx, cy, w, h = bboxes[i]
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x1 < x2 and y1 < y2:
                mask[b_idx, 0, y1:y2, x1:x2] = 1.0 
                
        t_feat_aligned = t_feat[:, :C, :, :] 
        s_fg = s_feat * mask
        t_fg = t_feat_aligned * mask
        fg_pixels = mask.sum() + 1e-6
        layer_loss = F.mse_loss(s_fg, t_fg, reduction='sum') / fg_pixels
        total_kd_loss += layer_loss
    return total_kd_loss


# ================= 2. 自定义蒸馏 Trainer =================
class MGD_DetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = None
        self.s_hook_feats = []
        self.t_hook_feats = []
        self.kd_weight = 0.5
        self.hook_layers = [21,22,23] 

# 🚀 核心改动：直接在这里重写，彻底干掉 YAML 构建！
    def get_model(self, cfg=None, weights=None, verbose=True):
        print(f"\n🚀 [拦截成功] 强行接管模型构建！拒绝使用 YAML。")
        
        # 1. 智能判断 weights 的类型
        if isinstance(weights, nn.Module):
            print("📂 框架已预加载模型实例，直接接管进行通道修复...")
            model = weights.float() # 强制转回 FP32，防止半精度报错
        elif isinstance(weights, str):
            print(f"📂 传入的是路径，正在加载真实的剪枝权重: {weights}")
            ckpt = torch.load(weights, map_location='cpu')
            if isinstance(ckpt, dict) and 'model' in ckpt:
                model = ckpt['model'].float()
            else:
                model = ckpt.float()
        else:
            raise TypeError(f"未知的 weights 类型: {type(weights)}")

        print("🔧 正在同步剪枝模型的通道属性...")
        
        # 2. 修复通道匹配问题 (防 fuse 和 loss 计算崩溃的核心)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.out_channels = m.weight.shape[0]
                m.in_channels = m.weight.shape[1] * m.groups
            elif isinstance(m, nn.BatchNorm2d):
                m.num_features = m.weight.shape[0]

        # 3. 适配 Trainer 环境
        model.to(self.device)
        model.args = self.args
        if hasattr(model, 'init_criterion'):
            model.criterion = model.init_criterion()
            
        print("✅ [适配完成] 剪枝模型结构强制同步完毕！\n")
        return model

    def setup_teacher(self):
        print("🎓 正在加载教师模型...")
        # 【注意：确保这里的教师模型路径正确】
        teacher = YOLO(r"runs/detect/prun/RSS_YOLO_SFPN3/weights/best.pt").model
        
        for param in teacher.parameters():
            param.requires_grad = False
        self.teacher_model = teacher.to(self.device).eval()
        
        def get_t_hook():
            def hook(module, inp, out):
                self.t_hook_feats.append(out)
            return hook
            
        def get_s_hook():
            def hook(module, inp, out):
                self.s_hook_feats.append(out)
            return hook

        for layer_idx in self.hook_layers:
            self.teacher_model.model[layer_idx].register_forward_hook(get_t_hook())
            self.model.model[layer_idx].register_forward_hook(get_s_hook())
            
        print(f"🔗 成功在第 {self.hook_layers} 层挂载特征捕获 Hook！")

    def train_step(self, batch):
        if self.teacher_model is None:
            self.setup_teacher()
            
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(self.amp):
            loss, loss_items = self.model(batch)
            with torch.no_grad():
                self.teacher_model(batch)
            
            kd_loss = compute_mask_guided_kd_loss(self.s_hook_feats, self.t_hook_feats, batch)
            total_loss = loss + self.kd_weight * kd_loss
            loss_items = torch.cat((loss_items, kd_loss.unsqueeze(0).detach())) 

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.s_hook_feats.clear()
        self.t_hook_feats.clear()
        return loss_items


# ================= 3. 引擎路由替换 =================
class MGD_YOLO(YOLO):
    @property
    def task_map(self):
        map_dict = super().task_map
        map_dict["detect"]["trainer"] = MGD_DetectionTrainer
        return map_dict


if __name__ == '__main__':
    print("🚀 启动掩码引导解耦蒸馏 (Mask-Guided KD) 流程...")
    
    # 传入你的 LAMP 剪枝模型路径
    student_model = MGD_YOLO(r"runs\detect\train4\weights\last.pt")
    
    student_model.train(
        data="my_VisDrone.yaml",
        epochs=400,            
        batch=8,
        patience=50,         
        imgsz=640,
        device=0,
        workers=2,
        amp=True,
        cache=True,
        optimizer="SGD",       
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,       
        warmup_momentum=0.8,  
        box_iou="SDCIoU",
        project='/3240410021/ultralytics-main/runs/detect/prun',
        name='mgd',
    )