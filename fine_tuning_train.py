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

# 1. 定义回调函数
def validate_interval(trainer):
    """
    一个回调函数，用于控制每隔N个epoch进行一次验证。
    trainer.epoch 从0开始计数。
    """
    # 设置验证频率，例如每5个epoch验证一次
    interval = 40
    # 判断是否为验证周期 (epoch 0, 4, 9, 14, ...)
    if trainer.epoch % interval == 0:
        # 启用验证
        trainer.args.val = True
    else:
        # 禁用验证
        trainer.args.val = False

if __name__ == '__main__':


    model3 = YOLO(r"runs/detect/CODrone/RSSYOLO_LITE/weights/best_lamp_prun_2.5.pt")   
    # model3.add_callback('on_train_epoch_end', validate_interval)
    model3.train(
        # data="my_VisDrone.yaml",
        data="CODrone.yaml",
        epochs=400,             # 中间恢复阶段不需要 400 轮，交由早停控制
        batch=8,
        # patience=50,           # 连续 10 轮不长点直接切断，进入下一轮剪枝
        imgsz=640,
        device=0,
        workers=2,
        amp=True,
        cache=True,
        optimizer="SGD",       # 如果恢复太慢，强烈建议换成 "AdamW" 试试
        lr0=0.01,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,       # 绝对不能为 0，给网络 3 轮时间适应残缺结构
        # warmup_momentum=0.8,   # 降低预热期的动量，防止梯度过激
        box_iou="SDCIoU",
        project='/3240410021/ultralytics-main/runs/detect/CODrone',
        name='RSSYOLO_LITE_PURNED_2.5',

    )

#     from ultralytics import YOLO
#     from ultralytics import settings
#     settings["tensorboard"]=True
#     model4 = YOLO("ultralytics/cfg/models/11/yolo11SFPN3.yaml")
#     model4.train(
#         # data="my_VisDrone.yaml",
#         data="AI_TOD.yaml", 
# #         data="CODrone.yaml",
#         epochs=400,
#         batch=8,
#         # batch=8,
#         imgsz=640,
#         device=0,
#         workers=2,
#         amp=True,
#         # patience=25,  #早停
#         cos_lr=True, # 使用余弦退火调度器
#         # optimizer="AdamW",  # 使用AdamW优化器
#         optimizer="SGD",
#         # lr0=0.007,  # 设置初始学习率为0.007
#         lr0=0.01,
#         lrf=0.01,
#         box_iou="CIoU",
#         warmup_epochs=3,
#         cache = True,
#         project="/3240410021/ultralytics-main/runs/detect/AI_TOD结果",
#         # resume=True,
#         name="SFPN3_1",
#     )