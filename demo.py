# import torch
# from ultralytics import YOLO

# yolo = YOLO("runs/detect/best.pt")
# model = yolo.model

# for name, m in model.named_modules():
#     if m.__class__.__name__ == "SACSP":
#         print(f"\n=== {name} (SACSP) ===")

#         for bname, bm in m.named_modules():
#             if isinstance(bm, torch.nn.Conv2d):
#                 print(f"  {bname}: in={bm.in_channels} out={bm.out_channels} groups={bm.groups}")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from ultralytics import YOLO

# ===================== 配置你的模型路径 =====================
# 未剪枝模型
model = YOLO(r"runs/detect/visdrone2019结果/IOU/SNAIOU/weights/best.pt")
# 剪枝模型（需要导出就替换这个路径）
# model = YOLO(r"runs/detect/best_taylor_pruned_r0.50.pt")

# 导出 TensorRT 引擎（GPU 极致加速）
model.export(
    format="engine",  # 导出为 TensorRT
    imgsz=640, 
    device=0,
    half=False,  # 开启 FP16 半精度，速度翻倍
    simplify=False,  # 简化模型，消除通道不对齐问题
    verbose=True       # 打印日志，能看到进度
)
print(model.model)
print(model.info())
