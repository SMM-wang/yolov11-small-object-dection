import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import time
import numpy as np
from ultralytics import YOLO

# 加载模型
# model = YOLO(r"runs/detect/visdrone2019结果/IOU/SNAIOU/weights/best.pt")
model = YOLO(r"runs/detect/best.engine")
# model = YOLO(r"runs/detect/visdrone2019结果/IOU/SNAIOU/weights/best.pt")
imgsz = 640

if __name__ == '__main__':
    # ===================== 官方val测速（最终正确结果，保留）=====================
    speed_results = model.val(
        data="my_VisDrone.yaml",
        imgsz=640,
        device="0",
        batch=1,
        cache=False,
        plots=False,
        save=False,
        verbose=False
    )
    pre = speed_results.speed['preprocess']
    infer = speed_results.speed['inference']
    post = speed_results.speed['postprocess']
    total_ms = pre + infer + post
    fps = 1000 / total_ms
    
    print("="*50)
    print(f"✅ 【官方标准】真实部署FPS: {fps:.2f}")
    print(f"单帧延迟: {total_ms:.2f}ms (预处理{pre:.1f}+推理{infer:.1f}+后处理{post:.1f})")
    print("="*50)

   