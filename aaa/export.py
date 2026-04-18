import sys
sys.path.insert(0, r"C:\workspace\python\ultralytics-main")
import torch
from ultralytics import YOLO
model = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\visdrone2019e结果\IOU\CIOU\weights\best.pt")
# model.export(format="onnx",half=True) # 导出为16半精度 ONNX 模型
model.export(format="ncnn",half=True)# 导出为16半精度 NCNN 模型


