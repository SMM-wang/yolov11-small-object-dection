
import sys
import os

from ultralytics import YOLO,RTDETR
# model = YOLO(r"./runs/detect/visdrone2019结果/IOU/CIOU/weights/best.pt")
# model = YOLO(r"runs/detect/visdrone2019结果/特征提取模块/MSALCSPv3/weights/best.pt")
model = YOLO(r"runs/detect/train2/weights/best.pt")
# model = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\visdrone2019e结果\IOU\NWF-IoU\weights\best.pt")

# model = YOLO(r"../runs/detect/visdrone2019e结果/NECK/YOLO11MSALCSP+SFPN(nearest,HWD)/weights/best.pt")
# model = RTDETR(r"C:\workspace\python\ultralytics-main\runs\detect\train18\weights\best.pt")
if __name__ == '__main__':

    train_results = model.val(
        # data="Insulator-Defect Detection.yaml",  # 数据集配置文件路径
        data="my_VisDrone.yaml",
        imgsz=640,  # 训练图像尺寸
        device="0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
        batch=24,
        cache=True,
        plots = True,
        save =True,
        #project = "C:/workspace/python/ultralytics-main/runs/detect",
        # classes=3
    )
    model.info()
