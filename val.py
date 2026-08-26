import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO,RTDETR
# model = YOLO(r"runs/detect/visdrone2019结果/train/weights/best.pt")
# model = YOLO(r"runs/detect/visdrone2019结果/特征提取模块/MSALCSPv3/weights/best.pt")
# model = YOLO(r"runs/detect/visdrone2019结果/train3/weights/best.pt")
# model = YOLO(r"runs/detect/CODrone结果/base/weights/best.pt")
model = YOLO(r"runs/detect/CODrone/RSS_YOLO3/weights/last.pt")
# model = YOLO(r"runs/detect/AT_TOD结果/train5/weights/best.pt")

# model = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\visdrone2019e结果\IOU\NWF-IoU\weights\best.pt")

# model = YOLO(r"../runs/detect/visdrone2019e结果/NECK/YOLO11MSALCSP+SFPN(nearest,HWD)/weights/best.pt")
# model = RTDETR(r"C:\workspace\python\ultralytics-main\runs\detect\train18\weights\best.pt")
if __name__ == '__main__':

    train_results = model.val(
        # data="Insulator-Defect Detection.yaml",  # 数据集配置文件路径
        # data="my_VisDrone.yaml",
#         data="AI_TOD.yaml", 
        data="CODrone.yaml",
        imgsz=640,  # 训练图像尺寸
        device="0",  # 运行设备（例如 'cpu', 0, [0,1,2,3]）
        batch=8,
        cache=True,
        plots = True,
        save =True,
        #project = "C:/workspace/python/ultralytics-main/runs/detect", 
        # classes=3,
        # conf=0.001,
        # iou=0.30,  # IOU阈值
        split='test',
    )
    model.info()
