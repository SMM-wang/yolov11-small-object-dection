
import sys
import os

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import torch

# 禁用代理
os.environ['NO_PROXY'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

# 支持的图片和视频文件扩展名
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')

print(torch.__version__)
print(torch.version.cuda)
def is_image_file(file_path):
    """判断是否为图片文件"""
    return file_path.lower().endswith(IMAGE_EXTENSIONS)


def is_video_file(file_path):
    """判断是否为视频文件"""
    return file_path.lower().endswith(VIDEO_EXTENSIONS)


def get_next_exp_folder(base_path):
    """获取下一个可用的exp文件夹名称"""
    base_path = Path(base_path)

    # 检查exp是否存在
    exp_path = base_path / "exp"
    if not exp_path.exists():
        return exp_path

    # 查找exp2, exp3, ...
    counter = 2
    while True:
        exp_path = base_path / f"exp{counter}"
        if not exp_path.exists():
            return exp_path
        counter += 1


# 加载模型
# model = YOLO(r"../runs/detect/visdrone2019e结果/NECK/YOLO11MSALCSP+ASFF/weights/best.pt")
# model = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\train11\weights\best.pt")
# model = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\visdrone2019e结果\IOU\NWF-IoU\weights\best.pt")
model = YOLO(r"./runs/detect/train3/weights/best.pt")
# model = YOLO(r"runs/detect/visdrone2019结果/特征提取模块/MSALCSPv3/weights/best.pt")
# 设置源路径 - 可改为图片路径、视频路径或文件夹路径
# source = "./aaa/test_picture/drone7.png"  # 图片示例
# source = r"./test_picture"
source = r"./aaa/test_picture/drone6.jpg"
# source = '../che.avi'  # 视频示例
# source = "测试图片"  # 文件夹示例
# source = "0"  # 摄像头示例

# 检查源是否存在
if not (source.isdigit() or os.path.exists(source)):
    print(f"错误: 源路径 '{source}' 不存在或无效")
    exit()

# 创建保存目录
save_dir = get_next_exp_folder("./runs/detect")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"检测结果将保存到: {save_dir}")

# 判断是否为视频流（摄像头或视频文件）
is_video = False
source_name = "frame"  # 默认名称

if source.isdigit():
    is_video = True  # 摄像头
    source_name = "camera"
elif os.path.isfile(source):
    is_video = is_video_file(source)  # 视频文件
    source_name = Path(source).stem  # 获取文件名（不含扩展名）
elif os.path.isdir(source):
    source_name = "batch"

# 打开摄像头（如果需要）
cap = None
if source.isdigit():
    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {source}")
        exit()

# 运行预测（关闭自动保存，我们手动保存）
results = model.predict(
    source=source,
    stream=is_video,  # 视频流需要流式处理
    save=False,  # 关闭自动保存
    # classes = [2]
)

# 处理预测结果
frame_count = 0
for result in results:
    # 获取原始图像
    frame = result.orig_img.copy()

    # 绘制检测框
    detected_frame = result.plot(
        line_width=1,  # 边界框线条粗细
        font_size=0.5,  # 字体大小
        labels=False,
        probs=True
    )

    # 手动保存检测结果
    if is_video:
        # 视频帧：使用帧编号命名
        save_path = save_dir / f"{source_name}_frame_{frame_count:06d}.jpg"
    else:
        # 图片：使用原文件名或批次编号
        if os.path.isfile(source):
            save_path = save_dir / f"{source_name}_detected.jpg"
        else:
            save_path = save_dir / f"{source_name}_{frame_count:04d}.jpg"

    cv2.imwrite(str(save_path), detected_frame)
    print(f"已保存: {save_path}")
    frame_count += 1

    # 显示结果
    cv2.imshow('YOLO 检测结果 (按q退出)', detected_frame)

    # 处理退出条件
    # 视频流按帧刷新（1ms延迟），图片等待按键
    wait_time = 1 if is_video else 0
    key = cv2.waitKey(wait_time)
    if key == ord('q') or key == 27:  # q键或ESC键退出
        break

# 释放资源
if cap is not None:
    cap.release()
cv2.destroyAllWindows()

print(f"\n检测完成！共处理 {frame_count} 帧/张图片")
print(f"所有结果已保存到: {save_dir}")
