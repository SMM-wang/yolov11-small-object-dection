import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO
import torch

import gc
import os

# 设置PyTorch CUDA内存分配器配置
if torch.cuda.is_available():
    # 减少内存碎片化
    torch.backends.cudnn.benchmark = True
    # 设置环境变量以优化内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def clear_cuda_memory(trainer_or_validator):
    """
    通用清理函数。
    Ultralytics 会将 trainer 或 validator 实例作为参数传入，
    虽然我们这里没用到该实例，但必须保留这个参数位。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()



def clear_cuda_memory_val_batch(validator):
    if not hasattr(validator, 'custom_counter'):
        validator.custom_counter = 0

    validator.custom_counter += 1

    # 每 50 次触发一次
    if validator.custom_counter % 1 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

# 在函数外部定义一个简单的计数器（或者利用 Python 的函数属性）
def clear_cuda_memory_batch(trainer):
    # 初始化计数器 (如果不存在)
    if not hasattr(trainer, 'custom_counter'):
        trainer.custom_counter = 0

    trainer.custom_counter += 1

    # 每 50 次触发一次
    if trainer.custom_counter % 1 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()








from ultralytics import settings
settings["tensorboard"]=True


if __name__ == '__main__':
    # visdrone2019
    model1 = YOLO("ultralytics/cfg/models/11/yolo11SACSP+SFPN.yaml")
    # model1 = YOLO("ultralytics/cfg/models/11/yolo11MSAL.yaml")   
    # model1 = YOLO("ultralytics/cfg/models/11/yolo11SFPN.yaml")   
    # model1 = YOLO(r"C:\workspace\python\ultralytics-main\runs\detect\train24\weights\best.pt")
    # model1.add_callback("on_train_batch_end", clear_cuda_memory_batch)
    # model1.add_callback("on_train_epoch_end", clear_cuda_memory)
    # model1.add_callback("on_val_batch_end", clear_cuda_memory_val_batch)

    model1.train(
        data="my_VisDrone.yaml",
        # data="AI_TODe.yaml", 
        epochs=1,
        batch=8,
        # batch=8,
        imgsz=640,
        device=0,
        workers=2,
        amp=True,
        # patience=25,  #早停
        cos_lr=True, # 使用余弦退火调度器
        # optimizer="AdamW",  # 使用AdamW优化器
        optimizer="SGD",
        # lr0=0.007,  # 设置初始学习率为0.007
        lr0=0.01,
        lrf=0.01,
        box_iou="SNAIoU",
        warmup_epochs=3,
        cache = True,
        #project="C:/workspace/python/ultralytics-main/runs/detect",
        # resume=True,
    )
   

    # model3 = YOLO("ultralytics/cfg/models/11/yolo11MSAL+SFPN.yaml")
    # model3.add_callback("on_train_epoch_end", clear_cuda_memory)
    # model3.add_callback("on_train_batch_end", clear_cuda_memory_batch)
    # model3.add_callback("on_val_batch_end", clear_cuda_memory_val_batch)
    # model3.train(
    #     # data="my_VisDrone_e.yaml",
    #     data="AI_TOD.yaml",
    #     epochs=400,
    #     batch=4,
    #     imgsz=640,
    #     device=0,
    #     workers=0,
    #     amp=True,
    #     patience=10,  #早停
    #     cos_lr=True, # 使用余弦退火调度器
    #     # optimizer="AdamW",  # 使用AdamW优化器
    #     optimizer="SGD",
    #     # lr0=0.007,  # 设置初始学习率为0.007
    #     # lr0=0.001,
    #     lrf=0.01,
    #     warmup_epochs=3,
    #     cache = True,
    #     project="C:/workspace/python/ultralytics-main/runs/detect",
    #     # resume=True,
    # )



    # model = YOLO("ultralytics/cfg/models/11/yolo11MSAL.yaml")
    # model.add_callback("on_train_epoch_end", clear_cuda_memory)
    # model.add_callback("on_train_batch_end", clear_cuda_memory_batch)
    # model.train(
    #     data="my_VisDrone.yaml",
    #     # data="AI_TOD.yaml",
    #     epochs=400,
    #     batch=8,
    #     imgsz=640,
    #     device=0,
    #     workers=0,
    #     amp=True,
    #     patience=10,  #早停
    #     cos_lr=True, # 使用余弦退火调度器
    #     optimizer="AdamW",  # 使用AdamW优化器
    #     # optimizer="SGD",
    #     lr0=0.007,  # 设置初始学习率为0.007
    #     # lr0=0.001,
    #     # lrf=0.01,
    #     warmup_epochs=3,
    #     cache = True,
    #     project=r"../runs/detect",
    #     # resume=True,
    # )

# Insulator-Defect Detection

    # model1 = YOLO("ultralytics/cfg/models/11/yolo11MSAL+CFPT.yaml")
    # # 添加自定义回调函数
    # model1.add_callback("on_train_epoch_end", clear_cuda_memory)
    # # model1.add_callback("on_train_batch_end", clear_cuda_memory_batch)
    #
    # model1.train(
    #     data="Insulator-Defect Detection.yaml",
    #     epochs=300,
    #     batch=8,
    #     imgsz=640,
    #     device=0,
    #     workers=0,
    #     amp=True,
    #     patience=20,  # 早停
    #     cos_lr=True,  # 使用余弦退火调度器
    #     optimizer="SGD",  # 使用AdamW优化器
    #     lr0=0.01,
    #     warmup_epochs=5,
    #     cache=True,
    #     project=r"../runs/detect",
    # )

    #yolo task=detect mode=train model=yolov10n.pt data=ultralytics/cfg/models/v10/yolov10n.yaml batch=8 epochs=50 imgsz=640 device=0 workers=0 data=VOC-test.yaml amp=True



