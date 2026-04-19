import os
import sys

# 添加项目根目录到 sys.path
project_root = r'c:\workspace\python\yolov11-small-object-dection'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
import datetime  # 新增：导入时间戳模块

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.nms import non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolov11_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        model, ckpt = load_checkpoint(weight, device=device)
        model_names = ckpt['model'].names
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.uint8)
        self.__dict__.update(locals())
        print("模型类别列表：", model_names)

    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.7)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 1)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, tuple(int(x) for x in color), 1,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return

        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        pred = self.model(tensor)[0]
        pred = self.post_process(pred)

        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(
                pred[:, :4].cpu().detach().numpy().astype(np.int32),
                img,
                grayscale_cam
            )

        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                box = data[:4]

                # YOLOv11/v8 的 non_max_suppression 输出格式
                # 标准格式: [x1, y1, x2, y2, conf, class_id]
                # 其中 conf 是置信度，class_id 是类别索引

                if len(data) >= 6:
                    conf = float(data[4])  # 第5个位置是置信度
                    class_id = int(data[5])  # 第6个位置是类别ID
                else:
                    # 兼容其他格式
                    conf = 0.0
                    class_id = 0

                # 确保 class_id 在有效范围内
                if class_id >= len(self.model_names):
                    class_id = 0

                cam_image = self.draw_detections(
                    box,
                    self.colors[class_id],
                    f'{self.model_names[class_id]} {conf:.2f}',
                    cam_image
                )

        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_root, grad_name):
        # 核心修改：生成唯一文件夹名（方法名 + 递增数字）
        counter = 1
        new_save_dir = os.path.join(save_root, f"{grad_name}_{counter}")  # 初始文件夹路径

        # 如果文件夹已存在，则递增数字直到找到未使用的编号
        while os.path.exists(new_save_dir):
            counter += 1
            new_save_dir = os.path.join(save_root, f"{grad_name}_{counter}")

        # 创建新文件夹（自动处理不存在的情况）
        os.makedirs(new_save_dir, exist_ok=True)
        print(f"本次结果将保存至：{new_save_dir}")

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                if not os.path.isfile(os.path.join(img_path, img_path_)):  # 跳过子文件夹
                    continue
                name = img_path_.rsplit('.')[0]
                end_name = img_path_.rsplit('.')[-1]
                self.process(
                    os.path.join(img_path, img_path_),  # 原图片路径（绝对路径更安全）
                    os.path.join(new_save_dir, f"{name}_{grad_name}.{end_name}")  # 新保存路径
                )
        else:
            name = os.path.basename(img_path).rsplit('.')[0]
            self.process(
                img_path,
                os.path.join(new_save_dir, f"{name}_{grad_name}.png")
            )


    # def __call__(self, img_path, save_path, grad_name):
    #     # remove dir if exist
    #     # if os.path.exists(save_path):
    #     #     shutil.rmtree(save_path)
    #     # make dir if not exist
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path, exist_ok=True)
    #
    #     if os.path.isdir(img_path):
    #         for img_path_ in os.listdir(img_path):
    #             name = img_path_.rsplit('.')[0]
    #             end_name = img_path_.rsplit('.')[-1]
    #             self.process(f'{img_path}/{img_path_}', f'{save_path}/{name}_{grad_name}.{end_name}')
    #     else:
    #         self.process(img_path, f'{save_path}/result_{grad_name}.png')


def get_params():
    # 绘制热力图方法列表
    grad_list = [
        # 'GradCAM',
        # 'GradCAMPlusPlus',
        # 'XGradCAM',
        'EigenCAM',
        # 'HiResCAM',
        # 'LayerCAM',
        # 'RandomCAM',
        # 'EigenGradCAM'
    ]
    # 推荐的层（基于你的模型结构）
    layers = [16, 19, 22]
    # layers = [16]
    # layers = [ 12, 13, 14]
    for grad_name in grad_list:
        params = {
            # 'weight': '../runs/detect/visdrone2019结果/YOLO11n/weights/best.pt',  # 训练好的权重路径
            'weight': 'runs/detect/visdrone2019结果/特征提取模块/SACSP/weights/best.pt',
            'device': 'cuda:0',  # cpu或者cuda:0
            'method': grad_name,
            'layer': layers,  # 计算梯度的层, 指定层的索引
            'backward_type': 'class',  # class, box, all
            'conf_threshold': 0.45,  # 置信度阈值默认0.45 nms
            'ratio': 0.2,  # 建议0.02-0.1，取前多少数据
            'show_box':False,  # 是否显示检测框
            'renormalize': False  # 是否优化热力图显示结果
        }
        yield params


if __name__ == '__main__':
    for each in get_params():
        model = yolov11_heatmap(**each)
        # model参数：1.图片路径 2.保存根目录 3.方法名
        model(r'aaa/test_picture/0000308_02201_d_0000316.jpg', 'runs/detect/梯度热力图/YOLO11sacsp', each['method'])
