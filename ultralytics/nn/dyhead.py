import math
from ultralytics.utils.tal import dist2bbox, make_anchors
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DYReLU(nn.Module):
    def __init__(self, inp, oup, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


class Conv3x3Norm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv3x3Norm, self).__init__()

        self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(num_groups=16, num_channels=out_channels)

    def forward(self, input, **kwargs):
        x = self.conv(input.contiguous(), **kwargs)
        x = self.bn(x)
        return x



#保留跨层级融合功能。通过投影层统一通道数，然后做特征金字塔融合。

class DyConvWithProjection(nn.Module):
    """支持不同通道数的动态卷积模块，带跨层级特征融合"""

    def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
        super(DyConvWithProjection, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 主卷积分支
        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(out_channels, out_channels, 1))  # 用于上采样的上层特征
        self.DyConv.append(conv_func(in_channels, out_channels, 1))  # 当前层
        self.DyConv.append(conv_func(out_channels, out_channels, 2))  # 用于下采样的下层特征

        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(inplace=True))

        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(out_channels, out_channels)
        self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)

        # 用于跨层级特征的offset生成
        self.offset_up = nn.Conv2d(out_channels, 27, kernel_size=3, stride=1, padding=1)
        self.offset_down = nn.Conv2d(out_channels, 27, kernel_size=3, stride=1, padding=1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, x_prev=None, x_next=None):
        """
        x: 当前层特征
        x_prev: 上一层特征（更高分辨率，已投影到统一通道）
        x_next: 下一层特征（更低分辨率，已投影到统一通道）
        """
        # 处理当前层
        offset_mask = self.offset(x)
        offset = offset_mask[:, :18, :, :]
        mask = offset_mask[:, 18:, :, :].sigmoid()
        conv_args = dict(offset=offset, mask=mask)

        temp_fea = [self.DyConv[1](x, **conv_args)]

        # 处理上一层特征（如果存在）- 更高分辨率，需要下采样
        if x_prev is not None:
            offset_mask_down = self.offset_down(x_prev)
            offset_down = offset_mask_down[:, :18, :, :]
            mask_down = offset_mask_down[:, 18:, :, :].sigmoid()
            conv_args_down = dict(offset=offset_down, mask=mask_down)
            temp_fea.append(self.DyConv[2](x_prev, **conv_args_down))

        # 处理下一层特征（如果存在）- 更低分辨率，需要上采样
        if x_next is not None:
            offset_mask_up = self.offset_up(x_next)
            offset_up = offset_mask_up[:, :18, :, :]
            mask_up = offset_mask_up[:, 18:, :, :].sigmoid()
            conv_args_up = dict(offset=offset_up, mask=mask_up)
            up_fea = self.DyConv[0](x_next, **conv_args_up)
            up_fea = F.interpolate(up_fea, size=[x.size(2), x.size(3)], mode='nearest')
            temp_fea.append(up_fea)

        # 特征融合
        attn_fea = []
        res_fea = []
        for fea in temp_fea:
            res_fea.append(fea)
            attn_fea.append(self.AttnConv(fea))

        res_fea = torch.stack(res_fea)
        spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
        mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

        return self.relu(mean_fea)


class Detect_DyHead(nn.Module):
    """带跨层级融合的YOLOv8 Detect head with DyHead"""
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        """初始化检测头，nc为类别数，ch为各层通道数元组，如(64, 128, 256)"""
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 统一输出通道数，使用最大通道数
        unified_channels = max(ch)

        # 投影层：将不同通道数统一到unified_channels
        self.channel_proj = nn.ModuleList()
        for c in ch:
            if c != unified_channels:
                self.channel_proj.append(nn.Conv2d(c, unified_channels, 1, bias=False))
            else:
                self.channel_proj.append(nn.Identity())

        # 为每个层级创建DyConv，统一使用unified_channels
        self.dyhead_tower = nn.ModuleList()
        for i in range(self.nl):
            self.dyhead_tower.append(
                DyConvWithProjection(unified_channels, unified_channels, conv_func=Conv3x3Norm)
            )

        # 输出投影：将unified_channels投影回原始通道数
        self.output_proj = nn.ModuleList()
        for c in ch:
            if c != unified_channels:
                self.output_proj.append(nn.Conv2d(unified_channels, c, 1, bias=False))
            else:
                self.output_proj.append(nn.Identity())

    def forward(self, x):
        # 1. 通道投影到统一维度
        x_proj = [self.channel_proj[i](feat) for i, feat in enumerate(x)]

        # 2. 应用DyConv，带跨层级融合
        x_dy = []
        for i in range(self.nl):
            x_prev = x_proj[i - 1] if i > 0 else None
            x_next = x_proj[i + 1] if i < self.nl - 1 else None
            x_dy.append(self.dyhead_tower[i](x_proj[i], x_prev, x_next))

        # 3. 投影回原始通道数
        x = [self.output_proj[i](feat) for i, feat in enumerate(x_dy)]

        # 4. 检测头处理
        shape = x[0].shape
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """初始化检测头的偏置"""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

#方案一 不进行融合
# class DyConvSingle(nn.Module):
#     """单层级的动态卷积模块，不做跨层级特征融合"""
#
#     def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
#         super(DyConvSingle, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # 只需要一个卷积分支处理当前层
#         self.conv = conv_func(in_channels, out_channels, 1)
#
#         self.AttnConv = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(out_channels, 1, kernel_size=1),
#             nn.ReLU(inplace=True))
#
#         self.h_sigmoid = h_sigmoid()
#         self.relu = DYReLU(out_channels, out_channels)
#         self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, 0, 0.01)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         # 生成offset和mask
#         offset_mask = self.offset(x)
#         offset = offset_mask[:, :18, :, :]
#         mask = offset_mask[:, 18:, :, :].sigmoid()
#         conv_args = dict(offset=offset, mask=mask)
#
#         # 应用可变形卷积
#         out = self.conv(x, **conv_args)
#
#         # 应用注意力
#         attn = self.AttnConv(out)
#         attn = self.h_sigmoid(attn)
#         out = out * attn
#
#         # 应用动态ReLU
#         out = self.relu(out)
#
#         return out
#
#
# class Detect_DyHead(nn.Module):
#     """YOLOv8 Detect head with DyHead (简化版，每层独立处理)"""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#
#     def __init__(self, nc=80, ch=()):
#         """初始化检测头，nc为类别数，ch为各层通道数元组，如(64, 128, 256)"""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#
#         # 为每个检测层创建独立的DyConv模块，匹配各自的通道数
#         self.dyhead_tower = nn.ModuleList()
#         for i in range(self.nl):
#             channel = ch[i]
#             self.dyhead_tower.append(
#                 DyConvSingle(channel, channel, conv_func=Conv3x3Norm)
#             )
#
#     def forward(self, x):
#         # 对每个层级独立应用DyConv
#         processed_features = []
#         for i, feature in enumerate(x):
#             processed_features.append(self.dyhead_tower[i](feature))
#
#         x = processed_features
#
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         shape = x[0].shape  # BCHW
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:
#             return x
#         elif self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#
#         if self.export and self.format in ('tflite', 'edgetpu'):
#             # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
#             # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
#             # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size
#
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)




#
# class DyConv(nn.Module):
#     def __init__(self, in_channels=256, out_channels=256, conv_func=Conv3x3Norm):
#         super(DyConv, self).__init__()
#
#         self.DyConv = nn.ModuleList()
#         self.DyConv.append(conv_func(in_channels, out_channels, 1))
#         self.DyConv.append(conv_func(in_channels, out_channels, 1))
#         self.DyConv.append(conv_func(in_channels, out_channels, 2))
#
#         self.AttnConv = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, 1, kernel_size=1),
#             nn.ReLU(inplace=True))
#
#         self.h_sigmoid = h_sigmoid()
#         self.relu = DYReLU(in_channels, out_channels)
#         self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.DyConv.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, 0, 0.01)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#         for m in self.AttnConv.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, 0, 0.01)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         next_x = {}
#         feature_names = list(x.keys())
#         for level, name in enumerate(feature_names):
#
#             feature = x[name]
#
#             offset_mask = self.offset(feature)
#             offset = offset_mask[:, :18, :, :]
#             mask = offset_mask[:, 18:, :, :].sigmoid()
#             conv_args = dict(offset=offset, mask=mask)
#
#             temp_fea = [self.DyConv[1](feature, **conv_args)]
#             if level > 0:
#                 temp_fea.append(self.DyConv[2](x[feature_names[level - 1]], **conv_args))
#             if level < len(x) - 1:
#                 input = x[feature_names[level + 1]]
#                 temp_fea.append(F.interpolate(self.DyConv[0](input, **conv_args),
#                                               size=[feature.size(2), feature.size(3)]))
#             attn_fea = []
#             res_fea = []
#             for fea in temp_fea:
#                 res_fea.append(fea)
#                 attn_fea.append(self.AttnConv(fea))
#
#             res_fea = torch.stack(res_fea)
#             spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
#             mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
#             next_x[name] = self.relu(mean_fea)
#
#         return next_x




# class Detect_DyHead(nn.Module):
#     """YOLOv8 Detect head for detection models."""
#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#
#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
#         self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
#         dyhead_tower = []
#         for i in range(self.nl):
#             channel = ch[i]
#             dyhead_tower.append(
#                 DyConv(
#                     channel,
#                     channel,
#                     conv_func=Conv3x3Norm,
#                 )
#             )
#         self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))
#
#     def forward(self, x):
#         tensor_dict = {i: tensor for i, tensor in enumerate(x)}
#         x = self.dyhead_tower(tensor_dict)
#         x = list(x.values())
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         shape = x[0].shape  # BCHW
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:
#             return x
#         elif self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape
#
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
#             box = x_cat[:, :self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4:]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
#         dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
#
#         if self.export and self.format in ('tflite', 'edgetpu'):
#             # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
#             # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
#             # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
#             img_h = shape[2] * self.stride[0]
#             img_w = shape[3] * self.stride[0]
#             img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
#             dbox /= img_size
#
#         y = torch.cat((dbox, cls.sigmoid()), 1)
#         return y if self.export else (y, x)
#
#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


