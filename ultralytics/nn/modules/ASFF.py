import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


class ASFF(nn.Module):
    """
    ASFF (Adaptively Spatial Feature Fusion) 单个尺度融合模块
    用于融合三个不同尺度的特征到当前 level。
    """

    def __init__(self, c1, c2, level=0):
        # c1: 输入通道列表，例如 [256, 512, 1024] (对应 P3, P4, P5)
        # c2: 当前层的输出通道数 (通常等于该层原本的通道数)
        # level: 当前模块负责融合的目标层级索引 (0, 1, 或 2)
        super().__init__()
        self.level = level
        self.dim = c1  # 输入的三层通道数列表
        self.inter_dim = self.dim[self.level]  # 当前目标层的通道数

        # 压缩通道用于计算权重 (为了减少计算量，通常设为 16 或与通道数相关)
        compress_c = 8 if self.inter_dim < 16 else 16

        # 权重计算分支
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)  # 输出3个权重图

        # 特征适配层 (Resize layers)
        self.convs = nn.ModuleList()
        for i in range(len(self.dim)):
            if self.level == i:
                # 同层：如果通道一致则恒等，否则 1x1 卷积调整
                self.convs.append(
                    nn.Identity() if self.dim[i] == self.inter_dim else Conv(self.dim[i], self.inter_dim, 1, 1))
            elif self.level < i:
                # 目标层较浅(大图)，来源层较深(小图) -> 上采样
                # 例如：Level 0 (160x160) <- Level 2 (40x40)
                factor = 2 ** (i - self.level)
                self.convs.append(nn.Sequential(
                    Conv(self.dim[i], self.inter_dim, 1, 1),
                    nn.Upsample(scale_factor=factor, mode='nearest')
                ))
            elif self.level > i:
                # 目标层较深(小图)，来源层较浅(大图) -> 下采样
                # 例如：Level 2 (40x40) <- Level 0 (160x160)
                stride = 2 ** (self.level - i)
                self.convs.append(Conv(self.dim[i], self.inter_dim, 3, stride))

    def forward(self, x):
        # x 必须是一个列表，包含 [P3_feat, P4_feat, P5_feat]
        h, w = x[self.level].shape[2:]  # 目标尺寸

        # 1. 调整所有特征到统一尺寸和通道
        resized_feats = []
        for i, layer in enumerate(self.convs):
            feat = layer(x[i])
            # 强制对齐尺寸 (处理 padding 导致的细微差异)
            if feat.shape[2:] != (h, w):
                feat = F.interpolate(feat, size=(h, w), mode='nearest')
            resized_feats.append(feat)

        # 2. 计算空间权重 alpha, beta, gamma
        # 权重必需基于当前尺度的特征计算，或者基于 resize 后的特征计算
        w0 = self.weight_level_0(resized_feats[0])
        w1 = self.weight_level_1(resized_feats[1])
        w2 = self.weight_level_2(resized_feats[2])

        weights = torch.cat([w0, w1, w2], dim=1)  # [B, C*3, H, W]
        weights = self.weight_levels(weights)  # [B, 3, H, W]
        weights = F.softmax(weights, dim=1)  # 归一化权重

        # 3. 加权融合
        # expand_as 确保广播机制正常
        out = (resized_feats[0] * weights[:, 0:1, :, :]) + \
              (resized_feats[1] * weights[:, 1:2, :, :]) + \
              (resized_feats[2] * weights[:, 2:3, :, :])

        return out