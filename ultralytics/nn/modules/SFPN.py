import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import HWD


from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2


class SFM(nn.Module):
    """
    Synthetic Fusion Module (SFM)
    论文来源: SFPN: SYNTHETIC FPN FOR OBJECT DETECTION [Section 3.1, Fig 3]
    """

    def __init__(self, channels):
        super(SFM, self).__init__()
        # 定义 HWD 下采样模块 (用于 2x 下采样)
        self.hwd = HWD(channels, channels)
        # 论文提到在neck之前将通道固定为112 (或其他值) 以方便融合
        # 这里的 conv-3x3 用于融合相加后的特征 [cite: 88, 72]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,groups= channels),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(),

        )
        # self.fusion_conv = C3k2(channels, channels,  shortcut=False)

    def forward(self, inputs, target_size=None):
        """
        参数:
            inputs (list of torch.Tensor): 包含三个可选输入的列表或元组。
                                           对应论文 Fig 3 中的三种输入来源 (Blue, Orange, Gray)。
                                           输入可以是 None，也可以是不同尺寸的特征图。
            target_size (tuple): 目标特征图尺寸 (H, W)。如果为 None，则默认对齐到 inputs 中第一个非空张量的尺寸。

        返回:
            out: 融合后的合成特征图 (Synthetic Feature Map)
        """

        # 1. 确定目标尺寸 (Target Size)
        if target_size is None:
            # # 过滤出所有非空的输入 (valid inputs)
            # valid_inputs = [x for x in inputs if x is not None]
            #
            # if not valid_inputs:
            #     raise ValueError("All inputs are None and no target_size provided.")
            #
            # # --- 修改逻辑开始 ---
            # # 找到空间尺寸最小的输入 (根据 H*W 面积判断)
            # smallest_input = min(valid_inputs, key=lambda x: x.shape[2] * x.shape[3])
            # min_h, min_w = smallest_input.shape[2:]
            #
            # # 计算 1.5 倍并取整
            # target_size = (int(min_h * 1.5), int(min_w * 1.5))
            target_size = inputs[0].shape[2:]

        fused_feature = None
        valid_inputs_count = 0

        # 2. 处理每个输入 (Linear Scaling & Element-wise Addition)
        # for x in inputs:
        #     if x is not None:
        #         # 线性缩放 (Linear Scaling): Upsample 或 Downsample
        #         # 论文提到 "linearly scaling inputs" (Fig 3 显示 linear upsample 1.5x 和 linear downsample 0.75x)
        #         if x.shape[2:] != target_size:
        #             # 使用双线性插值进行缩放 (即论文中的 linear scaling)
        #             x_resized = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        #         else:
        #             x_resized = x
        #
        #         # 逐像素相加 (Pixel-by-pixel addition)
        #         if fused_feature is None:
        #             fused_feature = x_resized
        #         else:
        #             fused_feature = fused_feature + x_resized
        #
        #         valid_inputs_count += 1
        for x in inputs:
            if x is not None:
                in_h, in_w = x.shape[2], x.shape[3]
                out_h, out_w = target_size

                # --- 核心修改逻辑 ---

                # 情况 A: 尺寸一致 -> 直接使用
                if (in_h, in_w) == (out_h, out_w):
                    x_resized = x

                # 情况 B: 输入比目标大 -> 下采样 (Downsample)
                # 优先使用 HWD (保留小目标细节)
                elif in_h > out_h:
                    # HWD 只能处理严格的 2 倍下采样
                    if in_h == out_h * 2 and in_w == out_w * 2:
                        x_resized = self.hwd(x)
                    else:
                        # 如果是 4倍 (如 P3->P5) 或其他比例，回退到 MaxPool
                        # MaxPool 比 Bilinear 更好，因为不会模糊掉小目标的高亮像素
                        x_resized = F.adaptive_max_pool2d(x, output_size=target_size)

                # 情况 C: 输入比目标小 -> 上采样 (Upsample)
                # 使用 Bilinear (平滑过渡)
                else:
                    # x_resized = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                    x_resized = F.interpolate(x, size=target_size, mode='nearest')

                # --- 累加融合 ---
                if fused_feature is None:
                    fused_feature = x_resized
                else:
                    fused_feature = fused_feature + x_resized

        if fused_feature is None:
            return None  # 或者返回全0张量，视具体实现而定

        # 3. 融合卷积 (Fusion with Conv-3x3)
        out = self.fusion_conv(fused_feature)

        return out


class SFB(nn.Module):
    """
    Synthetic Fusion Block (SFB)
    集成多层特征，进行一次完整的 SFPN 融合步骤。
    假设输入是三个层级: [P3, P4, P5] (小, 中, 大 尺寸)
    """

    def __init__(self, channels):
        super(SFB, self).__init__()
        # 为每个层级定义 SFM
        # 这里简化为 3 层结构 (P3, P4, P5)，对应论文的 SFPN-3
        # 如果要做 SFPN-5/9，需要在这里增加更多的 SFM 节点
        self.sfm_p3 = SFM(channels)  # 处理 P3 (大尺寸特征图)
        self.sfm_p4 = SFM(channels)  # 处理 P4 (中尺寸特征图)
        self.sfm_p5 = SFM(channels)  # 处理 P5 (小尺寸特征图)

    def forward(self, features):
        """
        features: list of tensors [p3, p4, p5]
        注意: YOLO通常顺序是 P3(80x80), P4(40x40), P5(20x20)
        """
        p3, p4, p5 = features


        p4_out = self.sfm_p4([p3, p4, p5], target_size=p4.shape[2:])

        p5_out = self.sfm_p5([p5, p4_out, None], target_size=p5.shape[2:])

        p3_out = self.sfm_p3([p3, p4_out, None], target_size=p3.shape[2:])

        return [p3_out, p4_out, p5_out]


class SFPN(nn.Module):
    # 参数顺序必须与 tasks.py 中构造的 args 列表一致:
    # args = [base_channels, out_channels, unified_channel, num_sfbs]
    def __init__(self, base_channels, out_channels, unified_channel=128, num_sfbs=3):
        super().__init__()

        # 1. 适配层
        self.adapters = nn.ModuleList([
            # Conv(c, unified_channel, 1) for c in base_channels
            Conv(c, unified_channel, 1) if c != unified_channel else nn.Identity()
            for c in base_channels
        ])

        # 2. SFB 堆叠
        self.sfbs = nn.ModuleList([
            SFB(unified_channel) for _ in range(num_sfbs)
        ])

        # 3. 输出层 (恢复到 YOLO Detect 需要的通道数)
        # SFPN 内部处理都是 unified_channel (256)，但 Detect 头可能期待 [256, 512, 1024]
        self.out_convs = nn.ModuleList([
            Conv(unified_channel, c, 1) for c in out_channels
        ])

    def forward(self, x):
        # x: [P3, P4, P5] from backbone

        # 统一通道
        feats = [adapter(f) for adapter, f in zip(self.adapters, x)]

        # SFPN 融合
        for sfb in self.sfbs:
            feats = sfb(feats)

        # 映射回输出通道
        # return feats
        return [conv(f) for conv, f in zip(self.out_convs, feats)]