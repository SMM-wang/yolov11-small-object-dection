import torch
import torch.nn as nn
import torch.nn.functional as F


from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2


class SFM(nn.Module):
    """
    Synthetic Fusion Module (SFM)
    论文来源: SFPN: SYNTHETIC FPN FOR OBJECT DETECTION [Section 3.1, Fig 3]
    """

    def __init__(self, channels):
        super(SFM, self).__init__()
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
        # 论文中 SFM 用于生成合成层 (例如 0.75x 尺寸)
        # if target_size is None:
        #     for x in inputs:
        #         if x is not None:
        #             target_size = x.shape[2:]
        #             break
        # 1. 确定目标尺寸 (Target Size)
        if target_size is None:
            # 过滤出所有非空的输入 (valid inputs)
            valid_inputs = [x for x in inputs if x is not None]

            if not valid_inputs:
                raise ValueError("All inputs are None and no target_size provided.")

            # --- 修改逻辑开始 ---
            # 找到空间尺寸最小的输入 (根据 H*W 面积判断)
            smallest_input = min(valid_inputs, key=lambda x: x.shape[2] * x.shape[3])
            min_h, min_w = smallest_input.shape[2:]

            # 计算 1.5 倍并取整
            target_size = (int(min_h * 1.5), int(min_w * 1.5))


        fused_feature = None
        valid_inputs_count = 0

        # 2. 处理每个输入 (Linear Scaling & Element-wise Addition)
        for x in inputs:
            if x is not None:
                # 线性缩放 (Linear Scaling): Upsample 或 Downsample
                # 论文提到 "linearly scaling inputs" (Fig 3 显示 linear upsample 1.5x 和 linear downsample 0.75x)
                if x.shape[2:] != target_size:
                    # 使用双线性插值进行缩放 (即论文中的 linear scaling)
                    x_resized = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
                else:
                    x_resized = x

                # 逐像素相加 (Pixel-by-pixel addition)
                if fused_feature is None:
                    fused_feature = x_resized
                else:
                    fused_feature = fused_feature + x_resized

                valid_inputs_count += 1

        if fused_feature is None:
            return None  # 或者返回全0张量，视具体实现而定

        # 3. 融合卷积 (Fusion with Conv-3x3)
        out = self.fusion_conv(fused_feature)

        return out


class SFB_5(nn.Module):
    """
    Synthetic Fusion Block (SFB)
    集成多层特征，进行一次完整的 SFPN 融合步骤。
    假设输入是三个层级: [P3, P4, P5] (小, 中, 大 尺寸)
    """

    def __init__(self, channels):
        super(SFB_5, self).__init__()
        # 为每个层级定义 SFM
        self.sfm_f3 = SFM(channels)  # 30
        self.sfm_f6 = SFM(channels)  # 60
        self.sfm_f2 = SFM(channels)  # 20
        self.sfm_f4 = SFM(channels)  # 40
        self.sfm_f8 = SFM(channels)  # 80


    def forward(self, features):
        """
        features: list of tensors [p8,p6,p4,p3,p2]
        注意: YOLO通常顺序是 P8(80x80),p6(60x60), P4(40x40),p3(30x30), P2(20x20)
        """

        f8, f6, f4, f3, f2 = features

        f3_out = self.sfm_f3([f3, f2, f4], target_size=f3.shape[2:])
        f6_out = self.sfm_f6([f6, f4, f8], target_size=f6.shape[2:])
        f2_out = self.sfm_f2([f2,f3_out, None], target_size=f2.shape[2:])
        f4_out = self.sfm_f4([f4, f3_out, f6_out], target_size=f4.shape[2:])
        f8_out = self.sfm_f8([f8, f6_out, None], target_size=f8.shape[2:])

        return [f8_out, f6_out, f4_out, f3_out, f2_out]


class SFPN_5(nn.Module):
    # 参数顺序必须与 tasks.py 中构造的 args 列表一致:
    # args = [base_channels, out_channels, unified_channel, num_sfbs]
    def __init__(self, base_channels, out_channels, unified_channel=128, num_sfbs=3):
        super().__init__()

        # # 1. 适配层
        # self.adapters = nn.ModuleList([
        #     Conv(c, unified_channel, 1) for c in base_channels
        # ])

        # 2. SFB 堆叠
        self.sfbs = nn.ModuleList([
            SFB_5(unified_channel) for _ in range(num_sfbs)
        ])

        # 3. 输出层 (恢复到 YOLO Detect 需要的通道数)

        self.out_convs = nn.ModuleList([
            Conv(unified_channel, c, 1) for c in out_channels
        ])

    def forward(self, x):
        # [f8_out, f6_out, f4_out, f3_out, f2_out]

        # # 统一通道
        # feats = [adapter(f) for adapter, f in zip(self.adapters, x)]

        # SFPN 融合
        for sfb in self.sfbs:
            feats = sfb(x)

        # 映射回输出通道
        # return feats
        feats = [feats[0], feats[2],feats[4]]
        return [conv(f) for conv, f in zip(self.out_convs, feats)]
# --- 简单测试用例 ---
if __name__ == "__main__":
    # 模拟论文 Fig 3 的场景
    # 假设 C=112 (论文中使用的通道数 )
    c = 112
    sfm = SFM(channels=c)
    sfb = SFB_5(channels=c)

    sfpn = SFPN_5(base_channels=[c, c, c], out_channels=[64, 128,256], unified_channel=c, num_sfbs=3)
    # 模拟三个不同尺度的输入
    # Input 1 (Top, Orange in Fig 3): 0.5H, 0.5W -> 需要 upsample 1.5x 到 0.75
    x1 = torch.randn(1, c, 20, 20)
    x2 = torch.randn(1, c, 30, 30)
    x3 = torch.randn(1, c, 40, 40)
    x4 = torch.randn(1, c, 60, 60)
    x5 = torch.randn(1, c, 80, 80)

    x = [x5, x4, x3, x2, x1]
    # x = [x1, x3]


    # output = sfb(x)
    output = sfpn(x)
    # output = sfm(x)

    # print(f"Output shape: {output.shape}")  # 应该为 (1, 112, 48, 48)
    # print(output[0].shape, output[1].shape, output[2].shape, output[3].shape, output[4].shape)
    print(output[0].shape, output[1].shape, output[2].shape)
    # print(output.shape)