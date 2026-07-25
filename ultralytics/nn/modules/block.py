# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",


    "MSALCSP",
    "SACSP",
    "My_Index",

    "RFD",
    "RFD_LITE",


)


class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1: int, cm: int, c2: int):
        """Initialize the StemBlock of PPHGNetV2.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize HGBlock with specified parameters.

        Args:
            c1 (int): Input channels.
            cm (int): Middle channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of LightConv or Conv blocks.
            lightconv (bool): Whether to use LightConv.
            shortcut (bool): Whether to use shortcut connection.
            act (nn.Module): Activation function.
        """
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        """Initialize the SPPF layer with given input/output channels and kernel size.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            n (int): Number of pooling iterations.
            shortcut (bool): Whether to use shortcut connection.

        Notes:
            This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        """Initialize the CSP Bottleneck with 1 convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of convolutions.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and residual connection to input tensor."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize a CSP Bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
#         """Initialize a CSP bottleneck with 2 convolutions.

#         Args:
#             c1 (int): Input channels.
#             c2 (int): Output channels.
#             n (int): Number of Bottleneck blocks.
#             shortcut (bool): Whether to use shortcut connections.
#             g (int): Groups for convolutions.
#             e (float): Expansion ratio.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward pass using split() instead of chunk()."""
#         y = self.cv1(x).split((self.c, self.c), 1)
#         y = [y[0], y[1]]
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions. Modified for Pruning."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        
        # --- 核心修改 1：物理拆分大卷积为两个独立卷积 ---
        # 原版: self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv1_left = Conv(c1, self.c, 1, 1)
        self.cv1_right = Conv(c1, self.c, 1, 1)
        # -----------------------------------------------
        
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        # --- 核心修改 2：分别独立计算，彻底消除 chunk 或 split ---
        y_left = self.cv1_left(x)
        y_right = self.cv1_right(x)
        
        y = [y_left, y_right]
        # -------------------------------------------------------
        
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        # 由于已经物理拆分，这个方法其实不再需要，但为了兼容性可以保留修改后的逻辑
        y_left = self.cv1_left(x)
        y_right = self.cv1_right(x)
        y = [y_left, y_right]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with cross-convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        """Initialize CSP Bottleneck with a single convolution.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepConv blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of RepC3 module."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with TransformerBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Transformer blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize C3 module with GhostBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Ghost bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/Efficient-AI-Backbones."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        """Initialize Ghost Bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize CSP Bottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1: int, c2: int, s: int = 1, e: int = 4):
        """Initialize ResNet block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            e (int): Expansion ratio.
        """
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1: int, c2: int, s: int = 1, is_first: bool = False, n: int = 1, e: int = 4):
        """Initialize ResNet layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            s (int): Stride.
            is_first (bool): Whether this is the first layer.
            n (int): Number of ResNet blocks.
            e (int): Expansion ratio.
        """
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1: int, c2: int, nh: int = 1, ec: int = 128, gc: int = 512, scale: bool = False):
        """Initialize MaxSigmoidAttnBlock.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            nh (int): Number of heads.
            ec (int): Embedding channels.
            gc (int): Guide channels.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaxSigmoidAttnBlock.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor.

        Returns:
            (torch.Tensor): Output tensor after attention.
        """
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, guide.shape[1], self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        ec: int = 128,
        nh: int = 1,
        gc: int = 512,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        """Initialize C2f module with attention mechanism.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            ec (int): Embedding channels for attention.
            nh (int): Number of heads for attention.
            gc (int): Guide channels for attention.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer with attention.

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk().

        Args:
            x (torch.Tensor): Input tensor.
            guide (torch.Tensor): Guide tensor for attention.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(
        self, ec: int = 256, ch: tuple[int, ...] = (), ct: int = 512, nh: int = 8, k: int = 3, scale: bool = False
    ):
        """Initialize ImagePoolingAttn module.

        Args:
            ec (int): Embedding channels.
            ch (tuple): Channel dimensions for feature maps.
            ct (int): Channel dimension for text embeddings.
            nh (int): Number of attention heads.
            k (int): Kernel size for pooling.
            scale (bool): Whether to use learnable scale parameter.
        """
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> torch.Tensor:
        """Forward pass of ImagePoolingAttn.

        Args:
            x (list[torch.Tensor]): List of input feature maps.
            text (torch.Tensor): Text embeddings.

        Returns:
            (torch.Tensor): Enhanced text embeddings.
        """
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """Batch Norm Contrastive Head using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize BNContrastiveHead.

        Args:
            embed_dims (int): Embedding dimensions for features.
        """
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """Fuse the batch normalization layer in the BNContrastiveHead module."""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Passes input out unchanged."""
        return x

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Forward function of contrastive learning with batch normalization.

        Args:
            x (torch.Tensor): Image features.
            w (torch.Tensor): Text features.

        Returns:
            (torch.Tensor): Similarity scores.
        """
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize RepBottleneck.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepCSP layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int, n: int = 1):
        """Initialize CSP-ELAN layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for RepCSP.
            n (int): Number of RepCSP blocks.
        """
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1: int, c2: int, c3: int, c4: int):
        """Initialize ELAN1 layer.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            c4 (int): Intermediate channels for convolutions.
        """
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1: int, c2: int):
        """Initialize AConv module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1: int, c2: int):
        """Initialize ADown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
        """
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1: int, c2: int, c3: int, k: int = 5):
        """Initialize SPP-ELAN block.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            c3 (int): Intermediate channels.
            k (int): Kernel size for max pooling.
        """
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1: int, c2s: list[int], k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize CBLinear module.

        Args:
            c1 (int): Input channels.
            c2s (list[int]): List of output channel sizes.
            k (int): Kernel size.
            s (int): Stride.
            p (int | None): Padding.
            g (int): Groups.
        """
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx: list[int]):
        """Initialize CBFuse module.

        Args:
            idx (list[int]): Indices for feature selection.
        """
        super().__init__()
        self.idx = idx

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through CBFuse layer.

        Args:
            xs (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Fused output tensor.
        """
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize CSP bottleneck layer with two convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C3f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed: int) -> None:
        """Initialize RepVGGDW module.

        Args:
            ed (int): Input and output channels.
        """
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """Fuse the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        if not hasattr(self, "conv1"):
            return  # already fused
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """Compact Inverted Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, e: float = 0.5, lk: bool = False):
        """Initialize the CIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            e (float): Expansion ratio.
            lk (bool): Whether to use RepVGGDW.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use large kernel. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(
        self, c1: int, c2: int, n: int = 1, shortcut: bool = False, lk: bool = False, g: int = 1, e: float = 0.5
    ):
        """Initialize C2fCIB module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of CIB modules.
            shortcut (bool): Whether to use shortcut connection.
            lk (bool): Whether to use large kernel.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """Initialize multi-head attention module.

        Args:
            dim (int): Input dimension.
            num_heads (int): Number of attention heads.
            attn_ratio (float): Attention ratio for key dimension.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Attention ratio for key dimension.
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        """Initialize PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass in PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


# class C2PSA(nn.Module):
#     """C2PSA module with attention mechanism for enhanced feature extraction and processing.

#     This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
#     capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

#     Attributes:
#         c (int): Number of hidden channels.
#         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
#         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
#         m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

#     Methods:
#         forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

#     Examples:
#         >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
#         >>> input_tensor = torch.randn(1, 256, 64, 64)
#         >>> output_tensor = c2psa(input_tensor)

#     Notes:
#         This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
#     """

#     def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
#         """Initialize C2PSA module.

#         Args:
#             c1 (int): Input channels.
#             c2 (int): Output channels.
#             n (int): Number of PSABlock modules.
#             e (float): Expansion ratio.
#         """
#         super().__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)

#         self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Process the input tensor through a series of PSA blocks.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor after processing.
#         """
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
#         b = self.m(b)
#         return self.cv2(torch.cat((a, b), 1))

class C2PSA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing. Modified for Pruning."""

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initialize the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        
        # --- 核心修改：物理拆分 cv1 为左右两个独立卷积 ---
        # 原版: self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv1_left = Conv(c1, self.c, 1, 1)
        self.cv1_right = Conv(c1, self.c, 1, 1)
        # ------------------------------------------------
        
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        # --- 核心修改：独立计算，彻底消除 split ---
        a = self.cv1_left(x)
        b = self.cv1_right(x)
        # ------------------------------------------
        
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature
    extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2fPSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1: int, c2: int, k: int, s: int):
        """Initialize SCDown module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (int): Kernel size.
            s (int): Stride.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution and downsampling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Downsampled output tensor.
        """
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and
    customize the model by truncating or unwrapping layers.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): Unwraps the model to a sequential containing all but the last `truncate` layers.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.
    """

    def __init__(
        self, model: str, weights: str = "DEFAULT", unwrap: bool = True, truncate: int = 2, split: bool = False
    ):
        """Load the model and weights from torchvision.

        Args:
            model (str): Name of the torchvision model to load.
            weights (str): Pre-trained weights to load.
            unwrap (bool): Whether to unwrap the model.
            truncate (int): Number of layers to truncate.
            split (bool): Whether to split the output.
        """
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor | list[torch.Tensor]): Output tensor or list of tensors.
        """
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y


class AAttn(nn.Module):
    """Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, area: int = 1):
        """Initialize an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through the area-attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention.
        """
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 1.2, area: int = 1):
        """Initialize an Area-attention block module.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights using a truncated normal distribution.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after area-attention and feed-forward processing.
        """
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        a2: bool = True,
        area: int = 1,
        residual: bool = False,
        mlp_ratio: float = 2.0,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through A2C2f layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network for transformer-based architectures."""

    def __init__(self, gc: int, ec: int, e: int = 4) -> None:
        """Initialize SwiGLU FFN with input dimension, output dimension, and expansion factor.

        Args:
            gc (int): Guide channels.
            ec (int): Embedding channels.
            e (int): Expansion factor.
        """
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU transformation to input features."""
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    """Residual connection wrapper for neural network modules."""

    def __init__(self, m: nn.Module) -> None:
        """Initialize residual module with the wrapped module.

        Args:
            m (nn.Module): Module to wrap with residual connection.
        """
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual connection to input features."""
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding module for feature enhancement."""

    def __init__(self, ch: list[int], c3: int, embed: int):
        """Initialize SAVPE module with channels, intermediate channels, and embedding dimension.

        Args:
            ch (list[int]): List of input channel dimensions.
            c3 (int): Intermediate channels.
            embed (int): Embedding dimension.
        """
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity()
            )
            for i, x in enumerate(ch)
        )

        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in {1, 2} else nn.Identity())
            for i, x in enumerate(ch)
        )

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x: list[torch.Tensor], vp: torch.Tensor) -> torch.Tensor:
        """Process input features and visual prompts to generate enhanced embeddings."""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class Proto26(Proto):
    """Ultralytics YOLO26 models mask Proto module for segmentation models."""

    def __init__(self, ch: tuple = (), c_: int = 256, c2: int = 32, nc: int = 80):
        """Initialize the Ultralytics YOLO models mask Proto module with specified number of protos and masks.

        Args:
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
            nc (int): Number of classes for semantic segmentation.
        """
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: torch.Tensor, return_semseg: bool = True) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = f(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = super().forward(self.feat_fuse(feat))
        if self.training and return_semseg:
            semseg = self.semseg(feat)
            return (p, semseg)
        return p

    def fuse(self):
        """Fuse the model for inference by removing the semantic segmentation head."""
        self.semseg = None


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model.

    References:
        https://arxiv.org/abs/1605.08803
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py
    """

    @staticmethod
    def nets():
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))

    @property
    def prior(self):
        """The prior distribution."""
        return torch.distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self):
        super().__init__()

        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))

        self.s = torch.nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = torch.nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate the log determinant of the Jacobian
        matrix.
        """
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det



class ShapeRouter(nn.Module):
    """
    形状路由器 (Shape Router)
    作用：根据输入特征像素点的局部上下文，动态预测该位置目标的形状偏置（横向 vs 纵向）。
    """
    def __init__(self, channels):
        super().__init__()
        # 极轻量的设计：通过 1x1 卷积判断每个像素点应该偏向横向还是纵向
        # 输出 2 个通道，分别代表 水平权重(H) 和 垂直权重(V)
        self.router = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 2, kernel_size=1)
        )

    def forward(self, x):
        # 预测形状 logits
        shape_logits = self.router(x) # (B, 2, H, W)
        # 使用 Softmax 确保横纵权重之和为 1 (Soft Routing)
        shape_weights = F.softmax(shape_logits, dim=1) 
        
        # 分离出水平(横向)权重和垂直(纵向)权重
        weight_h, weight_v = shape_weights.chunk(2, dim=1) # 分别为 (B, 1, H, W)
        return weight_h, weight_v


class SACSP(nn.Module):
    """
    形状自适应跨阶段局部网络 (Shape-Adaptive CSP, SA-CSP)
    核心亮点：利用非对称卷积构建横纵专家，并通过 ShapeRouter 实现像素级的形状自适应特征提取。
    """
    def __init__(self, in_channels, out_channels, expansion=0.5):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        # 确保通道数整除，避免拼接错位
        mid_channels = mid_channels if mid_channels % 2 == 0 else mid_channels + 1
        
        # CSP 通道切分
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 2 * mid_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(2 * mid_channels),
        #     nn.SiLU()
        # )
        self.conv1 = Conv(in_channels, 2 * mid_channels, k=1, s=1)
        
        # 1. 局部基础分支 (用于提供纯净的 Query)
        # self.local_branch = nn.Sequential(
        #     nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.SiLU(),
        # )
        self.local_branch = Conv(mid_channels, mid_channels, k=3, s=1, p=1)

        scale_ch = mid_channels // 2
        
        # 2. 形状自适应多尺度分支
        # 兜底极小目标分支 (无方向性)
        # self.scale1 = nn.Sequential(
        #     nn.Conv2d(mid_channels, scale_ch, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(scale_ch),
        #     nn.SiLU()
        # )
        self.scale1 = Conv(mid_channels, scale_ch, k=1, s=1)
        
        # --- 核心：形状自适应特征提取算子 ---
        # 实例化形状路由器
        self.shape_router = ShapeRouter(mid_channels)
        
        # 尺度 2：中等小目标形状提取
        self.shape_h2 = nn.Conv2d(mid_channels, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)) # 横向感知
        self.shape_v2 = nn.Conv2d(mid_channels, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0)) # 纵向感知
        self.shape_bn_act2 = nn.Sequential(
            nn.BatchNorm2d(scale_ch),
            nn.SiLU()
        )

        # 尺度 3：偏大小目标形状提取 (堆叠扩大感受野)
        self.shape_h3 = nn.Sequential(
            nn.Conv2d(mid_channels, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.SiLU(),
            nn.Conv2d(scale_ch, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1))
        )
        self.shape_v3 = nn.Sequential(
            nn.Conv2d(mid_channels, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.SiLU(),
            nn.Conv2d(scale_ch, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.shape_bn_act3 = nn.Sequential(
            nn.BatchNorm2d(scale_ch),
            nn.SiLU()
        )

        # 3. 交叉注意力特征提纯 (过滤残留背景)
        # 注意：这里需要你保留之前定义的 CrossAttention 类
        self.cross_att = GuidedCrossAttention(
            in_channels=mid_channels, 
            interact_in_channels=scale_ch * 3, 
            reduction=16 
        )

        # 4. 尾部融合输出
        # self.final_conv = nn.Conv2d(2 * mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        # self.final_bn = nn.BatchNorm2d(out_channels)
        # self.final_act = nn.SiLU()
        self.final_conv = Conv(2 * mid_channels, out_channels, k=1, s=1)

    def forward(self, x):
        # CSP 切分
        x1, x2 = self.conv1(x).chunk(2, 1)

        # 提取局部特征 (Query)
        local_feat = self.local_branch(x2)
        
        # 提取极小尺度特征
        s1 = self.scale1(x2)
        
        # --- 形状自适应计算 (Shape-Adaptive Forward) ---
        # 1. 路由器生成像素级形状权重
        w_h, w_v = self.shape_router(x2)
        
        # 保存第一组结果到 txt
        # with open('w_h_w_v_output.txt', 'a') as f:
        #     f.write(f"w_h: {w_h.cpu().detach().numpy().tolist()}\n")
        #     f.write(f"w_v: {w_v.cpu().detach().numpy().tolist()}\n\n")
        
        # 2. 动态自适应融合：用权重动态调节横向和纵向卷积的输出
        s2 = self.shape_bn_act2(w_h * self.shape_h2(x2) + w_v * self.shape_v2(x2))
        s3 = self.shape_bn_act3(w_h * self.shape_h3(x2) + w_v * self.shape_v3(x2))
        
        # 拼接形状自适应特征
        shape_adaptive_feat = torch.cat([s1, s2, s3], dim=1)

        # 交叉注意力提纯
        refined_feat = self.cross_att(local_feat, shape_adaptive_feat)

        # CSP 残差拼接与输出
        out = torch.cat([x1, refined_feat], dim=1)
        # out = self.final_act(self.final_bn(self.final_conv(out)))
        out = self.final_conv(out)

        return out



class GuidedCrossAttention(nn.Module):
    """
    真正的特征引导型交叉注意力 (Guided Cross-Attention)
    逻辑：利用纯净的局部特征(local_feat)生成掩码，去提纯多尺度特征(multi_scale_feat)
    """
    def __init__(self, in_channels, interact_in_channels, reduction=16):
        super().__init__()
        # in_channels: local_feat 的通道数
        # interact_in_channels: multi_scale_feat 的通道数 (拼接后的总通道数)

        # 1. 通道对齐：将多尺度特征映射到标准通道数
        # self.v_proj = nn.Sequential(
        #     nn.Conv2d(interact_in_channels, in_channels, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.SiLU()
        # )
        self.v_proj = Conv(interact_in_channels, in_channels, k=1, s=1)

        # 2. 空间交叉引导 (Spatial Cross-Guidance)
        # 核心逻辑：用纯净的局部特征来计算空间注意力权重
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 3. 通道交叉融合 (Channel Cross-Fusion)
        # 结合提取出的纯净特征，进行通道特征对齐
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, local_feat, multi_scale_feat):
        """
        local_feat: 感受野小，特征纯净，作为 Query
        multi_scale_feat: 感受野大，尺度丰富但含噪声，作为 Value
        """
        # 1. 映射 Value
        v = self.v_proj(multi_scale_feat)

        # =========================================================
        # 2. 真正的交叉空间过滤 (Cross Spatial Attention)
        # 用 local_feat 求 max 和 avg，生成空间掩码 (Mask)
        # =========================================================
        max_pool = torch.max(local_feat, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(local_feat, dim=1, keepdim=True)
        # s_mask 的高亮区域就是小目标所在的精准位置
        s_mask = self.spatial_gate(torch.cat([max_pool, avg_pool], dim=1))
        
        # 将局部特征生成的 Mask 作用于多尺度特征，过滤掉其周边的背景噪声
        v_spatially_filtered = v * s_mask

        # =========================================================
        # 3. 交叉通道融合 (Cross Channel Attention)
        # =========================================================
        # 将纯净特征和过滤后的多尺度特征拼接，评估通道重要性
        concat_feat = torch.cat([local_feat, v_spatially_filtered], dim=1)
        c_mask = self.channel_gate(concat_feat)
        
        # 在通道维度重新校准特征
        refined_feat = v_spatially_filtered * c_mask

        # =========================================================
        # 4. 残差输出
        # 保留底层的纯净特征，并加上提纯后的多尺度特征
        # =========================================================
        out = local_feat + refined_feat

        return out

# class MSALCSP(nn.Module):
#     """
#     基于数据驱动的 MSALCSPv3:
#     逻辑链：CSP切分 -> 非对称十字多尺度提取(匹配82.5%的非方形小目标) -> 交叉注意力(抑制背景) -> 拼接融合
#     """
#     def __init__(self, in_channels, out_channels, expansion=0.5):
#         super().__init__()
#         mid_channels = int(out_channels * expansion)
        
#         # CSP第一步：特征通道切分 (确保整除)
#         mid_channels = mid_channels if mid_channels % 2 == 0 else mid_channels + 1
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2 * mid_channels),
#             nn.SiLU()
#         )
        
#         # 1. 基础局部特征分支 (Query) - 极小感受野，提取核心像素点
#         self.local_branch = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU(),
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()
#         )

#         # 2. 非对称十字多尺度分支 (Cross-Strip Multi-Scale)
#         # 完美匹配你统计的 66.28% 纵向 和 16.24% 横向 目标
#         scale_ch = mid_channels // 2
        
#         # 尺度1：极小目标 (1x1，兜底)
#         self.scale1 = nn.Sequential(
#             nn.Conv2d(mid_channels, scale_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(scale_ch),
#             nn.SiLU()
#         )
        
#         # 尺度2：中等非对称目标 (分别提取 1x3 横向和 3x1 纵向，然后融合)
#         self.scale2_h = nn.Conv2d(mid_channels, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         self.scale2_v = nn.Conv2d(mid_channels, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         self.scale2_bn_act = nn.Sequential(
#             nn.BatchNorm2d(scale_ch),
#             nn.SiLU()
#         )

#         # 尺度3：偏大非对称目标 (利用两次堆叠扩大感受野，等效 1x5 和 5x1)
#         self.scale3_h = nn.Sequential(
#             nn.Conv2d(mid_channels, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1)),
#             nn.SiLU(),
#             nn.Conv2d(scale_ch, scale_ch, kernel_size=(1, 3), stride=1, padding=(0, 1))
#         )
#         self.scale3_v = nn.Sequential(
#             nn.Conv2d(mid_channels, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0)),
#             nn.SiLU(),
#             nn.Conv2d(scale_ch, scale_ch, kernel_size=(3, 1), stride=1, padding=(1, 0))
#         )
#         self.scale3_bn_act = nn.Sequential(
#             nn.BatchNorm2d(scale_ch),
#             nn.SiLU()
#         )

#         # 3. 交叉注意力融合
#         self.cross_att = GuidedCrossAttention(
#             in_channels=mid_channels, 
#             interact_in_channels=scale_ch * 3, 
#             reduction=16 
#         )

#         # 4. 尾部融合
#         self.final_conv = nn.Conv2d(2 * mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.final_bn = nn.BatchNorm2d(out_channels)
#         self.final_act = nn.SiLU()

#     def forward(self, x):
#         x1, x2 = self.conv1(x).chunk(2, 1)

#         # 提取局部 Query
#         local_feat = self.local_branch(x2)

#         # 非对称多尺度提取
#         s1 = self.scale1(x2)
        
#         # 十字融合：将横向(Horizontal)和纵向(Vertical)特征相加，剥离四个角的背景
#         s2 = self.scale2_bn_act(self.scale2_h(x2) + self.scale2_v(x2))
#         s3 = self.scale3_bn_act(self.scale3_h(x2) + self.scale3_v(x2))
        
#         multi_scale_feat = torch.cat([s1, s2, s3], dim=1) # (N, 1.5*mid_channels, H, W)

#         # 交叉注意力提纯
#         refined_multi_scale = self.cross_att(local_feat, multi_scale_feat)

#         # CSP 拼接与输出
#         out = torch.cat([x1, refined_multi_scale], dim=1)
#         out = self.final_act(self.final_bn(self.final_conv(out)))

#         return out    


class MSALCSP(nn.Module):
    """
    重构后的 MSALCSPV2:
    逻辑链：CSP切分 -> 精细化多尺度提取(匹配VisDrone小目标) -> 交叉注意力(抑制背景) -> 拼接融合
    """
    def __init__(self, in_channels, out_channels, expansion=0.5):
        super().__init__()
        mid_channels = int(out_channels * expansion)
        
        # CSP第一步：特征通道切分
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * mid_channels),
            nn.SiLU()
        )
        
        # 1. 基础局部特征分支 (Local Branch) - 感受野极小，提取最本真的像素信息
        # 替代原有的交叉注意力中额外的att_branch，让结构更紧凑
        self.local_branch = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )

        # 2. 多尺度特征分支 (Multi-Scale Branch) - 针对无人机尺度变化
        scale_ch = mid_channels // 2
        # 分支1：极小目标 (1x1 卷积，绝对局部信息，无周边背景干扰)
        self.scale1 = nn.Sequential(
            nn.Conv2d(mid_channels, scale_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(scale_ch),
            nn.SiLU()
        )
        # 分支2：中小目标 (3x3 标准卷积，常规感受野)
        self.scale2 = nn.Sequential(
            nn.Conv2d(mid_channels, scale_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(scale_ch),
            nn.SiLU()
        )
        # 摒弃膨胀卷积！改用 5x5 感受野 (通过两个3x3堆叠，避免网格效应，保留局部细粒度)
        self.scale3 = nn.Sequential(
            nn.Conv2d(mid_channels, scale_ch, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(scale_ch, scale_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(scale_ch),
            nn.SiLU()
        )

        # 3. 交叉注意力融合 (Cross Attention Fusion)
        # 逻辑：利用纯净的local_branch特征作为Query，去提取多尺度特征(Key/Value)中的有效信息
        # 注意：此处你需要确保你的 CrossAttention 支持 Q 维度等于 K/V 维度的操作
        self.cross_att = GuidedCrossAttention(
            in_channels=mid_channels, 
            interact_in_channels=scale_ch * 3, # 3个多尺度分支拼接
            reduction=16 
        )

        # 4. 尾部融合
        self.final_conv = nn.Conv2d(2 * mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.final_bn = nn.BatchNorm2d(out_channels)
        self.final_act = nn.SiLU()

    def forward(self, x):
        # CSP 切分
        x1, x2 = self.conv1(x).chunk(2, 1)

        # 提取极局部特征 (作为 Query，也作为残差连接的一部分)
        local_feat = self.local_branch(x2)

        # 提取多尺度特征 (无空洞，防漏检)
        s1 = self.scale1(x2)
        s2 = self.scale2(x2)
        s3 = self.scale3(x2)
        multi_scale_feat = torch.cat([s1, s2, s3], dim=1) # (N, 1.5*mid_channels, H, W)

        # 核心逻辑：用纯净的局部信息(Query) 去 提纯 多尺度信息中的复杂背景噪声
        refined_multi_scale = self.cross_att(local_feat, multi_scale_feat)

        # CSP 融合：保留跨阶段的宏观特征 x1，与提纯后的特征组合
        out = torch.cat([x1, refined_multi_scale], dim=1)

        out = self.final_conv(out)
        out = self.final_bn(out)
        out = self.final_act(out)

        return out



# class DynamicGating(nn.Module):
#     """动态门控融合：根据特征重要性分配权重"""

#     def __init__(self, in_channels, num_branches=3):
#         super().__init__()
#         self.num_branches = num_branches
#         self.gate = nn.Sequential(
#             nn.Conv2d(in_channels, num_branches, kernel_size=1, stride=1, padding=0),
#             nn.Sigmoid()
#         )
#     def forward(self, branches):
#         avg_feat = torch.mean(torch.stack(branches), dim=0)  #(3,n,c,h,w)-> (N, C, H, W)
#         weights = self.gate(avg_feat)  # (N, num_branches, H, W)
#         # concatenated_feat = torch.cat(branches, dim=1)
#         # weights = self.gate(concatenated_feat)
#         fused = 0.
#         for i in range(self.num_branches):
#             branch_weight = weights[:, i:i + 1, ...].expand_as(branches[i])
#             fused += branches[i] * branch_weight
#         return fused


# class CrossAttention(nn.Module):
#     """交叉注意力：结合通道与空间注意力，并引入分支特征交互（彻底解决通道匹配）"""

#     def __init__(self, in_channels, interact_in_channels, reduction=12):
#         super().__init__()
#         # 通道注意力
#         self.channel_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
#             nn.SiLU(),
#             nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#         # 空间注意力
#         self.spatial_att = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )

#         self.bn = nn.BatchNorm2d(in_channels)
#         self.act = nn.SiLU()

#         # 分支交互卷积：输入通道=interact_in_channels（2×in_chan）
#         self.branch_interact = nn.Sequential(
#             nn.Conv2d(interact_in_channels, in_channels, kernel_size=1),
#         )
#     def forward(self, x, other_feats):
#         # 分支交互（通道已匹配）
#         interact_feat = self.branch_interact(other_feats)
#         x = x + interact_feat
#         # x = interact_feat  #不要分支
#         x = self.bn(x)
#         x = self.act(x)

#         # 通道注意力
#         c_att = self.channel_att(x)
#         x = x * c_att
#         # 空间注意力
#         max_pool = torch.max(x, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x, dim=1, keepdim=True)
#         s_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
#         x = x * s_att
#         return x

#1K chunk 3k(1,2,3)CBS3   
# class MSALCSP(nn.Module):
#     "MSALCSP V1代表的是Multi-Scale Attention Layer with CSP（多尺度注意力层+CSP）"
#     def __init__(self, in_channels, out_channels, expansion=0.5):
#         super().__init__()
#         # 计算中间通道数（确保能被3整除，避免多尺度分支拼接问题）
#         mid_channels = int(out_channels *  expansion)
#         mid_channels = mid_channels if mid_channels % 3 == 0 else mid_channels + (3 - mid_channels % 3)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 2*mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2*mid_channels),
#             nn.SiLU()
#         )
#         # 1. 基础特征分支（轻量化深度可分离卷积）
#         self.base_branch = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
#             # nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU(),
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()
#         )

#         # 2. 多尺度扩张分支（小目标适配：3个并行子分支）

#         self.scale_branch_ch = mid_channels // 3  # 每个子分支通道数（确保整除）
#         self.total_scale_ch = self.scale_branch_ch * 3  # 拼接后总通道数

#         self.scale_pre = nn.Sequential(
#             nn.Conv2d(mid_channels, self.total_scale_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(self.total_scale_ch),
#             nn.SiLU()
#         )
#         # 中大目标分支：3×3 dilation=2（感受野5*5，避免过大）感受野大小=(kernel_size−1)×dilation+1
#         self.scale_branch3 = nn.Sequential(
#             # nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=3, dilation=3)
#         )
#         # 中小目标分支：3×3 dilation=1（感受野3*3）
#         self.scale_branch2 = nn.Sequential(
#             # nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=2, dilation=2)
#         )
#         # 极小目标分支：仅1×1（感受野1×1，保留细节）
#         self.scale_branch1 = nn.Sequential(
#             # nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=1, dilation=1)
#         )
#         self.scale_postprocess = nn.Sequential(
#             nn.Conv2d(self.total_scale_ch, mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()
#         )

#         # 3. 交叉注意力分支（通道匹配优化）
#         self.att_branch = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU(),
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU()
#         )
#         # 初始化CrossAttention：
#         self.cross_att = CrossAttention(
#             in_channels=mid_channels,
#             interact_in_channels=2 * mid_channels,  #基础分支+多尺度分支
#             reduction=12  # 小目标优化：增强通道注意力细粒度
#         )


#         # 4. 动态融合与残差连接
#         self.dynamic_fusion = DynamicGating(mid_channels, num_branches=3)

#         self.final_conv = nn.Conv2d(2*mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.final_bn = nn.BatchNorm2d(out_channels)
#         # self.final_act = nn.Mish()

#         self.final_act = nn.SiLU()
#     def forward(self, x):

#         x1 ,x2 = self.conv1(x).chunk(2, 1)


#         # 基础分支特征提取
#         base_feat = self.base_branch(x2)  # (N, mid_channels, H, W)

#         scale_shared = self.scale_pre(x2)
#         scale_1, scale_2, scale_3 = scale_shared.chunk(3, 1)
#         # 多尺度分支特征提取与融合

#         scale_feat1 = self.scale_branch1(scale_1)
#         scale_feat2 = self.scale_branch2(scale_2)
#         scale_feat3 = self.scale_branch3(scale_3)

#         scale_feat_cat = torch.cat([scale_feat1, scale_feat2, scale_feat3], dim=1)
#         scale_feat = self.scale_postprocess(scale_feat_cat)  # (N, mid_channels, H, W)

#         # 交叉注意力分支特征增强
#         att_feat = self.att_branch(x2)  # (N, mid_channels, H, W)
#         other_feats = torch.cat([base_feat, scale_feat], dim=1)  # (N, 2×mid_channels, H, W)
#         # other_feats = [base_feat, scale_feat]
#         att_feat = self.cross_att(att_feat, other_feats)# (N,mid_channels, H, W)
#         # att_feat = self.cross_att(None, other_feats)

#         # 动态融合
#         fused = self.dynamic_fusion([base_feat, scale_feat, att_feat])

#         out = torch.cat([x1,fused],1)

#         out = self.final_conv(out)
#         out = self.final_bn(out)
#         out = self.final_act(out)

#         return out


# class MSALCSP(nn.Module):  #1,2,5 论文用
#     "MSALCSP代表的是Multi-Scale Attention Layer with CSP（多尺度注意力层+CSP）"
#     def __init__(self, in_channels, out_channels, expansion=0.5):
#         super().__init__()
#         # 计算中间通道数（确保能被3整除，避免多尺度分支拼接问题）
#         mid_channels = int(out_channels *  expansion)
#         mid_channels = mid_channels if mid_channels % 3 == 0 else mid_channels + (3 - mid_channels % 3)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 2*mid_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(2*mid_channels),
#             nn.SiLU()
#         )
#         # 1. 基础特征分支（轻量化深度可分离卷积）
#         self.base_branch = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
#             nn.BatchNorm2d(mid_channels),
#         )
#
#         # 2. 多尺度扩张分支（小目标适配：3个并行子分支）
#
#         self.scale_branch_ch = mid_channels // 3  # 每个子分支通道数（确保整除）
#         self.total_scale_ch = self.scale_branch_ch * 3  # 拼接后总通道数
#
#         # self.scale_pre = nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0)
#
#         # 中大目标分支：3×3 dilation=2（感受野5*5，避免过大）感受野大小=(kernel_size−1)×dilation+1
#         self.scale_branch2 = nn.Sequential(
#             nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=5, dilation=5)
#         )
#         # 中小目标分支：3×3 dilation=1（感受野3*3）
#         self.scale_branch1 = nn.Sequential(
#             nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=2, dilation=2)
#         )
#         # 极小目标分支：仅1×1（感受野1×1，保留细节）
#         self.scale_branch3 = nn.Sequential(
#             nn.Conv2d(mid_channels, self.scale_branch_ch, kernel_size=1, stride=1, padding=0),
#             nn.Conv2d(self.scale_branch_ch, self.scale_branch_ch, kernel_size=3, stride=1, padding=1, dilation=1)
#         )
#         self.scale_postprocess = nn.Sequential(
#             nn.BatchNorm2d(self.total_scale_ch),
#             nn.Conv2d(self.total_scale_ch, mid_channels, kernel_size=1, stride=1, padding=0)
#         )
#
#         # 3. 交叉注意力分支（通道匹配优化）
#         self.att_branch = nn.Sequential(
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels),
#             nn.BatchNorm2d(mid_channels)
#         )
#         # 初始化CrossAttention：
#         self.cross_att = CrossAttention(
#             in_channels=mid_channels,
#             interact_in_channels=2 * mid_channels,  #基础分支+多尺度分支
#             reduction=12  # 小目标优化：增强通道注意力细粒度
#         )
#
#
#         # 4. 动态融合与残差连接
#         self.dynamic_fusion = DynamicGating(mid_channels, num_branches=3)
#
#         self.final_conv = nn.Conv2d(2*mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
#         self.final_bn = nn.BatchNorm2d(out_channels)
#         # self.final_act = nn.Mish()
#
#         self.final_act = nn.SiLU()
#     def forward(self, x):
#
#         x1 ,x2 = self.conv1(x).chunk(2, 1)
#
#
#         # 基础分支特征提取
#         base_feat = self.base_branch(x2)  # (N, mid_channels, H, W)
#
#         # scale_shared = self.scale_pre(x2)
#         # 多尺度分支特征提取与融合
#         scale_feat1 = self.scale_branch1(x2)
#         scale_feat2 = self.scale_branch2(x2)
#         scale_feat3 = self.scale_branch3(x2)
#         # scale_feat1 = self.scale_branch1(scale_shared)
#         # scale_feat2 = self.scale_branch2(scale_shared)
#         # scale_feat_cat = torch.cat([scale_feat1, scale_feat2, scale_shared], dim=1)
#
#         scale_feat_cat = torch.cat([scale_feat1, scale_feat2, scale_feat3], dim=1)
#         scale_feat = self.scale_postprocess(scale_feat_cat)  # (N, mid_channels, H, W)
#
#         # 交叉注意力分支特征增强
#         att_feat = self.att_branch(x2)  # (N, mid_channels, H, W)
#         other_feats = torch.cat([base_feat, scale_feat], dim=1)  # (N, 2×mid_channels, H, W)
#         att_feat = self.cross_att(att_feat, other_feats)# (N,mid_channels, H, W)
#
#         # 动态融合
#         fused = self.dynamic_fusion([base_feat, scale_feat, att_feat])
#
#         out = torch.cat([x1,fused],1)
#
#         out = self.final_conv(out)
#         out = self.final_bn(out)
#         out = self.final_act(out)
#
#         return out

class My_Index(nn.Module):
    """从列表或字典中选择特定索引/键的模块"""
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return x[self.idx]
        elif isinstance(x, dict):
            return x[self.idx] if self.idx in x else list(x.values())[self.idx]
        else:
            raise TypeError(f"Index: expected list/tuple/dict, but got {type(x)}")
class SPD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 加上 bias=False，因为后面紧跟了 BatchNorm，偏置不起作用还会占参数
        # self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1, bias=False)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.SiLU()
        self.conv_fusion = Conv(in_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        # x = self.act(self.batch_norm(x))
        return x

# Deep feature downsampling C
class RFD(nn.Module):
    """
    RFD - 极致轻量 & 完美剪枝版
    所有中间深度特征提取严格保持 in_channels 维度，杜绝通道扩张引起的参数暴增与剪枝撕裂。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1. 辅助分支 (CutD): 负责无损直通，输出直接对齐 out_channels
        self.spd_c = SPD(in_channels=in_channels, out_channels=out_channels)
        
        # 2. 前置特征提取: 严格保持 in_channels (等宽 DWConv，剪枝绝对安全)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.SiLU()
        # )
        self.conv = DWConv(in_channels, in_channels, 3, 1)
        
        # 3. DWConvD 下采样分支: 严格保持 in_channels
        # self.conv_x = nn.Sequential(
        #     nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.SiLU()
        # )
        self.conv_x = DWConv(in_channels, in_channels, 3, 2)

        # 4. MaxD 下采样分支: 严格保持 in_channels
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(in_channels)
        
        # 5. 融合层 (Fusion)
        # 拼接维度计算：cut_c(out) + conv_x(in) + max_m(in) = out_channels + 2 * in_channels
        fusion_in_channels = out_channels + 2 * in_channels
        # self.fusion = nn.Sequential(
        #     nn.Conv2d(fusion_in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.SiLU()
        # )
        self.fusion = Conv(fusion_in_channels, out_channels, 1, 1)

    def forward(self, x):       
        # 1. 下路：SPD 直接降维并转换通道
        c = self.spd_c(x)       
        
        # 2. 上路前置：维持原通道提取特征
        x_main = self.conv(x)        
        
        # 3. 上路衍生分支 1：DW 下采样
        x_dw = self.conv_x(x_main)      
        
        # 4. 上路衍生分支 2：池化下采样
        m = self.max_m(x_main)       
        m = self.batch_norm_m(m)
        
        # 5. 最终融合
        out = torch.cat([c, x_dw, m], dim=1)  
        out = self.fusion(out)             
        
        return out
    
#     2C
# class RFD(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         mid_channels = 2 * in_channels

#         # 1. SPD 辅助分支
#         self.spd_c = SPD(in_channels=in_channels, out_channels=mid_channels)
        
#         # =========================================================
#         # 🌟 修复核心：将带有倍增的 groups 卷积解耦为 PW + DW
#         # =========================================================
#         self.conv = nn.Sequential(
#             # 第一步：1x1 点卷积 (PW) 负责维度扩张 C -> 2C (无 groups 限制，随便剪)
#             nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.SiLU(),
#             # 第二步：3x3 深度卷积 (DW) 负责空间提取 (此时 in=out=mid_channels，完美等宽 DW)
#             nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
#         )
#         # 注意：因为移到了 Sequential 里，原本独立的 bn 和 act 就可以删掉了
        
#         # 3. DWConvD 分支 (等宽 DW，完美安全)
#         self.conv_x = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
#         self.batch_norm_x = nn.BatchNorm2d(mid_channels)
#         self.act_x = nn.SiLU()
        
#         # 4. MaxD 分支
#         self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.batch_norm_m = nn.BatchNorm2d(mid_channels)
        
#         # 5. Cat+Conv 融合层
#         self.fusion = nn.Sequential(
#             nn.Conv2d(3 * mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.GELU ()
#         )

#     def forward(self, x):       
#         c = x                   
        
#         # 走解耦后的特征提取分支 (直接过 Sequential)
#         x = self.conv(x)        
#         m = x                   
        
#         # 1. SPD 下路
#         c = self.spd_c(c)       
        
#         # 2. DWConvD 上路分支
#         x = self.conv_x(x)      
#         x = self.act_x(self.batch_norm_x(x))
        
#         # 3. MaxD 上路分支
#         m = self.max_m(m)       
#         m = self.batch_norm_m(m)
        
#         # 4. Concat + conv 融合
#         out = torch.cat([c, x, m], dim=1)  
#         out = self.fusion(out)             
        
#         return out

class RFD_LITE(nn.Module):
    """
    UltraLight-RFD (极致轻量+剪枝完美对齐版)
    等效 SPD 核心，使用 2x2 深度可分离卷积 (无重叠)
    空间域并行特征提取并做加法融合
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # --- 步骤 1: 空间域极致轻量多频段提取 (保持 in_channels 维度，参数极低) ---
        
        # 分支 A: 等效 SPD 核心，使用 2x2 深度可分离卷积 (无重叠)
        # 参数量: in_channels * 4
        self.branch_spd = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False)
        
        # 分支 B: 局部平滑感受野，使用 3x3 深度可分离卷积
        # 参数量: in_channels * 9
        self.branch_dw = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        
        # 分支 C: 高频显著特征提取 (无参数)
        self.branch_max = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 空间特征融合后的批归一化
        self.bn_spatial = nn.BatchNorm2d(in_channels)
        self.act_spatial = nn.SiLU()

        # --- 步骤 2: 跨通道信息交互与升维 ---
        # 唯一的计算大头，参数量: in_channels * out_channels * 1
        self.pw_conv = Conv(in_channels, out_channels, k=1, s=1)

    def forward(self, x):
        # 1. 空间域并行特征提取并做加法融合 (计算量极小，内存完全连续)
        x_spatial = self.branch_spd(x) + self.branch_dw(x) + self.branch_max(x)
        x_spatial = self.act_spatial(self.bn_spatial(x_spatial))
        
        # 2. 跨通道信息交互与升维
        # LAMP 剪枝算法只需无脑剪裁这个 pw_conv，绝对不会出现通道对齐崩溃！
        return self.pw_conv(x_spatial)