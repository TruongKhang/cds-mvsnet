# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=3, padding=1, res_op=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 2 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_op = res_op

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x) if self.res_op else self.drop_path(x)
        return x


class ConvNeXtFPN(nn.Module):
    """
    ConvNeXt+FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = Block
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1 = nn.Conv2d(1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = LayerNorm(initial_dim, eps=1e-6, data_format="channels_first")

        self.layer1 = block(block_dims[0], drop_path=0., layer_scale_init_value=0)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer3_outconv = nn.Conv2d(block_dims[3], block_dims[2], kernel_size=1, padding=0, bias=False)
        self.layer3_outconv2 = nn.Sequential(
            block(block_dims[2], drop_path=0., layer_scale_init_value=0),
            block(block_dims[2], drop_path=0., layer_scale_init_value=0),
            # ConvBlock(block_dims[2], block_dims[2]),
            # conv3x3(block_dims[2], block_dims[2]),
        )
        self.norm_outlayer3 = LayerNorm(block_dims[2], eps=1e-6, data_format="channels_first")
        self.layer2_outconv = nn.Conv2d(block_dims[2], block_dims[1], kernel_size=1, padding=0, bias=False)
        self.layer2_outconv2 = nn.Sequential(
            block(block_dims[1], drop_path=0., layer_scale_init_value=0),
            # ConvBlock(block_dims[2], block_dims[1]),
            # conv3x3(block_dims[1], block_dims[1]),
        )
        self.layer1_outconv = nn.Conv2d(block_dims[1], block_dims[0], kernel_size=1, padding=0, bias=False)
        self.layer1_outconv2 = nn.Sequential(
            block(block_dims[0], drop_path=0., layer_scale_init_value=0),
            block(block_dims[0], drop_path=0., layer_scale_init_value=0),
            # ConvBlock(block_dims[1], block_dims[0]),
            # conv3x3(block_dims[0], block_dims[0]),
        )
        self.norm_outlayer1 = LayerNorm(block_dims[0], eps=1e-6, data_format="channels_first")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)

    def _make_layer(self, block, dim, stride=1):
        layer1 = nn.Sequential(nn.Conv2d(self.in_planes, dim, kernel_size=3, padding=1, stride=stride, bias=False),
                               LayerNorm(dim, eps=1e-6,
                                         data_format="channels_first"))  # block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, drop_path=0., layer_scale_init_value=0)  # block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.bn1(self.conv1(x))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out_2x = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv2(x3 + self.layer3_outconv(x4_out_2x))

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv2(x2 + self.layer2_outconv(x3_out_2x))

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv2(x1 + self.layer1_outconv(x2_out_2x))

        return [self.norm_outlayer3(x3_out), self.norm_outlayer1(x1_out)]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x