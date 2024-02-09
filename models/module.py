import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

from timm.models.layers import trunc_normal_, DropPath
sys.path.append("..")

from models.dynamic_conv import DynamicConv
from models.convnext import LayerNorm, LayerNorm3d


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", dynamic=False, **kwargs):
        super(Conv2d, self).__init__()

        if dynamic:
            self.conv = DynamicConv(in_channels, out_channels, size_kernels=kernel_size, stride=stride, bias=(not bn), **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs)
        self.dynamic = dynamic
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.InstanceNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x, epipole=None, temperature=0.001):
        if self.dynamic:
            #feat, epipole, temperature = x
            y, norm_curv = self.conv(x, epipole=epipole, temperature=temperature)
        else:
            y = self.conv(x)
        # y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.leaky_relu(y, 0.1, inplace=True)
        out = (y, norm_curv) if self.dynamic else y
        return out

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


# class Conv2d(nn.Module):
#     """Applies a 2D convolution (optionally with batch normalization and relu activation)
#     over an input signal composed of several input planes.

#     Attributes:
#         conv (nn.Module): convolution module
#         bn (nn.Module): batch normalization module
#         relu (bool): whether to activate by relu

#     Notes:
#         Default momentum for batch normalization is set to be 0.01,

#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                 layer_scale_init_value=0, res_op=True, dynamic=False, **kwargs):
#         super(Conv2d, self).__init__()

#         if dynamic:
#             self.conv = DynamicConv(in_channels, out_channels, size_kernels=kernel_size, stride=stride, bias=False, **kwargs)
#         else:
#             groups = in_channels if in_channels == out_channels else 1
#             self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False, groups=groups, **kwargs)
#         self.dynamic = dynamic

#         self.norm = LayerNorm(out_channels, eps=1e-6)
#         self.pwconv1 = nn.Linear(out_channels, 2 * out_channels) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(2 * out_channels, out_channels)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.res_op = res_op

#     def forward(self, x, epipole=None, temperature=0.001):
#         input = x
#         if self.dynamic:
#             #feat, epipole, temperature = x
#             x, norm_curv = self.conv(x, epipole=epipole, temperature=temperature)
#         else:
#             x = self.conv(x)
        
#         x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
#         x = input + x if self.res_op else x

#         out = (x, norm_curv) if self.dynamic else x
#         return out

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.trunc_normal_(m.weight, std=.02)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 act=True, norm=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not norm), **kwargs)
        self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if norm else None
        self.act = nn.GELU() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 act=True, norm=True, bn_momentum=0.1, **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not norm), **kwargs)
        self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if norm else None
        self.act = nn.GELU() if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, padding=1, stride=1, norm="bn"):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm2d(planes, eps=1e-6)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(planes, affine=True)
        elif norm == "ln":
            self.norm = LayerNorm(planes, data_format="channels_first")
        else:
            self.norm = None

        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv(x)
        if self.norm:
            y = self.norm(y)
        y = self.act(y)
        return y


# class FeatureNet(nn.Module):
#     def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
#         super(FeatureNet, self).__init__()
#         assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
#         print("*************feature extraction arch mode:{}****************".format(arch_mode))
#         self.arch_mode = arch_mode
#         self.stride = stride
#         self.base_channels = base_channels
#         self.num_stage = num_stage

#         self.conv00 = Conv2d(3, base_channels, (3, 7, 11), 1, dynamic=True)
#         self.conv01 = Conv2d(base_channels, base_channels, (3, 5, 7), 1, dynamic=True)

#         self.downsample1 = Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)
#         self.conv10 = Conv2d(base_channels*2, base_channels*2, (3, 5), 1, dynamic=True)
#         self.conv11 = Conv2d(base_channels*2, base_channels*2, (3, 5), 1, dynamic=True)

#         self.downsample2 = Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)
#         self.conv20 = Conv2d(base_channels*4, base_channels*4, (3, 5), 1, dynamic=True)
#         self.conv21 = Conv2d(base_channels*4, base_channels*4, (3, 5), 1, dynamic=True)

#         self.downsample3 = Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1)
#         self.conv30 = Conv2d(base_channels*8, base_channels*8, (1, 3), 1, dynamic=True)
#         self.conv31 = Conv2d(base_channels*8, base_channels*8, (1, 3), 1, dynamic=True)
#         self.out0 = DynamicConv(base_channels*8, base_channels*8, size_kernels=(1, 3))
#         # self.act0 = nn.Sequential(nn.InstanceNorm2d(base_channels*8), nn.Tanh())
#         self.act0 = LayerNorm(base_channels*8, data_format="channels_first")
#         self.out_channels = [base_channels * 8]

#         self.inner0 = Conv2d(base_channels * 12, base_channels * 4, 3, padding=1)
#         self.inner1 = Conv2d(base_channels * 6, base_channels * 2, 3, padding=1)
#         self.inner2 = Conv2d(base_channels * 3, base_channels * 2, 3, padding=1)

#         self.out1 = DynamicConv(base_channels*4, base_channels*4, size_kernels=(3, 5))
#         # self.act1 = nn.Sequential(nn.InstanceNorm2d(base_channels*4), nn.Tanh())
#         self.act1 = LayerNorm(base_channels*4, data_format="channels_first")
#         self.out2 = DynamicConv(base_channels*2, base_channels*2, size_kernels=(3, 5))
#         # self.act2 = nn.Sequential(nn.InstanceNorm2d(base_channels*2), nn.Tanh())
#         self.act2 = LayerNorm(base_channels*2, data_format="channels_first")
#         self.out3 = DynamicConv(base_channels*2, base_channels*2, size_kernels=(3, 5))
#         # self.act3 = nn.Sequential(nn.InstanceNorm2d(base_channels*2), nn.Tanh())
#         self.act3 = LayerNorm(base_channels*2, data_format="channels_first")

#         self.out_channels.append(base_channels*4)
#         self.out_channels.append(base_channels*2)
#         self.out_channels.append(base_channels*2)

#     def forward(self, x, epipole=None, temperature=0.001):
#         conv00, nc00 = self.conv00(x, epipole, temperature)
#         conv01, nc01 = self.conv01(conv00, epipole, temperature)
#         down_conv0, down_epipole0 = self.downsample1(conv01), epipole / 2
#         conv10, nc10 = self.conv10(down_conv0, down_epipole0, temperature)
#         conv11, nc11 = self.conv11(conv10, down_epipole0, temperature)
#         down_conv1, down_epipole1 = self.downsample2(conv11), epipole / 4
#         conv20, nc20 = self.conv20(down_conv1, down_epipole1, temperature)
#         conv21, nc21 = self.conv21(conv20, down_epipole1, temperature)
#         down_conv2, down_epipole2 = self.downsample3(conv21), epipole / 8
#         conv30, nc30 = self.conv30(down_conv2, down_epipole2, temperature)
#         conv31, nc31 = self.conv31(conv30, down_epipole2, temperature)
        
#         intra_feat = conv31
#         out, nc32 = self.out0(intra_feat, epipole=down_epipole2, temperature=temperature)
#         out = self.act0(out)
#         nc_sum = (nc30**2 + nc31**2 + nc32**2) / 3
#         outputs = {}
#         outputs["stage1"] = out, nc_sum, nc32.abs()

#         intra_feat = torch.cat((F.interpolate(intra_feat, scale_factor=2, mode="bilinear"), conv21), dim=1)
#         intra_feat = self.inner0(intra_feat)
#         out, nc22 = self.out1(intra_feat, epipole=down_epipole1, temperature=temperature)
#         out = self.act1(out)
#         nc_sum = (nc20 ** 2 + nc21**2 + nc22 ** 2) / 3
#         outputs["stage2"] = out, nc_sum, nc22.abs()

#         intra_feat = torch.cat((F.interpolate(intra_feat, scale_factor=2, mode="bilinear"), conv11), dim=1)
#         intra_feat = self.inner1(intra_feat)
#         out, nc12 = self.out2(intra_feat, epipole=down_epipole0, temperature=temperature)
#         out = self.act2(out)
#         nc_sum = (nc10 ** 2 + nc11 ** 2 + nc12 ** 2) / 3
#         outputs["stage3"] = out, nc_sum, nc12.abs()

#         intra_feat = torch.cat((F.interpolate(intra_feat, scale_factor=2, mode="bilinear"), conv01), dim=1)
#         intra_feat = self.inner2(intra_feat)
#         out, nc02 = self.out3(intra_feat, epipole=epipole, temperature=temperature)
#         out = self.act3(out)
#         nc_sum = (nc00 ** 2 + nc01 ** 2 + nc02 ** 2) / 3
#         outputs["stage4"] = out, nc_sum, nc02.abs()

#         return outputs


class FeatureNet(nn.Module):
    """
    FPN, output resolution are 1/8 and 1/2.
    Each block has 2 layers.
    """

    def __init__(self, base_channels):
        super().__init__()
        # Config
        block = ConvBlock
        initial_dim = base_channels
        block_dims = [base_channels*2, base_channels*4, base_channels*8, base_channels*16]

        self.layer0 = self._make_layer(block, 3, initial_dim, kernel_size=7, padding=3, stride=1) # ConvBlock(1, initial_dim, kernel_size=7, padding=3, stride=2)
        self.layer1 = self._make_layer(block, initial_dim, block_dims[0], stride=2)  # 1/2
        self.layer2 = self._make_layer(block, block_dims[0], block_dims[1], stride=2)  # 1/4
        self.layer3 = self._make_layer(block, block_dims[1], block_dims[2], stride=2)  # 1/8
        self.layer4 = self._make_layer(block, block_dims[2], block_dims[3], stride=2)  # 1/16

        self.layer3_outconv = nn.Conv2d(block_dims[2], block_dims[3], kernel_size=1, padding=0, bias=False)
        self.layer3_outconv2 = nn.Sequential(
            ConvBlock(block_dims[3], block_dims[2]),
            # ConvBlock(block_dims[2], block_dims[2]),
            nn.Conv2d(block_dims[2], block_dims[2], kernel_size=3, padding=1, bias=False),
        )
        # self.norm_outlayer3 = LayerNorm(block_dims[2], eps=1e-6, data_format="channels_first")

        self.layer2_outconv = nn.Conv2d(block_dims[1], block_dims[2], kernel_size=1, padding=0, bias=False)
        self.layer2_outconv2 = nn.Sequential(
            ConvBlock(block_dims[2], block_dims[1]),
            nn.Conv2d(block_dims[1], block_dims[1], kernel_size=3, padding=1, bias=False),
        )
        self.layer1_outconv = nn.Conv2d(block_dims[0], block_dims[1], kernel_size=1, padding=0, bias=False)
        self.layer1_outconv2 = nn.Sequential(
            ConvBlock(block_dims[1], block_dims[0]),
            nn.Conv2d(block_dims[0], block_dims[0], kernel_size=3, padding=1, bias=False),
        )
        # self.norm_outlayer1 = LayerNorm(block_dims[0], eps=1e-6, data_format="channels_first")
        self.layer0_outconv = nn.Conv2d(initial_dim, block_dims[0], kernel_size=3, padding=1, bias=False)
        self.layer0_outconv2 = nn.Sequential(
            ConvBlock(block_dims[0], initial_dim),
            nn.Conv2d(initial_dim, initial_dim, kernel_size=3, padding=1, bias=False),
        )

        self.apply(self._init_weights)

        self.out_channels = block_dims[::-1][1:] + [initial_dim]

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_dim, out_dim, kernel_size=3, padding=1, stride=1):
        layer1 = block(in_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride)
        layer2 = block(out_dim, out_dim, stride=1)
        layers = (layer1, layer2)
        return nn.Sequential(*layers)

    def forward(self, x):
        # ResNet Backbone
        x0 = self.layer0(x) #self.act(self.bn1(self.conv1(x))))
        x1 = self.layer1(x0)  # 1/2
        x2 = self.layer2(x1)  # 1/4
        x3 = self.layer3(x2)  # 1/8
        x4 = self.layer4(x3)  # 1/16

        # FPN
        x4_out_2x = F.interpolate(x4, scale_factor=2., mode='bilinear', align_corners=True)
        x3_out = self.layer3_outconv(x3)
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out + x2_out_2x)

        x1_out_2x = F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True)
        x0_out = self.layer0_outconv(x0)
        x0_out = self.layer0_outconv2(x0_out + x1_out_2x)

        return {"stage1": x3_out, "stage2": x2_out, "stage3": x1_out, "stage4": x0_out}


# class CostRegNet(nn.Module):
#     def __init__(self, in_channels, base_channels, last_layer=True, full_res=False):
#         super(CostRegNet, self).__init__()
#         self.last_layer = last_layer
#         self.conv0 = Conv3d(in_channels, base_channels, padding=1)

#         self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
#         self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

#         self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
#         self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

#         self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
#         self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

#         if full_res:
#             self.conv7 = nn.Sequential(Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1),
#                                        Conv3d(base_channels*4, base_channels*4, padding=1))
#             self.conv9 = nn.Sequential(Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1),
#                                        Conv3d(base_channels*2, base_channels*2, padding=1))
#             self.conv11 = nn.Sequential(Deconv3d(base_channels * 2, base_channels, stride=2, padding=1, output_padding=1),
#                                        Conv3d(base_channels, base_channels, padding=1))
#         else:
#             self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

#             self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

#             self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

#         if self.last_layer:
#             if full_res:
#                 self.prob = nn.Sequential(Conv3d(base_channels, base_channels, padding=1), nn.Conv3d(base_channels, 1, 1, stride=1, bias=False))
#             else:
#                 self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

#     def forward(self, x):
#         conv0 = self.conv0(x)
#         conv2 = self.conv2(self.conv1(conv0))
#         conv4 = self.conv4(self.conv3(conv2))
#         x = self.conv6(self.conv5(conv4))
#         x = conv4 + self.conv7(x)
#         x = conv2 + self.conv9(x)
#         x = conv0 + self.conv11(x)
#         if self.last_layer:
#             x = self.prob(x)
#         return x
    
class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True, n_levels=3):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer
        self.n_levels = n_levels

        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1) if n_levels > 1 else nn.Identity()
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1) if n_levels > 1 else nn.Identity()


        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1) if n_levels > 2 else nn.Identity()
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1) if n_levels > 2 else nn.Identity()

        self.conv7 = nn.Sequential(Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels*4, base_channels*4, padding=1)) if n_levels > 2 else nn.Identity()
        self.conv9 = nn.Sequential(Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels*2, base_channels*2, padding=1)) if n_levels > 1 else nn.Identity()
        self.conv11 = nn.Sequential(Deconv3d(base_channels * 2, base_channels, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels, base_channels, padding=1))

        if self.last_layer:
            self.prob = nn.Sequential(Conv3d(base_channels, base_channels, padding=1), nn.Conv3d(base_channels, 1, 1, stride=1, bias=False))

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x) if self.n_levels > 2 else self.conv7(x)
        x = conv2 + self.conv9(x) if self.n_levels > 1 else self.conv9(x)
        x = conv0 + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def conf_regression(p, n=4):
    ndepths = p.size(1)
    with torch.no_grad():
        # photometric confidence
        prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, (n-1)//2, n//2]),
                                            (n, 1, 1), stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(p.detach(), depth_values=torch.arange(ndepths, device=p.device, dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=ndepths - 1)
        conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1))
    return conf.squeeze(1)


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=1000.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    nl = (ndepth - 1) // 2
    nr = ndepth - 1 - nl
    cur_depth_min = (cur_depth - nl * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + nr * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = torch.ones_like(cur_depth) * depth_inteval_pixel #(cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    delta = (depth_range_samples - min_depth).clamp(min=0)
    depth_range_samples = min_depth + delta
    delta = (depth_range_samples - max_depth).clamp(max=0)
    depth_range_samples = max_depth + delta
    return depth_range_samples #.clamp(min=min_depth, max=max_depth)


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=1000.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:

        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

    return depth_range_samples


if __name__ == "__main__":
    # some testing code, just IGNORE it

    feature_net = FeatureNet(8).to(torch.device('cuda'))
    out = feature_net(torch.rand(2, 3, 512, 640).float().cuda(), torch.rand(2, 2).float().cuda())

    # pass
