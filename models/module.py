import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("..")
# from utils import local_pcd

from models.dynamic_conv import DynamicConv


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


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


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


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            y = self.bn(y)
        if self.relu:
            y = F.relu(y, inplace=True)
        return y

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


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
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


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
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv00 = Conv2d(3, base_channels, (3, 7, 11), 1, dynamic=True)
        self.conv01 = Conv2d(base_channels, base_channels, (3, 5, 7), 1, dynamic=True)

        self.downsample1 = Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1)
        self.conv10 = Conv2d(base_channels*2, base_channels*2, (3, 5), 1, dynamic=True)
        self.conv11 = Conv2d(base_channels*2, base_channels*2, (3, 5), 1, dynamic=True)

        self.downsample2 = Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1)
        self.conv20 = Conv2d(base_channels*4, base_channels*4, (1, 3), 1, dynamic=True)
        self.conv21 = Conv2d(base_channels*4, base_channels*4, (1, 3), 1, dynamic=True)

        self.out1 = DynamicConv(base_channels*4, base_channels*4, size_kernels=(1, 3))
        self.act1 = nn.Sequential(nn.InstanceNorm2d(base_channels*4), nn.Tanh())
        self.out_channels = [base_channels*4]

        self.inner1 = Conv2d(base_channels * 6, base_channels * 2, 1) #nn.Sequential(nn.Conv2d(base_channels * 6, base_channels*2, 1, bias=True), nn.ReLU(inplace=True))
        self.inner2 = Conv2d(base_channels * 3, base_channels, 1) #nn.Sequential(nn.Conv2d(base_channels * 3, base_channels, 1, bias=True), nn.ReLU(inplace=True))

        self.out2 = DynamicConv(base_channels*2, base_channels*2, size_kernels=(1, 3))  # nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
        self.act2 = nn.Sequential(nn.InstanceNorm2d(base_channels*2), nn.Tanh())
        self.out3 = DynamicConv(base_channels, base_channels, size_kernels=(1, 3))  # nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
        self.act3 = nn.Sequential(nn.InstanceNorm2d(base_channels), nn.Tanh())
        self.out_channels.append(base_channels*2)
        self.out_channels.append(base_channels)

    def forward(self, x, epipole=None, temperature=0.001):
        conv00, nc00 = self.conv00(x, epipole, temperature)
        conv01, nc01 = self.conv01(conv00, epipole, temperature)
        down_conv0, down_epipole0 = self.downsample1(conv01), epipole / 2
        conv10, nc10 = self.conv10(down_conv0, down_epipole0, temperature)
        conv11, nc11 = self.conv11(conv10, down_epipole0, temperature)
        down_conv1, down_epipole1 = self.downsample2(conv11), epipole / 4
        conv20, nc20 = self.conv20(down_conv1, down_epipole1, temperature)
        conv21, nc21 = self.conv21(conv20, down_epipole1, temperature)

        intra_feat = conv21
        outputs = {}
        out, nc22 = self.out1(intra_feat, epipole=down_epipole1, temperature=temperature)
        out = self.act1(out)
        nc_sum = (nc20 ** 2 + nc21**2 + nc22 ** 2) / 3
        outputs["stage1"] = out, nc_sum, nc22.abs()

        intra_feat = torch.cat((F.interpolate(intra_feat, scale_factor=2, mode="nearest"), conv11), dim=1)
        intra_feat = self.inner1(intra_feat)
        out, nc12 = self.out2(intra_feat, epipole=down_epipole0, temperature=temperature)
        out = self.act2(out)
        nc_sum = (nc10 ** 2 + nc11 ** 2 + nc12 ** 2) / 3
        outputs["stage2"] = out, nc_sum, nc12.abs()

        intra_feat = torch.cat((F.interpolate(out, scale_factor=2, mode="nearest"), conv01), dim=1)
        intra_feat = self.inner2(intra_feat)
        out, nc02 = self.out3(intra_feat, epipole=epipole, temperature=temperature)
        out = self.act3(out)
        nc_sum = (nc00 ** 2 + nc01 ** 2 + nc02 ** 2) / 3
        outputs["stage3"] = out, nc_sum, nc02.abs()

        return outputs

# class FeatureNet(nn.Module):
#     def __init__(self, base_channels, arch_mode="unet"):
#         super(FeatureNet, self).__init__()
#         assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
#         print("*************feature extraction arch mode:{}****************".format(arch_mode))
#         self.arch_mode = arch_mode
#         self.base_channels = base_channels
#
#         convs = [Conv2d(3, base_channels, 3, 1, padding=1, dynamic=True)]
#         for _ in range(4):
#             convs.append(Conv2d(base_channels, base_channels, 3, 1, padding=1, dynamic=True))
#         self.convs = nn.ModuleList(convs)
#         self.out = nn.Conv2d(base_channels, base_channels, 1, bias=False)
#         self.out_channels = base_channels
#
#     def forward(self, x, epipole=None):
#         for conv in self.convs:
#             x = conv((x, epipole))
#         out = self.out(x)
#         return out


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels, last_layer=True, full_res=False):
        super(CostRegNet, self).__init__()
        self.last_layer = last_layer
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        if full_res:
            self.conv7 = nn.Sequential(Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels*4, base_channels*4, padding=1))
            self.conv9 = nn.Sequential(Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels*2, base_channels*2, padding=1))
            self.conv11 = nn.Sequential(Deconv3d(base_channels * 2, base_channels, stride=2, padding=1, output_padding=1),
                                       Conv3d(base_channels, base_channels, padding=1))
        else:
            self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

            self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

            self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        if self.last_layer:
            if full_res:
                self.prob = nn.Sequential(Conv3d(base_channels, base_channels, padding=1), nn.Conv3d(base_channels, 1, 1, stride=1, bias=False))
            else:
                self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        if self.last_layer:
            x = self.prob(x)
        return x


class Refinement(nn.Module):
    """Depth map refinement network"""

    def __init__(self):
        """Initialize"""

        super(Refinement, self).__init__()

        # img: [B,3,H,W]
        #self.conv0 = nn.Sequential(ConvBnReLU(in_channels=3, out_channels=8),
        #                           ConvBnReLU(in_channels=8, out_channels=8),
        #                           ConvBnReLU(in_channels=8, out_channels=8))
        self.conv0 = ConvBnReLU(in_channels=3, out_channels=8)
        # depth map:[B,1,H/2,W/2]
        #self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(inplace=True))
        self.conv1 = ConvBnReLU(in_channels=1, out_channels=8)
        #self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True))
        self.conv2 = ConvBnReLU(in_channels=8, out_channels=8)
        self.deconv = nn.ConvTranspose2d(
            in_channels=8, out_channels=8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(in_channels=16, out_channels=8)
                                   #ConvBnReLU(in_channels=8, out_channels=8),
                                   #ConvBnReLU(in_channels=8, out_channels=8))
        #self.conv3 = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
        #                           nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True),
        #                           nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True))
        self.res = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(
        self, img: torch.Tensor, depth_0: torch.Tensor, depth_min: torch.Tensor, depth_max: torch.Tensor
    ) -> torch.Tensor:
        """Forward method
        Args:
            img: input reference images (B, 3, H, W)
            depth_0: current depth map (B, 1, H//2, W//2)
            depth_min: pre-defined minimum depth (B, )
            depth_max: pre-defined maximum depth (B, )
        Returns:
            depth: refined depth map (B, 1, H, W)
        """

        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (
            depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) * 10

        conv0 = self.conv0(img)
        #deconv = F.relu(self.deconv(self.conv2(self.conv1(depth))), inplace=True)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = (F.interpolate(depth, scale_factor=2, mode="bilinear", align_corners=True) + res) / 10
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(batch_size, 1, 1, 1)

        return depth



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
        prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n//2 - 1, n//2]),
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
    # import sys
    # sys.path.append("../")
    # from datasets import find_dataset_def
    # from torch.utils.data import DataLoader
    # import numpy as np
    # import cv2
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt
    #
    # # MVSDataset = find_dataset_def("colmap")
    # # dataset = MVSDataset("../data/results/ford/num10_1/", 3, 'test',
    # #                      128, interval_scale=1.06, max_h=1250, max_w=1024)
    #
    # MVSDataset = find_dataset_def("dtu_yao")
    # num_depth = 48
    # dataset = MVSDataset("../data/DTU/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
    #                      3, num_depth, interval_scale=1.06 * 192 / num_depth)
    #
    # dataloader = DataLoader(dataset, batch_size=1)
    # item = next(iter(dataloader))
    #
    # imgs = item["imgs"][:, :, :, ::4, ::4]  #(B, N, 3, H, W)
    # # imgs = item["imgs"][:, :, :, :, :]
    # proj_matrices = item["proj_matrices"]   #(B, N, 2, 4, 4) dim=N: N view; dim=2: index 0 for extr, 1 for intric
    # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :]
    # # proj_matrices[:, :, 1, :2, :] = proj_matrices[:, :, 1, :2, :] * 4
    # depth_values = item["depth_values"]     #(B, D)
    #
    # imgs = torch.unbind(imgs, 1)
    # proj_matrices = torch.unbind(proj_matrices, 1)
    # ref_img, src_imgs = imgs[0], imgs[1:]
    # ref_proj, src_proj = proj_matrices[0], proj_matrices[1:][0]  #only vis first view
    #
    # src_proj_new = src_proj[:, 0].clone()
    # src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
    # ref_proj_new = ref_proj[:, 0].clone()
    # ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])



    # warped_imgs = homo_warping(src_imgs[0], src_proj_new, ref_proj_new, depth_values)
    #
    # ref_img_np = ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255
    # cv2.imwrite('../tmp/ref.png', ref_img_np)
    # cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    #
    # for i in range(warped_imgs.shape[2]):
    #     warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
    #     img_np = warped_img[0].detach().cpu().numpy()
    #     img_np = img_np[:, :, ::-1] * 255
    #
    #     alpha = 0.5
    #     beta = 1 - alpha
    #     gamma = 0
    #     img_add = cv2.addWeighted(ref_img_np, alpha, img_np, beta, gamma)
    #     cv2.imwrite('../tmp/tmp{}.png'.format(i), np.hstack([ref_img_np, img_np, img_add])) #* ratio + img_np*(1-ratio)]))

    feature_net = FeatureNet(8).to(torch.device('cuda'))
    out = feature_net(torch.rand(2, 3, 512, 640).float().cuda(), torch.rand(2, 2).float().cuda())

    # pass
