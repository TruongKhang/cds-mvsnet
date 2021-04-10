import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair


def skew_matrix(vector3d):
    batch_size = vector3d.size(0)
    s = torch.zeros((batch_size, 3, 3), dtype=vector3d.dtype, device=vector3d.device)
    s[:, 0, 1] = - vector3d[:, 2]
    s[:, 0, 2] = vector3d[:, 1]
    s[:, 1, 0] = vector3d[:, 2]
    s[:, 1, 2] = - vector3d[:, 0]
    s[:, 2, 0] = - vector3d[:, 1]
    s[:, 2, 1] = vector3d[:, 0]
    return s


def compute_Fmatrix(cam_params1, cam_params2):
    intr1, extr1 = cam_params1[:, 1, :3, :3], cam_params1[:, 0, :3, :4]
    intr2, extr2 = cam_params2[:, 1, :3, :3], cam_params2[:, 0, :3, :4]

    rot1, trans1 = extr1[:, :3, :3], extr1[:, :3, [3]]
    rot2, trans2 = extr2[:, :3, :3], extr2[:, :3, [3]]
    cam_center1 = - torch.inverse(rot1) @ trans1
    # cam_center1 = torch.cat((cam_center1, torch.ones_like(cam_center1[:, [0], :])), dim=1)
    cam_center2 = - torch.inverse(rot2) @ trans2

    proj1 = torch.matmul(intr1, rot1)
    proj2 = torch.matmul(intr2, rot2)
    # print(proj2.size(), cam_center1.size())
    cam_center12 = torch.matmul(proj2, cam_center1 - cam_center2)
    print(cam_center12.size())
    smatrix = skew_matrix(cam_center12.squeeze(2))
    print(smatrix.size(), proj1.size(), proj2.size())
    Fmatrix = smatrix @ proj2 @ torch.inverse(proj1)

    return Fmatrix


def compute_epipole(Fmatrix):
    c = 1e3
    eq1 = c * Fmatrix[:, 0] + Fmatrix[:, 1] + Fmatrix[:, 2] # [B, 3]
    eq2 = c * Fmatrix[:, 0] - Fmatrix[:, 1] - Fmatrix[:, 2] # [B, 3]
    eq = torch.stack((eq1, eq2), dim=1) # [B, 2, 3]
    epipole = - torch.inverse(eq[:, :, :2]) @ eq[:, :, [2]] # [B, 2, 1]
    return epipole.squeeze(2)


class GaussFilter2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, padding_mode='zeros', filter_type=None):
        self.filter_size = kernel_size
        super(GaussFilter2d, self).__init__(in_channels, out_channels, _pair(kernel_size), _pair(stride), _pair(padding),
                                            _pair(dilation), False, _pair(0), groups, bias, padding_mode)

        y, x = torch.meshgrid([torch.arange(-(kernel_size - 1) // 2 - 1, (kernel_size - 1) // 2, dtype=torch.float32),
                               torch.arange(-(kernel_size - 1) // 2 - 1, (kernel_size - 1) // 2, dtype=torch.float32)])
        sigma = float(self.filter_size / 3)

        gauss_kernel = torch.exp(- (x**2 + y**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        if filter_type == 'x':
            self.weight = -x / (sigma**2) * gauss_kernel
        elif filter_type == 'y':
            self.weight = -y / (sigma**2) * gauss_kernel
        elif filter_type == 'xx':
            self.weight = (sigma**2 - x**2) / (sigma**4) * gauss_kernel
        elif filter_type == 'xy':
            self.weight = x * y / (sigma**4) * gauss_kernel
        elif filter_type == 'yy':
            self.weight = (sigma ** 2 - y ** 2) / (sigma ** 4) * gauss_kernel
        else:
            self.weight = gauss_kernel
        self.weight = self.weight.unsqueeze(0).unsqueeze(0).repeat(out_channels, in_channels, 1, 1)

    def forward(self, img):
        filtered_img = F.conv2d(img, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return filtered_img


class DynamicConv(nn.Module):
    def __init__(self, in_c, out_c, size_kernels=(3, 5, 7), stride=1, thresh_scale=0.01):
        super(DynamicConv, self).__init__()
        self.size_kernels = size_kernels
        self.thresh_scale = thresh_scale
        self.convs = nn.ModuleList([nn.Conv2d(in_c, out_c, k, padding=(k-1)//2, stride=stride) for k in self.size_kernels])

    def forward(self, feature_vol, epipole=None):
        surface = torch.mean(feature_vol.detach(), dim=1)
        batch_size, height, width = surface.shape[0], surface.shape[2], surface.shape[3]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=surface.device),
                               torch.arange(0, width, dtype=torch.float32, device=surface.device)])
        x, y = x.contigous(), y.contigous()
        epipole = epipole.unsqueeze(-1).unsqueeze(-1) # [B, 2, 1, 1]
        u = x.unsqueeze(0).unsqueeze(0) - epipole[:, [0], :, :] # [B, 1, H, W]
        v = y.unsqueeze(0).unsqueeze(0) - epipole[:, [1], :, :] # [B, 1, H, W]
        normed_uv = torch.sqrt(u**2 + v**2)
        u, v = u / normed_uv, v / normed_uv

        selected_conv = self.convs[-1]

        for idx, s in enumerate(self.size_kernels):
            new_s = s * 3
            dx = GaussFilter2d(1, 1, new_s, padding=(new_s-1)//2, filter_type='x')(surface)
            dy = GaussFilter2d(1, 1, new_s, padding=(new_s-1)//2, filter_type='y')(surface)
            dxx = GaussFilter2d(1, 1, new_s, padding=(new_s-1)//2, filter_type='xx')(surface)
            dxy = GaussFilter2d(1, 1, new_s, padding=(new_s-1)//2, filter_type='xy')(surface)
            dyy = GaussFilter2d(1, 1, new_s, padding=(new_s-1)//2, filter_type='yy')(surface)

            E, F, G = 1 + dx**2, dx*dy, 1 + dy**2
            normed = torch.sqrt(1 + dx**2 + dy**2)
            L, M, N = dxx / normed, dxy / normed, dyy / normed

            k = (u**2*L + 2*u*v*M + v**2*N) / (u**2*E + 2*u*v*F + v**2*G)
            if k > self.thresh_scale:
                selected_conv = self.convs[idx]
                break

        filtered_feature = selected_conv(feature_vol)
        return filtered_feature


def read_cam_file(filename, interval_scale=1.0):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics[:2, :] /= 4.0
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_interval = float(lines[11].split()[1])

    if len(lines[11].split()) >= 3:
        num_depth = lines[11].split()[2]
        depth_max = depth_min + int(float(num_depth)) * depth_interval
        depth_interval = (depth_max - depth_min) / 192

    depth_interval *= interval_scale

    return intrinsics, extrinsics, depth_min, depth_interval


if __name__ == '__main__':
    from PIL import Image
    from torchvision.transforms import ToTensor

    ref_img = Image.open(r"D:\lab\dtu\scan4\images\00000000.jpg")
    ref_img = ToTensor()(ref_img)
    src_img = Image.open("D:/lab/dtu/scan4/images/00000010.jpg")
    src_img = ToTensor()(src_img)

    ref_intr, ref_extr, _, _ = read_cam_file("D://lab/dtu/scan4/cams/00000000_cam.txt")
    src_intr, src_extr, _, _ = read_cam_file("D://lab/dtu/scan4/cams/00000010_cam.txt")

    ref_cam_params = torch.zeros(1, 2, 4, 4)
    ref_cam_params[0, 0, :4, :4] = torch.tensor(ref_extr, dtype=torch.float32)
    ref_cam_params[0, 1, :3, :3] = torch.tensor(ref_intr, dtype=torch.float32)
    ref_cam_params = ref_cam_params.cuda()

    src_cam_params = torch.zeros(1, 2, 4, 4)
    src_cam_params[0, 0, :4, :4] = torch.tensor(src_extr, dtype=torch.float32)
    src_cam_params[0, 1, :3, :3] = torch.tensor(src_intr, dtype=torch.float32)
    src_cam_params = src_cam_params.cuda()

    Fmatrix = compute_Fmatrix(ref_cam_params, src_cam_params)
    ref_epipole = compute_epipole(Fmatrix)
    src_epipole = compute_epipole(torch.transpose(Fmatrix, 1, 2))
    print("reference epipole: ", ref_epipole)
    print("src epipole: ", src_epipole)

    # compare with opencv result
    import cv2
    np_Fmatrix = Fmatrix[0].cpu().numpy()
    src_line = cv2.computeCorrespondEpilines(np.zeros((1, 1, 2)), 1, np_Fmatrix)
    src_line = src_line.reshape(-1, 3)
    src_point = np.array([0, -src_line[0][2] / src_line[0][1]], dtype=np.float32)
    ref_line = cv2.computeCorrespondEpilines(src_point.reshape((1, 1, 2)), 2, np_Fmatrix)
    ref_line = ref_line.reshape(-1, 3)
    print(ref_line[0][:2])

    print(ref_epipole / torch.sqrt((ref_epipole**2).sum()))







