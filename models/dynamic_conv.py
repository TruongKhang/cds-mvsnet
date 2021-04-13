import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time


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
    # print(cam_center12.size())
    smatrix = skew_matrix(cam_center12.squeeze(2))
    # print(smatrix.size(), proj1.size(), proj2.size())
    Fmatrix = smatrix @ proj2 @ torch.inverse(proj1)

    return Fmatrix


def compute_epipole(Fmatrix):
    c = 1e3
    eq1 = c * Fmatrix[:, 0] + Fmatrix[:, 1] + Fmatrix[:, 2] # [B, 3]
    eq2 = c * Fmatrix[:, 0] - Fmatrix[:, 1] - Fmatrix[:, 2] # [B, 3]
    eq = torch.stack((eq1, eq2), dim=1) # [B, 2, 3]
    epipole = - torch.inverse(eq[:, :, :2]) @ eq[:, :, [2]] # [B, 2, 1]
    return epipole.squeeze(2)


class GaussFilter2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, device=None):
        self.filter_size = kernel_size
        super(GaussFilter2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.y, self.x = torch.meshgrid([torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1, dtype=torch.float32, device=device),
                                         torch.arange(-(kernel_size - 1) // 2, (kernel_size - 1) // 2 + 1, dtype=torch.float32, device=device)])

    def forward(self, img):
        sigma = float(self.filter_size / 9 * 1.2)

        gauss_kernel = torch.exp(- (self.x ** 2 + self.y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

        fx = -self.x / (sigma ** 2) * gauss_kernel
        fy = -self.y / (sigma ** 2) * gauss_kernel
        fxx = (self.x ** 2 - sigma ** 2) / (sigma ** 4) * gauss_kernel
        fxy = self.x * self.y / (sigma ** 4) * gauss_kernel
        fyy = (self.y ** 2 - sigma ** 2) / (sigma ** 4) * gauss_kernel
        weight = torch.stack((fx, fy, fxx, fxy, fyy))
        weight = weight.unsqueeze(1).repeat(1, self.in_channels, 1, 1) / self.in_channels
        start = time()
        filtered_results = F.conv2d(img, weight, None, self.stride, self.padding, self.dilation, self.groups)
        print(time() - start)
        dx, dy, dxx, dxy, dyy = torch.unbind(filtered_results, dim=1)
        return dx, dy, dxx, dxy, dyy


class DynamicConv(nn.Module):
    def __init__(self, in_c, out_c, size_kernels=(3, 5, 7), stride=1, bias=True, thresh_scale=0.005, **kwargs):
        super(DynamicConv, self).__init__()
        self.size_kernels = size_kernels
        self.thresh_scale = thresh_scale
        self.convs = nn.ModuleList([nn.Conv2d(in_c, out_c, k, padding=(k-1)//2, stride=stride, bias=bias) for k in self.size_kernels])

    def forward(self, feature_vol, epipole=None):
        surface = torch.mean(feature_vol.detach(), dim=1, keepdim=True)
        batch_size, height, width = surface.shape[0], surface.shape[2], surface.shape[3]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=surface.device),
                               torch.arange(0, width, dtype=torch.float32, device=surface.device)])
        x, y = x.contiguous(), y.contiguous()
        epipole_map = epipole.unsqueeze(-1).unsqueeze(-1) # [B, 2, 1, 1]
        u = x.unsqueeze(0).unsqueeze(0) - epipole_map[:, [0], :, :] # [B, 1, H, W]
        v = y.unsqueeze(0).unsqueeze(0) - epipole_map[:, [1], :, :] # [B, 1, H, W]
        normed_uv = torch.sqrt(u**2 + v**2)
        u, v = u / normed_uv, v / normed_uv

        # selected_conv = self.convs[-1]
        filtered_result = 0.0
        sum_mask = torch.zeros_like(surface)
        # t11, t12, t13 = 0.0, 0.0, 0.0
        for idx, s in enumerate(self.size_kernels):
            new_s = s * 3
            if idx == (len(self.size_kernels) - 1):
                sum_mask += 1
            else:
                # start1 = time()
                gauss_filter = GaussFilter2d(1, 1, new_s, padding=(new_s - 1) // 2, device=surface.device)
                dx, dy, dxx, dxy, dyy = gauss_filter(surface)
                # start2 = time()
                # t11 = start2 - start1 + t11
                E, F, G = 1 + dx ** 2, dx * dy, 1 + dy ** 2
                normed = torch.sqrt(E + G - 1)  # 1 + dx**2 + dy**2)
                L, M, N = dxx / normed, dxy / normed, dyy / normed
                # start3 = time()
                # t12 = start3 - start2 + t12
                k = (u ** 2 * L + 2 * u * v * M + v ** 2 * N) / (u ** 2 * E + 2 * u * v * F + v ** 2 * G + 1e-10)
                sum_mask = sum_mask + (k.abs() > self.thresh_scale).float()
                # t13 = time() - start3 + t13
            filtered_result = filtered_result + (sum_mask == 1).float() * self.convs[idx](feature_vol)
        return filtered_result #, sum_mask, t11, t12, t13


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
    ref_img = ToTensor()(ref_img).cuda()
    src_img = Image.open("D:/lab/dtu/scan4/images/00000010.jpg")
    src_img = ToTensor()(src_img).cuda()

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

    # gauss_filter = GaussFilter2d(3, 1, 9, padding=4, device=ref_img.device)
    # filtered_img = gauss_filter(ref_img.unsqueeze(0), filter_type='xx')
    import matplotlib.pyplot as plt
    # import matplotlib
    # cmap = matplotlib.cm.gray
    # plt.imshow(filtered_img.squeeze(0).squeeze(0).cpu().numpy())
    # plt.colorbar()
    # plt.show()

    from time import time
    conv = DynamicConv(3, 1, size_kernels=(3, 5, 7, 9), thresh_scale=0.005)
    conv = conv.to(torch.device('cuda'))
    t1 = time()
    conv_img, mask, t11, t12, t13 = conv(ref_img.unsqueeze(0), epipole=ref_epipole)
    t2 = time()
    conv = nn.Conv2d(3, 1, 7, padding=3).to(torch.device('cuda'))
    t3 = time()
    conv_img = conv(ref_img.unsqueeze(0))
    t4 = time()
    print("Time of Dynamic Conv: ", t2-t1, t11, t12, t13)
    print("Time of original Conv: ", t4-t3)
    plt.imshow(mask.squeeze(0).squeeze(0).cpu().numpy())
    plt.colorbar()
    plt.show()






