import torch
import torch.nn as nn


def skew_matrix(vector3d):
    batch_size = vector3d.size(0)
    s = torch.zeros((batch_size, 3, 3), dtype=vector3d.dtype, device=vector3d.dtype)
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
    proj1 = torch.matmul(intr1, extr1)
    proj2 = torch.matmul(intr2, extr2)
    cam_center1 = torch.inverse(extr1[:, :, :3]) @ extr1[:, :, [3]]
    cam_center2 = torch.inverse(extr2[:, :, :3]) @ extr2[:, :, [3]]
    smatrix = skew_matrix(proj2 @ cam_center1)
    Fmatrix = smatrix @ proj2 @ torch.inverse(proj1)

    return Fmatrix


def compute_epipole(Fmatrix):
    c = 1e3
    eq1 = c * Fmatrix[:, 0] + Fmatrix[:, 1] + Fmatrix[:, 2] # [B, 3]
    eq2 = c * Fmatrix[:, 0] - Fmatrix[:, 1] - Fmatrix[:, 2] # [B, 3]
    eq = torch.stack((eq1, eq2), dim=1) # [B, 2, 3]
    epipole = - torch.inverse(eq[:, :, :2]) @ eq[:, :, [2]] # [B, 2, 1]
    return epipole.squeeze(2)


class DynamicConv(nn.Module):
    def __init__(self, in_c, out_c, size_kernels=(1, 3, 5)):
        super(DynamicConv, self).__init__()
        self.size_kernels = size_kernels
        self.convs = nn.ModuleList([nn.Conv2d(in_c, out_c, k, padding=(k-1)//2) for k in self.size_kernels])

    def forward(self, feature_vol, epipole=None):
        surface = torch.mean(feature_vol.detach(), dim=1)
        



