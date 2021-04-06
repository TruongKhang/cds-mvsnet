########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import numpy as np
from scipy.stats import poisson
from scipy import signal


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


# The proposed Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus',
                 init_method='k', stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, False, _pair(0), groups, bias, padding_mode)

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()
        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / s
        cout = cout.view(sz)

        return nconv, cout

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'p':  # Poisson
            mu = self.kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)

        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels) + 0.01)


# My modification is in this class
# Non-negativity enforcement class
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_pre', Parameter(weight.data))
        setattr(module, name, fn._pos(getattr(module, name + '_pre')))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs):
        # if module.training:
        #     weight = getattr(module, self.name)
        # del module._parameters[self.name]
        pos_weight = self._pos(getattr(module, self.name + '_pre'))
        setattr(module, self.name, pos_weight)
        # else:
        #     pass

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0], p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return


class NormCNN(nn.Module):

    def __init__(self, pos_fn=None, num_channels=4):
        super().__init__()

        self.pos_fn = pos_fn

        self.nconv1 = NConv2d(1, num_channels, (5, 5), pos_fn, 'p', padding=2)
        self.nconv2 = NConv2d(num_channels, num_channels, (5, 5), pos_fn, 'p', padding=2)
        self.nconv3 = NConv2d(num_channels, num_channels, (5, 5), pos_fn, 'p', padding=2)

        self.nconv4 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)
        self.nconv5 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)
        self.nconv6 = NConv2d(2 * num_channels, num_channels, (3, 3), pos_fn, 'p', padding=1)

        self.nconv7 = NConv2d(num_channels, 1, (1, 1), pos_fn, 'k')

    def pad_and_concat(self, x1, x2):
        h1, w1 = x1.size(2), x1.size(3)
        h2, w2 = x2.size(2), x2.size(3)
        dl = (w1 - w2) // 2
        dr = w1 - w2 - dl
        dt = (h1 - h2) // 2
        db = h1 - h2 - dt
        x2_padded = F.pad(x2, [dl, dr, dt, db], mode='replicate')
        return torch.cat((x1, x2_padded), dim=1)

    def forward(self, x0, c0):
        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)
        x1, c1 = self.nconv3(x1, c1)

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = retrieve_elements_from_indices(x1, idx)
        c1_ds /= 4

        x2_ds, c2_ds = self.nconv2(x1_ds, c1_ds)
        x2_ds, c2_ds = self.nconv3(x2_ds, c2_ds)

        # Downsample 2
        ds = 2
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)
        x2_dss = retrieve_elements_from_indices(x2_ds, idx)
        c2_dss /= 4

        x3_ds, c3_ds = self.nconv2(x2_dss, c2_dss)

        # Downsample 3
        ds = 2
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)
        x3_dss = retrieve_elements_from_indices(x3_ds, idx)
        c3_dss /= 4
        x4_ds, c4_ds = self.nconv2(x3_dss, c3_dss)

        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')
        # x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds, x4), 1), torch.cat((c3_ds, c4), 1))
        x34_ds, c34_ds = self.nconv4(self.pad_and_concat(x3_ds, x4), self.pad_and_concat(c3_ds, c4))

        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        # x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds, x34), 1), torch.cat((c2_ds, c34), 1))
        x23_ds, c23_ds = self.nconv5(self.pad_and_concat(x2_ds, x34), self.pad_and_concat(c2_ds, c34))

        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')
        # xout, cout = self.nconv6(torch.cat((x23, x1), 1), torch.cat((c23, c1), 1))
        xout, cout = self.nconv6(self.pad_and_concat(x23, x1), self.pad_and_concat(c23, c1))

        xout, cout = self.nconv7(xout, cout)

        return xout, cout


class PretrainedCNN(nn.Module):

    def __init__(self, pos_fn=None, num_channels=2):
        super().__init__()

        self.pos_fn = pos_fn

        self.navg1 = self.navg_layer((5, 5), 3, 1, num_channels, 'p', True)
        self.navg2 = self.navg_layer((5, 5), 3, num_channels, num_channels, 'p', True)
        self.navg3 = self.navg_layer((5, 5), 3, num_channels, num_channels, 'p', True)
        self.navg4 = self.navg_layer((1, 1), 3, num_channels, 1, 'p', True)

        self.navg34 = self.navg_layer((3, 3), 3, 2 * num_channels, num_channels, 'p', True)
        self.navg23 = self.navg_layer((3, 3), 3, 2 * num_channels, num_channels, 'p', True)
        self.navg12 = self.navg_layer((3, 3), 3, 2 * num_channels, num_channels, 'p', True)

        self.bias1 = nn.Parameter(torch.zeros(num_channels) + 0.01)
        self.bias2 = nn.Parameter(torch.zeros(num_channels) + 0.01)
        self.bias3 = nn.Parameter(torch.zeros(num_channels) + 0.01)
        self.bias4 = nn.Parameter(torch.zeros(1) + 0.01)

        self.bias34 = nn.Parameter(torch.zeros(num_channels) + 0.01)
        self.bias23 = nn.Parameter(torch.zeros(num_channels) + 0.01)
        self.bias12 = nn.Parameter(torch.zeros(num_channels) + 0.01)

    def forward(self, x0, c0):

        x1, c1 = self.navg_forward(self.navg1, c0, x0, self.bias1)

        x1, c1 = self.navg_forward(self.navg2, c1, x1, self.bias2)

        x1, c1 = self.navg_forward(self.navg3, c1, x1, self.bias3)

        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)
        x1_ds = torch.zeros(c1_ds.size()).cuda()
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])

        c1_ds /= 4

        x2_ds, c2_ds = self.navg_forward(self.navg2, c1_ds, x1_ds, self.bias2)

        x2_ds, c2_ds = self.navg_forward(self.navg3, c2_ds, x2_ds, self.bias3)

        ds = 2
        c2_dss, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)

        x2_dss = torch.zeros(c2_dss.size()).cuda()
        for i in range(x2_dss.size(0)):
            for j in range(x2_dss.size(1)):
                x2_dss[i, j, :, :] = x2_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_dss /= 4

        x3_ds, c3_ds = self.navg_forward(self.navg2, c2_dss, x2_dss, self.bias2)

        # x3_ds, c3_ds = self.navg_forward(self.navg3, c3_ds, x3_ds, self.bias3)

        ds = 2
        c3_dss, idx = F.max_pool2d(c3_ds, ds, ds, return_indices=True)

        x3_dss = torch.zeros(c3_dss.size()).cuda()
        for i in range(x3_dss.size(0)):
            for j in range(x3_dss.size(1)):
                x3_dss[i, j, :, :] = x3_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_dss /= 4

        x4_ds, c4_ds = self.navg_forward(self.navg2, c3_dss, x3_dss, self.bias2)

        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest')
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')

        x34_ds, c34_ds = self.navg_forward(self.navg34, torch.cat((c3_ds, c4), 1), torch.cat((x3_ds, x4), 1),
                                           self.bias34)

        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')

        x23_ds, c23_ds = self.navg_forward(self.navg23, torch.cat((c2_ds, c34), 1), torch.cat((x2_ds, x34), 1),
                                           self.bias23)

        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')

        xout, cout = self.navg_forward(self.navg12, torch.cat((c23, c1), 1), torch.cat((x23, x1), 1), self.bias12)

        xout, cout = self.navg_forward(self.navg4, cout, xout, self.bias4)

        return xout, cout

    def navg_forward(self, navg, c, x, b, eps=1e-20, restore=False):

        # Normalized Averaging
        ca = navg(c)
        xout = torch.div(navg(x * c), ca + eps)

        # Add bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(xout)
        xout = xout + b

        if restore:
            cm = (c == 0).float()
            xout = torch.mul(xout, cm) + torch.mul(1 - cm, x)

        # Propagate confidence
        # cout = torch.ne(ca, 0).float()
        cout = ca
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = navg.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)

        cout = cout / s

        cout = cout.view(sz)
        k = k.view(k_sz)

        return xout, cout

    def navg_layer(self, kernel_size, init_stdev=0.5, in_channels=1, out_channels=1, initalizer='x', pos=False,
                   groups=1):

        navg = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                         padding=(kernel_size[0] // 2, kernel_size[1] // 2), bias=False, groups=groups)

        weights = navg.weight

        if initalizer == 'x':  # Xavier
            torch.nn.init.xavier_uniform(weights)
        elif initalizer == 'k':
            torch.nn.init.kaiming_uniform(weights)
        elif initalizer == 'p':
            mu = kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.from_numpy(w).float().cuda()
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(out_channels, 1, 1, 1)
            w = w.repeat(1, in_channels, 1, 1)
            weights.data = w + torch.rand(w.shape).cuda()

        return navg


if __name__ == '__main__':
    ncnn = NormCNN(pos_fn='softplus')
    print(ncnn.__str__())
