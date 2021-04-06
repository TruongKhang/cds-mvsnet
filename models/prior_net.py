import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from models.nconv import NormCNN

path_to_dir = os.path.abspath(os.path.dirname(__file__))


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # m.weight.data.normal_(0, 1e-3)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        # m.weight.data.normal_(0, 1e-3)
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.ReLU(inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers


def convt_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights:
    for m in layers.modules():
        init_weights(m)

    return layers


class PriorNet(nn.Module):
    def __init__(self, bn=False):
        super(PriorNet, self).__init__()

        self.bn = bn
        # self.feature_root = feature_root
        #self.conf_estimator = UNetSP(2, 1, m=4)

        self.nconv = NormCNN(pos_fn='softplus')

        self.depth_refinement = nn.Sequential(UNet(1, 32, 32, 1, batchnorms=self.bn),
                                              ResidualBlock(32),
                                              ResidualBlock(32),
                                              ResidualBlock(32),
                                              nn.Conv2d(32, 1, 1))
        self.conf_refinement = UNetSP(1, 1)

    def forward(self, depths, confs):
        # (img has shape: (batch_size, h, w)) (grayscale)
        # (sparse has shape: (batch_size, h, w))

        num_views = depths.size(1)

        #mean_depth = torch.mean(depths, dim=1)

        prop_depths, prop_confs = [], []
        for i in range(num_views):
            #diff_depth = torch.abs(depths[:, i, ...] - mean_depth)
            #est_conf = self.conf_estimator(torch.cat((confs[:, i, ...], diff_depth), dim=1))
            #dout, cout = self.nconv(depths[:, i, ...], est_conf)

            dout, cout = self.nconv(depths[:, i, ...], confs[:, i, ...])
            prop_depths.append(dout)
            prop_confs.append(cout)
        prop_depths = torch.stack(prop_depths, dim=1)
        prop_confs = torch.stack(prop_confs, dim=1)

        normed_prop_conf = torch.sum(prop_confs, dim=1)
        avg_depth = torch.sum(prop_depths * prop_confs, dim=1) / (normed_prop_conf + 1e-16)
        avg_conf = normed_prop_conf / num_views

        refined_depth = self.depth_refinement(avg_depth)
        refined_conf = self.conf_refinement(avg_conf)

        return refined_depth, refined_conf


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, bn=False):
        super(ResidualBlock, self).__init__()
        layers = [nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)]
        if bn:
            layers.append(nn.BatchNorm2d(channels, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(channels, affine=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        return out


class UNetSP(nn.Module):
    def __init__(self, n_channels, n_classes, m=8):
        super().__init__()
        self.inc = inconv(n_channels, m*4)
        self.down1 = down(m*4, m*4)
        self.down2 = down(m*4, m*8)
        self.down3 = down(m*8, m*8)
        #self.down4 = down(128, 128)
        #self.up1 = up(256, 64)
        self.up2 = up(m*8+m*8, m*8)
        self.up3 = up(m*8+m*4, m*4)
        self.up4 = up(m*4+m*4, m*4)
        self.outc = outconv(m*4, n_classes)

    def forward(self, x):
        x1 = self.inc(x) #32
        x2 = self.down1(x1) #64
        x3 = self.down2(x2) #64
        x4 = self.down3(x3) #128
        #x5 = self.down4(x4) #128
        #x = self.up1(x5, x4) #128
        x = self.up2(x4, x3) #128
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.softplus(x)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    Basic UNet building block, calling itself recursively.
    Note that the final output does not have a ReLU applied.
    """

    def __init__(self, Cin, F, Cout, depth, batchnorms=True):
        super().__init__()
        self.F = F
        self.depth = depth

        if batchnorms:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(Cout),
            )
        else:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
            )

        if depth > 1:
            self.process = UNet(F, 2 * F, F, depth - 1, batchnorms=batchnorms)
        else:
            if batchnorms:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                )
            else:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):
        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = F.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        result = self.post(torch.cat((features, upsampled), dim=1))

        return result


if __name__ == '__main__':
    # imgs = torch.rand((1, 3, 480, 752)).float().cuda()
    # sdmaps = torch.rand((1, 1, 480, 752)).float().cuda()
    # cfds_0 = torch.rand((1, 1, 480, 752)).float().cuda()
    # Es = torch.eye(4).unsqueeze(0).cuda()
    # Ks = torch.rand((1, 3, 3)).float().cuda()
    # m = DepthCompletionNet()
    # print(m)
    # m = m.cuda()
    # prev_state = torch.zeros(1, 1, 480, 752).float().cuda(), torch.zeros(1, 1, 480, 752).float().cuda()
    # depth, cfd, init_depth, init_cfd = m((imgs, sdmaps), prev_state=prev_state)
    # print(depth.size(), cfd.size(), init_depth.size(), init_cfd.size())
    data = torch.rand((1, 3, 1, 64, 64)), torch.rand((1, 3, 1, 64, 64))
    # model = UNet(32, 32, 32, depth=3, batchnorms=False) #
    model = DepthRefinementNet()
    print(model.__str__())
    out = model(data[0], data[1])
    print(out[0].size(), out[1].size())






