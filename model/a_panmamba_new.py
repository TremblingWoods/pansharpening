import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .refine import Refine


class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        resi = self.relu_1(self.conv_1(x))
        out_1, out_2 = torch.chunk(resi, 2, dim=1)
        resi = torch.cat([self.norm(out_1), out_2], dim=1)
        resi = self.relu_2(self.conv_2(resi))
        # input = torch.cat([x,resi],dim=1)
        # out = self.conv_3(input)
        return x + resi


class ConvFuse(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvFuse, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.out = HinResBlock(out_size, out_size)

    def forward(self, ms, pan):
        out = self.conv1(torch.cat([ms, pan], dim=1))
        return out + self.out(out)


class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = 32
        self.base_filter = base_filter
        self.pan_encoder = nn.Sequential(nn.Conv2d(1, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                         HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.ms_encoder = nn.Sequential(nn.Conv2d(4, base_filter, 3, 1, 1), HinResBlock(base_filter, base_filter),
                                        HinResBlock(base_filter, base_filter), HinResBlock(base_filter, base_filter))
        self.deep_fusion1 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion2 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion3 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion4 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion5 = ConvFuse(base_filter * 2, base_filter)
        self.pan_feature_extraction = nn.Sequential(*[HinResBlock(base_filter, base_filter) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[HinResBlock(base_filter, base_filter) for i in range(8)])
        self.output = Refine(base_filter, 4)

    def forward(self, ms, _, pan):
        ms_bic = F.interpolate(ms, scale_factor=4)
        ms_f = self.ms_encoder(ms_bic)
        b, c, h, w = ms_f.shape
        pan_f = self.pan_encoder(pan)
        ms_f = self.ms_feature_extraction(ms_f)
        pan_f = self.pan_feature_extraction(pan_f)
        ms_f = self.deep_fusion1(ms_f, pan_f)
        ms_f = self.deep_fusion2(ms_f, pan_f)
        ms_f = self.deep_fusion3(ms_f, pan_f)
        ms_f = self.deep_fusion4(ms_f, pan_f)
        ms_f = self.deep_fusion5(ms_f, pan_f)
        hrms = self.output(ms_f) + ms_bic
        return hrms
