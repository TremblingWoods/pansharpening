# -*- coding = utf-8 -*-
'''
@Time: 2023/7/19 17:36
@Author: Caoke
@File: pannet_ht.py
@Software: PyCharm2023
Description: 2000 epoch, decay 1000 x0.1, batch_size = 128, learning_rate = 1e-2, patch_size = 33, MSE
'''


import os
import torch
import torch.nn as nn
import torch.optim as optim
from model.base_net import *
from torchvision.transforms import *
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args):
        super(Net, self).__init__()  # 调用父类nn.Module对参数进行初始化
        out_channels = 4
        base_filter = 64
        num_channels = 5
        self.args = args
        n_resblocks = 4  # 使用6个res块叠加

        res_block_s1 = [
            ConvBlock(num_channels, 32, 9, 1, 4, activation='prelu', norm=None, bias=False)
            # ConvBlock(48, 32, 3, 1, 1, activation=None, norm=None, bias=False),
        ]

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.deconv = nn.ConvTranspose2d(in_channels=num_channels, out_channels=num_channels, kernel_size=8, stride=4,
                                         padding=2, bias=True)

        for i in range(n_resblocks):
            res_block_s1.append(ResnetBlock(32, 3, 1, 1, 0.2, activation='prelu', norm=None))
        res_block_s1.append(ConvBlock(32, out_channels, 3, 1, 1, activation='prelu', norm=None, bias=False))
        self.res_block_s1 = nn.Sequential(*res_block_s1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)  # 标准化xavier初始化
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.xavier_uniform_(m.weight, gain=1)  # 标准化xavier初始化
                if m.bias is not None:
                    m.bias.data.zero_()

    #  WV3_data，WV2_data，GF2_data的四个图像相同
    # （1）n是样本的意思，此时batch-size=64，修改batch-size=4后第一维度变为4
    # （2）c是通道的意思，上面有说，即像素点取值形式
    # （3）h和w很好理解，就是所谓的图像像素的高度，w是图像像素的宽度。
    #  ms_image:tensor(64,4,128,128)
    #  b_ms=bms_image:tensor(64,4,128,128)
    #  l_ms=lms_image:tensor(64,4,32,32)
    #  x_pan=pan_image:tensor(64,1,128,128)

    # num_channels在pnn中设置为7因为是4+1+1+1 （4通道为RGB+近红外）
    # 此时仅使用了4+1因此修改为5，输入通道

    def forward(self, l_ms, b_ms, x_pan):  # lms_image, bms_image, pan_image
        # 高频分量使用转置卷积得到
        highpass_pan = x_pan - F.interpolate(F.interpolate(x_pan, scale_factor=1 / 4, mode='bicubic'), scale_factor=4,
                                             mode='bicubic')
        highpass_lms = F.interpolate(l_ms - F.interpolate(b_ms, scale_factor=1 / 4, mode='bicubic'), scale_factor=4,
                                     mode='bicubic')
        x_f = torch.cat([highpass_lms, highpass_pan], 1)
        x_f = self.res_block_s1(x_f)
        x_f = torch.add(x_f, b_ms)
        return x_f
