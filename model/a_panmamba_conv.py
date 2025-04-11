import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.refine import Refine

#对单个输入信息进行特征增强, 单个信息残差链接特征增强
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


#特征融合，先把ms和pan进行通道拼接，然后用卷积提取整体信息，最后在调用残差链接增强输出信息
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

        #上采样
        ms_bic = F.interpolate(ms, scale_factor=4)
        #conv编码
        ms_f = self.ms_encoder(ms_bic) #(B, 32, H, W)
        b, c, h, w = ms_f.shape
        #conv编码
        pan_f = self.pan_encoder(pan) #(B, 32, H, W)

        #分别特征提取
        ms_f = self.ms_feature_extraction(ms_f) #通过 8 次残差操作逐步增强特征表达能力
        pan_f = self.pan_feature_extraction(pan_f)

        #不断把pan的信息融合到ms里，其中pan在上一步之后就不变了
        ms_f = self.deep_fusion1(ms_f, pan_f)
        ms_f = self.deep_fusion2(ms_f, pan_f)
        ms_f = self.deep_fusion3(ms_f, pan_f)
        ms_f = self.deep_fusion4(ms_f, pan_f)
        ms_f = self.deep_fusion5(ms_f, pan_f)

        #这一步当作数据增强的暂时先不动：
        hrms = self.output(ms_f) + ms_bic
        return hrms

#wv2 200eopch
# ################## reference comparision #######################
# metrics:    PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q
# deep    [41.3444, 0.968, 0.0236, 0.9897, 0.9729, 0.7688]
# ################## reference comparision #######################
# ################## no reference comparision ####################
# metrics:    D_lamda,  D_s,    QNR
# deep     [0.065, 0.12, 0.8236]
# ################## no reference comparision ####################

if __name__ == "__main__":
    # l_ms:torch.Size([4, 4, 32, 32])
    # x_pan:torch.Size([4, 1, 128, 128])
    # b_ms: torch.Size([4, 4, 128, 128])
    lms = torch.randn((4, 4, 32, 32))
    pan = torch.randn((4, 1, 128, 128))
    bms = torch.randn((4, 4, 128, 128))
    net = Net()
    out = net(lms,bms,pan)
    print(out.size())