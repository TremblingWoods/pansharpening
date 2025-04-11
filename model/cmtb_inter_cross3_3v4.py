# -*- coding = utf-8 -*-
# @Time: 2023/8/10 17:36
# @Author: Hutao
# @File: CMTB_inter_cross3_3.py
# @Software: PyCharm2023

import torch
from torch import nn, einsum
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules import InvertibleConv1x1
import torch.nn.functional as F
from model.base_net import *
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Crossmodality(nn.Module):
    def __init__(self):
        super(Crossmodality, self).__init__()
        # 在构造函数中定义神经网络的层次结构
        num_channels = 32
        self.conv_1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=True)
        self.act_1 = nn.GELU()
        self.conv_2 = nn.Conv2d(num_channels*2, num_channels, kernel_size=1, padding=0, bias=True)
        self.act_2 = nn.GELU()
        self.conv_3 = nn.Conv2d(num_channels, 1, kernel_size=3, padding=1, bias=True)
        self.conv_4 = nn.Conv2d(num_channels, 1, kernel_size=1, padding=0, bias=True)
        self.IN = nn.InstanceNorm2d(num_features=num_channels)
    def forward(self, pan, ms):
        # 在 forward 方法中使用输入参数进行计算
        lms_conv = self.conv_1(ms)
        lms_act = self.act_1(lms_conv)
        x_h1_cat = torch.cat((lms_conv, pan), dim=1)
        x_h = self.act_2(self.conv_2(x_h1_cat))
        beta = self.conv_3(x_h)
        gamma = self.conv_4(x_h)
        # 计算通道维度上的均值
        mean = torch.mean(ms, dim=(0, 2, 3))
        # 计算通道维度上的方差
        variance = torch.var(ms, dim=(0, 2, 3))
        # 将 gamma 和 beta 进行广播，使其与张量 x_h 具有相同的形状
        gamma_broadcasted = gamma.expand_as(x_h)
        beta_broadcasted = beta.expand_as(x_h)
        # 将 variance 和 mean 进行扩展，以适应 O 的形状
        variance_expanded = variance.view(1, 32, 1, 1)
        mean_expanded = mean.view(1, 32, 1, 1)
        pan = self.IN(pan)
        updated_pan = pan * (variance_expanded + gamma_broadcasted) + mean_expanded + beta_broadcasted
        # print('Crossmodality done')
        return updated_pan

class AttentionTransformerBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, attn_window_size=None, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, stride=False, relative_pos_embedding=False, deform_relative_pos_embedding=False,
                 deformable=False, learnable=True, restart_regression=None, img_size=(1,1)):
        super().__init__()
        # if shift:
        #     self.shift_size = window_size // 2
        # else:
        #     self.shift_size = 0
        self.ws = window_size
        self.dim = dim
        self.shift_size = shift_size
        self.img_size = img_size
        self.mlp_ratio = mlp_ratio
        self.norm1_pan = norm_layer(dim, data_format="channels_first")
        # self.norm1_lms = norm_layer(dim, data_format="channels_first")
        self.learnable = learnable
        self.restart_regression = restart_regression
        # deformable = deformable > 0
        self.deformable = deformable
        self.pos = nn.Conv2d(dim, dim, window_size // 2 * 2 + 1, 1, window_size // 2, groups=dim, bias=True)
        #关键点
        self.attn = DeformableWindowAttention(
                dim, num_heads=num_heads, window_size=window_size, attn_window_size=attn_window_size, shift_size=self.shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=deform_relative_pos_embedding, img_size=img_size, learnable=learnable, restart_regression=restart_regression,)
        # self.local = nn.Conv2d(dim, dim, window_size//2*2+1, 1, window_size//2, groups=dim, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, stride=stride)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer,
                       drop=drop, stride=stride)
        self.norm3 = norm_layer(dim, data_format="channels_first")
        self.norm3_2 = norm_layer(dim, data_format="channels_first")
        # print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, out_dim, stride, shift_size, num_heads))

    def forward(self, pan, lms):

        pan = pan + self.pos(pan)
        lms = lms + self.pos(lms)

        attn_out = self.attn(self.norm1_pan(pan),self.norm1_pan(lms))

        # print(attn_out)
        pan = pan + self.drop_path(attn_out[0])
        pan_atten = pan + self.drop_path(self.mlp(self.norm3(pan)))
        lms = lms+ self.drop_path_2(attn_out[1])
        lms_atten = lms + self.drop_path_2(self.mlp(self.norm3_2(lms)))

        return pan_atten ,lms_atten


class DeformableWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., relative_pos_embedding=False, learnable=True, restart_regression=None,
                 attn_window_size=None, shift_size=0, img_size=(1, 1)):
        super().__init__()
        self.img_size = img_size
        self.num_heads = num_heads
        self.dim = dim
        self.relative_pos_embedding = True
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.attn_ws = attn_window_size or self.ws
        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * 2, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        self.left_size = self.img_size
        if min(self.img_size) <= self.ws:
            self.shift_size = 0


        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            # print(window_size,attn_window_size)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + window_size - 1) * (window_size + window_size - 1),
                            num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.attn_ws)
            coords_w = torch.arange(self.attn_ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.attn_ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            # print('The relative_pos_embedding is used')

    def forward(self, x, lms):
        b, _, h, w = x.shape

        shortcut = x
        shortcut_lms = lms

        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left
        expand_h, expand_w = h + padding_top + padding_down, w + padding_left + padding_right
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2,
                                                                                                       1).unsqueeze(
            0)  # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_h - 1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,1).reshape(1, 2,window_num_h,self.ws,window_num_w,self.ws)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                               1).reshape(
            1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)
        # base_coords = window_reference+window_coords
        base_coords = image_reference


        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            coords = base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)
        if self.learnable:
            sampling_offsets = self.sampling_offsets(x)
            sampling_offsets = sampling_offsets.reshape(b * self.num_heads, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (expand_w // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (expand_h // self.ws)
            # print("sampling_offsets",sampling_offsets.shape)

            sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(b * self.num_heads, 2, window_num_h, window_num_w)
            # print("sampling_scales",sampling_scales.shape)

            coords = coords + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None,
                                                                                        :, None]
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b * self.num_heads, self.attn_ws * window_num_h,
                                                                 self.attn_ws * window_num_w, 2)

        qkv = self.qkv(shortcut).reshape(b, 3, self.num_heads, self.dim // self.num_heads, h, w).transpose(1,
                                                                                                           0).reshape(
            3 * b * self.num_heads, self.dim // self.num_heads, h, w)

        qkv_lms = self.qkv(shortcut_lms).reshape(b, 3, self.num_heads, self.dim // self.num_heads, h, w).transpose(1,
                                                                                                                   0).reshape(
            3 * b * self.num_heads, self.dim // self.num_heads, h, w)
        # if self.shift_size > 0:
        qkv = torch.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(3,
                                                                                                             b * self.num_heads,
                                                                                                             self.dim // self.num_heads,
                                                                                                             h + padding_td,
                                                                                                             w + padding_lr)
        qkv_lms = torch.nn.functional.pad(qkv_lms, (padding_left, padding_right, padding_top, padding_down)).reshape(3,
                                                                                                             b * self.num_heads,
                                                                                                             self.dim // self.num_heads,
                                                                                                             h + padding_td,
                                                                                                             w + padding_lr)
        q_pan, k, v = qkv[0], qkv[1], qkv[2]
        q, k_lms, v_lms = qkv_lms[0], qkv_lms[1], qkv_lms[2]
        k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
        v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)
        # print("QKV setting done!")
        # select_mask = q.not_equal(0).sum(dim=(0,1), keepdim=True).not_equal(0)
        q_pan = q_pan.reshape(b, self.num_heads, self.dim // self.num_heads, window_num_h, self.ws, window_num_w,
                      self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
                                                                    self.ws * self.ws, self.dim // self.num_heads)
        q = q.reshape(b, self.num_heads, self.dim // self.num_heads, window_num_h, self.ws, window_num_w,
                      self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
                                                                    self.ws * self.ws, self.dim // self.num_heads)
        k = k_selected.reshape(b, self.num_heads, self.dim // self.num_heads, window_num_h, self.attn_ws, window_num_w,
                               self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w,
                                                                                  self.num_heads,
                                                                                  self.attn_ws * self.attn_ws,
                                                                                  self.dim // self.num_heads)
        v = v_selected.reshape(b, self.num_heads, self.dim // self.num_heads, window_num_h, self.attn_ws, window_num_w,
                               self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w,
                                                                                  self.num_heads,
                                                                                  self.attn_ws * self.attn_ws,
                                                                                  self.dim // self.num_heads)

        dots_lms = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots_lms += relative_position_bias.unsqueeze(0)

        attn_lms = dots_lms.softmax(dim=-1)
        out_lms = attn_lms @ v

        out_lms = rearrange(out_lms, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                        hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
        # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out_lms = out_lms[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out = self.proj(out_lms)
        out_lms = self.proj_drop(out_lms)

        dots_pan = (q_pan @ k.transpose(-2, -1)) * self.scale
        # dots = dots + mask
        #       b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.attn_ws*self.attn_ws, self.dim//self.num_heads

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots_pan += relative_position_bias.unsqueeze(0)

        attn_pan = dots_pan.softmax(dim=-1)
        out_pan = attn_pan @ v

        out_pan = rearrange(out_pan, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                            hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
        # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out_pan = out_pan[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out_pan = self.proj(out_pan)
        out_pan = self.proj_drop(out_pan)

        return out_pan,out_lms



class HinResBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.2, use_HIN=True):
        super(HinResBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        # self.conv_3 = nn.Conv2d(in_size+in_size,out_size,3,1,1)
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
        return self.identity(x)+resi
class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.process = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x
class Refine(nn.Module):

    def __init__(self,n_feat,out_channel):
        super(Refine, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             # CALayer(n_feat,4),
             # CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out
class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=LayerNorm, img_size=(0,0)):
        super().__init__()
        self.input_resolution = img_size
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim, data_format="channels_first")
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input dim={self.dim}, out dim={self.out_dim}"

    def flops(self) -> float:
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.out_dim
        return flops

def upsample(x, h, w):
    return F.interpolate(x, size=[h, w], mode='bicubic', align_corners=True)

class Net(nn.Module):
    def __init__(self, num_channels=4,channels=None,base_filter=None,args=None):
        super(Net, self).__init__()
        channels = base_filter = 32
        # self.encoder1 = HinResBlock()
        self.msconv = nn.Conv2d(4, channels, 3, 1, 1)  # conv for ms
        self.relu1 = nn.LeakyReLU()
        self.panconv = nn.Conv2d(1, channels, 3, 1, 1)
        self.relu2 = nn.LeakyReLU()
        self.encoderlms1 = nn.Sequential(HinResBlock(channels,channels),HinResBlock(channels,channels),HinResBlock(channels,channels))
        self.encoderpan1 = nn.Sequential(HinResBlock(channels,channels),HinResBlock(channels,channels),HinResBlock(channels,channels))
        self.crossmodel1 = Crossmodality()
        self.crossmodel2 = Crossmodality()
        self.CMTB1 = AttentionTransformerBlock(dim=channels, out_dim=channels, num_heads=8, window_size=8,
                                                    attn_window_size=None, shift_size=0, mlp_ratio=2., qkv_bias=True,
                                                    qk_scale=None, drop=0.1, attn_drop=0.1,
                                                    drop_path=0.1, act_layer=nn.GELU, norm_layer=LayerNorm, stride=False,
                                                    relative_pos_embedding=False, deform_relative_pos_embedding=False,
                                                    deformable=False, learnable=True, restart_regression=None,
                                                    img_size=(128, 128))
        self.patition1 = PatchMerging(channels,2*channels)
        self.patition1_2 = PatchMerging(channels, 2*channels)
        self.CMTB2 = AttentionTransformerBlock(dim=2*channels, out_dim=2*channels, num_heads=8, window_size=8,
                                                    attn_window_size=None, shift_size=0, mlp_ratio=2., qkv_bias=True,
                                                    qk_scale=None, drop=0., attn_drop=0.1,
                                                    drop_path=0.1, act_layer=nn.GELU, norm_layer=LayerNorm, stride=False,
                                                    relative_pos_embedding=False, deform_relative_pos_embedding=False,
                                                    deformable=False, learnable=True, restart_regression=None,
                                                    img_size=(64, 64))
        self.patition2 = PatchMerging(2*channels,4*channels)
        self.patition2_2 = PatchMerging(2*channels, 4*channels)

        self.CMTB3 = AttentionTransformerBlock(dim=4*channels, out_dim=4*channels, num_heads=8, window_size=8,
                                                    attn_window_size=None, shift_size=0, mlp_ratio=2., qkv_bias=True,
                                                    qk_scale=None, drop=0., attn_drop=0.0,
                                                    drop_path=0.1, act_layer=nn.GELU, norm_layer=LayerNorm, stride=False,
                                                    relative_pos_embedding=False, deform_relative_pos_embedding=False,
                                                    deformable=False, learnable=True, restart_regression=None,
                                                    img_size=(32, 32))

        self.encoderpan2 = HinResBlock(channels, channels)
        self.encoderlms2 = HinResBlock(channels, channels)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self.up1pan = nn.Sequential(nn.Conv2d(4*channels,8*channels,1,1,0),nn.PixelShuffle(2))
        self.up1ms = nn.Sequential(nn.Conv2d(4*channels,8*channels,1,1,0),nn.PixelShuffle(2))
        self.up2pan = nn.Sequential(nn.Conv2d(4*channels,4*channels,1,1,0),nn.PixelShuffle(2))
        self.up2ms = nn.Sequential(nn.Conv2d(4*channels,4*channels,1,1,0),nn.PixelShuffle(2))
        self.fuse = nn.Sequential(InvBlock(HinResBlock, 2*channels, channels),nn.Conv2d(2*channels,channels,1,1,0))
        self.relu = nn.LeakyReLU()
        self.conv_out = Refine(channels, 4)


    def forward(self, ms,_,pan):
        # ms  - low-resolution multi-spectral image [N,C,h,w]
        # pan - high-resolution panchromatic image [N,1,H,W]
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape
        seq = []
        mHR = upsample(ms, M, N) # size 4
        ms = mHR
        msf = self.relu1(self.msconv(ms)) #(4->channels) # size 4
        panf = self.relu2(self.panconv(pan)) #(1->channels)
        msf  = self.encoderlms1(msf)
        panf = self.encoderpan1(panf)
        msf = self.encoderlms2(msf)
        panf = self.encoderpan2(panf)

        panf = self.crossmodel2(panf,msf)

        pan_attn , lms_attn = self.CMTB1(panf, msf)
        pan_1,ms_1 = pan_attn,lms_attn #size 4

        pan_attn = self.patition1(pan_attn)
        lms_attn = self.patition1_2(lms_attn)
        pan_attn, lms_attn = self.CMTB2(pan_attn, lms_attn)

        pan_2,ms_2 = pan_attn,lms_attn #size 2
        pan_attn = self.patition2(pan_attn)
        lms_attn = self.patition2_2(lms_attn)

        panf, msf = self.CMTB3(pan_attn, lms_attn)

        pan_3,ms_3 = panf,msf #size1
        pan_2_2,ms_2_2 = self.up1pan(pan_3),self.up1ms(ms_3)
        pan_1_2,ms_1_2 = self.up2pan(torch.cat([pan_2,pan_2_2],dim=1)),self.up2ms(torch.cat([ms_2,ms_2_2],dim=1))
        panf,msf =pan_1+pan_1_2,ms_1_2+ms_1
        # panf = self.pixel_shuffle(panf)
        # msf = self.pixel_shuffle(msf)
        HR = self.conv_out(self.relu(self.fuse(torch.cat((panf,msf), dim=1))))
        return HR +ms

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


