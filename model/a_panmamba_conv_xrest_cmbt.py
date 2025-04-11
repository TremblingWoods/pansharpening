import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.refine import Refine
from torch import einsum
from pdb import set_trace as stx
import numbers
from einops import rearrange
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modules import InvertibleConv1x1
from model.base_net import *
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class OCAB(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias):
        super(OCAB, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size = window_size,
            rel_size = window_size + (self.overlap_win_size - window_size),
            dim_head = self.dim_head
        )
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)
        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = self.window_size, p2 = self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c = self.inner_dim), (ks, vs))
        # print(f'qs.shape:{qs.shape}, ks.shape:{ks.shape}, vs.shape:{vs.shape}')
        #split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', head = self.num_spatial_heads), (qs, ks, vs))
        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn = spatial_attn.softmax(dim=-1)
        out = (spatial_attn @ vs)
        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', head = self.num_spatial_heads,
                        h = h // self.window_size, w = w // self.window_size, p1 = self.window_size, p2 = self.window_size)
        # merge spatial and channel
        out = self.project_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, num_spatial_heads,
                 spatial_dim_head, ffn_expansion_factor, bias, LayerNorm_type,num_heads,attn_window_size,
                 shift_size,qkv_bias,qk_scale,attn_drop,drop,deform_relative_pos_embedding,img_size,
                 learnable,restart_regression):
        super(TransformerBlock, self).__init__()

        self.channel_attn = ChannelAttention(dim, num_channel_heads, bias)
        self.spatial_attn = OCAB(dim, window_size, overlap_ratio, num_spatial_heads, spatial_dim_head, bias)

        self.attn = DeformableWindowAttention(
            dim, num_heads=num_heads, window_size=window_size, attn_window_size=attn_window_size,
            shift_size= shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
            relative_pos_embedding=deform_relative_pos_embedding, img_size=img_size,
            learnable=learnable, restart_regression=restart_regression)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)


    def forward(self, x):

        #TSAB
        x = x + self.channel_attn(self.norm1(x))#TSA
        x = x + self.channel_ffn(self.norm2(x)) #FFN

        x = x + self.spatial_attn(self.norm3(x))#SSA
        x = x + self.spatial_ffn(self.norm4(x)) #FFN

        #SSAB
        ret,_ = self.attn(self.norm3(x),self.norm3(x)) #SSA 替换成可变窗口
        x = x + ret
        x = x + self.spatial_ffn(self.norm4(x)) #FFN

        return x


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
        window_size = 8
        overlap_ratio = 0.5
        channel_heads = 1
        spatial_heads = 2
        spatial_dim_head = 16
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        num_heads = 8
        attn_window_size = None
        shift_size = 0
        qkv_bias = True
        qk_scale = None
        attn_drop = 0.1
        drop = 0.1
        deform_relative_pos_embedding = False
        img_size = (128, 128)
        learnable = True
        restart_regression = None
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.out = TransformerBlock(dim=out_size, window_size=window_size, overlap_ratio=overlap_ratio,
                         num_channel_heads=channel_heads,
                         num_spatial_heads=spatial_heads, spatial_dim_head=spatial_dim_head,
                         ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type,num_heads = num_heads,attn_window_size =attn_window_size ,
                         shift_size =shift_size ,qkv_bias = qkv_bias,qk_scale = qk_scale ,attn_drop = attn_drop,
                         drop=drop,deform_relative_pos_embedding=deform_relative_pos_embedding,img_size=img_size,
                         learnable=learnable,restart_regression=restart_regression)
    def forward(self, ms, pan):
        out = self.conv1(torch.cat([ms, pan], dim=1))
        return out + self.out(out)



class Net(nn.Module):
    def __init__(self, num_channels=None, base_filter=None, args=None):
        super(Net, self).__init__()
        base_filter = dim = 48
        window_size = 8
        overlap_ratio = 0.5
        channel_heads = 1
        spatial_heads = 2
        spatial_dim_head = 16
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        num_heads=8
        attn_window_size=None
        shift_size=0
        qkv_bias=True
        qk_scale=None
        attn_drop=0.1
        drop=0.1
        deform_relative_pos_embedding=False
        img_size=(128, 128)
        learnable=True
        restart_regression=None
        self.base_filter = base_filter
        self.pan_encoder = nn.Sequential(nn.Conv2d(1, base_filter, 3, 1, 1),
                        TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                         num_channel_heads=channel_heads,
                         num_spatial_heads=spatial_heads, spatial_dim_head=spatial_dim_head,
                         ffn_expansion_factor=ffn_expansion_factor,
                         bias=bias, LayerNorm_type=LayerNorm_type,num_heads = num_heads,attn_window_size =attn_window_size ,
                         shift_size =shift_size ,qkv_bias = qkv_bias,qk_scale = qk_scale ,attn_drop = attn_drop,
                         drop=drop,deform_relative_pos_embedding=deform_relative_pos_embedding,img_size=img_size,
                         learnable=learnable,restart_regression=restart_regression),
                                         TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                                                          num_channel_heads=channel_heads,
                                                          num_spatial_heads=spatial_heads,
                                                          spatial_dim_head=spatial_dim_head,
                                                          ffn_expansion_factor=ffn_expansion_factor,
                                                          bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                                                          attn_window_size=attn_window_size,
                                                          shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          attn_drop=attn_drop,
                                                          drop=drop,
                                                          deform_relative_pos_embedding=deform_relative_pos_embedding,
                                                          img_size=img_size,
                                                          learnable=learnable, restart_regression=restart_regression),
                                         TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                                                          num_channel_heads=channel_heads,
                                                          num_spatial_heads=spatial_heads,
                                                          spatial_dim_head=spatial_dim_head,
                                                          ffn_expansion_factor=ffn_expansion_factor,
                                                          bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                                                          attn_window_size=attn_window_size,
                                                          shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          attn_drop=attn_drop,
                                                          drop=drop,
                                                          deform_relative_pos_embedding=deform_relative_pos_embedding,
                                                          img_size=img_size,
                                                          learnable=learnable, restart_regression=restart_regression))

        self.ms_encoder = nn.Sequential(nn.Conv2d(4, base_filter, 3, 1, 1),
                                        TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                                                         num_channel_heads=channel_heads,
                                                         num_spatial_heads=spatial_heads,
                                                         spatial_dim_head=spatial_dim_head,
                                                         ffn_expansion_factor=ffn_expansion_factor,
                                                         bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                                                         attn_window_size=attn_window_size,
                                                         shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         attn_drop=attn_drop,
                                                         drop=drop,
                                                         deform_relative_pos_embedding=deform_relative_pos_embedding,
                                                         img_size=img_size,
                                                         learnable=learnable, restart_regression=restart_regression),
                                        TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                                                         num_channel_heads=channel_heads,
                                                         num_spatial_heads=spatial_heads,
                                                         spatial_dim_head=spatial_dim_head,
                                                         ffn_expansion_factor=ffn_expansion_factor,
                                                         bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                                                         attn_window_size=attn_window_size,
                                                         shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         attn_drop=attn_drop,
                                                         drop=drop,
                                                         deform_relative_pos_embedding=deform_relative_pos_embedding,
                                                         img_size=img_size,
                                                         learnable=learnable, restart_regression=restart_regression),
                                        TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                                                         num_channel_heads=channel_heads,
                                                         num_spatial_heads=spatial_heads,
                                                         spatial_dim_head=spatial_dim_head,
                                                         ffn_expansion_factor=ffn_expansion_factor,
                                                         bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                                                         attn_window_size=attn_window_size,
                                                         shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         attn_drop=attn_drop,
                                                         drop=drop,
                                                         deform_relative_pos_embedding=deform_relative_pos_embedding,
                                                         img_size=img_size,
                                                         learnable=learnable, restart_regression=restart_regression))


        self.deep_fusion1 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion2 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion3 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion4 = ConvFuse(base_filter * 2, base_filter)
        self.deep_fusion5 = ConvFuse(base_filter * 2, base_filter)
        self.pan_feature_extraction = nn.Sequential(*[
            TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                             num_channel_heads=channel_heads,
                             num_spatial_heads=spatial_heads, spatial_dim_head=spatial_dim_head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                             attn_window_size=attn_window_size,
                             shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                             drop=drop, deform_relative_pos_embedding=deform_relative_pos_embedding, img_size=img_size,
                             learnable=learnable, restart_regression=restart_regression) for i in range(8)])
        self.ms_feature_extraction = nn.Sequential(*[
            TransformerBlock(dim=dim, window_size=window_size, overlap_ratio=overlap_ratio,
                             num_channel_heads=channel_heads,
                             num_spatial_heads=spatial_heads, spatial_dim_head=spatial_dim_head,
                             ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type, num_heads=num_heads,
                             attn_window_size=attn_window_size,
                             shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                             drop=drop, deform_relative_pos_embedding=deform_relative_pos_embedding, img_size=img_size,
                             learnable=learnable, restart_regression=restart_regression)for i in range(8)])
        self.output = Refine(base_filter, 4)

    def forward(self, ms, _, pan):
        # 输入维度说明：
        # ms: 低分辨率多光谱图像 (batch_size, 4, H_lr, W_lr)，如 (B,4,32,32)
        # pan: 高分辨率全色图像 (batch_size, 1, H_hr, W_hr)，如 (B,1,128,128)
        # _: 可能为占位符（如低分辨率PAN），实际未使用
        # 上采样：低分辨率MS→高分辨率尺寸（对齐PAN）
        ms_bic = F.interpolate(ms, scale_factor=4)  # 输出维度 (B,4,128,128)

        # MS特征编码：4通道→32通道
        ms_f = self.ms_encoder(ms_bic)  # 维度 (B,32,128,128)

        # PAN特征编码：1通道→32通道
        pan_f = self.pan_encoder(pan)  # 维度 (B,32,128,128)

        # 特征提取：各自通过8层残差块（保持维度）
        ms_f = self.ms_feature_extraction(ms_f)  # 保持 (B,32,128,128)
        pan_f = self.pan_feature_extraction(pan_f)

        # 多级融合：将PAN特征逐步融合到MS特征
        ms_f = self.deep_fusion1(ms_f, pan_f)
        ms_f = self.deep_fusion2(ms_f, pan_f)
        ms_f = self.deep_fusion3(ms_f, pan_f)
        ms_f = self.deep_fusion4(ms_f, pan_f)
        ms_f = self.deep_fusion5(ms_f, pan_f)

        # 输出层：32通道→4通道，并加上上采样的MS（残差结构）
        hrms = self.output(ms_f) + ms_bic  # 输出维度 (B,4,128,128)
        return hrms



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