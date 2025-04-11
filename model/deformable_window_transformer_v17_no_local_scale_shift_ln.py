from torch.nn.modules.batchnorm import BatchNorm2d
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules import padding
from einops import rearrange, repeat
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from mmcv_custom import load_checkpoint
# from mmdet.utils import get_root_logger
# from ..builder import BACKBONES

eps = 1e-5

'''
compared with deformable window transformer, deformable scale is changed to extention rather than scale factor
'''

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
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
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

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

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, qkv_bias=True, qk_scale=None, 
                attn_drop=0., proj_drop=0., relative_pos_embedding=False, shift_size=0, img_size=(1,1)):
        super().__init__()
        self.img_size = img_size
        self.num_heads = num_heads
        self.dim = dim
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.attn_ws = window_size
        
        # if self.img_size[0] % self.ws != 0:
        # self.padding_bottom = (self.ws - self.img_size[0] % self.ws) % self.ws
        # if self.img_size[1] % self.ws != 0:
        # self.padding_right = (self.ws - self.img_size[1] % self.ws) % self.ws
        # self.shuffle = shuffle
        self.shift_size = shift_size % self.ws
        self.left_size = self.img_size
        if min(self.img_size) <= self.ws:
            self.shift_size = 0

        # if self.shift_size > 0:
        #     self.padding_bottom = (self.ws - self.shift_size + self.padding_bottom) % self.ws
        #     self.padding_right = (self.ws - self.shift_size + self.padding_right) % self.ws

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=qkv_bias)
        

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x, coords=None):
        b, c, h, w = x.shape
        shortcut = x
        # assert h == self.img_size[0]
        # assert w == self.img_size[1]
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left

        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))
        window_num_h = (h+padding_top+padding_down) // self.ws
        window_num_w = (w+padding_left+padding_right) // self.ws
        
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.dim // self.num_heads, h+padding_top+padding_down, w+padding_left+padding_right).transpose(1, 0).reshape(3, b*self.num_heads, self.dim // self.num_heads, h+padding_top+padding_down, w+padding_left+padding_right)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.dim//self.num_heads)
        k = k.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.dim//self.num_heads)
        v = v.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.dim//self.num_heads)
        # q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        out = out[:, :, padding_top:h+padding_top, padding_left:w+padding_left]
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, None

    def flops(self, ):
        N = self.ws * self.ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        flops *= (self.img_size[0] / self.ws * self.img_size[1] / self.ws)
        return flops

class DeformableWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, qkv_bias=True, qk_scale=None, 
            attn_drop=0., proj_drop=0., relative_pos_embedding=False, learnable=True, restart_regression=None,
            attn_window_size=None, shift_size=0, img_size=(1,1)):
        super().__init__()
        self.img_size = img_size
        self.num_heads = num_heads
        self.dim = dim
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.attn_ws = attn_window_size or self.ws
        
        # if self.img_size[0] % self.ws != 0:
        # self.padding_bottom = (self.ws - self.img_size[0] % self.ws) % self.ws
        # if self.img_size[1] % self.ws != 0:
        # self.padding_right = (self.ws - self.img_size[1] % self.ws) % self.ws
        # assert self.img_size[0] % self.ws == 0
        # assert self.img_size[1] % self.ws == 0
        # self.sampling_offsets = nn.Linear(dim, self.num_heads * 3)
        # self.sampling_offsets = nn.Sequential(*[nn.Conv2d(dim, 32, kernel_size=window_size, stride=window_size), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, self.num_heads*2, kernel_size=1, stride=1)])
        # self.sampling_scales = nn.Sequential(*[nn.Conv2d(dim, 32, kernel_size=window_size, stride=window_size), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, self.num_heads*2, kernel_size=1, stride=1)])
        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(), 
                nn.Conv2d(dim, self.num_heads*2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(), 
                nn.Conv2d(dim, self.num_heads*2, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        self.left_size = self.img_size
        if min(self.img_size) <= self.ws:
            self.shift_size = 0

        # if self.shift_size > 0:
        #     self.padding_bottom = (self.ws - self.shift_size + self.padding_bottom) % self.ws
        #     self.padding_right = (self.ws - self.shift_size + self.padding_right) % self.ws

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + attn_window_size - 1) * (window_size + attn_window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

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
            print('The relative_pos_embedding is used')

        # h, w = self.img_size
        # if self.shift_size > 0:
        #     # self.valid_h = []
        #     h, w = h+self.ws, w+self.ws
        # h, w = h + self.shift_size + self.padding_bottom, w + self.shift_size + self.padding_right
        # image_reference_w = torch.linspace(-1, 1, w)
        # image_reference_h = torch.linspace(-1, 1, h)
        # if self.shift_size > 0:
        #     self.valid_h = (image_reference_h[self.shift_size], image_reference_h[-1-(self.ws-self.shift_size)])
        #     self.valid_w = (image_reference_w[self.shift_size], image_reference_w[-1-(self.ws-self.shift_size)])
        # else:
        #     self.valid_h = (image_reference_h[0], image_reference_h[-1])
        #     self.valid_w = (image_reference_w[0], image_reference_w[-1])
        # self.valid_h = (image_reference_h[self.shift_size], image_reference_h[-1-self.padding_bottom])
        # self.valid_w = (image_reference_w[self.shift_size], image_reference_w[-1-self.padding_right])
        # image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        # window_reference = WindowProcessor(image_reference=image_reference, shift=self.shift_size, 
        #                                     ws=self.ws, h=h, w=w, pooling=nn.functional.avg_pool2d, return_type='pooling')
        # window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        # window_num_h, window_num_w = window_reference.shape[-2:]
        # window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)
        # self.register_buffer('window_reference', window_reference)

        # mask = torch.ones(1, 1, h, w)
        # mask = WindowProcessor(mask, shift=self.shift_size, ws=self.ws, h=h, w=w, return_type='padding')
        # select_mask = torch.zeros(1, 1, h, w)
        # select_mask[:, :, self.shift_size:h-self.shift_size, self.shift_size:w-self.shift_size] = 1
        # self.register_buffer('select_mask', select_mask)

        # base_coords_h = torch.arange(self.attn_ws) * 2 * self.ws / self.attn_ws / (h-1)
        # base_coords_h = (base_coords_h - base_coords_h.mean())
        # base_coords_w = torch.arange(self.attn_ws) * 2 * self.ws / self.attn_ws / (w-1)
        # base_coords_w = (base_coords_w - base_coords_w.mean())

        # expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        # assert expanded_base_coords_h.shape[0] == window_num_h
        # assert expanded_base_coords_h.shape[1] == self.attn_ws
        # expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        # assert expanded_base_coords_w.shape[0] == window_num_w
        # assert expanded_base_coords_w.shape[1] == self.attn_ws
        # expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        # expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        # coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)
        # self.register_buffer('base_coords', window_reference+coords)
        # self.register_buffer('coords', coords)

    def forward(self, x, coords=None):
        b, _, h, w = x.shape
        shortcut = x
        # assert h == self.img_size[0]
        # assert w == self.img_size[1]
        # if self.shift_size > 0:
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left
        expand_h, expand_w = h+padding_top+padding_down, w+padding_left+padding_right
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_w-1)
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
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)
        # base_coords = window_reference+window_coords
        base_coords = image_reference
        # self.register_buffer('base_coords', window_reference+coords)
        # self.register_buffer('coords', coords)

        # image_reference_w = torch.linspace(-1, 1, w).to(x.device)
        # image_reference_h = torch.linspace(-1, 1, h).to(x.device)
        # image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        # window_reference = WindowProcessor(image_reference=image_reference, shift=self.shift_size, 
        #                                     ws=self.ws, h=h, w=w, pooling=nn.functional.avg_pool2d, return_type='pooling')
        # window_reference = self.window_reference.clone()
        # window_num_h, window_num_w = self.base_coords.shape[-4], self.base_coords.shape[-2]

        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            coords = base_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1)
        if self.learnable:
            sampling_offsets = self.sampling_offsets(x)
            sampling_offsets = sampling_offsets.reshape(b*self.num_heads, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)
            
            sampling_scales = self.sampling_scales(x)       #B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(b*self.num_heads, 2, window_num_h, window_num_w)
            
            coords = coords + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b*self.num_heads, self.attn_ws*window_num_h, self.attn_ws*window_num_w, 2)

        # elif coords is None:
        #     coords = self.base_coords.repeat(b, 1, 1, 1, 1, 1)

        qkv = self.qkv(shortcut).reshape(b, 3, self.num_heads, self.dim // self.num_heads, h, w).transpose(1, 0).reshape(3*b*self.num_heads, self.dim // self.num_heads, h, w)
        # if self.shift_size > 0:
        qkv = torch.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(3, b*self.num_heads, self.dim // self.num_heads, h+padding_td, w+padding_lr)
        # else:
        #     qkv = qkv.reshape(3, b*self.num_heads, self.dim // self.num_heads, h, w)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
        v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

        # select_mask = q.not_equal(0).sum(dim=(0,1), keepdim=True).not_equal(0)
        q = q.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.dim//self.num_heads)
        k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
        v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
        # mask: b*self.num_heads, self.attn_ws*window_num_h, self.attn_ws*window_num_w
        # mask = mask.reshape(b, self.num_heads, self.attn_ws, window_num_h, self.attn_ws, window_num_w).permute(0, 3, 5, 1, 2, 4).reshape(b*window_num_h*window_num_w, self.num_heads, 1, self.attn_ws*self.attn_ws)
        
        dots = (q @ k.transpose(-2, -1)) * self.scale
        # dots = dots + mask
        #       b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.attn_ws*self.attn_ws, self.dim//self.num_heads

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
            # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out = out[:, :, padding_top:h+padding_top, padding_left:w+padding_left]
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, coords
    
    def _clip_grad(self, grad_norm):
        # print('clip grads of the model for selection')
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)
        # constant_(self.sampling_offsets.weight.data, 0.)
        # thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        # for i in range(self.n_points):
        #     grid_init[:, :, i, :] *= i + 1
        # with torch.no_grad():
        #     self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

    def flops(self, ):
        N = self.ws * self.ws
        M = self.attn_ws * self.attn_ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * M
        #  x = (attn @ v)
        flops += self.num_heads * N * M * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        if self.learnable:
            # pooling
            flops_sampling += h * w * self.dim
            # regressing the shift and scale
            flops_sampling += 2 * (h/self.ws + w/self.ws) * self.num_heads*2 * self.dim
            # calculating the coords
            flops_sampling += h/self.ws * self.attn_ws * w/self.ws * self.attn_ws * 2
        # grid sampling attended features
        flops_sampling += h/self.ws * self.attn_ws * w/self.ws * self.attn_ws * self.dim
        
        flops += flops_sampling

        return flops

class Block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, attn_window_size=None, shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
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
        self.norm1 = norm_layer(dim, data_format="channels_first")
        self.learnable = learnable
        self.restart_regression = restart_regression
        # deformable = deformable > 0
        self.deformable = deformable
        if not deformable:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, window_size=window_size, shift_size=self.shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding, img_size=img_size)
        else:
            self.pos = nn.Conv2d(dim, dim, window_size//2*2+1, 1, window_size//2, groups=dim, bias=True)
            self.attn = DeformableWindowAttention(
                dim, num_heads=num_heads, window_size=window_size, attn_window_size=attn_window_size, shift_size=self.shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=deform_relative_pos_embedding, img_size=img_size, learnable=learnable, restart_regression=restart_regression,)
        # self.local = nn.Conv2d(dim, dim, window_size//2*2+1, 1, window_size//2, groups=dim, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, stride=stride)
        self.norm3 = norm_layer(dim, data_format="channels_first")
        print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, out_dim, stride, shift_size, num_heads))

    def forward(self, x, coords):
        if self.deformable:
            x = x + self.pos(x)
        attn_x, coords = self.attn(self.norm1(x), coords)
        x = x + self.drop_path(attn_x)
        # x = x + self.local(self.norm2(x)) # local connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, coords

    def flops(self, ):
        flops = 0
        if self.deformable:
            flops += self.img_size[0] * self.img_size[1] * (self.ws//2*2+1 ** 2) * self.dim
        flops += self.attn.flops()
        # norm
        flops += self.img_size[0] * self.img_size[1] * self.dim * 2

        # flops += self.img_size[0] * self.img_size[1] * (self.ws//2*2+1 ** 2) * self.dim
        
        flops += 2*self.img_size[0] * self.img_size[1] * self.dim * self.dim * self.mlp_ratio
        # # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        # # attn = (q @ k.transpose(-2, -1))
        # flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # #  x = (attn @ v)
        # flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # # x = self.proj(x)
        # flops += N * self.dim * self.dim
        return flops

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

class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, attn_window_size=None, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, relative_pos_embedding=False, deform_relative_pos_embedding=False, deformable=False, 
                 cycle=100, shift=True, group=100, img_size=None, use_checkpoint=False):
        super().__init__()
        # assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim, img_size=img_size)
        else:
            self.patch_partition = None

        self.img_size = img_size
        self.cycle = cycle
        self.group = group
        self.shift = shift
        self.use_checkpoint = use_checkpoint
        if self.shift:
            shift_size_choice = [0, window_size // 2]
        else:
            shift_size_choice = [0]
        self.layers = nn.ModuleList([])
        for idx in range(layers):
            shift_size = shift_size_choice[idx % len(shift_size_choice)]
            if window_size >= self.img_size[0]:
                shift_size = 0
            learnable = deformable and (idx % self.cycle < self.cycle // 2)
            restart_regression = deformable and (idx % self.group == 0 or idx % self.group == 1)
            self.layers.append(
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, attn_window_size=attn_window_size, shift_size=shift_size, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding, deform_relative_pos_embedding=deform_relative_pos_embedding, deformable=deformable, learnable=learnable, restart_regression=restart_regression,
                      img_size=img_size)
            )

    def forward(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)
            
        # for regular_block, shifted_block in self.layers:
        #     x = regular_block(x)
        #     x = shifted_block(x)
        coords = [None, None]
        for idx in range(len(self.layers)):
            if self.use_checkpoint:
                x, _coords = checkpoint.checkpoint(self.layers[idx], x, coords[idx%2])
            else:
                x, _coords = self.layers[idx](x, coords[idx%2])
            coords[idx%2] = _coords
        return x

    def flops(self, ):
        flops = 0
        if self.patch_partition:
            flops += self.patch_partition.flops()
        for block in self.layers:
            flops += block.flops()
        return flops

class PatchEmbedding(nn.Module):
    def __init__(self, in_chans=3, out_channels=48, img_size=None, patch_size=4, norm_layer=None):
        self.img_size = img_size
        # self.inter_channel = inter_channel
        self.in_chans = in_chans
        self.out_channel = out_channels
        self.patch_size = patch_size
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=out_channels, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = nn.Identity()
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(inter_channel),
        #     nn.ReLU6(inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU6(inplace=True)
        # )
        # self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x = self.conv3(self.conv2(self.conv1(x)))
        x = self.conv(x)
        x = self.norm(x)
        return x

    def flops(self, ) -> float:
        flops = 0
        flops += 3 * self.out_channel * self.img_size[0] * self.img_size[1]
        if type(self.norm) != nn.Identity:
            flops += self.out_channel * self.img_size[0] * self.img_size[1]
        # flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        # flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        # flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        # flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        # flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops


@BACKBONES.register_module()
class DeformableWindowTransformerV17NoLocalScaleShiftLN(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=96, mlp_ratio=4., layers=[2,2,6,2], num_heads=[3,6,12,24], 
                relative_pos_embedding=True, deform_relative_pos_embedding=False, window_size=7, attn_window_size=None, qkv_bias=True, qk_scale=None, drop_rate=0., 
                attn_drop_rate=0., drop_path_rate=0., has_pos_embed=False, deformable=None, cycle=100, group=100, shift=True, 
                                out_indices=(0, 1, 2, 3), frozen_stages=-1, load_ema=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.load_ema = load_ema
        self.out_indices = out_indices
        # self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        deformable = deformable or (True, True, True, False)
        self.use_checkpoint = use_checkpoint
        # num_features for consistency with other models
        self.cycle = cycle
        self.group = group
        self.shift = shift
        # assert (self.group % self.cycle == 0 or self.group % self.cycle == self.group)
        self.has_pos_embed = has_pos_embed
        if type(embed_dim) == int:
            self.embed_dim = [embed_dim * 2**i for i in range(4)]
        else:
            self.embed_dim = embed_dim
            assert len(self.embed_dim) == 4
        self.mlp_ratio = mlp_ratio
        if type(mlp_ratio) == int or type(mlp_ratio) == float:
            self.mlp_ratio = [mlp_ratio for _ in range(4)]
        for i in range(4):
            assert self.embed_dim[i] % num_heads[i] == 0
        # dims = [i*token_dim for i in num_heads]
        dims = self.embed_dim

        for layer_i in self.out_indices:
            layer = LayerNorm(self.embed_dim[layer_i], data_format="channels_first")
            layer_name = f'norm{layer_i}'
            self.add_module(layer_name, layer)

        self.window_size = window_size if type(window_size) != int else [window_size for i in range(4)]
        attn_window_size = attn_window_size or window_size
        self.attn_window_size = attn_window_size if type(attn_window_size) != int else [attn_window_size for i in range(4)]

        num_patches = (img_size*img_size) // 16
        if type(img_size) == int:
            img_size = np.array([img_size, img_size])
        self.img_size = img_size

        embed_dim = dims[0]
        # self.num_features = self.embed_dim = embed_dim
        self.to_token = PatchEmbedding(in_chans=in_chans, out_channels=dims[0], img_size=img_size, patch_size=4)

        if self.has_pos_embed:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([])
        for i in range(4):
            self.layers.append(
                StageModule(layers[i], embed_dim, dims[i], num_heads[i], window_size=self.window_size[i], attn_window_size=self.attn_window_size[i],
                            mlp_ratio=self.mlp_ratio[i], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                            relative_pos_embedding=relative_pos_embedding, deform_relative_pos_embedding=deform_relative_pos_embedding, deformable=deformable[i], cycle=cycle, 
                            shift=shift, group=group, img_size=img_size // 4 // (2 ** i), use_checkpoint=use_checkpoint
                            )
            )
            embed_dim = dims[i]

        # self.norm = LayerNorm(embed_dim, data_format="channels_first")
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier head
        # self.head = nn.Linear(dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)
        # self.apply(self.reset_parameters)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        # def _init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)
        def _init_weights(m):
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def reset_parameters(m):
            if hasattr(m, '_reset_parameters'):
                m._reset_parameters()

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            self.apply(reset_parameters)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger, load_ema=self.load_ema)
        elif pretrained is None:
            self.apply(_init_weights)
            self.apply(reset_parameters)
        else:
            raise TypeError('pretrained must be a str or None')

    def clip_grad(self, m):
        if hasattr(m, '_clip_grad'):
            m._clip_grad(0.05)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.to_token(x)
        b, c, h, w = x.shape

        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)

        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)

                # out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(x_out)

        return tuple(outs)

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x

    def flops(self, ):
        flops = self.to_token.flops()
        idx = 0
        for layer in self.layers:
            idx += 1
            flops += layer.flops()
            print(f'flops for layer {idx}: {layer.flops()}')
        idx = 0
        for layer in self.layers:
            idx += 1
            n_parameters = sum(p.numel() for p in layer.parameters() if p.requires_grad)
            print(f'params for layer {idx}: {n_parameters}')
        flops += self.embed_dim[-1] * self.img_size[0] * self.img_size[1] / 32
        # flops += self.embed_dim[-1] * self.num_classes
        return flops

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeformableWindowTransformerV17NoLocalScaleShiftLN, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        print(f'freeze stages: {self.frozen_stages}')
        # if self.frozen_stages >= 0:
            # self.patch_embed.eval()
            # for param in self.patch_embed.parameters():
            #     param.requires_grad = False

        # if self.frozen_stages >= 1 and self.ape:
        #     self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
