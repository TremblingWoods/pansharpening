import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch.nn.init import trunc_normal_


# -------------------------------
# DeformableWindowAttention (W-MSA 模块)
# -------------------------------
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
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + window_size - 1) * (window_size + window_size - 1),
                            num_heads))
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
                                                                                                       1).unsqueeze(0)
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_h - 1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1).reshape(-1)
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1).reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                        1).reshape(1, 2,
                                                                                                                   window_num_h,
                                                                                                                   self.ws,
                                                                                                                   window_num_w,
                                                                                                                   self.ws)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                               1).reshape(
            1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)
        base_coords = image_reference

        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            coords = base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)
        if self.learnable:
            sampling_offsets = self.sampling_offsets(x)
            sampling_offsets = sampling_offsets.reshape(b * self.num_heads, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (expand_w // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (expand_h // self.ws)
            sampling_scales = self.sampling_scales(x)
            sampling_scales = sampling_scales.reshape(b * self.num_heads, 2, window_num_h, window_num_w)
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
        qkv = torch.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(
            3, b * self.num_heads, self.dim // self.num_heads, h + padding_td, w + padding_lr)
        qkv_lms = torch.nn.functional.pad(qkv_lms, (padding_left, padding_right, padding_top, padding_down)).reshape(
            3, b * self.num_heads, self.dim // self.num_heads, h + padding_td, w + padding_lr)
        q_pan, k, v = qkv[0], qkv[1], qkv[2]
        q, k_lms, v_lms = qkv_lms[0], qkv_lms[1], qkv_lms[2]
        k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
        v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

        q_pan = q_pan.reshape(b, self.num_heads, self.dim // self.num_heads, window_num_h, self.ws, window_num_w,
                              self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w,
                                                                            self.num_heads,
                                                                            self.ws * self.ws,
                                                                            self.dim // self.num_heads)
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
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots_lms += relative_position_bias.unsqueeze(0)

        attn_lms = dots_lms.softmax(dim=-1)
        out_lms = attn_lms @ v

        out_lms = rearrange(out_lms, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                            hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        out_lms = out_lms[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out = self.proj(out_lms)
        out_lms = self.proj_drop(out_lms)

        dots_pan = (q_pan @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots_pan += relative_position_bias.unsqueeze(0)

        attn_pan = dots_pan.softmax(dim=-1)
        out_pan = attn_pan @ v

        out_pan = rearrange(out_pan, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                            hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        out_pan = out_pan[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out_pan = self.proj(out_pan)
        out_pan = self.proj_drop(out_pan)

        return out_pan, out_lms


# -------------------------------
# DWT 和 IDWT 模块：自适应小波变换
# -------------------------------
class WaveletTransform(nn.Module):
    def __init__(self, channels):
        super(WaveletTransform, self).__init__()
        self.channels = channels
        sqrt2 = math.sqrt(2)
        # 初始化1D滤波器，后续可学习
        self.fL = nn.Parameter(torch.tensor([1 / sqrt2, 1 / sqrt2], dtype=torch.float32))
        self.fH = nn.Parameter(torch.tensor([1 / sqrt2, -1 / sqrt2], dtype=torch.float32))

    def dwt(self, x):
        """
        对输入特征图 x 进行离散小波变换 (DWT)。
        将 x 分解为四个子带: X_LL, X_LH, X_HL, X_HH
        """
        B, C, H, W = x.shape
        fL = self.fL
        fH = self.fH
        kernel_LL = torch.outer(fL, fL)
        kernel_LH = torch.outer(fL, fH)
        kernel_HL = torch.outer(fH, fL)
        kernel_HH = torch.outer(fH, fH)
        kernel_LL = kernel_LL.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_LH = kernel_LH.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_HL = kernel_HL.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_HH = kernel_HH.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        X_LL = F.conv2d(x, kernel_LL, stride=2, groups=C)
        X_LH = F.conv2d(x, kernel_LH, stride=2, groups=C)
        X_HL = F.conv2d(x, kernel_HL, stride=2, groups=C)
        X_HH = F.conv2d(x, kernel_HH, stride=2, groups=C)
        return X_LL, X_LH, X_HL, X_HH

    def idwt(self, X_LL, X_LH, X_HL, X_HH):
        """
        对四个子带进行逆离散小波变换 (IDWT)，重构原始特征图。
        """
        B, C, H, W = X_LL.shape
        fL = self.fL
        fH = self.fH
        kernel_LL = torch.outer(fL, fL)
        kernel_LH = torch.outer(fL, fH)
        kernel_HL = torch.outer(fH, fL)
        kernel_HH = torch.outer(fH, fH)
        kernel_LL = kernel_LL.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_LH = kernel_LH.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_HL = kernel_HL.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        kernel_HH = kernel_HH.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        X_rec_LL = F.conv_transpose2d(X_LL, kernel_LL, stride=2, groups=C)
        X_rec_LH = F.conv_transpose2d(X_LH, kernel_LH, stride=2, groups=C)
        X_rec_HL = F.conv_transpose2d(X_HL, kernel_HL, stride=2, groups=C)
        X_rec_HH = F.conv_transpose2d(X_HH, kernel_HH, stride=2, groups=C)
        X_rec = X_rec_LL + X_rec_LH + X_rec_HL + X_rec_HH
        return X_rec

    def forward(self, x):
        return self.dwt(x)


# -------------------------------
# 卷积模块：用于高频信息增强
# -------------------------------
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


# -------------------------------
# Wavelet-Enhanced Attention Block (WAB)
# -------------------------------
class WaveletEnhancedAttentionBlock(nn.Module):
    def __init__(self, channels, wmsa_module):
        """
        channels: 输入特征的通道数
        wmsa_module: 已经实现的 W-MSA 模块（这里传入的是 DeformableWindowAttention 实例）
        """
        super(WaveletEnhancedAttentionBlock, self).__init__()
        self.channels = channels
        self.wmsa = wmsa_module

        # 初始化小波变换模块（DWT/IDWT）
        self.wavelet = WaveletTransform(channels)

        # 对高频分量分别采用卷积模块进行处理
        self.conv_module_lh = ConvModule(channels, channels)
        self.conv_module_hl = ConvModule(channels, channels)
        self.conv_module_hh = ConvModule(channels, channels)

        # 用 1x1 卷积融合重构后的特征
        self.idwt_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        # 1. 小波分解
        X_LL, X_LH, X_HL, X_HH = self.wavelet.dwt(x)

        # 2. 分别处理低频与高频信息
        # 对低频分量采用 W-MSA 提取全局信息
        # 注意：此处调整为传入相同的 X_LL 作为两个参数，输出中选用 out_pan
        out_pan, _ = self.wmsa(X_LL, X_LL)
        X_LL = out_pan

        # 对高频分量采用卷积模块增强局部细节
        X_LH = self.conv_module_lh(X_LH)
        X_HL = self.conv_module_hl(X_HL)
        X_HH = self.conv_module_hh(X_HH)

        # 3. 小波逆变换重构特征图
        x_rec = self.wavelet.idwt(X_LL, X_LH, X_HL, X_HH)

        # 4. 通过 1x1 卷积进一步融合
        out = self.idwt_conv(x_rec)
        return out


# -------------------------------
# 示例：如何使用 WaveletEnhancedAttentionBlock 与 DeformableWindowAttention
# -------------------------------
if __name__ == "__main__":
    # 定义参数
    B, C, H, W = 2, 64, 128, 128
    dummy_input = torch.randn(B, C, H, W)

    # 初始化 W-MSA 模块：这里传入相应参数
    wmsa_module = DeformableWindowAttention(dim=C, num_heads=4, window_size=8, img_size=(H, W))

    # 初始化 Wavelet-Enhanced Attention Block
    wab = WaveletEnhancedAttentionBlock(channels=C, wmsa_module=wmsa_module)

    # 前向传播
    output = wab(dummy_input)
    print("Output shape:", output.shape)
