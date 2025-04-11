import torch
# from swin_transformer import *
import torch
from torch import nn, einsum
import numpy as np
from einops import rearrange, repeat

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x

class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        x = self.patch_partition(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
class Decoder(nn.Module):
    def __init__(self, channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                 window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True):
        super().__init__()

        self.Decoder = nn.Sequential(
            #TT
            StageModule(in_channels=hidden_dim*3, hidden_dimension=hidden_dim*3, layers=layers[0],
                        downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            StageModule(in_channels=hidden_dim*3, hidden_dimension=hidden_dim*3, layers=layers[0],
                        downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            #Upsample
            nn.Upsample(scale_factor=2),
            #CC
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            #RefineLayer
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h):
        HRMS = self.Decoder(h)
        return HRMS

class Net(nn.Module):
    def __init__(self, num_channels=4,base_filter=64,args=None, channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                 window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True):
        super().__init__()

        self.Encoder = nn.Sequential(
            #CC
            nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            #Downsample
            nn.AvgPool2d(2, 2),
            #TT
            StageModule(in_channels=hidden_dim*2, hidden_dimension=hidden_dim*2, layers=layers[1],
                        downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            StageModule(in_channels=hidden_dim*2, hidden_dimension=hidden_dim*2, layers=layers[1],
                        downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        )
        self.decoder = Decoder(channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                  window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True)

    def forward(self, LRMS,_, PAN):
        l = self.Encoder(nn.functional.interpolate(LRMS, scale_factor=4))
        p = self.Encoder(torch.repeat_interleave(PAN, dim=1, repeats=4))
        l_unique, l_common = torch.chunk(l, 2, 1)
        p_unique, p_common = torch.chunk(p, 2, 1)
        h = torch.cat([l_unique, (l_common + p_common) / 2, p_unique], 1)
        HRMS = self.decoder(h)
        return  HRMS

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
