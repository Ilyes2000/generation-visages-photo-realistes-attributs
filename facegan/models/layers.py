import math
import torch
from torch import nn
import torch.nn.functional as F

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lr_mul = lr_mul
        self.scale = (1.0 / math.sqrt(in_features)) * self.lr_mul
    def forward(self, x):
        return F.linear(x, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None)

class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, style_dim, up=False, demodulate=True):
        super().__init__()
        self.in_ch = in_ch; self.out_ch = out_ch; self.kernel = kernel
        self.up = up; self.demodulate = demodulate; self.pad = kernel // 2
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.mod = EqualizedLinear(style_dim, in_ch, bias=True)
    def forward(self, x, style):
        b, c, h, w = x.shape
        style = self.mod(style).view(b, 1, self.in_ch, 1, 1)
        weight = self.weight * style
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2,3,4]) + 1e-8)
            weight = weight * d.view(b, self.out_ch, 1, 1, 1)
        x = x.reshape(1, -1, h, w)
        weight = weight.reshape(b * self.out_ch, self.in_ch, self.kernel, self.kernel)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        out = F.conv2d(x, weight, padding=self.pad, groups=b)
        out = out.reshape(b, self.out_ch, out.shape[-2], out.shape[-1])
        return out

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, image, noise=None):
        if noise is None:
            noise = image.new_empty(image.size(0), 1, image.size(2), image.size(3)).normal_()
        return image + self.weight * noise

class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, style_dim, up=False, demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel, style_dim, up=up, demodulate=demodulate)
        self.noise = NoiseInjection(out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x, style, noise=None):
        x = self.conv(x, style)
        x = self.noise(x, noise=noise)
        x = self.act(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, in_ch, style_dim):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, 3, 1, style_dim, up=False, demodulate=False)
    def forward(self, x, style):
        return self.conv(x, style)
