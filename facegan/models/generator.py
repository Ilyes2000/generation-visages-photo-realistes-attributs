# facegan/models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

# -----------------------------
# Mapping (z,c) -> w
# -----------------------------
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, n_layers=8):
        super().__init__()
        layers = []
        in_dim = z_dim + c_dim
        for i in range(n_layers):
            layers += [
                nn.Linear(in_dim if i == 0 else w_dim, w_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)  # normalisation stylegan
        return self.mlp(x)

# -----------------------------
# ModulatedConv2d (StyleGAN2)
# -----------------------------
class ModulatedConv2d(nn.Module):
    """
    Convolution modulée/démodulée par style (StyleGAN2).
    Implémentation per-sample via grouped conv (groups=B).
    """
    def __init__(self, c_in, c_out, w_dim, k=3, up=False):
        super().__init__()
        self.c_in   = c_in
        self.c_out  = c_out
        self.k      = k
        self.up     = up
        self.weight = nn.Parameter(torch.randn(1, c_out, c_in, k, k))  # [1, Cout, Cin, k, k]
        self.affine = nn.Linear(w_dim, c_in)  # -> style s (par canal d'entrée)
        self.bias   = nn.Parameter(torch.zeros(c_out))

        # init kaiming
        nn.init.kaiming_normal_(self.weight, a=0.2)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x, w, noise=None):
        """
        x: [B, Cin, H, W]
        w: [B, w_dim] -> affine -> s: [B, Cin]
        """
        B, Cin, H, W = x.shape
        assert Cin == self.c_in, f"Cin mismatch: x has {Cin}, layer expects {self.c_in}"

        # upsample éventuel (avant conv)
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")

        # modulation
        s = self.affine(w)                   # [B, Cin]
        s = s.view(B, 1, Cin, 1, 1) + 1.0    # [B, 1, Cin, 1, 1] (1 + modulation)

        # poids par-échantillon : [B, Cout, Cin, k, k]
        w_mod = self.weight * s

        # demodulation
        d = torch.rsqrt((w_mod**2).sum(dim=[2,3,4], keepdim=True) + 1e-8)  # [B, Cout, 1,1,1]
        w_mod = w_mod * d

        # conv groupée
        # x: [1, B*Cin, H, W], w: [B*Cout, Cin, k, k], groups=B
        x_ = x.view(1, B * Cin, x.shape[2], x.shape[3])
        w_ = w_mod.view(B * self.c_out, self.c_in, self.k, self.k)
        y  = F.conv2d(x_, w_, bias=None, stride=1, padding=self.k//2, groups=B)
        y  = y.view(B, self.c_out, y.shape[2], y.shape[3])

        if noise is not None:
            y = y + noise
        y = y + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(y, 0.2, inplace=True)

# -----------------------------
# StyledConv + ToRGB
# -----------------------------
class StyledConv(nn.Module):
    def __init__(self, c_in, c_out, w_dim, up=False):
        super().__init__()
        self.conv = ModulatedConv2d(c_in, c_out, w_dim, k=3, up=up)

    def forward(self, x, w):
        return self.conv(x, w, noise=None)

class ToRGB(nn.Module):
    def __init__(self, c_in, w_dim):
        super().__init__()
        self.conv = ModulatedConv2d(c_in, 3, w_dim, k=1, up=False)

    def forward(self, x, w):
        # pas d'activation après ToRGB ici; on ajoutera tanh en sortie G
        y = self.conv(x, w, noise=None)
        return y

# -----------------------------
# Generator (256x256)
# -----------------------------
class Generator(nn.Module):
    def __init__(self, w_dim=256, base_ch=64, out_res=256):
        super().__init__()
        assert out_res == 256, "Ce G minimal est paramétré pour 256x256."
        self.w_dim = w_dim

        # Constante initiale 4x4
        self.const = nn.Parameter(torch.randn(1, base_ch*8, 4, 4))

        ch = base_ch
        # 4x4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.blocks = nn.ModuleList([
            StyledConv(ch*8, ch*8, w_dim, up=False),  # 4
            StyledConv(ch*8, ch*8, w_dim, up=False),  # 4
            StyledConv(ch*8, ch*4, w_dim, up=True),   # 8
            StyledConv(ch*4, ch*4, w_dim, up=False),  # 8
            StyledConv(ch*4, ch*2, w_dim, up=True),   # 16
            StyledConv(ch*2, ch*2, w_dim, up=False),  # 16
            StyledConv(ch*2, ch,   w_dim, up=True),   # 32
            StyledConv(ch,   ch,   w_dim, up=False),  # 32
            StyledConv(ch,   ch,   w_dim, up=True),   # 64
            StyledConv(ch,   ch,   w_dim, up=False),  # 64
            StyledConv(ch,   ch,   w_dim, up=True),   # 128
            StyledConv(ch,   ch,   w_dim, up=False),  # 128
            StyledConv(ch,   ch,   w_dim, up=True),   # 256
            StyledConv(ch,   ch,   w_dim, up=False),  # 256
        ])
        self.torgb = ToRGB(ch, w_dim)

        # init
        nn.init.normal_(self.const, mean=0.0, std=1.0)

    def forward(self, w):
        B = w.shape[0]
        x = self.const.repeat(B, 1, 1, 1)
        for blk in self.blocks:
            x = blk(x, w)
        rgb = self.torgb(x, w)
        return torch.tanh(rgb)  # [-1,1]
