import torch
import torch.nn.functional as F
from torch import nn

class AugmentPipe(nn.Module):
    """
    Mini-pipe ADA différentiable, sans opérations in-place.
    On laisse p=0 par défaut sur CPU.
    """
    def __init__(self, p=0.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(p)))

    def forward(self, x):
        if self.p.item() <= 0:
            return x
        B, C, H, W = x.shape
        out = x
        # flip
        mask = (torch.rand(B, device=x.device) < self.p).float().view(B,1,1,1)
        out = mask * torch.flip(out, dims=[3]) + (1 - mask) * out
        # léger jitter de luminosité (additif)
        j = (torch.rand(B,1,1,1, device=x.device) - 0.5) * 0.05 * self.p.item()
        out = out + j
        out = torch.clamp(out, -1.0, 1.0)
        return out

class AdaAug:
    def __init__(self, augment_pipe, target=0.6, speed=0.1):
        self.pipe = augment_pipe; self.target = float(target); self.speed = float(speed); self.mu = 0.0
    def update(self, d_real_out):
        with torch.no_grad():
            signs = (d_real_out > 0).float().mean().item()
            self.mu = 0.99 * self.mu + 0.01 * signs
            delta = (self.mu - self.target) * self.speed
            new_p = self.pipe.p + delta; self.pipe.set_p(new_p)
            return self.pipe.p
