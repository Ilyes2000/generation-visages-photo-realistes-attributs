# facegan/models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- blocs de base ----------

class ResDown(nn.Module):
    """Bloc résiduel avec downsample (StyleGAN2-like, simple)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.skip  = nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.act   = nn.LeakyReLU(0.2, inplace=True)
        self.avg   = nn.AvgPool2d(2)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.avg(self.conv2(h))
        s = self.avg(self.skip(x))
        return self.act(h + s)

class MinibatchStdDev(nn.Module):
    """
    Canal Minibatch-Std-Dev : calcule l'écart-type par groupe
    et concatène 1 canal supplémentaire.
    """
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.shape
        g = min(self.group_size, N) if N > 1 else 1
        if g > 1 and N % g == 0:
            y = x.view(g, -1, C, H, W)
            y = y - y.mean(dim=0, keepdim=True)
            y = torch.sqrt((y**2).mean(dim=0) + self.eps)
            y = y.mean(dim=[1,2,3], keepdim=True)           # (N/g,1,1,1)
            y = y.repeat(g, 1, H, W)                        # (N,1,H,W)
            x = torch.cat([x, y], dim=1)                    # (N,C+1,H,W)
        else:
            # si minibatch=1, on ajoute un canal 0
            x = torch.cat([x, torch.zeros(N,1,H,W, device=x.device, dtype=x.dtype)], dim=1)
        return x

# Alias de compatibilité (si d’autres fichiers utilisent MinibatchStd)
MinibatchStd = MinibatchStdDev

# ---------- discriminateur avec projection ----------

class ProjectionDiscriminator(nn.Module):
    """
    Discriminateur conditionnel par projection :
      logit(x,y) = <f(x), e(y)> + b
    où f(x) est un pooling (somme) des features ; e(y) est un embedding des labels.
    """
    def __init__(self,
                 channels=64,
                 num_blocks=4,
                 n_age=5, n_gender=2, n_eth=7,
                 d_age=16, d_gender=8, d_eth=16):
        super().__init__()

        c = channels
        feats = []
        # suppose entrée 3x256x256 ; 4 blocs -> 16x16 (approx)
        feats += [nn.Conv2d(3, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(num_blocks):
            feats += [ResDown(c, c*2)]
            c *= 2
        self.body = nn.Sequential(*feats)

        # IMPORTANT : utiliser MinibatchStdDev (avec alias MinibatchStd disponible)
        self.minibatch = MinibatchStdDev()
        self.final_conv = nn.Conv2d(c+1, c, 3, 1, 1)  # +1: canal minibatch
        self.final_act  = nn.LeakyReLU(0.2, inplace=True)
        self.fc = nn.Linear(c, 1)                     # tête non-conditionnelle

        # embeddings des labels + projection vers dim features c
        self.emb_age    = nn.Embedding(n_age, d_age)
        self.emb_gender = nn.Embedding(n_gender, d_gender)
        self.emb_eth    = nn.Embedding(n_eth, d_eth)
        self.to_proj = nn.Linear(d_age + d_gender + d_eth, c)

        # init propre
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def embed_labels(self, age, gender, eth):
        e = torch.cat([
            self.emb_age(age),
            self.emb_gender(gender),
            self.emb_eth(eth)
        ], dim=1)
        return self.to_proj(e)  # (B, c)

    def forward(self, x, age, gender, eth):
        h = self.body(x)
        h = self.minibatch(h)
        h = self.final_act(self.final_conv(h))
        feat = h.sum(dim=[2,3])               # global sum pooling (B, c)

        out_uncond = self.fc(feat)            # (B, 1)
        e = self.embed_labels(age, gender, eth)          # (B, c)
        proj = torch.sum(feat * e, dim=1, keepdim=True)  # (B,1)
        return out_uncond + proj
