import torch
from torch import nn
from .layers import EqualizedLinear

class LabelEmbedder(nn.Module):
    def __init__(self, num_age_bins=5, num_gender=2, num_ethnicity=7, emb_dim=32, out_dim=128):
        super().__init__()
        self.emb_age = nn.Embedding(num_age_bins, emb_dim)
        self.emb_gender = nn.Embedding(num_gender, emb_dim)
        self.emb_eth = nn.Embedding(num_ethnicity, emb_dim)
        self.proj = nn.Sequential(
            EqualizedLinear(emb_dim * 3, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, age, gender, eth):
        ea = self.emb_age(age); eg = self.emb_gender(gender); ee = self.emb_eth(eth)
        e = torch.cat([ea, eg, ee], dim=1)
        return self.proj(e)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, c_dim=128, w_dim=512, num_layers=8, lr_mul=0.01):
        super().__init__()
        dims = [z_dim + c_dim] + [w_dim] * num_layers
        layers = []
        for i in range(num_layers):
            layers.append(EqualizedLinear(dims[i], dims[i+1], lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, z, c):
        z = z / (z.pow(2).mean(dim=1, keepdim=True) + 1e-8).sqrt()
        x = torch.cat([z, c], dim=1)
        return self.net(x)
