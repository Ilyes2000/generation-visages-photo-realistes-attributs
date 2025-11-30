# facegan/train.py
import os, argparse, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .data.dataset import FaceAttrsDataset
from .models.generator import Generator, MappingNetwork
from .models.discriminator import ProjectionDiscriminator
from .augment import AugmentPipe

# --------- pertes (StyleGAN2 "logistic") ---------
def d_loss_logistic(d_real, d_fake):
    # D veut: real -> +inf ; fake -> -inf
    return F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

def g_loss_logistic(d_fake):
    # G veut: fake -> +inf
    return F.softplus(-d_fake).mean()

# (optionnel) clip grad pour stabilité CPU
def clip_grads(params, max_norm=5.0):
    torch.nn.utils.clip_grad_norm_(params, max_norm)

# --------- builder modèles ----------
def setup_models(args, device):
    d_age, d_gen, d_eth = 16, 8, 16
    c_dim, z_dim, w_dim = (d_age+d_gen+d_eth), 128, 256

    G = Generator(w_dim=w_dim, base_ch=32 if args.lite else 64).to(device)
    D = ProjectionDiscriminator(
        channels=32 if args.lite else 64,
        num_blocks=4,
        n_age=5, n_gender=2, n_eth=7,
        d_age=d_age, d_gender=d_gen, d_eth=d_eth
    ).to(device)

    mapper = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim,
                            n_layers=4 if args.lite else 8).to(device)
    label_emb = torch.nn.ModuleDict({
        "age": torch.nn.Embedding(5, d_age),
        "gen": torch.nn.Embedding(2, d_gen),
        "eth": torch.nn.Embedding(7, d_eth),
    }).to(device)

    for m in list(G.modules()) + list(mapper.modules()) + list(label_emb.modules()):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if getattr(m, "bias", None) is not None:
                torch.nn.init.zeros_(m.bias)

    g_params = sum(p.numel() for p in G.parameters())/1e6
    d_params = sum(p.numel() for p in D.parameters())/1e6
    print(f"G params: {g_params:.2f}M | D params: {d_params:.2f}M | device={device}")

    return G, D, mapper, label_emb

# --------- split train/val ----------
def make_loaders(csv, batch_size):
    df = pd.read_csv(csv)
    if "split" in df.columns and df["split"].isin(["train","val","test"]).any():
        ds_train = FaceAttrsDataset(csv, split="train")
        # val prioritaire, sinon test, sinon petit split auto
        if "val" in df["split"].values:
            ds_val = FaceAttrsDataset(csv, split="val")
        elif "test" in df["split"].values:
            ds_val = FaceAttrsDataset(csv, split="test")
        else:
            ds_full = FaceAttrsDataset(csv, split=None)
            n = len(ds_full); n_val = max(1, int(0.05*n))
            idx = np.random.permutation(n)
            ds_val = Subset(ds_full, idx[:n_val])
    else:
        ds_full = FaceAttrsDataset(csv, split=None)
        n = len(ds_full); n_val = max(1, int(0.05*n))
        idx = np.random.permutation(n)
        ds_train = Subset(ds_full, idx[n_val:])
        ds_val   = Subset(ds_full, idx[:n_val])

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(emaG, D, mapper, label_emb, loader, device):
    emaG.eval()
    D.eval()
    total_d, total_g, n = 0.0, 0.0, 0
    for real_img, age, gender, eth in loader:
        real_img = real_img.to(device)
        age, gender, eth = age.to(device), gender.to(device), eth.to(device)

        # D(real)
        d_real = D(real_img, age, gender, eth)

        # G(fake) via emaG pour validation
        z = torch.randn(real_img.size(0), 128, device=device)
        c = torch.cat([label_emb["age"](age), label_emb["gen"](gender), label_emb["eth"](eth)], dim=1)
        w = mapper(z, c)
        fake_img = emaG(w)
        d_fake = D(fake_img, age, gender, eth)

        d_loss = d_loss_logistic(d_real, d_fake).item()
        g_loss = g_loss_logistic(d_fake).item()
        total_d += d_loss * real_img.size(0)
        total_g += g_loss * real_img.size(0)
        n += real_img.size(0)
    return total_d / max(1,n), total_g / max(1,n)

def main(args):
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader = make_loaders(args.csv, args.batch_size)

    G, D, mapper, label_emb = setup_models(args, device)
    emaG = Generator(w_dim=256, base_ch=32 if args.lite else 64).to(device)
    emaG.load_state_dict(G.state_dict())
    for p in emaG.parameters(): p.requires_grad = False

    # Optims: D un peu plus lent que G pour éviter qu'il écrase G
    g_opt = optim.Adam(list(G.parameters()) + list(mapper.parameters()) + list(label_emb.parameters()),
                       lr=2e-4, betas=(0.0, 0.99))
    d_opt = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.99))

    aug = AugmentPipe(p=0.0).to(device)  # ADA off sur CPU

    it = 0
    eval_every = max(100, args.batch_size*10)

    G.train(); D.train()
    t0 = time.time()

    while it < args.iters:
        for real_img, age, gender, eth in train_loader:
            it += 1
            real_img = real_img.to(device)
            age, gender, eth = age.to(device), gender.to(device), eth.to(device)

            # ---------- D step ----------
            z = torch.randn(real_img.size(0), 128, device=device)
            c = torch.cat([label_emb["age"](age), label_emb["gen"](gender), label_emb["eth"](eth)], dim=1)
            w = mapper(z, c)
            fake_img = G(w)

            d_real = D(aug(real_img), age, gender, eth)
            d_fake = D(aug(fake_img.detach()), age, gender, eth)
            d_loss = d_loss_logistic(d_real, d_fake)

            # R1 de temps en temps
            if it % 16 == 0:
                real_img.requires_grad_(True)
                d_real_r1 = D(aug(real_img), age, gender, eth)
                grad = torch.autograd.grad(outputs=d_real_r1.sum(), inputs=real_img, create_graph=True)[0]
                r1 = grad.pow(2).reshape(grad.size(0), -1).sum(1).mean()
                d_loss = d_loss + args.r1_gamma * 0.5 * r1
                real_img.requires_grad_(False)

            d_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            clip_grads(D.parameters(), 5.0)
            d_opt.step()

            # ---------- G step ----------
            z = torch.randn(real_img.size(0), 128, device=device)
            c = torch.cat([label_emb["age"](age), label_emb["gen"](gender), label_emb["eth"](eth)], dim=1)
            w = mapper(z, c)
            fake_img = G(w)
            d_fake_for_g = D(aug(fake_img.clone()), age, gender, eth)  # clone pour éviter in-place
            g_loss = g_loss_logistic(d_fake_for_g)

            g_opt.zero_grad(set_to_none=True)
            g_loss.backward()
            clip_grads(list(G.parameters()) + list(mapper.parameters()) + list(label_emb.parameters()), 5.0)
            g_opt.step()

            # EMA
            with torch.no_grad():
                for p_ema, p in zip(emaG.parameters(), G.parameters()):
                    p_ema.copy_(p_ema * 0.999 + p * (1 - 0.999))

            if it % 100 == 0:
                dt = time.time() - t0; t0 = time.time()
                print(f"[{it}/{args.iters}] train D: {d_loss.item():.4f} | G: {g_loss.item():.4f}  ({dt:.1f}s)")

            if it % eval_every == 0:
                val_d, val_g = evaluate(emaG, D, mapper, label_emb, val_loader, device)
                print(f"    -> val D: {val_d:.4f} | val G: {val_g:.4f}")

            if it >= args.iters:
                break

    torch.save({"G": G.state_dict(), "emaG": emaG.state_dict()},
               os.path.join(args.out_dir, "last.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lite", type=lambda s: s.lower() in ["1","true","yes"], default=True)
    p.add_argument("--r1_gamma", type=float, default=1.0)
    args = p.parse_args()
    main(args)
