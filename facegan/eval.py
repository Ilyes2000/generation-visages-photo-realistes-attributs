"""
Evaluate FID (clean-fid) and diversity via LPIPS.
- Requires: pip install clean-fid lpips
Usage:
python -m facegan.eval --ckpt path/to/ckpt.pt --csv path/to/attrs.csv --real_split val --n_gen 5000 --batch 64
"""
import os, argparse, torch
from .models.generator import Generator
from .models.mapping import MappingNetwork, LabelEmbedder
from .data.dataset import FacesAttrDataset
from .utils import load_checkpoint

def gen_images(G, mapper, label_emb, device, z_dim, n, out_dir, batch=64, seed=123):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=device).manual_seed(seed)
    wrote = 0; idx = 0
    while wrote < n:
        m = min(batch, n - wrote)
        z = torch.randn(m, z_dim, generator=g, device=device)
        age = torch.randint(0, label_emb.emb_age.num_embeddings, (m,), generator=g, device=device)
        gender = torch.randint(0, label_emb.emb_gender.num_embeddings, (m,), generator=g, device=device)
        eth = torch.randint(0, label_emb.emb_eth.num_embeddings, (m,), generator=g, device=device)
        c = label_emb(age, gender, eth); w = mapper(z, c)
        num_layers = len(G.convs); w = w.unsqueeze(1).repeat(m, num_layers, 1)
        imgs = G(w).cpu(); imgs = (imgs.clamp(-1,1)+1)/2.0
        for i in range(m):
            path = os.path.join(out_dir, f"{idx+i:08d}.png")
            import torchvision.utils as vutils
            vutils.save_image(imgs[i], path)
        wrote += m; idx += m
    return out_dir

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(w_dim=args.w_dim).to(device)
    mapper = MappingNetwork(z_dim=args.z_dim, c_dim=args.c_dim, w_dim=args.w_dim).to(device)
    label_emb = LabelEmbedder(num_age_bins=args.num_age_bins, num_gender=args.num_gender, num_ethnicity=args.num_ethnicity, emb_dim=args.emb_dim, out_dim=args.c_dim).to(device)

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    G.load_state_dict(ckpt['G']); mapper.load_state_dict(ckpt['mapper']); label_emb.load_state_dict(ckpt['label_emb'])
    G.eval(); mapper.eval(); label_emb.eval()

    # Generate fakes
    tmp_fake = os.path.join(args.tmp_dir, "fake"); os.makedirs(args.tmp_dir, exist_ok=True)
    gen_images(G, mapper, label_emb, device, args.z_dim, args.n_gen, tmp_fake, batch=args.batch)

    # Real images for clean-fid
    ds = FacesAttrDataset(args.csv, split=args.real_split, image_size=args.image_size)
    tmp_real = os.path.join(args.tmp_dir, "real"); os.makedirs(tmp_real, exist_ok=True)
    import shutil
    for i in range(min(len(ds), args.n_gen)):
        path = ds.df.iloc[i]['path']; dst = os.path.join(tmp_real, f"{i:08d}.png"); shutil.copy(path, dst)

    # FID
    try:
        from cleanfid import fid
        score = fid.compute_fid(tmp_real, tmp_fake, num_workers=4)
        print(f"FID = {score:.3f}")
    except Exception as e:
        print("Install clean-fid: pip install clean-fid"); print("Error:", e)

    # LPIPS diversity
    try:
        import lpips, PIL.Image as Image, torchvision.transforms as TT, numpy as np
        lpips_model = lpips.LPIPS(net='vgg').to(device)
        files = sorted([os.path.join(tmp_fake, f) for f in os.listdir(tmp_fake) if f.endswith('.png')])
        choose = files[:min(512, len(files))]
        tfm = TT.Compose([TT.Resize((args.image_size, args.image_size)), TT.ToTensor()])
        imgs = [tfm(Image.open(p).convert("RGB")).to(device) for p in choose]
        X = torch.stack(imgs, 0)
        n_pairs = min(1000, X.size(0)//2); dists = []
        for i in range(n_pairs):
            a = X[i*2:i*2+1]; b = X[i*2+1:i*2+2]
            d = lpips_model(a*2-1, b*2-1).mean().item(); dists.append(d)
        print(f"LPIPS diversity ≈ {np.mean(dists):.3f} ± {np.std(dists):.3f}")
    except Exception as e:
        print("Install lpips: pip install lpips"); print("Error:", e)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--real_split", type=str, default="val")
    ap.add_argument("--tmp_dir", type=str, default="./eval_tmp")
    ap.add_argument("--n_gen", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--z_dim", type=int, default=512)
    ap.add_argument("--w_dim", type=int, default=512)
    ap.add_argument("--c_dim", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--num_age_bins", type=int, default=5)
    ap.add_argument("--num_gender", type=int, default=2)
    ap.add_argument("--num_ethnicity", type=int, default=7)
    args = ap.parse_args(); main(args)
