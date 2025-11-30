import os, argparse, torch, math
from .models.generator import Generator
from .models.mapping import MappingNetwork, LabelEmbedder
from .utils import save_grid, load_checkpoint

def main(args):
    device = torch.device("cpu") if args.device=="cpu" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    arch = ckpt.get("arch", {"g_ch": {4:512,8:512,16:512,32:256,64:128,128:64,256:32}})

    G = Generator(w_dim=args.w_dim, channels=arch["g_ch"]).to(device)
    mapper = MappingNetwork(z_dim=args.z_dim, c_dim=args.c_dim, w_dim=args.w_dim).to(device)
    label_emb = LabelEmbedder(num_age_bins=args.num_age_bins, num_gender=args.num_gender, num_ethnicity=args.num_ethnicity,
                              emb_dim=args.emb_dim, out_dim=args.c_dim).to(device)

    G.load_state_dict(ckpt['G']); mapper.load_state_dict(ckpt['mapper']); label_emb.load_state_dict(ckpt['label_emb'])
    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.n, args.z_dim, device=device)
        age = torch.full((args.n,), args.age_bin, dtype=torch.long, device=device)
        gender = torch.full((args.n,), args.gender, dtype=torch.long, device=device)
        eth = torch.full((args.n,), args.ethnicity, dtype=torch.long, device=device)
        c = label_emb(age, gender, eth); w = mapper(z, c)
        w = w.unsqueeze(1).repeat(args.n, len(G.convs), 1)
        imgs = G(w).cpu()
        save_grid(imgs, os.path.join(args.out_dir, f"gen_a{args.age_bin}_g{args.gender}_e{args.ethnicity}.png"),
                  nrow=int(math.isqrt(args.n)))
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./samples")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--z_dim", type=int, default=256)
    ap.add_argument("--w_dim", type=int, default=256)
    ap.add_argument("--c_dim", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=32)
    ap.add_argument("--num_age_bins", type=int, default=5)
    ap.add_argument("--num_gender", type=int, default=2)
    ap.add_argument("--num_ethnicity", type=int, default=7)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--age_bin", type=int, default=2)
    ap.add_argument("--gender", type=int, default=0)
    ap.add_argument("--ethnicity", type=int, default=3)
    args = ap.parse_args(); main(args)
