import os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import gradio as gr

# --- Anti OMP crash (Windows – évite libiomp doublon + limite threads BLAS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]   # .../stylegan2_cond
EXT  = ROOT / "ext"
OUT  = ROOT / "runs" / "gradio_out"
OUT.mkdir(parents=True, exist_ok=True)

# Repos / checkpoints
SG2_DIR = EXT / "stylegan2-ada-pytorch"
SG2_PKL = EXT / "models" / "ffhq.pkl"

SGV2_DIR  = EXT / "stargan-v2"
SGV2_CKPT = EXT / "stargan_ckpt" / "100000_nets_ema.ckpt"

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256

# ---------- Utils ----------
def to_uint8_tensor(img):  # [-1,1] -> [0,255] uint8 CHW
    x = img.detach()
    if x.min() < 0: x = (x.clamp(-1,1)+1)/2
    return (x*255.0).clamp(0,255).byte()

def to_pil(img):  # [-1,1] or [0,1] CHW -> PIL
    x = img.detach().cpu()
    if x.min() < 0: x = (x.clamp(-1,1)+1)/2
    x = (x*255.0).clamp(0,255).byte().permute(1,2,0).numpy()
    return Image.fromarray(x)

def pil_to_tensor(pil, size=256):
    im = pil.convert("RGB").resize((size,size), Image.LANCZOS)
    t = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255.0
    return t

def build_prompt(age, gender, ethnicity):
    age_map = {"18-29":"young adult", "30-39":"adult", "40-49":"middle-aged",
               "50-59":"older adult", "60+":"senior"}
    g_map   = {"Male":"man", "Female":"woman"}
    eth_map = {"White":"white", "Black":"black", "Latino/Hispanic":"latino",
               "East Asian":"east asian", "Southeast Asian":"southeast asian",
               "Indian":"indian", "Middle Eastern":"middle eastern"}
    age_en = age_map.get(age, "adult")
    g_en   = g_map.get(gender, "person")
    e_en   = eth_map.get(ethnicity, "")
    s = f"a portrait photo of a {age_en} {g_en}"
    if e_en: s += f" of {e_en} ethnicity"
    s += ", studio lighting, high detail, neutral background, photorealistic"
    return s

# ---------- 1) StyleGAN2-ADA (FFHQ) ----------
def load_stylegan2():
    if not SG2_DIR.exists():
        return None, f"Repo NVIDIA manquant: {SG2_DIR}"
    if not SG2_PKL.exists():
        return None, f"Checkpoint manquant: {SG2_PKL}"
    sys.path.insert(0, str(SG2_DIR))
    try:
        import dnnlib, legacy  # fournis par le repo
        with open(SG2_PKL, "rb") as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)  # type: ignore
        G.eval()
        return G, None
    except Exception as e:
        return None, f"Échec chargement StyleGAN2-ADA: {e}"

# CLIP preprocess en tenseurs (différentiable)
_CLIP_MU  = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
_CLIP_SIG = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)

def clip_encode_image_tensor(model, img_tensor):  # img: [-1,1] Bx3xHxW
    x = (img_tensor.clamp(-1,1)+1)/2            # [0,1]
    x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
    x = (x - _CLIP_MU.to(x.device)) / _CLIP_SIG.to(x.device)
    return model.encode_image(x)

def sg2_generate_random(age, gender, ethnicity, seed, steps, lr):
    G, err = load_stylegan2()
    if G is None: return None, err

    import clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

    prompt = build_prompt(age, gender, ethnicity)
    with torch.no_grad():
        txt = clip.tokenize([prompt]).to(DEVICE)
        t_feat = clip_model.encode_text(txt)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)

    torch.manual_seed(int(seed))
    z = torch.randn(1, G.z_dim, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=float(lr))

    for _ in range(int(steps)):  # petit guidage CLIP
        img = G(z, None, truncation_psi=0.8, noise_mode="const")  # [-1,1] 1x3xHxW
        i_feat = clip_encode_image_tensor(clip_model, img)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        loss = -(i_feat * t_feat).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    with torch.no_grad():
        out = G(z, None, truncation_psi=0.8, noise_mode="const")[0]
    return to_pil(out), f"SG2 + CLIP | prompt='{prompt}'"

def sg2_edit_upload(uploaded, age, gender, ethnicity, inv_steps, inv_lr, edit_steps, edit_lr):
    G, err = load_stylegan2()
    if G is None: return None, err
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)

    prompt = build_prompt(age, gender, ethnicity)
    with torch.no_grad():
        txt = clip.tokenize([prompt]).to(DEVICE)
        t_feat = clip_model.encode_text(txt)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)

    # inversion naïve sur z (MSE image) — très approximatif (CPU lent)
    x_tgt = pil_to_tensor(uploaded, IMG_SIZE).to(DEVICE)*2-1
    z = torch.randn(1, G.z_dim, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([z], lr=float(inv_lr))
    for _ in range(int(inv_steps)):
        x_hat = G(z, None, truncation_psi=0.8, noise_mode="const")
        loss = F.mse_loss(x_hat, x_tgt.unsqueeze(0))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    # guidage CLIP
    opt = torch.optim.Adam([z], lr=float(edit_lr))
    for _ in range(int(edit_steps)):
        img = G(z, None, truncation_psi=0.8, noise_mode="const")
        i_feat = clip_encode_image_tensor(clip_model, img)
        i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
        loss = -(i_feat * t_feat).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    with torch.no_grad():
        out = G(z, None, truncation_psi=0.8, noise_mode="const")[0]
    return to_pil(out), f"Inversion + CLIP | prompt='{prompt}'"

# ---------- 2) StarGAN v2 (genre) ----------
def load_starganv2():
    if not SGV2_DIR.exists():
        return None, None, f"Repo StarGAN v2 manquant: {SGV2_DIR}"
    if not SGV2_CKPT.exists():
        return None, None, f"Checkpoint manquant: {SGV2_CKPT}"
    sys.path.insert(0, str(SGV2_DIR))
    from munch import Munch
    import importlib, torch
    core_model = importlib.import_module("core.model")
    args = Munch({"img_size":256,"style_dim":64,"w_hpf":1,"latent_dim":16,"num_domains":2})
    nets = core_model.build_model(args)    # DataParallel
    G = nets.generator; M = nets.mapping_network

    # Charger ckpt (clé varie selon release)
    d = torch.load(str(SGV2_CKPT), map_location="cpu")
    if "model_ema" in d: d = d["model_ema"]
    elif "nets_ema" in d: d = d["nets_ema"]

    def strip(sd):
        out={}
        for k,v in sd.items():
            out[k[7:]] = v if k.startswith("module.") else v
        return out
    g_sd = strip(d.get("generator", d.get("G_ema", {})))
    m_sd = strip(d.get("mapping_network", d.get("M_ema", {})))
    G.module.load_state_dict(g_sd, strict=False)
    M.module.load_state_dict(m_sd, strict=False)
    G.to(DEVICE).eval(); M.to(DEVICE).eval()
    return G, M, None

@torch.no_grad()
def sgv2_translate_gender(uploaded, target_gender, n_styles, seed):
    G, M, err = load_starganv2()
    if G is None: return None, err
    if uploaded is None: return None, "Charge une image d'abord."
    y = 1 if target_gender=="Female" else 0  # 0: male, 1: female (convention courante)
    x = pil_to_tensor(uploaded, 256).to(DEVICE)*2-1
    x = x.unsqueeze(0)

    torch.manual_seed(int(seed))
    ims=[]
    for _ in range(int(n_styles)):
        z = torch.randn(1,16,device=DEVICE)
        y_t = torch.tensor([y],dtype=torch.long,device=DEVICE)
        try:
            s = M(y_t, z)  # selon version
        except:
            s = M(z, y_t)
        out = G(x, s)  # [-1,1]
        ims.append(((out[0].clamp(-1,1)+1)/2).cpu())
    row = torch.cat(ims, dim=2)  # concat horizontal
    return to_pil(row), f"StarGAN v2 → {target_gender} ({n_styles} styles)"

# ---------- 3) Stable Diffusion (img2img) ----------
_SD_I2I = None
def ensure_sd_img2img():
    global _SD_I2I
    if _SD_I2I is not None: return _SD_I2I, None
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float32
        )
        _SD_I2I = pipe.to(DEVICE)
        return _SD_I2I, None
    except Exception as e:
        return None, f"Échec chargement SD2.1 img2img: {e}. (Astuce: 'huggingface-cli login')"

@torch.no_grad()
def sd_img2img(uploaded, age, gender, ethnicity, seed, steps, guidance, strength):
    pipe, err = ensure_sd_img2img()
    if pipe is None: return None, err
    if uploaded is None: return None, "Charge une image d'abord."
    prompt = build_prompt(age, gender, ethnicity)
    neg = "blurry, low quality, watermark, text, deformed"
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    im = uploaded.convert("RGB").resize((512,512), Image.LANCZOS)
    out = pipe(
        prompt=prompt, negative_prompt=neg,
        image=im, strength=float(strength),
        num_inference_steps=int(steps), guidance_scale=float(guidance),
        generator=gen
    ).images[0]
    return out, f"SD2.1 img2img | prompt='{prompt}' | strength={strength}"

# ---------- UI ----------
AGES     = ["18-29","30-39","40-49","50-59","60+"]
GENDERS  = ["Male","Female"]
ETHS     = ["White","Black","Latino/Hispanic","East Asian","Southeast Asian","Indian","Middle Eastern"]

with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
               title="Faces: StyleGAN2 / StarGAN v2 / SD img2img") as demo:
    gr.Markdown(
        "# 🎭 Faces Playground\n"
        "- **Onglet 1** : StyleGAN2-ADA (FFHQ) → génération/édition par mini-guidage CLIP (sans fichiers .npy de directions).\n"
        "- **Onglet 2** : StarGAN v2 (CelebA-HQ) → translation *Genre* sur image uploadée.\n"
        "- **Onglet 3** : Stable Diffusion 2-1 (img2img) → modifie une image uploadée selon Âge/Genre/Ethnie.\n"
        "> Requiert localement: `ext/models/ffhq.pkl` et `ext/stargan_ckpt/100000_nets_ema.ckpt`."
    )

    with gr.Tabs():
        with gr.Tab("StyleGAN2-ADA (FFHQ)"):
            with gr.Row():
                with gr.Column(scale=1):
                    age_a    = gr.Dropdown(AGES, value="30-39", label="Âge")
                    gender_a = gr.Dropdown(GENDERS, value="Male", label="Genre")
                    eth_a    = gr.Dropdown(ETHS, value="White", label="Ethnie")
                    seed_a   = gr.Slider(0,9999,42,step=1,label="Seed")
                    steps_a  = gr.Slider(1,60,30,step=1,label="Étapes CLIP")
                    lr_a     = gr.Slider(1e-4,5e-2,1e-2,step=1e-4,label="LR (z)")
                    btn_gen  = gr.Button("🎲 Générer (aléatoire + CLIP)")
                with gr.Column(scale=2):
                    out_a = gr.Image(label="Image générée", type="pil")
                    log_a = gr.Textbox(label="Log", interactive=False)

            gr.Markdown("### Éditer une image existante (inversion naïve + CLIP)")
            with gr.Row():
                with gr.Column(scale=1):
                    up_a     = gr.Image(label="Importer image (≈256x256)", type="pil")
                    inv_s    = gr.Slider(5,300,60,step=5,label="Inversion itérations")
                    inv_lr   = gr.Slider(1e-4,5e-2,5e-3,step=1e-4,label="Inversion LR")
                    edt_s    = gr.Slider(1,60,30,step=1,label="Édition itérations")
                    edt_lr   = gr.Slider(1e-4,5e-2,1e-2,step=1e-4,label="Édition LR")
                    btn_edt  = gr.Button("🖼️ Éditer")
                with gr.Column(scale=2):
                    out_b = gr.Image(label="Image éditée", type="pil")
                    log_b = gr.Textbox(label="Log", interactive=False)

            btn_gen.click(sg2_generate_random,
                          [age_a, gender_a, eth_a, seed_a, steps_a, lr_a],
                          [out_a, log_a])
            btn_edt.click(sg2_edit_upload,
                          [up_a, age_a, gender_a, eth_a, inv_s, inv_lr, edt_s, edt_lr],
                          [out_b, log_b])

        with gr.Tab("StarGAN v2 (genre)"):
            gr.Markdown("Checkpoint requis: `ext/stargan_ckpt/100000_nets_ema.ckpt` (CelebA-HQ).")
            with gr.Row():
                with gr.Column(scale=1):
                    up_b   = gr.Image(label="Importer image (visage)", type="pil")
                    gen_b  = gr.Dropdown(GENDERS, value="Female", label="Genre cible")
                    nsty   = gr.Slider(1,6,4,step=1,label="# styles/variantes")
                    seed_b = gr.Slider(0,9999,7,step=1,label="Seed style")
                    btn_b  = gr.Button("🔁 Traduire le genre (StarGAN v2)")
                with gr.Column(scale=2):
                    out_c = gr.Image(label="Résultat (grille)", type="pil")
                    log_c = gr.Textbox(label="Log", interactive=False)

            def _sgv2(img, g, n, s):
                if not SGV2_CKPT.exists():
                    return None, f"Checkpoint manquant: {SGV2_CKPT}"
                return sgv2_translate_gender(img, g, n, s)

            btn_b.click(_sgv2, [up_b, gen_b, nsty, seed_b], [out_c, log_c])

        with gr.Tab("Stable Diffusion 2-1 (img2img)"):
            with gr.Row():
                with gr.Column(scale=1):
                    up_c    = gr.Image(label="Importer image (sera redimensionnée 512x512)", type="pil")
                    age_c   = gr.Dropdown(AGES, value="30-39", label="Âge")
                    gen_c   = gr.Dropdown(GENDERS, value="Female", label="Genre")
                    eth_c   = gr.Dropdown(ETHS, value="East Asian", label="Ethnie")
                    seed_c  = gr.Slider(0,9999,123,step=1,label="Seed")
                    steps_c = gr.Slider(5,50,25,step=1,label="# steps")
                    cfg_c   = gr.Slider(1.0,12.0,7.0,step=0.5,label="Guidance")
                    str_c   = gr.Slider(0.1,0.95,0.55,step=0.05,label="Strength (degré de changement)")
                    btn_c   = gr.Button("✨ Modifier (SD 2-1 img2img)")
                with gr.Column(scale=2):
                    out_d = gr.Image(label="Image modifiée", type="pil")
                    log_d = gr.Textbox(label="Log", interactive=False)

            btn_c.click(sd_img2img, [up_c, age_c, gen_c, eth_c, seed_c, steps_c, cfg_c, str_c],
                        [out_d, log_d])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
