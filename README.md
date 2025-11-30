# 🧪 FairFace cStyleGAN2-ADA (Conditionnel) — Démo Gradio + StarGAN-v2 & SD2.1

Ce dépôt permet de **préparer vos données FairFace**, **entraîner un StyleGAN2-ADA conditionnel** (Âge / Genre / Ethnie), **charger des checkpoints pré-entraînés** (FFHQ, StarGAN-v2), et **lancer une interface Gradio** (thème sombre) pour générer/éditer des visages.  
En bonus : **Stable Diffusion 2.1 img2img** pour l’édition guidée par texte.

---

## 🔰 TL;DR (démarrage rapide)

### Windows (Conda)
```powershell
conda create -n facegan python=3.10 -y
conda activate facegan
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ftfy regex tqdm
pip install "git+https://github.com/openai/CLIP.git"
pip install diffusers transformers accelerate safetensors

$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"

# (Option) SD2.1 privé : huggingface-cli login
```

### Linux / macOS (venv)
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
# CUDA si GPU :
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# ou CPU :
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install ftfy regex tqdm "git+https://github.com/openai/CLIP.git"
pip install diffusers transformers accelerate safetensors
```

---

## 🗂️ Arborescence conseillée
```
stylegan2_cond/
├── apps/
│   └── gradio_demo.py
├── data/
│   └── fairface_filtered/
│       ├── images/               # vos images (centrées, 256x256)
│       └── labels.csv            # filename,age_bin,gender,ethnicity
├── ext/
│   ├── stargan_ckpt/             # optionnel (StarGAN-v2)
│   │   └── 100000_nets_ema.ckpt
│   └── sd/                       # optionnel (cache diffusers)
├── runs/
│   ├── training/                 # logs/ckpts d'entraînement SG2-ADA
│   └── demo_outputs/             # images sorties Gradio
├── models/
│   ├── sg2ada_fairface.pkl       # snapshot G_ema (si entraîné)
│   └── ffhq.pkl                  # optionnel (pré-entraîné NVIDIA)
├── requirements.txt
└── README.md
```

---

## 📦 `requirements.txt` minimal
```txt
numpy>=1.23
pillow>=10.0
opencv-python
torchmetrics
matplotlib
tqdm
pandas
scikit-image
gradio==4.44.1
fastapi<0.115
starlette<0.39
gradio_client>=0.16
munch
einops
protobuf<5
```

> 💡 Si vous voyez **`OMP: Error #15`** (Windows), exportez avant de lancer :  
> `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`.

---

## 🧹 Préparer FairFace (→ 256×256)

1. Téléchargez **FairFace** et un **CSV d’annotations** (`age`, `gender`, `race`).  
2. Filtrez et exportez en **256×256** centrés visage.  
3. Créez `data/fairface_filtered/images/*.jpg` + `data/fairface_filtered/labels.csv` :

```csv
filename,age_bin,gender,ethnicity
000001.jpg,0,1,3
000002.jpg,2,0,6
...
```

- `age_bin ∈ {0..4}` (ex: 18–29, 30–39, 40–49, 50–59, 60+)  
- `gender ∈ {0: Male, 1: Female}`  
- `ethnicity ∈ {0..6}` (White, Black, Latino_Hispanic, East_Asian, Southeast_Asian, Indian, Middle_Eastern)

---

## 🏋️‍♀️ Entraîner StyleGAN2-ADA (conditionnel)

### (Option) Convertir en dataset `.zip` style NVIDIA
```bash
python dataset_tool.py --source=data/fairface_filtered/images                        --dest=data/fairface_256.zip                        --resolution=256x256
```

### Lancer l’entraînement
```bash
python train.py   --outdir=runs/training/sg2ada_fairface256   --data=data/fairface_256.zip   --gpus=1 --batch=64 --cfg=stylegan2 --cbase=32768 --cmax=512   --gamma=10 --kimg=6000 --snap=50 --cond=1
```

À la fin, copiez le meilleur snapshot en :
```
models/sg2ada_fairface.pkl
```

---

## ▶️ Lancer l’UI Gradio

### Windows PowerShell
```powershell
conda activate facegan
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
cd apps
python gradio_demo.py ^
  --fairface_pkl "..\models\sg2ada_fairface.pkl" ^
  --ffhq_pkl "..\modelsfhq.pkl" ^
  --stargan_ckpt "..\ext\stargan_ckpt@000_nets_ema.ckpt" ^
  --sd_model "stabilityai/stable-diffusion-2-1" ^
  --out_dir "..
uns\demo_outputs"
```

### Linux / macOS
```bash
source .venv/bin/activate  # ou conda activate facegan
cd apps
python gradio_demo.py   --fairface_pkl ../models/sg2ada_fairface.pkl   --ffhq_pkl ../models/ffhq.pkl   --stargan_ckpt ../ext/stargan_ckpt/100000_nets_ema.ckpt   --sd_model "stabilityai/stable-diffusion-2-1"   --out_dir ../runs/demo_outputs
```

- Ouverture par défaut : **http://127.0.0.1:7860**  
- Si votre `localhost` est bloqué : lancer avec `share=True` (ou corriger le proxy).

---

## 🖱️ Utilisation de l’UI

### Panneau 1 — StyleGAN2-ADA (FairFace)
- Upload image (optionnel) ou génération *from scratch*.
- Sélectionnez **Âge / Genre / Ethnie** → **Générer**.
- Sorties dans `runs/demo_outputs/`.

### Panneau 2 — StarGAN-v2 (CelebA-HQ) *(optionnel)*
- Nécessite `ext/stargan_ckpt/100000_nets_ema.ckpt`.
- Édite surtout **Âge** et **Genre** selon le checkpoint.

### Panneau 3 — Stable Diffusion 2.1 img2img *(optionnel)*
- Upload image + sliders d’attributs → prompt auto-généré.
- Paramètre `strength` (0.3–0.6 = conserve la structure, change les traits).

---

## 🧩 Checkpoints & ressources

- **FFHQ** : `models/ffhq.pkl` (test inférence SG2 générique).  
- **StarGAN-v2** : `ext/stargan_ckpt/100000_nets_ema.ckpt`.  
- **SD2.1** : `stabilityai/stable-diffusion-2-1` (token HF si requis).

---

## 🛠️ Dépannage (FAQ)

- **`OMP: Error #15` (Windows)** → Exportez : `KMP_DUPLICATE_LIB_OK=TRUE` + limites de threads.  
- **Gradio → `TypeError: argument of type 'bool' is not iterable`** → Épinglez : `gradio==4.44.1`, `fastapi<0.115`, `starlette<0.39>`.  
- **`ModuleNotFoundError: No module named 'clip'`** → Installez Git puis : `pip install "git+https://github.com/openai/CLIP.git"` (ou `open-clip-torch`).  
- **`FileNotFoundError: ... sg2ada_fairface.pkl`** → Vérifiez `--fairface_pkl`.  
- **Diffusers / HF** → `huggingface-cli login` si le modèle n’est pas public.  
- **CUDA non détecté** → Build PyTorch **compatible** avec votre CUDA.

---

## 🔁 Reproductibilité

- Logguez `seed` + hyperparamètres (`batch`, `kimg`, `gamma`, `cond`) dans `runs/training/...`.  
- Exportez **FID / pertes / R1 / ADA** depuis vos notebooks.  
- Versionnez les scripts + snapshots `.pkl`.

---

## ⚖️ Éthique & conformité

- Datasets licites (FairFace, CelebA-HQ, FFHQ).  
- Transparence des **biais** (analyse par sous-groupes).  
- Pas d’usage sur personnes réelles sans consentement.

---

## 📜 Licence

- Code de recherche : **licence académique non-commerciale** (adaptez si besoin).  
- Respectez les licences **StyleGAN2-ADA**, **StarGAN-v2**, **Diffusers/Stable Diffusion**.

---

## 🧱 (Optionnel) Scripts prêts à l’emploi

### `scripts/setup_windows.ps1`
```powershell
param(
  [string]$EnvName="facegan",
  [string]$CudaIndexUrl="https://download.pytorch.org/whl/cu121"
)
conda create -n $EnvName python=3.10 -y
conda activate $EnvName
pip install --index-url $CudaIndexUrl torch torchvision torchaudio
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install ftfy regex tqdm
pip install "git+https://github.com/openai/CLIP.git"
pip install diffusers transformers accelerate safetensors
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
Write-Host "OK"
```

### `scripts/setup_unix.sh`
```bash
#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install ftfy regex tqdm "git+https://github.com/openai/CLIP.git"
pip install diffusers transformers accelerate safetensors
echo "OK"
```

### `scripts/run_gradio_windows.ps1`
```powershell
param(
  [string]$FairfacePkl="..\models\sg2ada_fairface.pkl",
  [string]$FfhqPkl="..\modelsfhq.pkl",
  [string]$StarGanCkpt="..\ext\stargan_ckpt@000_nets_ema.ckpt",
  [string]$SdModel="stabilityai/stable-diffusion-2-1",
  [string]$OutDir="..
uns\demo_outputs"
)
conda activate facegan
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:OMP_NUM_THREADS="1"; $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"
cd apps
python gradio_demo.py --fairface_pkl $FairfacePkl --ffhq_pkl $FfhqPkl --stargan_ckpt $StarGanCkpt --sd_model $SdModel --out_dir $OutDir
```

### `scripts/run_gradio_unix.sh`
```bash
#!/usr/bin/env bash
set -e
source .venv/bin/activate 2>/dev/null || conda activate facegan
cd apps
python gradio_demo.py   --fairface_pkl ../models/sg2ada_fairface.pkl   --ffhq_pkl ../models/ffhq.pkl   --stargan_ckpt ../ext/stargan_ckpt/100000_nets_ema.ckpt   --sd_model "stabilityai/stable-diffusion-2-1"   --out_dir ../runs/demo_outputs
```

---

## 📌 Récapitulatif commandes

```bash
# (Windows) créer env + installer
conda create -n facegan python=3.10 -y
conda activate facegan
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install -r requirements.txt
pip install ftfy regex tqdm "git+https://github.com/openai/CLIP.git"
pip install diffusers transformers accelerate safetensors

# (Option) préparer dataset stylegan
python dataset_tool.py --source=data/fairface_filtered/images --dest=data/fairface_256.zip --resolution=256x256

# entraîner SG2-ADA conditionnel
python train.py --outdir=runs/training/sg2ada_fairface256 --data=data/fairface_256.zip   --gpus=1 --batch=64 --cfg=stylegan2 --cbase=32768 --cmax=512 --gamma=10 --kimg=6000 --snap=50 --cond=1

# lancer Gradio
cd apps
python gradio_demo.py --fairface_pkl ../models/sg2ada_fairface.pkl --ffhq_pkl ../models/ffhq.pkl   --stargan_ckpt ../ext/stargan_ckpt/100000_nets_ema.ckpt --sd_model "stabilityai/stable-diffusion-2-1"   --out_dir ../runs/demo_outputs
```
