# facegan/data/dataset.py
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd
import os

class FaceAttrsDataset(Dataset):
    def __init__(self, csv, split=None):
        df = pd.read_csv(csv)
        # Ne filtre que si split demandé ET colonne présente
        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.paths = df["path"].tolist()
        self.age = torch.tensor(df["age_bin"].values, dtype=torch.long)
        self.gender = torch.tensor(df["gender"].values, dtype=torch.long)
        self.eth = torch.tensor(df["ethnicity"].values, dtype=torch.long)

        self.t = T.Compose([
            T.Resize((256,256), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(0.5, 0.5)
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.t(img)
        return img, self.age[idx], self.gender[idx], self.eth[idx]

# Aliases de compat
try: FacesAttrDataset
except NameError: FacesAttrDataset = FaceAttrsDataset
try: FaceAttrsDataset
except NameError: FaceAttrsDataset = FacesAttrDataset