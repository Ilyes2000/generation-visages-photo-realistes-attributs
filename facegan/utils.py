import os, torch
from torchvision.utils import save_image

def save_grid(tensor, path, nrow=8):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensor = (tensor.clamp(-1,1) + 1)/2.0
    save_image(tensor, path, nrow=nrow)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kwargs, path)

def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
