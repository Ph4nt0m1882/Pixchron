import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

class PixelArtDataset(Dataset):
    """
    Dataset PyTorch personnalisé pour charger des images Pixel Art avec canal Alpha (RGBA).
    """
    def __init__(self, hf_dataset_name="Falah/pixel_art", split="train", image_size=64):
        print(f"Téléchargement/Chargement du dataset HF : {hf_dataset_name}")
        # Charge le dataset depuis HuggingFace
        self.dataset = load_dataset(hf_dataset_name, split=split)
        self.image_size = image_size
        
        # Transformation : on évite toute interpolation qui créerait du flou (anti-aliasing)
        # On force le nearest neighbor pour le redimensionnement.
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() # Convertit en tenseur [C, H, W] et normalise entre 0 et 1
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # La colonne de l'image s'appelle souvent 'image' dans les datasets HF
        if 'image' in item:
            img = item['image']
        elif 'img' in item:
            img = item['img']
        else:
            # Si le nom est différent, on prend la première valeur qui est une image PIL
            img = next(val for val in item.values() if isinstance(val, Image.Image))
            
        # Forcer la conversion en RGBA pour avoir 4 canaux
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
            
        # Appliquer les transformations
        tensor_img = self.transform(img)
        
        # Normaliser entre -1 et 1 (standard pour la diffusion)
        tensor_img = (tensor_img * 2) - 1.0
        
        return tensor_img

def get_dataloader(batch_size=16, image_size=64):
    dataset = PixelArtDataset(image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == "__main__":
    # Test rapide
    dataset = PixelArtDataset()
    print(f"Nombre d'images : {len(dataset)}")
    sample = dataset[0]
    print(f"Shape d'un sample : {sample.shape}") # Devrait être [4, 64, 64]
    print(f"Min: {sample.min()}, Max: {sample.max()}")
