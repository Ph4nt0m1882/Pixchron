import os
import json
from PIL import Image
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class Dataset:
        pass
    class DataLoader:
        pass

class MultimodalPixelArtDataset(Dataset):
    """
    Dataset PyTorch pour charger les données d'entraînement multimodales de Pixchron.
    Associe chaque sprite pixel art cible (RGBA) avec son conditionnement :
    - Texte (Caption VLM)
    - Image de référence (Photo/Art réel)
    - Croquis d'esquisse (Pencil Sketch)
    """

    def __init__(
        self,
        root_dir: str = "datasets_multimodal",
        modality: str = "all",
        image_size: int = 64
    ):
        self.root_dir = root_dir
        self.modality = modality
        self.image_size = image_size
        self.metadata_file = os.path.join(root_dir, "metadata.jsonl")
        
        self.items = []
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if modality == "all" or entry.get("modality") == modality:
                            self.items.append(entry)
        else:
            print(f"⚠️ Fichier {self.metadata_file} introuvable.")

        # Transformations : Préservation absolue des pixels nets pour la cible (NEAREST)
        if HAS_TORCH:
            self.transform_target = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            
            # Pour les conditions (photos réelles ou croquis), interpolation douce standard
            self.transform_cond = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor()
            ])
        else:
            self.transform_target = None
            self.transform_cond = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # 1. Chargement de l'image cible (Pixel Art)
        target_path = os.path.join(self.root_dir, item["target_image"])
        try:
            target_img = Image.open(target_path).convert("RGBA")
        except Exception:
            target_img = Image.new("RGBA", (self.image_size, self.image_size), (0, 0, 0, 0))
            
        if HAS_TORCH:
            target_tensor = self.transform_target(target_img)
            target_tensor = (target_tensor * 2.0) - 1.0
            cond_tensor = torch.zeros((3, self.image_size, self.image_size))
        else:
            target_tensor = np.array(target_img)
            cond_tensor = np.zeros((self.image_size, self.image_size, 3))
            
        has_cond_image = False
        
        cond_rel_path = item.get("reference_image") or item.get("sketch_image")
        if cond_rel_path:
            full_cond_path = os.path.join(self.root_dir, cond_rel_path)
            if os.path.exists(full_cond_path):
                try:
                    cond_img = Image.open(full_cond_path).convert("RGB")
                    if HAS_TORCH:
                        cond_tensor = self.transform_cond(cond_img)
                        cond_tensor = (cond_tensor * 2.0) - 1.0
                    else:
                        cond_tensor = np.array(cond_img)
                    has_cond_image = True
                except Exception:
                    pass

        return {
            "target": target_tensor,
            "modality": item.get("modality", "text"),
            "caption": item.get("caption", ""),
            "cond_image": cond_tensor,
            "has_cond_image": has_cond_image,
            "category": item.get("category", "")
        }

class PixelArtDataset(Dataset):
    """Dataset historique simplifié pour tester avec HuggingFace."""
    def __init__(self, hf_dataset_name="huggan/pokemon", split="train", image_size=64):
        from datasets import load_dataset
        self.dataset = load_dataset(hf_dataset_name, split=split)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item.get('image') or item.get('img')
        if not isinstance(img, Image.Image):
            img = next(val for val in item.values() if isinstance(val, Image.Image))
        img = img.convert('RGBA')
        tensor_img = self.transform(img)
        return (tensor_img * 2) - 1.0

def get_multimodal_dataloader(root_dir="datasets_multimodal", modality="all", batch_size=16, image_size=64):
    dataset = MultimodalPixelArtDataset(root_dir=root_dir, modality=modality, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == "__main__":
    ds = MultimodalPixelArtDataset()
    print(f"Chargé avec succès : {len(ds)} échantillons multimodaux.")
