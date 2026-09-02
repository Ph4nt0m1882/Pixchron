import os
from typing import Optional, Tuple
from PIL import Image
import numpy as np

class NeuralBackgroundRemover:
    """
    Suppresseur d'arrière-plan neural haute précision pour sprites et pixel art.
    Supporte RMBG-1.4 / RMBG-2.0 sur GPU CUDA avec seuillage net pour éviter les halos semi-transparents.
    Possède un fallback algorithmique (flood-fill & proximité des coins).
    """

    def __init__(
        self,
        model_name: str = "briaai/RMBG-1.4",
        device: str = "cuda",
        alpha_threshold: float = 0.5,
        use_neural: bool = True
    ):
        self.model_name = model_name
        self.device = device
        self.alpha_threshold = alpha_threshold
        self.use_neural = use_neural
        self.model = None
        
        if self.use_neural:
            self._init_neural_model()

    def _init_neural_model(self):
        """Charge le modèle RMBG sur GPU si disponible."""
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
            
            if not torch.cuda.is_available() and self.device == "cuda":
                self.device = "cpu"
                
            print(f"🎨 Chargement du modèle de détourage {self.model_name} sur {self.device}...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device).eval()
            print("✅ Modèle de détourage neural prêt !")
        except Exception as e:
            print(f"⚠️ RMBG non disponible ({e}). Utilisation du détourage algorithmique par défaut.")
            self.model = None

    def remove_background(self, img: Image.Image) -> Image.Image:
        """
        Supprime le fond de l'image et renvoie une image RGBA avec canal Alpha net.
        """
        # Si déjà en RGBA avec des zones transparentes significatives
        if img.mode == "RGBA":
            alpha_ch = np.array(img)[:, :, 3]
            if np.mean(alpha_ch == 0) > 0.1:
                return img # Déjà proprement détouré
                
        # 1. Utilisation du modèle Neural si disponible
        if self.model is not None:
            try:
                return self._remove_neural(img)
            except Exception as e:
                print(f"Erreur détourage neural : {e}, bascule sur algorithmique.")
                
        # 2. Fallback algorithmique (coins + flood fill)
        return self._remove_algorithmic(img)

    def _remove_neural(self, img: Image.Image) -> Image.Image:
        """Exécute RMBG sur l'image."""
        import torch
        import torchvision.transforms.functional as TF
        
        orig_size = img.size
        # RMBG attend une entrée normalisée de 1024x1024
        input_img = img.convert("RGB").resize((1024, 1024), Image.Resampling.BILINEAR)
        tensor_img = TF.to_tensor(input_img).unsqueeze(0).to(self.device)
        tensor_img = TF.normalize(tensor_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        with torch.no_grad():
            preds = self.model(tensor_img)[-1].sigmoid().cpu()
            
        mask = preds[0].squeeze()
        # Seuillage net pour pixel art (pas de semi-transparence floue)
        binary_mask = (mask > self.alpha_threshold).float().numpy()
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8)).resize(orig_size, Image.Resampling.NEAREST)
        
        rgba = img.convert("RGBA")
        rgba.putalpha(mask_img)
        return rgba

    def _remove_algorithmic(self, img: Image.Image, tolerance: float = 25.0) -> Image.Image:
        """Détourage géométrique par les coins et bords."""
        arr = np.array(img.convert("RGBA"))
        h, w = arr.shape[:2]
        
        corners = [arr[0, 0, :3], arr[0, w-1, :3], arr[h-1, 0, :3], arr[h-1, w-1, :3]]
        ref_bg = np.median(corners, axis=0)
        
        # Vérifier si les 4 coins sont proches
        if all(np.linalg.norm(c.astype(float) - ref_bg) < tolerance for c in corners):
            diff = np.linalg.norm(arr[:, :, :3].astype(float) - ref_bg, axis=-1)
            bg_mask = diff < tolerance
            arr[bg_mask, 3] = 0
            
        return Image.fromarray(arr)

if __name__ == "__main__":
    remover = NeuralBackgroundRemover(use_neural=False)
    print("Module NeuralBackgroundRemover initialisé avec succès.")
