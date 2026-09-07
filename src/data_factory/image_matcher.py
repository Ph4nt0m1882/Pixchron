import os
import glob
import urllib.parse
import urllib.request
import json
import time
from typing import Optional, List, Tuple
from PIL import Image
import numpy as np

class ImageReferenceMatcher:
    """
    Apparie un sprite pixel art avec la photo ou l'illustration réelle la plus sémantiquement
    proche via CLIP ou SigLIP pour l'entraînement de génération Image-to-PixelArt.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        raw_scraped_dir: str = "datasets_ready",
        device: str = "cuda",
        enable_web_fallback: bool = True
    ):
        self.model_name = model_name
        self.raw_scraped_dir = raw_scraped_dir
        self.device = device
        self.enable_web_fallback = enable_web_fallback
        self.model = None
        self.processor = None
        self._init_clip()

    def _init_clip(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModel
            
            if not torch.cuda.is_available() and self.device == "cuda":
                self.device = "cpu"
                
            print(f"🖼️ Chargement du modèle de similarité visuelle {self.model_name} sur {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            print("✅ Modèle CLIP / SigLIP prêt !")
        except Exception as e:
            print(f"⚠️ CLIP non initialisé ({e}). Mode fallback heuristique activé.")

    def find_best_reference(
        self,
        pixel_art_img: Image.Image,
        category: str,
        target_path: Optional[str] = None
    ) -> Image.Image:
        """
        Trouve la meilleure image de référence pour ce sprite.
        1. Cherche dans le répertoire brut local datasets_ready/<category>/
        2. Sinon, cherche 3 candidats web (photo réelle) et prend le meilleur selon CLIP.
        """
        candidates = self._get_local_candidates(category, target_path)
        
        if not candidates and self.enable_web_fallback:
            candidates = self._fetch_web_candidates(category)
            
        if not candidates:
            # Fallback synthétique si aucun candidat : version lissée/filtrée du sujet
            return self._create_fallback_reference(pixel_art_img)

        # Si CLIP n'est pas chargé ou 1 seul candidat
        if self.model is None or len(candidates) == 1:
            return candidates[0]

        # Sélection du meilleur candidat via score cosinus CLIP
        best_img = self._rank_candidates_with_clip(pixel_art_img, candidates)
        return best_img

    def _get_local_candidates(self, category: str, target_path: Optional[str]) -> List[Image.Image]:
        candidates = []
        cat_dir = os.path.join(self.raw_scraped_dir, category)
        if not os.path.exists(cat_dir):
            return candidates
            
        target_name = os.path.basename(target_path) if target_path else ""
        
        # Parcourir les fichiers bruts (jpg, webp, png)
        for f in os.listdir(cat_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.jpg', '.jpeg', '.webp', '.png') and f != target_name:
                p = os.path.join(cat_dir, f)
                try:
                    cand = Image.open(p).convert("RGB")
                    # On privilégie les images plus grandes (photos/art réel)
                    if cand.size[0] >= 64 and cand.size[1] >= 64:
                        candidates.append(cand)
                    if len(candidates) >= 15:
                        break
                except Exception:
                    continue
        return candidates

    def _fetch_web_candidates(self, category: str, count: int = 3) -> List[Image.Image]:
        """Recherche rapide DuckDuckGo d'images réelles pour la catégorie."""
        candidates = []
        clean_cat = category.replace("_", " ").strip()
        query = f"photo illustration {clean_cat}"
        
        try:
            url = f"https://duckduckgo.com/i.js?q={urllib.parse.quote(query)}&o=json"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data.get("results", [])
                
            for res in results[:count]:
                img_url = res.get("image")
                if img_url:
                    try:
                        img_req = urllib.request.Request(img_url, headers=headers)
                        with urllib.request.urlopen(img_req, timeout=5) as r:
                            cand = Image.open(r).convert("RGB")
                            candidates.append(cand)
                    except Exception:
                        continue
        except Exception:
            pass
            
        return candidates

    def _create_fallback_reference(self, pixel_art_img: Image.Image) -> Image.Image:
        """Génère une référence douce si aucune photo externe n'est trouvée."""
        from PIL import ImageFilter
        rgb = pixel_art_img.convert("RGB")
        upscaled = rgb.resize((256, 256), Image.Resampling.BILINEAR)
        blurred = upscaled.filter(ImageFilter.SMOOTH_MORE)
        return blurred

    def _rank_candidates_with_clip(self, target_img: Image.Image, candidates: List[Image.Image]) -> Image.Image:
        import torch
        
        try:
            target_rgb = target_img.convert("RGB").resize((224, 224))
            cand_rgbs = [c.convert("RGB").resize((224, 224)) for c in candidates]
            
            all_images = [target_rgb] + cand_rgbs
            inputs = self.processor(images=all_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                # Supporte CLIP et SigLIP
                if hasattr(self.model, "get_image_features"):
                    features = self.model.get_image_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    features = outputs.image_embeds if hasattr(outputs, "image_embeds") else outputs.last_hidden_state[:, 0]
                    
            # Normalisation
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            target_feat = features[0:1]
            cand_feats = features[1:]
            
            similarities = torch.matmul(cand_feats, target_feat.T).squeeze(-1).cpu().numpy()
            best_idx = int(np.argmax(similarities))
            return candidates[best_idx]
        except Exception as e:
            return candidates[0]

if __name__ == "__main__":
    matcher = ImageReferenceMatcher()
    print("Module ImageReferenceMatcher initialisé avec succès.")
