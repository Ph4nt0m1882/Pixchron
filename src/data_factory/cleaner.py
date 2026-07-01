import math
from PIL import Image
import numpy as np
from typing import Tuple, Optional
from schema import PixelArtMetadata

class PixelArtCleaner:
    def __init__(self, max_colors_allowed=256):
        self.max_colors = max_colors_allowed

    def extract_palette_size(self, img_array: np.ndarray) -> int:
        pixels = img_array.reshape(-1, img_array.shape[-1])
        unique_colors = np.unique(pixels, axis=0)
        return len(unique_colors)

    def detect_pixel_scale(self, img_array: np.ndarray) -> int:
        """
        Détecte l'échelle d'upscaling de l'image (ex: si chaque pixel est dessiné comme un bloc 2x2 ou 3x3).
        Si l'image contient des 'mixels' (mélange de tailles de pixels, très courant dans les faux jeux rétro),
        cette fonction retournera 1 (car le PGCD sera 1).
        """
        # Trouver où les couleurs changent horizontalement
        diffs_h = np.any(img_array[:, 1:] != img_array[:, :-1], axis=-1)
        # Trouver les distances entre les changements (longueurs des blocs de même couleur)
        run_lengths_h = []
        for row in diffs_h:
            changes = np.where(row)[0]
            if len(changes) > 1:
                run_lengths_h.extend(np.diff(changes))
                
        # Trouver où les couleurs changent verticalement
        diffs_v = np.any(img_array[1:, :] != img_array[:-1, :], axis=-1)
        run_lengths_v = []
        for col in diffs_v.T:
            changes = np.where(col)[0]
            if len(changes) > 1:
                run_lengths_v.extend(np.diff(changes))
                
        all_runs = run_lengths_h + run_lengths_v
        if not all_runs:
            return 1 # Image unie
            
        # Le PGCD de toutes les longueurs de segments nous donne la taille du plus petit "pixel" dessiné
        true_scale = all_runs[0]
        for run in all_runs[1:]:
            true_scale = math.gcd(true_scale, run)
            if true_scale == 1:
                break # Dès qu'on trouve 1, on sait qu'il y a de vrais pixels 1x1 (ou des mixels)
                
        return true_scale

    def process_image(self, filepath: str, source_url: str = "", license_type: str = "private_train") -> Tuple[Optional[Image.Image], Optional[PixelArtMetadata]]:
        try:
            img = Image.open(filepath).convert("RGBA")
            img_array = np.array(img)
        except Exception as e:
            print(f"Erreur de lecture de {filepath}: {e}")
            return None, None

        width, height = img.size
        
        # 0. Rejet immédiat si l'image est trop petite à la base
        if width < 4 or height < 4:
            print(f"REJET: {filepath} est trop petite nativement ({width}x{height}).")
            return None, None
            
        # 1. Détection des faux pixels (Upscaling brutal sans nettoyage ou Mixels)
        scale = self.detect_pixel_scale(img_array)
        if scale > 1:
            new_width = max(1, width // scale)
            new_height = max(1, height // scale)
            img = img.resize((new_width, new_height), Image.Resampling.NEAREST)
            img_array = np.array(img)
            width, height = new_width, new_height
            print(f"[{filepath}] Échelle {scale}x détectée. Réduction à la résolution native : {width}x{height}")
            
        if width < 4 or height < 4:
            print(f"REJET: {filepath} est trop petite après réduction ({width}x{height}).")
            return None, None
            
        # 1. Vérification stricte du pixel art (Nombre de couleurs)
        palette_size = self.extract_palette_size(img_array)
        if palette_size > self.max_colors:
            print(f"REJET: {filepath} contient {palette_size} couleurs (anti-aliasing ou photo détectée).")
            return None, None
            
        # 2. Détection d'un fond solide pour le remplacer par de la transparence
        corners = [
            img_array[0, 0], 
            img_array[0, width-1], 
            img_array[height-1, 0], 
            img_array[height-1, width-1]
        ]
        has_background = False
        if all(np.array_equal(corners[0], c) for c in corners):
            bg_color = corners[0]
            if bg_color[3] > 0:
                has_background = True
                mask = np.all(img_array == bg_color, axis=-1)
                img_array[mask] = [0, 0, 0, 0]
                img = Image.fromarray(img_array)

        # 3. Création des métadonnées avec le schéma strict
        metadata = PixelArtMetadata(
            license_type=license_type,
            dataset_bucket="safe_hf" if license_type in ["public_domain", "cc-by"] else "private_train",
            description="", 
            width=width,
            height=height,
            has_background=not has_background, 
            palette_size=palette_size,
            is_animation=False,
            frames=1,
            source_url=source_url
        )

        return img, metadata

        return img, metadata

if __name__ == "__main__":
    # Petit test bidon pour valider la compilation
    cleaner = PixelArtCleaner()
    print("Module Cleaner initialisé avec succès. Prêt à traiter des To de données !")
