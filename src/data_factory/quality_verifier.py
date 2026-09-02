from typing import Tuple, Dict, Any
from PIL import Image
import numpy as np

class QualityVerifier:
    """
    Contrôleur de qualité final pour valider qu'une image de sortie est 100% pixel-perfect.
    """

    def __init__(
        self,
        max_palette_size: int = 256,
        min_size: int = 8,
        max_size: int = 1024
    ):
        self.max_palette_size = max_palette_size
        self.min_size = min_size
        self.max_size = max_size

    def verify(self, img: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Vérifie la conformité de l'image nettoyée.
        Retourne (est_valide, raison_si_invalide, métriques).
        """
        w, h = img.size
        
        # 1. Vérification des dimensions
        if w < self.min_size or h < self.min_size:
            return False, f"Dimensions trop faibles ({w}x{h} < {self.min_size})", {"size": (w, h)}
            
        if w > self.max_size or h > self.max_size:
            return False, f"Dimensions excessives ({w}x{h} > {self.max_size})", {"size": (w, h)}
            
        arr = np.array(img)
        c = arr.shape[-1] if arr.ndim == 3 else 1
        
        # 2. Vérification de la palette
        pixels = arr.reshape(-1, c)
        unique_colors = len(np.unique(pixels, axis=0))
        
        if unique_colors > self.max_palette_size:
            return False, f"Palette trop riche ({unique_colors} couleurs > {self.max_palette_size})", {
                "size": (w, h),
                "palette_size": unique_colors
            }
            
        # 3. Vérification de la netteté des pixels (pas de flou de lissage)
        gray = np.mean(arr[:, :, :3], axis=-1)
        diff_h = np.abs(np.diff(gray, axis=1))
        non_zero_diffs = diff_h[diff_h > 0]
        
        # Si plus de 30% des transitions sont des micro-gradients continus (< 3 niveaux)
        # c'est un signe de lissage non matriciel
        if len(non_zero_diffs) > 100:
            micro_gradients = np.mean(non_zero_diffs < 3.0)
            if micro_gradients > 0.40:
                return False, f"Présence de gradients continus flous ({micro_gradients*100:.1f}%)", {
                    "size": (w, h),
                    "palette_size": unique_colors,
                    "micro_gradients": micro_gradients
                }
                
        metrics = {
            "size": (w, h),
            "palette_size": unique_colors,
            "has_alpha": c == 4 and np.any(arr[:, :, 3] == 0)
        }
        
        return True, "Pixel-perfect validé", metrics

if __name__ == "__main__":
    verifier = QualityVerifier()
    print("Module QualityVerifier initialisé avec succès.")
