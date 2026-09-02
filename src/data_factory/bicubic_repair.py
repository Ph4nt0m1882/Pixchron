import math
from typing import Tuple, Optional
from PIL import Image, ImageFilter
import numpy as np

class BicubicPixelRepairer:
    """
    Moteur de défloutage et de recalage de grille pour le pixel art lissé ou interpolé sur le web (ex: Archange).
    Restaure la netteté matricielle et quantifie les couleurs en palette indexée propre.
    """

    def __init__(
        self,
        target_palette_colors: int = 32,
        sharpen_strength: float = 1.8
    ):
        self.target_palette_colors = target_palette_colors
        self.sharpen_strength = sharpen_strength

    def repair(self, img: Image.Image, estimated_scale: float = 0.0) -> Tuple[Image.Image, float]:
        """
        Défloute, ré-aligne sur la grille discrète et quantifie l'image.
        Retourne (image_réparée_1x, echelle_utilisée).
        """
        orig_w, orig_h = img.size
        
        # 1. Estimation de la grille si non fournie
        if estimated_scale <= 1.0:
            estimated_scale = self._estimate_continuous_scale(img)
            
        # 2. Accentuation ciblée (Unsharp Masking) pour durcir les transitions floutées
        sharpened = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=int(self.sharpen_strength * 100), threshold=3))
        
        # 3. Calcul de la résolution native exacte
        target_w = max(8, int(round(orig_w / estimated_scale)))
        target_h = max(8, int(round(orig_h / estimated_scale)))
        
        # 4. Échantillonnage à la résolution native
        downscaled = sharpened.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # 5. Quantification en palette indexée nette (sans diffusion d'erreur Floyd-Steinberg pour éviter le bruit)
        has_alpha = downscaled.mode == "RGBA"
        if has_alpha:
            alpha = np.array(downscaled)[:, :, 3]
            rgb = downscaled.convert("RGB")
            quantized_rgb = rgb.quantize(colors=self.target_palette_colors, method=Image.Quantize.FASTOCTREE)
            result = quantized_rgb.convert("RGBA")
            result.putalpha(Image.fromarray(alpha))
        else:
            result = downscaled.quantize(colors=self.target_palette_colors, method=Image.Quantize.MEDIANCUT).convert("RGB")
            
        return result, estimated_scale

    def _estimate_continuous_scale(self, img: Image.Image) -> float:
        """Détecte l'échelle de redimensionnement continu via l'autocorrélation spectrale."""
        arr = np.array(img.convert("RGB"))
        gray = np.mean(arr, axis=-1)
        
        diff_h = np.abs(np.diff(gray, axis=1)) > 10
        density_x = np.sum(diff_h, axis=0).astype(float) - np.mean(diff_h)
        ac_x = np.correlate(density_x, density_x, mode='full')[len(density_x) - 1:]
        
        peaks = []
        for k in range(2, min(48, len(ac_x) - 1)):
            if ac_x[k] > ac_x[k-1] and ac_x[k] > ac_x[k+1] and ac_x[k] > 0:
                peaks.append((k, ac_x[k]))
                
        if not peaks:
            return 1.0
            
        max_p = max(p[1] for p in peaks)
        sig_peaks = [p for p in peaks if p[1] > 0.25 * max_p]
        if not sig_peaks:
            return 1.0
            
        first_k = sig_peaks[0][0]
        if len(sig_peaks) > 1 and sig_peaks[1][0] == first_k + 1:
            w1 = sig_peaks[0][1]
            w2 = sig_peaks[1][1]
            return float((first_k * w1 + (first_k + 1) * w2) / (w1 + w2))
            
        return float(first_k)

if __name__ == "__main__":
    repairer = BicubicPixelRepairer()
    print("Module BicubicPixelRepairer initialisé avec succès.")
