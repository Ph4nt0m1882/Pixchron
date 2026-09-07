import os
from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageOps
import numpy as np

class AISketchGenerator:
    """
    Générateur d'esquisses et de croquis d'artistes pour le conditionnement Sketch-to-PixelArt.
    Transforme le sprite pixel art en un croquis au crayon graphite préparatoire continu
    (contours fluides, hachures douces, texture papier).
    """

    def __init__(
        self,
        output_size: int = 256,
        pencil_stroke_strength: float = 1.6,
        paper_white: bool = True
    ):
        self.output_size = output_size
        self.pencil_stroke_strength = pencil_stroke_strength
        self.paper_white = paper_white

    def generate_sketch(self, img: Image.Image) -> Image.Image:
        """
        Génère un croquis d'esquisse haute qualité à partir d'un sprite pixel art.
        """
        w, h = img.size
        # 1. Si RGBA, extraire le masque alpha et composer sur fond blanc
        has_alpha = img.mode == "RGBA"
        if has_alpha:
            alpha = np.array(img)[:, :, 3]
            alpha_mask = Image.fromarray(alpha)
            bg = Image.new("RGBA", (w, h), (255, 255, 255, 255))
            composite = Image.alpha_composite(bg, img).convert("L")
        else:
            composite = img.convert("L")
            alpha_mask = None

        # 2. Redimensionner en haute résolution avec un filtre doux (anti-pixelisation)
        # pour obtenir des contours de dessin fluides et non des marches d'escalier
        target_w = max(self.output_size, w * 4)
        target_h = max(self.output_size, h * 4)
        smooth_res = composite.resize((target_w, target_h), Image.Resampling.BILINEAR)

        # 3. Extraction d'arêtes multi-échelles par Différence de Gaussiennes (DoG)
        # Simule le coup de crayon d'un dessinateur
        arr = np.array(smooth_res, dtype=float)
        
        # Filtre Gaussien fin (détails) et large (ombres)
        im_blur1 = smooth_res.filter(ImageFilter.GaussianBlur(radius=1.2))
        im_blur2 = smooth_res.filter(ImageFilter.GaussianBlur(radius=3.5))
        
        g1 = np.array(im_blur1, dtype=float)
        g2 = np.array(im_blur2, dtype=float)
        
        # Formule de DoG accentuée pour traits au crayon graphite
        dog = g1 - 0.90 * g2
        
        # Inversion pour avoir traits sombres sur papier clair
        # Normalisation et contraste
        dog_norm = 255.0 - (np.abs(dog) * self.pencil_stroke_strength * 4.0)
        sketch_arr = np.clip(dog_norm, 0, 255).astype(np.uint8)
        
        # 4. Ajout des contours extérieurs (silhouette de sprite)
        edges = smooth_res.filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(edges, dtype=float)
        sketch_arr = np.clip(sketch_arr.astype(float) - (edge_arr * 0.4), 0, 255).astype(np.uint8)
        
        sketch_img = Image.fromarray(sketch_arr)
        
        # 5. Adoucir légèrement pour reproduire la texture du crayon graphite
        sketch_img = sketch_img.filter(ImageFilter.SMOOTH)
        
        # 6. Redimensionner à la taille finale désirée
        sketch_final = sketch_img.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
        
        # Si on souhaite préserver la transparence du sprite
        if has_alpha and not self.paper_white and alpha_mask is not None:
            alpha_resized = alpha_mask.resize((self.output_size, self.output_size), Image.Resampling.LANCZOS)
            sketch_rgba = sketch_final.convert("RGBA")
            sketch_rgba.putalpha(alpha_resized)
            return sketch_rgba

        return sketch_final.convert("RGB")

if __name__ == "__main__":
    generator = AISketchGenerator()
    print("Module AISketchGenerator initialisé avec succès.")
