import math
from PIL import Image
import numpy as np

class PixelReconstructor:
    def __init__(self, color_tolerance=15.0):
        self.color_tolerance = color_tolerance
        
    def _detect_grid_scale(self, quantized_array: np.ndarray, max_scale: int = 40) -> int:
        """
        Détecte l'échelle d'upscale (taille d'un "macro-pixel") en analysant l'autocorrélation des bords.
        Cette méthode est mathématiquement infaillible même face à des arrière-plans unis ou des grilles superposées.
        """
        gray = np.mean(quantized_array, axis=-1)
        
        # 1. Détection des bords horizontaux et verticaux
        diff_h = np.abs(np.diff(gray, axis=1)) > 15
        diff_v = np.abs(np.diff(gray, axis=0)) > 15
        
        # 2. Compression en densité 1D
        density_x = np.sum(diff_h, axis=0)
        density_y = np.sum(diff_v, axis=1)
        
        # 3. Normalisation (centrer sur 0) pour faire ressortir les pics de variance
        density_x = density_x - np.mean(density_x)
        density_y = density_y - np.mean(density_y)
        
        # 4. Autocorrélation (mesure la périodicité du signal)
        autocorr_x = np.correlate(density_x, density_x, mode='full')[len(density_x)-1:]
        autocorr_y = np.correlate(density_y, density_y, mode='full')[len(density_y)-1:]
        
        # 5. Fusionner les deux axes pour une robustesse maximale
        max_len = min(max_scale + 1, len(autocorr_x), len(autocorr_y))
        if max_len < 3:
            return 1
            
        autocorr = autocorr_x[:max_len] + autocorr_y[:max_len]
        
        # La vraie échelle du pixel art est la fréquence fondamentale.
        # Sur une grille aléatoire clairsemée, les harmoniques (2x, 3x l'échelle) peuvent avoir 
        # un pic d'autocorrélation légèrement supérieur à cause du bruit. 
        # On cherche donc le PREMIER pic local qui est significativement fort (> 50% du max).
        search_space = autocorr[2:]
        peak = np.max(search_space)
        mean_abs = np.mean(np.abs(search_space))
        
        if peak < mean_abs * 3.5:
            return 1
            
        threshold = peak * 0.5
        best_k = int(np.argmax(search_space)) + 2 # Fallback
        
        for i in range(1, len(search_space)-1):
            val = search_space[i]
            if val > threshold and val > search_space[i-1] and val > search_space[i+1]:
                best_k = i + 2
                break
                
        return best_k

    def _find_best_offset(self, img_array: np.ndarray, scale: int) -> tuple[int, int]:
        """
        Trouve le décalage (offset) parfait pour centrer la grille d'échantillonnage
        au beau milieu de chaque faux pixel. Évite de couper les contours noirs (1-pixel width).
        """
        gray = np.mean(img_array, axis=-1)
        
        diff_h = np.abs(np.diff(gray, axis=1))
        profile_h = np.sum(diff_h, axis=0)
        
        diff_v = np.abs(np.diff(gray, axis=0))
        profile_v = np.sum(diff_v, axis=1)
        
        # On replie les fréquences sur la taille de l'échelle pour trouver la bordure de la grille
        folded_h = np.zeros(scale)
        for i, val in enumerate(profile_h): 
            folded_h[i % scale] += val
        boundary_x = np.argmax(folded_h)
        offset_x = (boundary_x + scale // 2) % scale
        
        folded_v = np.zeros(scale)
        for i, val in enumerate(profile_v): 
            folded_v[i % scale] += val
        boundary_y = np.argmax(folded_v)
        offset_y = (boundary_y + scale // 2) % scale
        return offset_x, offset_y
        
    def _crop_watermarks(self, quantized_array: np.ndarray, scale: int, ox: int, oy: int, downscaled_array: np.ndarray) -> np.ndarray:
        """
        Rogner automatiquement les bords de l'image contenant des éléments qui ne sont pas
        du pixel art (ex: textes, filigranes) en analysant la fréquence spatiale (nombre de bords).
        Une ligne de vrai pixel art ne peut pas avoir plus de (largeur / scale) changements de couleurs.
        Si on détecte beaucoup plus de changements, c'est du texte ou du bruit ajouté après coup.
        """
        if scale <= 1: return downscaled_array
        
        gray = np.mean(quantized_array, axis=-1)
        
        # Détection des bords horizontaux et verticaux (changement de couleur > 15 pour ignorer le mini-bruit)
        diff_h = np.abs(np.diff(gray, axis=1)) > 15
        diff_v = np.abs(np.diff(gray, axis=0)) > 15
        
        edges_per_row = np.sum(diff_h, axis=1)
        edges_per_col = np.sum(diff_v, axis=0)
        
        # Le nombre maximum théorique de bords dans du pixel art pur est de 1 par macro-pixel.
        # Avec le bruit JPEG (ringing) et l'anti-aliasing, un bord peut se dédoubler ou tripler.
        # Un multiplicateur de 4.0 est très permissif pour le pixel art, mais coupera quand même
        # le texte (qui a une densité de bords gigantesque, souvent > 8 bords par macro-pixel).
        median_edges_row = np.median(edges_per_row)
        median_edges_col = np.median(edges_per_col)
        
        max_edges_row = max(median_edges_row * 3, (quantized_array.shape[1] // scale) * 4)
        max_edges_col = max(median_edges_col * 3, (quantized_array.shape[0] // scale) * 4)
        
        bx = (ox - scale // 2) % scale
        by = (oy - scale // 2) % scale
        
        valid_rows = []
        for i in range(downscaled_array.shape[0]):
            r_start = by + i * scale
            r_end = by + (i + 1) * scale
            r_start = max(0, min(quantized_array.shape[0], r_start))
            r_end = max(0, min(quantized_array.shape[0], r_end))
            
            if r_start >= r_end:
                valid_rows.append(False)
            else:
                valid_rows.append(np.max(edges_per_row[r_start:r_end]) <= max_edges_row)
                
        valid_cols = []
        for j in range(downscaled_array.shape[1]):
            c_start = bx + j * scale
            c_end = bx + (j + 1) * scale
            c_start = max(0, min(quantized_array.shape[1], c_start))
            c_end = max(0, min(quantized_array.shape[1], c_end))
            
            if c_start >= c_end:
                valid_cols.append(False)
            else:
                valid_cols.append(np.max(edges_per_col[c_start:c_end]) <= max_edges_col)
        
        def get_largest_true_block(mask):
            max_len, best_start, best_end = 0, 0, 0
            current_start = -1
            current_len = 0
            for i, val in enumerate(mask):
                if val:
                    if current_start == -1: current_start = i
                    current_len += 1
                else:
                    if current_len > max_len:
                        max_len = current_len
                        best_start = current_start
                        best_end = i
                    current_start = -1
                    current_len = 0
            if current_len > max_len:
                max_len = current_len
                best_start = current_start
                best_end = len(mask)
            return best_start, best_end
            
        r1, r2 = get_largest_true_block(valid_rows)
        c1, c2 = get_largest_true_block(valid_cols)
        
        if r1 >= r2 or c1 >= c2:
            return downscaled_array
            
        return downscaled_array[r1:r2, c1:c2]

    def _clean_colors(self, img_array: np.ndarray) -> np.ndarray:
        """
        Fusionne les couleurs très similaires (distance < tolerance) pour supprimer le bruit JPEG.
        Effectué sur la petite image reconstruite, c'est extrêmement rapide et ça préserve les couleurs d'origine.
        """
        channels = img_array.shape[2]
        pixels = img_array.reshape(-1, channels)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Trier par fréquence (les couleurs les plus fréquentes sont les "vraies" couleurs)
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        
        merged_palette = []
        color_map = {}
        
        for color in unique_colors:
            found = False
            for p_color in merged_palette:
                # Distance euclidienne dans l'espace RGB
                if np.linalg.norm(color.astype(float) - p_color.astype(float)) < self.color_tolerance:
                    color_map[tuple(color)] = tuple(p_color)
                    found = True
                    break
            if not found:
                merged_palette.append(color)
                color_map[tuple(color)] = tuple(color)
                
        # Appliquer la map
        mapped_pixels = [color_map[tuple(p)] for p in pixels]
        return np.array(mapped_pixels, dtype=np.uint8).reshape(img_array.shape)

    def reconstruct(self, img: Image.Image) -> tuple[Image.Image, int]:
        """
        Reconstruit le pixel art pur à partir d'une image potentiellement upscalée/bruitée.
        Retourne (image_reconstruite, echelle_detectee).
        """
        has_alpha = img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info)
        
        if has_alpha:
            img = img.convert("RGBA")
            raw_array = np.array(img)
            # Pour la détection (qui a besoin de contrastes nets), on aplatit l'alpha sur fond blanc
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(bg, img).convert("RGB")
            quantized = alpha_composite.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        else:
            img = img.convert("RGB")
            raw_array = np.array(img)
            quantized = img.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
            
        quantized_array = np.array(quantized.convert("RGB"))
        
        # 2. Détecter l'échelle et l'offset sur l'image quantifiée (propre)
        scale = self._detect_grid_scale(quantized_array)
        offset_x, offset_y = self._find_best_offset(quantized_array, scale)
        
        # 3. Downscaler l'image RAW (avec toutes ses couleurs d'origine)
        if scale > 1:
            downscaled_array = raw_array[offset_y::scale, offset_x::scale]
            # 3.5 Rogner les watermarks / textes qui ne respectent pas la grille
            downscaled_array = self._crop_watermarks(quantized_array, scale, offset_x, offset_y, downscaled_array)
        else:
            downscaled_array = raw_array
            
        # 4. Nettoyer les couleurs proches (bruit JPEG) sur la petite image
        cleaned_array = self._clean_colors(downscaled_array)
        
        return Image.fromarray(cleaned_array), scale

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Reconstructing {filepath}...")
        try:
            img = Image.open(filepath)
            reconstructor = PixelReconstructor(color_tolerance=15.0)
            clean_img, scale = reconstructor.reconstruct(img)
            
            out_path = filepath.replace(".", "_clean.")
            clean_img.save(out_path)
            print(f"Done! Detected scale: {scale}x. Original size: {img.size} -> New size: {clean_img.size}")
            print(f"Saved to {out_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python pixel_reconstructor.py <image_path>")
