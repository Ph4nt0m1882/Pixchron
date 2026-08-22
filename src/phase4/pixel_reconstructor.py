import math
from io import BytesIO
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageSequence
import numpy as np

class PixelReconstructor:
    """
    Moteur de reconstruction et de nettoyage haute précision pour Pixel Art.
    
    Capacités :
    1. Détection robuste de l'échelle d'upscaling (macro-pixels) par autocorrélation spectrale de grille.
    2. Calcul du décalage (offset) de phase pour un échantillonnage sous-pixel centré.
    3. Rognage des textes parasites, filigranes et étiquettes non-pixel art en bordure.
    4. Débruitage des artefacts de compression JPEG par clustering/fusion de palette.
    5. Détection de fond solide (ou bruité) et conversion optionnelle en transparence Alpha.
    6. Support complet des images statiques (PNG/JPG/WEBP) et des animations (GIF multi-frames).
    """

    def __init__(
        self,
        color_tolerance: float = 15.0,
        max_colors: int = 256,
        min_native_size: int = 6,
        remove_background: bool = False,
        bg_tolerance: float = 25.0
    ):
        self.color_tolerance = color_tolerance
        self.max_colors = max_colors
        self.min_native_size = min_native_size
        self.remove_background = remove_background
        self.bg_tolerance = bg_tolerance

    def _detect_grid_scale(self, quantized_array: np.ndarray, max_scale: int = 48) -> int:
        """
        Détecte la taille d'un macro-pixel (ex: 2x, 3x, 4x, 8x, 16x) par autocorrélation
        des profils de gradient horizontal et vertical.
        """
        gray = np.mean(quantized_array[:, :, :3], axis=-1)
        h, w = gray.shape
        
        # Détection des transitions franches (> 15 pour filtrer le micro-bruit)
        diff_h = np.abs(np.diff(gray, axis=1)) > 15
        diff_v = np.abs(np.diff(gray, axis=0)) > 15
        
        density_x = np.sum(diff_h, axis=0).astype(float)
        density_y = np.sum(diff_v, axis=1).astype(float)
        
        if len(density_x) < 4 or len(density_y) < 4:
            return 1
            
        # Normalisation (centrage sur 0)
        density_x -= np.mean(density_x)
        density_y -= np.mean(density_y)
        
        # Autocorrélation 1D
        autocorr_x = np.correlate(density_x, density_x, mode='full')[len(density_x) - 1:]
        autocorr_y = np.correlate(density_y, density_y, mode='full')[len(density_y) - 1:]
        
        max_len = min(max_scale + 1, len(autocorr_x), len(autocorr_y))
        if max_len < 4:
            return 1
            
        autocorr = autocorr_x[:max_len] + autocorr_y[:max_len]
        
        # Recherche du premier pic périodique significatif
        search_space = autocorr[2:]
        if len(search_space) == 0:
            return 1
            
        peak = np.max(search_space)
        mean_abs = np.mean(np.abs(search_space))
        
        if peak < mean_abs * 2.8 or peak <= 0:
            return 1 # Pas de périodicité macro-pixel claire -> 1x
            
        threshold = peak * 0.45
        best_scale = 1
        
        for i in range(1, len(search_space) - 1):
            val = search_space[i]
            if val > threshold and val > search_space[i - 1] and val >= search_space[i + 1]:
                best_scale = i + 2
                break
                
        if best_scale == 1 and peak > mean_abs * 3.5:
            best_scale = int(np.argmax(search_space)) + 2
            
        return max(1, best_scale)

    def _find_best_offset(self, quantized_array: np.ndarray, scale: int) -> Tuple[int, int]:
        """
        Trouve le décalage (offset X/Y) pour que l'échantillonnage tombe pile au centre
        des macro-pixels, évitant ainsi d'amputer les contours noirs d'un pixel de large.
        """
        if scale <= 1:
            return 0, 0
            
        gray = np.mean(quantized_array[:, :, :3], axis=-1)
        
        diff_h = np.abs(np.diff(gray, axis=1))
        profile_h = np.sum(diff_h, axis=0)
        
        diff_v = np.abs(np.diff(gray, axis=0))
        profile_v = np.sum(diff_v, axis=1)
        
        folded_h = np.zeros(scale)
        for i, val in enumerate(profile_h):
            folded_h[i % scale] += val
        boundary_x = int(np.argmax(folded_h))
        offset_x = (boundary_x + scale // 2) % scale
        
        folded_v = np.zeros(scale)
        for i, val in enumerate(profile_v):
            folded_v[i % scale] += val
        boundary_y = int(np.argmax(folded_v))
        offset_y = (boundary_y + scale // 2) % scale
        
        return offset_x, offset_y

    def _crop_watermarks(self, quantized_array: np.ndarray, scale: int, ox: int, oy: int, downscaled_array: np.ndarray) -> np.ndarray:
        """
        Rogne les bordures contenant du texte (haute densité de bords anormale pour du pixel art).
        """
        if scale <= 1 or downscaled_array.shape[0] < 8 or downscaled_array.shape[1] < 8:
            return downscaled_array
            
        gray = np.mean(quantized_array[:, :, :3], axis=-1)
        diff_h = np.abs(np.diff(gray, axis=1)) > 15
        diff_v = np.abs(np.diff(gray, axis=0)) > 15
        
        edges_per_row = np.sum(diff_h, axis=1)
        edges_per_col = np.sum(diff_v, axis=0)
        
        median_edges_row = np.median(edges_per_row)
        median_edges_col = np.median(edges_per_col)
        
        max_edges_row = max(median_edges_row * 3.5, (quantized_array.shape[1] // scale) * 4.0)
        max_edges_col = max(median_edges_col * 3.5, (quantized_array.shape[0] // scale) * 4.0)
        
        bx = (ox - scale // 2) % scale
        by = (oy - scale // 2) % scale
        
        valid_rows = []
        for i in range(downscaled_array.shape[0]):
            r_start = max(0, min(quantized_array.shape[0], by + i * scale))
            r_end = max(0, min(quantized_array.shape[0], by + (i + 1) * scale))
            if r_start >= r_end:
                valid_rows.append(False)
            else:
                valid_rows.append(bool(np.max(edges_per_row[r_start:r_end]) <= max_edges_row))
                
        valid_cols = []
        for j in range(downscaled_array.shape[1]):
            c_start = max(0, min(quantized_array.shape[1], bx + j * scale))
            c_end = max(0, min(quantized_array.shape[1], bx + (j + 1) * scale))
            if c_start >= c_end:
                valid_cols.append(False)
            else:
                valid_cols.append(bool(np.max(edges_per_col[c_start:c_end]) <= max_edges_col))
                
        def get_largest_true_block(mask):
            max_len, best_start, best_end = 0, 0, 0
            curr_start = -1
            curr_len = 0
            for idx, val in enumerate(mask):
                if val:
                    if curr_start == -1:
                        curr_start = idx
                    curr_len += 1
                else:
                    if curr_len > max_len:
                        max_len = curr_len
                        best_start = curr_start
                        best_end = idx
                    curr_start = -1
                    curr_len = 0
            if curr_len > max_len:
                max_len = curr_len
                best_start = curr_start
                best_end = len(mask)
            return best_start, best_end
            
        r1, r2 = get_largest_true_block(valid_rows)
        c1, c2 = get_largest_true_block(valid_cols)
        
        if (r2 - r1) < self.min_native_size or (c2 - c1) < self.min_native_size:
            return downscaled_array
            
        return downscaled_array[r1:r2, c1:c2]

    def _clean_colors(self, img_array: np.ndarray) -> np.ndarray:
        """
        Débruite la palette de couleurs en regroupant les nuances proches (artefacts JPEG).
        Préserve la transparence Alpha.
        """
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, c)
        
        # Nettoyer les pixels quasi-transparents
        if c == 4:
            transparent_mask = pixels[:, 3] < 20
            pixels[transparent_mask] = [0, 0, 0, 0]
            
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Tri par fréquence décroissante (les couleurs dominantes sont les vraies couleurs)
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        
        merged_palette = []
        color_map = {}
        
        for color in unique_colors:
            # Si le pixel est totalement transparent, mapper vers (0,0,0,0)
            if c == 4 and color[3] == 0:
                color_map[tuple(color)] = (0, 0, 0, 0)
                continue
                
            found = False
            for p_color in merged_palette:
                if c == 4 and p_color[3] == 0:
                    continue
                    
                # Distance RGB pondérée (ou RGBA si alpha partiel)
                rgb_dist = np.linalg.norm(color[:3].astype(float) - p_color[:3].astype(float))
                
                alpha_close = True
                if c == 4:
                    alpha_close = abs(int(color[3]) - int(p_color[3])) < 30
                    
                if rgb_dist < self.color_tolerance and alpha_close:
                    color_map[tuple(color)] = tuple(p_color)
                    found = True
                    break
                    
            if not found:
                merged_palette.append(color)
                color_map[tuple(color)] = tuple(color)
                
        mapped_pixels = np.array([color_map[tuple(p)] for p in pixels], dtype=np.uint8)
        return mapped_pixels.reshape(h, w, c)

    def _handle_background(self, img_array: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Détecte si l'image possède un arrière-plan uni ou semi-uni (avec bruit)
        et le convertit proprement en canal Alpha transparent si remove_background=True.
        """
        h, w, c = img_array.shape
        if c == 3:
            # Convertir en RGBA
            alpha_ch = np.full((h, w, 1), 255, dtype=np.uint8)
            img_array = np.concatenate([img_array, alpha_ch], axis=-1)
            c = 4
            
        # Analyser les 4 coins
        corners = [
            img_array[0, 0],
            img_array[0, w - 1],
            img_array[h - 1, 0],
            img_array[h - 1, w - 1]
        ]
        
        # Si les 4 coins sont déjà transparents
        if all(cn[3] == 0 for cn in corners):
            return img_array, True
            
        # Calculer la couleur de fond candidate (médiane des 4 coins)
        corner_colors = np.array([cn[:3] for cn in corners], dtype=float)
        ref_bg = np.median(corner_colors, axis=0)
        
        # Vérifier si les 4 coins sont cohérents avec cette couleur
        corners_match = all(np.linalg.norm(cn[:3] - ref_bg) < self.bg_tolerance for cn in corners)
        
        has_solid_bg = False
        if corners_match and self.remove_background:
            has_solid_bg = True
            
            # Éliminer le fond par flood fill / masque de proximité
            rgb_diff = np.linalg.norm(img_array[:, :, :3].astype(float) - ref_bg, axis=-1)
            bg_mask = rgb_diff < self.bg_tolerance
            
            # Application de la transparence
            img_array[bg_mask] = [0, 0, 0, 0]
            
        return img_array, has_solid_bg

    def reconstruct_frame(self, frame_img: Image.Image) -> Tuple[Optional[Image.Image], int, Dict[str, Any]]:
        """
        Traite et nettoie une frame individuelle.
        Retourne (image_nettoyée, echelle_detectee, infos_qualite).
        """
        has_alpha = frame_img.mode in ('RGBA', 'LA') or (frame_img.mode == 'P' and 'transparency' in frame_img.info)
        
        if has_alpha:
            img = frame_img.convert("RGBA")
            raw_array = np.array(img)
            # Composite sur fond blanc pour calcul des gradients
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(bg, img).convert("RGB")
            quantized = alpha_composite.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
        else:
            img = frame_img.convert("RGB")
            raw_array = np.array(img)
            quantized = img.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
            
        quantized_array = np.array(quantized.convert("RGB"))
        orig_w, orig_h = img.size
        
        if orig_w < self.min_native_size or orig_h < self.min_native_size:
            return None, 1, {"rejected": True, "reason": f"Image trop petite ({orig_w}x{orig_h})"}
            
        # 1. Détection de l'échelle et de l'offset
        scale = self._detect_grid_scale(quantized_array)
        offset_x, offset_y = self._find_best_offset(quantized_array, scale)
        
        # 2. Échantillonnage centré
        if scale > 1:
            downscaled_array = raw_array[offset_y::scale, offset_x::scale]
            downscaled_array = self._crop_watermarks(quantized_array, scale, offset_x, offset_y, downscaled_array)
        else:
            downscaled_array = raw_array
            
        new_h, new_w = downscaled_array.shape[:2]
        if new_w < self.min_native_size or new_h < self.min_native_size:
            return None, scale, {"rejected": True, "reason": f"Résolution native trop basse ({new_w}x{new_h})"}
            
        # 3. Débruitage des couleurs JPEG
        cleaned_array = self._clean_colors(downscaled_array)
        
        # 4. Gestion du fond / Transparence
        cleaned_array, has_bg = self._handle_background(cleaned_array)
        
        # 5. Vérification du nombre de couleurs (Rejet des photos / peintures lisses)
        unique_colors_cnt = len(np.unique(cleaned_array.reshape(-1, cleaned_array.shape[-1]), axis=0))
        
        if unique_colors_cnt > self.max_colors and scale == 1:
            # Si c'est en 1x et qu'il y a plus de max_colors, c'est très probablement une photo / illustration
            return None, scale, {
                "rejected": True, 
                "reason": f"Palette trop riche ({unique_colors_cnt} couleurs > {self.max_colors}), photo/dessin lisse détecté"
            }
            
        result_img = Image.fromarray(cleaned_array)
        
        stats = {
            "rejected": False,
            "original_size": (orig_w, orig_h),
            "clean_size": (new_w, new_h),
            "scale": scale,
            "palette_size": unique_colors_cnt,
            "has_background_removed": has_bg
        }
        
        return result_img, scale, stats

    def reconstruct(self, img_input: Image.Image) -> Tuple[Optional[Image.Image], int, Dict[str, Any]]:
        """
        Point d'entrée principal pour images statiques et GIFs animés.
        """
        is_animated = getattr(img_input, "is_animated", False) and getattr(img_input, "n_frames", 1) > 1
        
        if not is_animated:
            return self.reconstruct_frame(img_input)
            
        # Traitement pour animation GIF multi-frames
        frames = []
        durations = []
        scales = []
        
        first_frame = True
        detected_scale = 1
        
        for frame in ImageSequence.Iterator(img_input):
            duration = frame.info.get('duration', 100)
            durations.append(duration)
            
            clean_f, f_scale, stats = self.reconstruct_frame(frame)
            if clean_f is None:
                return None, 1, stats
                
            if first_frame:
                detected_scale = f_scale
                first_frame = False
                
            frames.append(clean_f)
            scales.append(f_scale)
            
        if not frames:
            return None, 1, {"rejected": True, "reason": "Aucune frame valide dans le GIF"}
            
        # Assemblage du GIF nettoyé
        out_io = BytesIO()
        frames[0].save(
            out_io,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            disposal=2
        )
        out_io.seek(0)
        final_gif = Image.open(out_io)
        
        overall_stats = {
            "rejected": False,
            "is_animated": True,
            "frames": len(frames),
            "original_size": img_input.size,
            "clean_size": frames[0].size,
            "scale": detected_scale,
            "palette_size": 256
        }
        
        return final_gif, detected_scale, overall_stats

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Reconstruction de {filepath}...")
        try:
            raw = Image.open(filepath)
            rec = PixelReconstructor(color_tolerance=15.0, remove_background=True)
            res, sc, st = rec.reconstruct(raw)
            if res:
                out = filepath.rsplit(".", 1)[0] + "_cleaned.png"
                res.save(out)
                print(f"Succès ! Échelle: {sc}x | Taille originale: {st['original_size']} -> Propre: {st['clean_size']}")
                print(f"Sauvegardé dans: {out}")
            else:
                print(f"Rejeté: {st.get('reason')}")
        except Exception as e:
            print(f"Erreur: {e}")
    else:
        print("Usage: python pixel_reconstructor.py <chemin_image>")
