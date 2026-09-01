import math
from io import BytesIO
from typing import Tuple, Optional, Dict, Any
from PIL import Image, ImageSequence
import numpy as np

class PixelReconstructor:
    """
    Moteur de reconstruction et de nettoyage haute précision pour Pixel Art.
    
    Capacités :
    1. Détection de fréquence fondamentale de grille par analyse de premier cluster de pics d'autocorrélation.
    2. Centrage sous-pixel des échantillons pour préserver les outlines 1-px.
    3. Rejet strict des photos physiques (papier quadrillé, cahier à carreaux, perles Hama, broderies).
    4. Débruitage de palette de couleurs (fusion des artefacts JPEG).
    5. Détection de fond uni et conversion optionnelle en transparence Alpha.
    6. Support des animations GIF et des images statiques.
    """

    def __init__(
        self,
        color_tolerance: float = 15.0,
        max_colors: int = 256,
        min_native_size: int = 6,
        remove_background: bool = False,
        bg_tolerance: float = 25.0,
        strict_mode: bool = True,
        max_intra_block_std: float = 24.0,
        min_psnr: float = 16.0,
        max_mae: float = 20.0
    ):
        self.color_tolerance = color_tolerance
        self.max_colors = max_colors
        self.min_native_size = min_native_size
        self.remove_background = remove_background
        self.bg_tolerance = bg_tolerance
        self.strict_mode = strict_mode
        self.max_intra_block_std = max_intra_block_std
        self.min_psnr = min_psnr
        self.max_mae = max_mae

    def _detect_grid_scale(self, quantized_array: np.ndarray, max_scale: int = 48) -> Tuple[float, bool]:
        """
        Détecte l'échelle fondamentale de la grille en trouvant le PREMIER cluster de pics significatifs.
        """
        gray = np.mean(quantized_array[:, :, :3], axis=-1)
        h, w = gray.shape
        
        diff_h = np.abs(np.diff(gray, axis=1)) > 10
        diff_v = np.abs(np.diff(gray, axis=0)) > 10
        
        density_x = np.sum(diff_h, axis=0).astype(float)
        density_y = np.sum(diff_v, axis=1).astype(float)
        
        if len(density_x) < 4 or len(density_y) < 4:
            return 1.0, True
            
        density_x -= np.mean(density_x)
        density_y -= np.mean(density_y)
        
        ac_x = np.correlate(density_x, density_x, mode='full')[len(density_x) - 1:]
        ac_y = np.correlate(density_y, density_y, mode='full')[len(density_y) - 1:]
        
        # On ne cherche pas d'échelle plus grande que w // 8
        effective_max = min(max_scale + 1, len(ac_x), len(ac_y), max(4, min(w, h) // 4))
        if effective_max < 4:
            return 1.0, True
            
        autocorr = ac_x[:effective_max] + ac_y[:effective_max]
        
        peaks = []
        for k in range(2, effective_max - 1):
            val = autocorr[k]
            left = autocorr[k - 1]
            right = autocorr[k + 1]
            
            if val > left and val > right and val > 0:
                prominence = val - max(left, right)
                peaks.append((k, val, prominence))
                
        if not peaks:
            return 1.0, True
            
        max_peak_val = max(p[1] for p in peaks)
        significant_peaks = [p for p in peaks if p[1] > 0.25 * max_peak_val]
        
        if not significant_peaks:
            return 1.0, True
            
        first_k, first_val, first_prom = significant_peaks[0]
        
        if len(significant_peaks) > 1 and significant_peaks[1][0] == first_k + 1:
            w1 = first_val
            w2 = significant_peaks[1][1]
            fractional_k = (first_k * w1 + (first_k + 1) * w2) / (w1 + w2)
            return float(fractional_k), False
            
        return float(first_k), True

    def _find_best_offset(self, quantized_array: np.ndarray, scale: int) -> Tuple[int, int]:
        """
        Trouve le décalage (offset X/Y) pour que l'échantillonnage tombe au centre des macro-pixels.
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

    def _check_intra_block_variance(self, raw_array: np.ndarray, scale: int, ox: int, oy: int) -> float:
        """
        Mesure l'écart-type moyen à l'intérieur des macro-pixels.
        """
        if scale <= 1:
            return 0.0
            
        h, w = raw_array.shape[:2]
        rgb = raw_array[:, :, :3].astype(float)
        
        stds = []
        for y in range(oy, h - scale + 1, scale):
            for x in range(ox, w - scale + 1, scale):
                block = rgb[y:y + scale, x:x + scale]
                block_std = np.mean(np.std(block, axis=(0, 1)))
                stds.append(block_std)
                
        if not stds:
            return 0.0
            
        return float(np.median(stds))

    def _check_roundtrip_residual(self, raw_array: np.ndarray, downscaled_array: np.ndarray, scale: int, ox: int, oy: int) -> Tuple[float, float]:
        """
        Vérifie la fidélité de reconstruction cyclique (MAE & PSNR).
        """
        if scale <= 1:
            return 0.0, 99.0
            
        down_h, down_w = downscaled_array.shape[:2]
        orig_h, orig_w = raw_array.shape[:2]
        
        crop_h = min(orig_h - oy, down_h * scale)
        crop_w = min(orig_w - ox, down_w * scale)
        
        if crop_h < 4 or crop_w < 4:
            return 99.0, 0.0
            
        orig_crop = raw_array[oy:oy + crop_h, ox:ox + crop_w, :3].astype(float)
        
        up_img = Image.fromarray(downscaled_array).resize((down_w * scale, down_h * scale), Image.Resampling.NEAREST)
        up_arr = np.array(up_img)[:crop_h, :crop_w, :3].astype(float)
        
        diff = np.abs(orig_crop - up_arr)
        mae = float(np.mean(diff))
        mse = float(np.mean(diff ** 2))
        
        psnr = 10.0 * math.log10((255.0 ** 2) / (mse + 1e-6)) if mse > 0 else 99.0
        return mae, psnr

    def _clean_colors(self, img_array: np.ndarray) -> np.ndarray:
        """
        Débruite la palette de couleurs en regroupant les nuances proches (artefacts JPEG).
        """
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, c)
        
        if c == 4:
            transparent_mask = pixels[:, 3] < 20
            pixels[transparent_mask] = [0, 0, 0, 0]
            
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        sorted_indices = np.argsort(-counts)
        unique_colors = unique_colors[sorted_indices]
        
        merged_palette = []
        color_map = {}
        
        for color in unique_colors:
            if c == 4 and color[3] == 0:
                color_map[tuple(color)] = (0, 0, 0, 0)
                continue
                
            found = False
            for p_color in merged_palette:
                if c == 4 and p_color[3] == 0:
                    continue
                    
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
        Détecte et convertit les fonds solides en transparence Alpha.
        """
        h, w, c = img_array.shape
        if c == 3:
            alpha_ch = np.full((h, w, 1), 255, dtype=np.uint8)
            img_array = np.concatenate([img_array, alpha_ch], axis=-1)
            c = 4
            
        corners = [
            img_array[0, 0],
            img_array[0, w - 1],
            img_array[h - 1, 0],
            img_array[h - 1, w - 1]
        ]
        
        if all(cn[3] == 0 for cn in corners):
            return img_array, True
            
        corner_colors = np.array([cn[:3] for cn in corners], dtype=float)
        ref_bg = np.median(corner_colors, axis=0)
        
        corners_match = all(np.linalg.norm(cn[:3] - ref_bg) < self.bg_tolerance for cn in corners)
        
        has_solid_bg = False
        if corners_match and self.remove_background:
            has_solid_bg = True
            rgb_diff = np.linalg.norm(img_array[:, :, :3].astype(float) - ref_bg, axis=-1)
            bg_mask = rgb_diff < self.bg_tolerance
            img_array[bg_mask] = [0, 0, 0, 0]
            
        return img_array, has_solid_bg

    def reconstruct_frame(self, frame_img: Image.Image) -> Tuple[Optional[Image.Image], float, Dict[str, Any]]:
        """
        Traite une frame avec détection d'échelle fondamentale et filtrage qualité.
        """
        has_alpha = frame_img.mode in ('RGBA', 'LA') or (frame_img.mode == 'P' and 'transparency' in frame_img.info)
        
        if has_alpha:
            img = frame_img.convert("RGBA")
            raw_array = np.array(img)
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(bg, img).convert("RGB")
            quantized = alpha_composite.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
        else:
            img = frame_img.convert("RGB")
            raw_array = np.array(img)
            quantized = img.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
            
        quantized_array = np.array(quantized.convert("RGB"))
        orig_w, orig_h = img.size
        
        if orig_w < self.min_native_size or orig_h < self.min_native_size:
            return None, 1.0, {"rejected": True, "reason": f"Image trop petite ({orig_w}x{orig_h})", "original_size": (orig_w, orig_h)}
            
        # 1. Détection de l'échelle fondamentale
        scale, is_integer = self._detect_grid_scale(quantized_array)
        int_scale = int(round(scale))
        
        # Garde-fou : si le downscaling produit une image trop petite (< min_native_size), considérer comme 1x
        if orig_w / max(1.0, scale) < self.min_native_size or orig_h / max(1.0, scale) < self.min_native_size:
            scale = 1.0
            int_scale = 1
            is_integer = True

        # 2. Test anti-photos physique (cahier à carreaux)
        if self.strict_mode and int_scale > 1:
            ox, oy = self._find_best_offset(quantized_array, int_scale)
            intra_std = self._check_intra_block_variance(raw_array, int_scale, ox, oy)
            if intra_std > self.max_intra_block_std:
                return None, scale, {
                    "rejected": True,
                    "reason": f"Faux pixel art (Photo/Papier : variance intra-cases = {intra_std:.1f} > {self.max_intra_block_std})",
                    "original_size": (orig_w, orig_h)
                }
        else:
            ox, oy = 0, 0

        # 3. Échantillonnage centré ou resampling
        if is_integer and int_scale > 1:
            ox, oy = self._find_best_offset(quantized_array, int_scale)
            downscaled_array = raw_array[oy::int_scale, ox::int_scale]
        elif not is_integer and scale > 1.5:
            nat_w = max(self.min_native_size, int(round(orig_w / scale)))
            nat_h = max(self.min_native_size, int(round(orig_h / scale)))
            resampled = img.resize((nat_w, nat_h), Image.Resampling.LANCZOS)
            downscaled_array = np.array(resampled)
        else:
            downscaled_array = raw_array
            
        new_h, new_w = downscaled_array.shape[:2]
        if new_w < self.min_native_size or new_h < self.min_native_size:
            return None, scale, {"rejected": True, "reason": f"Résolution native trop basse ({new_w}x{new_h})", "original_size": (orig_w, orig_h)}
            
        # 4. Test de résidu cyclique (uniquement pour les échelles entières nettes)
        if self.strict_mode and is_integer and int_scale > 1:
            mae, psnr = self._check_roundtrip_residual(raw_array, downscaled_array, int_scale, ox, oy)
            if psnr < self.min_psnr or mae > self.max_mae:
                return None, scale, {
                    "rejected": True,
                    "reason": f"Faux pixel art (Résidu élevé : MAE={mae:.1f}, PSNR={psnr:.1f}dB)",
                    "original_size": (orig_w, orig_h)
                }
            
        # 5. Débruitage des couleurs JPEG
        cleaned_array = self._clean_colors(downscaled_array)
        
        # 6. Gestion du fond / Transparence
        cleaned_array, has_bg = self._handle_background(cleaned_array)
        
        # 7. Vérification de la palette (uniquement sur 1x non downscalé)
        unique_colors_cnt = len(np.unique(cleaned_array.reshape(-1, cleaned_array.shape[-1]), axis=0))
        if unique_colors_cnt > self.max_colors and is_integer and int_scale == 1:
            return None, scale, {
                "rejected": True, 
                "reason": f"Palette trop riche ({unique_colors_cnt} couleurs > {self.max_colors}), illustration lisse",
                "original_size": (orig_w, orig_h)
            }
            
        result_img = Image.fromarray(cleaned_array)
        
        stats = {
            "rejected": False,
            "original_size": (orig_w, orig_h),
            "clean_size": (new_w, new_h),
            "scale": scale,
            "is_integer_scale": is_integer,
            "palette_size": unique_colors_cnt,
            "has_background_removed": has_bg
        }
        
        return result_img, scale, stats

    def reconstruct(self, img_input: Image.Image) -> Tuple[Optional[Image.Image], float, Dict[str, Any]]:
        """
        Point d'entrée principal pour images statiques et GIFs.
        """
        is_animated = getattr(img_input, "is_animated", False) and getattr(img_input, "n_frames", 1) > 1
        
        if not is_animated:
            return self.reconstruct_frame(img_input)
            
        frames = []
        durations = []
        scales = []
        
        first_frame = True
        detected_scale = 1.0
        
        for frame in ImageSequence.Iterator(img_input):
            duration = frame.info.get('duration', 100)
            durations.append(duration)
            
            clean_f, f_scale, stats = self.reconstruct_frame(frame)
            if clean_f is None:
                return None, 1.0, stats
                
            if first_frame:
                detected_scale = f_scale
                first_frame = False
                
            frames.append(clean_f)
            scales.append(f_scale)
            
        if not frames:
            return None, 1.0, {"rejected": True, "reason": "Aucune frame valide dans le GIF", "original_size": img_input.size}
            
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
