import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import numpy as np

class SpriteSlicer:
    """
    Découpeur intelligent de fiches de présentation et de planches multi-sprites (ex: Exagide).
    Détecte les composantes connexes, élimine les signatures/logos d'artistes
    et extrait chaque sprite individuel sur fond transparent en résolution native 1:1.
    """

    def __init__(
        self,
        min_sprite_area: int = 400,
        padding: int = 2,
        logo_area_ratio_threshold: float = 0.04
    ):
        self.min_sprite_area = min_sprite_area
        self.padding = padding
        self.logo_area_ratio_threshold = logo_area_ratio_threshold

    def slice_presentation_card(
        self, 
        img: Image.Image, 
        macro_scale: int = 3
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Analyse une fiche multi-sprites et extrait chaque sprite individuel.
        Retourne une liste de tuples : [(sprite_img_nettoyee_1x, bounding_box), ...]
        """
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        h, w = arr.shape[:2]
        
        # 1. Détecter le fond pour créer un masque binaire de premier plan
        corners = [arr[0, 0, :3], arr[0, w-1, :3], arr[h-1, 0, :3], arr[h-1, w-1, :3]]
        ref_bg = np.median(corners, axis=0)
        
        # Masque de premier plan : pixels différents du fond ou alpha > 0
        diff_bg = np.linalg.norm(arr[:, :, :3].astype(float) - ref_bg, axis=-1)
        fg_mask = (diff_bg > 30.0) | (arr[:, :, 3] == 0)
        
        # Si l'image a déjà un canal alpha transparent
        if np.any(arr[:, :, 3] == 0):
            fg_mask = arr[:, :, 3] > 20
            
        # 2. Composantes connexes simples (via flood-fill / étiquetage)
        visited = np.zeros((h, w), dtype=bool)
        boxes = []
        
        # Sous-échantillonnage de la recherche pour rapidité
        step = max(1, macro_scale)
        for y in range(0, h, step):
            for x in range(0, w, step):
                if fg_mask[y, x] and not visited[y, x]:
                    # Inondation (BFS) pour trouver le contour de ce sprite
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    queue = [(y, x)]
                    visited[y, x] = True
                    cluster_size = 0
                    
                    while queue:
                        cy, cx = queue.pop(0)
                        cluster_size += 1
                        
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)
                        
                        # Voisins avec saut de pas macro_scale
                        for dy, dx in [(-step, 0), (step, 0), (0, -step), (0, step)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and fg_mask[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                                
                    bw = max_x - min_x + 1
                    bh = max_y - min_y + 1
                    area = bw * bh
                    
                    if area >= self.min_sprite_area:
                        boxes.append((min_x, min_y, max_x, max_y, area))

        if not boxes:
            return [(img, (0, 0, w, h))]
            
        # 3. Filtrer les logos, signatures d'artistes ou textes
        # On calcule l'aire maximale d'un sprite principal
        max_area = max(b[4] for b in boxes)
        valid_boxes = []
        for bx1, by1, bx2, by2, area in boxes:
            # Rejeter les petites signatures (ex: SoulJun < 5% de la taille d'un sprite)
            if area >= max_area * self.logo_area_ratio_threshold:
                valid_boxes.append((bx1, by1, bx2, by2))
                
        # 4. Découper et extraire chaque sprite
        extracted_sprites = []
        for x1, y1, x2, y2 in valid_boxes:
            # Ajouter un petit padding propre
            px1 = max(0, x1 - self.padding * macro_scale)
            py1 = max(0, y1 - self.padding * macro_scale)
            px2 = min(w, x2 + self.padding * macro_scale + 1)
            py2 = min(h, y2 + self.padding * macro_scale + 1)
            
            sprite_crop = rgba.crop((px1, py1, px2, py2))
            
            # Détourage local du fond
            crop_arr = np.array(sprite_crop)
            crop_diff = np.linalg.norm(crop_arr[:, :, :3].astype(float) - ref_bg, axis=-1)
            crop_arr[crop_diff < 30.0, 3] = 0
            clean_sprite = Image.fromarray(crop_arr)
            
            # Downscale centré par macro_scale si nécessaire
            if macro_scale > 1:
                cw, ch = clean_sprite.size
                nw = max(4, cw // macro_scale)
                nh = max(4, ch // macro_scale)
                clean_sprite = clean_sprite.resize((nw, nh), Image.Resampling.NEAREST)
                
            extracted_sprites.append((clean_sprite, (px1, py1, px2, py2)))
            
        return extracted_sprites

if __name__ == "__main__":
    slicer = SpriteSlicer()
    print("Module SpriteSlicer initialisé avec succès.")
