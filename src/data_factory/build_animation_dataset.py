import os
import tarfile
import json
import math
from io import BytesIO
from PIL import Image

import numpy as np

def auto_slice_spritesheet(img):
    """
    Découpe une spritesheet en détectant mathématiquement les lignes transparentes
    pour trouver la grille parfaite, sans couper les sprites.
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Masque d'opacité
    if img_array.shape[2] == 4:
        opaque_mask = img_array[:, :, 3] > 0
    else:
        bg_color = img_array[0, 0]
        opaque_mask = np.any(img_array != bg_color, axis=-1)
        
    cols_sum = np.sum(opaque_mask, axis=0)
    rows_sum = np.sum(opaque_mask, axis=1)
    
    def get_valid_divisions(length, proj_sum):
        valid_divs = []
        for divs in range(1, min(length, 64) + 1):
            if length % divs == 0:
                step = length // divs
                is_valid = True
                # On tolère 0 pixel opaque sur la ligne de coupe exacte
                for k in range(1, divs):
                    if proj_sum[k * step] > 0:
                        is_valid = False
                        break
                if is_valid:
                    valid_divs.append(divs)
        return valid_divs
        
    valid_cols = get_valid_divisions(w, cols_sum)
    valid_rows = get_valid_divisions(h, rows_sum)
    
    best_cols = max(valid_cols) if valid_cols else 1
    best_rows = max(valid_rows) if valid_rows else 1
    
    frame_w = w // best_cols
    frame_h = h // best_rows
    
    frames = []
    for y in range(best_rows):
        for x in range(best_cols):
            box = (x * frame_w, y * frame_h, (x + 1) * frame_w, (y + 1) * frame_h)
            frame = img.crop(box)
            
            # Rejet des frames 100% vides
            frame_arr = np.array(frame)
            if frame_arr.shape[2] == 4:
                if np.sum(frame_arr[:, :, 3] > 0) == 0:
                    continue
            else:
                if np.sum(np.any(frame_arr != img_array[0, 0], axis=-1)) == 0:
                    continue
                    
            frames.append(frame)
            
    return frames, best_cols, best_rows

def extract_animations(input_dir, output_dir):
    """
    Parcourt les .tar, extrait les spritesheets, les découpe, et les sauvegarde pour revue.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tar_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.tar'):
                tar_files.append(os.path.join(root, f))
                
    if not tar_files:
        print(f"Aucun fichier .tar trouvé dans {input_dir}")
        return
        
    print(f"Trouvé {len(tar_files)} archives .tar à analyser.")
    
    extracted_count = 0
    for tar_path in tar_files:
        print(f"\nLecture de {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                members = tar.getmembers()
                # Grouper par préfixe (ex: image_001.png et image_001.json)
                items = {}
                for m in members:
                    base = m.name.split('.')[0]
                    ext = m.name.split('.')[-1]
                    if base not in items:
                        items[base] = {}
                    items[base][ext] = m
                    
                for base, files in items.items():
                    if 'json' in files and 'png' in files:
                        json_file = tar.extractfile(files['json'])
                        if not json_file: continue
                        
                        try:
                            metadata = json.loads(json_file.read().decode('utf-8'))
                        except json.JSONDecodeError:
                            continue
                            
                        if metadata.get('is_animation', False) and metadata.get('frames', 1) > 1:
                            frames_count = metadata['frames']
                            
                            img_file = tar.extractfile(files['png'])
                            if not img_file: continue
                            
                            img = Image.open(BytesIO(img_file.read())).convert("RGBA")
                            
                            # Découpage par vision par ordinateur
                            frames, cols, rows = auto_slice_spritesheet(img)
                            
                            if len(frames) > 1:
                                real_frames_count = len(frames)
                                # Sauvegarde pour revue manuelle
                                prefix = f"{os.path.basename(tar_path).split('.')[0]}_{base}"
                                
                                # 1. La Frame 1 (Condition)
                                frame1_path = os.path.join(output_dir, f"{prefix}_frame1.png")
                                frames[0].save(frame1_path)
                                
                                # 2. Le GIF animé (Target)
                                gif_path = os.path.join(output_dir, f"{prefix}_anim.gif")
                                frames[0].save(
                                    gif_path, 
                                    save_all=True, 
                                    append_images=frames[1:], 
                                    duration=100, # 10 fps par défaut (peut être ajusté)
                                    loop=0,
                                    disposal=2 # Efface la frame précédente
                                )
                                
                                # Enrichissement de la description pour le contrôle de l'IA (Image-to-Animation)
                                duration_seconds = real_frames_count * 0.1 # 10 FPS par défaut
                                action = "l'action" # Idéalement extrait de la description, ou générique
                                anim_prompt = f" Animation sur {real_frames_count} frames sur {duration_seconds:.1f} secondes."
                                
                                # On met à jour le nombre réel de frames
                                metadata['frames'] = real_frames_count
                                # On ajoute la consigne à la fin de la description existante
                                metadata['description'] = metadata.get('description', '') + anim_prompt
                                
                                # 3. Le JSON (Metadata)
                                json_path = os.path.join(output_dir, f"{prefix}.json")
                                with open(json_path, 'w', encoding='utf-8') as jf:
                                    json.dump(metadata, jf, indent=2)
                                    
                                extracted_count += 1
                                print(f"  -> Extrait : {prefix} ({cols}x{rows} grid, {real_frames_count} frames trouvées)")
                                
        except Exception as e:
            print(f"Erreur sur l'archive {tar_path}: {e}")
            
    print(f"\nTerminé ! {extracted_count} animations extraites pour revue dans le dossier '{output_dir}'.")
    print("Prenez le temps de supprimer les GIF ratés. Nous pourrons ensuite packer les survivants.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extracteur d'Animations depuis WebDataset")
    parser.add_argument("--input", type=str, required=True, help="Dossier contenant les .tar de la Phase 1/2")
    parser.add_argument("--output", type=str, default="review_animations", help="Dossier de sortie pour la revue manuelle")
    args = parser.parse_args()
    
    extract_animations(args.input, args.output)
