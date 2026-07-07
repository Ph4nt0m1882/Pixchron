import os
import tarfile
import json
import math
from io import BytesIO
from PIL import Image

def guess_grid(width, height, total_frames):
    """
    Tente de deviner la disposition (colonnes, lignes) de la spritesheet 
    en cherchant des frames les plus carrées possibles.
    """
    best_cols, best_rows = 1, total_frames
    best_diff = float('inf')
    
    # Heuristique : chercher les diviseurs stricts d'abord
    for cols in range(1, width + 1):
        if width % cols != 0: continue
        for rows in range(1, height + 1):
            if height % rows != 0: continue
            
            # La grille doit contenir au moins le nombre de frames (parfois il y a des cases vides à la fin)
            if total_frames <= cols * rows <= total_frames + (cols - 1):
                frame_w = width // cols
                frame_h = height // rows
                
                # En pixel art, les frames sont souvent carrées ou avec des ratios simples
                diff = abs(frame_w - frame_h)
                if diff < best_diff:
                    best_diff = diff
                    best_cols, best_rows = cols, rows
                    
    if best_diff == float('inf'):
        # Fallback basique : ligne ou colonne simple
        if width >= height:
            return total_frames, 1
        else:
            return 1, total_frames
            
    return best_cols, best_rows

def slice_spritesheet(img, cols, rows, total_frames):
    """
    Découpe l'image en frames individuelles.
    """
    frame_w = img.width // cols
    frame_h = img.height // rows
    frames = []
    
    count = 0
    for y in range(rows):
        for x in range(cols):
            if count >= total_frames:
                break
            box = (x * frame_w, y * frame_h, (x + 1) * frame_w, (y + 1) * frame_h)
            frame = img.crop(box)
            frames.append(frame)
            count += 1
            
    return frames

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
                            
                            # Découpage
                            cols, rows = guess_grid(img.width, img.height, frames_count)
                            frames = slice_spritesheet(img, cols, rows, frames_count)
                            
                            if frames:
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
                                duration_seconds = frames_count * 0.1 # 10 FPS par défaut
                                action = "l'action" # Idéalement extrait de la description, ou générique
                                anim_prompt = f" Animation sur {frames_count} frames sur {duration_seconds:.1f} secondes."
                                
                                # On ajoute la consigne à la fin de la description existante
                                metadata['description'] = metadata.get('description', '') + anim_prompt
                                
                                # 3. Le JSON (Metadata)
                                json_path = os.path.join(output_dir, f"{prefix}.json")
                                with open(json_path, 'w', encoding='utf-8') as jf:
                                    json.dump(metadata, jf, indent=2)
                                    
                                extracted_count += 1
                                print(f"  -> Extrait : {prefix} ({cols}x{rows} grid, {frames_count} frames)")
                                
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
