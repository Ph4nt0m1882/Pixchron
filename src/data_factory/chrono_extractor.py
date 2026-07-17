import os
import json
from PIL import Image, ImageDraw

def process_chrono_sprites():
    in_dir = "scratch/ct"
    out_dir = "datasets_ready/chrono"
    os.makedirs(out_dir, exist_ok=True)
    
    # Rédiger les descriptions riches
    descriptions = {
        'Crono': 'Pixel art rétro (SNES). Crono, jeune héros avec des cheveux rouges en pointe (façon Super Saiyan), portant une tunique verte, un pantalon clair et un katana.',
        'Marle': 'Pixel art rétro (SNES). Marle, princesse rebelle blonde avec un pantalon blanc bouffant et un haut blanc/cyan.',
        'Lucca': 'Pixel art rétro (SNES). Lucca, jeune inventrice avec des cheveux violets courts, de grandes lunettes rondes et un casque orange.',
        'Frog': 'Pixel art rétro (SNES). Frog (Grenouille), grand chevalier batracien vert bipède avec une cape, maniant une épée.',
        'Robo': 'Pixel art rétro (SNES). Robo, robot humanoïde jaune-orangé avec une carrure massive et un œil central lumineux.',
        'Ayla': 'Pixel art rétro (SNES). Ayla, jeune femme préhistorique blonde avec une peau bronzée et une tenue en peau de bête minimaliste.',
        'Magus': 'Pixel art rétro (SNES). Magus, puissant sorcier avec de longs cheveux gris très clairs, la peau pâle et une grande cape bleue/violette foncé.'
    }
    
    count = 0
    for name, desc in descriptions.items():
        img_path = os.path.join(in_dir, f"{name}.png")
        if not os.path.exists(img_path):
            print(f"Skipping {name}, file not found.")
            continue
            
        img = Image.open(img_path).convert("RGBA")
        data = img.load()
        w, h = img.size
        
        # Le fond est le pixel en haut à gauche
        bg_color = data[0, 0][:3]
        
        for y in range(h):
            for x in range(w):
                if data[x, y][:3] == bg_color:
                    data[x, y] = (0, 0, 0, 0)
                    
        # BFS pour trouver les sprites
        visited = set()
        sprites = []
        for y in range(h):
            for x in range(w):
                if data[x, y][3] > 0 and (x, y) not in visited:
                    q = [(x, y)]
                    visited.add((x, y))
                    min_x, max_x, min_y, max_y = x, x, y, y
                    while q:
                        cx, cy = q.pop(0)
                        min_x = min(min_x, cx); max_x = max(max_x, cx)
                        min_y = min(min_y, cy); max_y = max(max_y, cy)
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                if data[nx, ny][3] > 0 and (nx, ny) not in visited:
                                    visited.add((nx, ny))
                                    q.append((nx, ny))
                    sprites.append((min_x, min_y, max_x, max_y))
                    
        # Portrait : plus grand sprite carré en haut de l'image (y < 100)
        portraits = [s for s in sprites if s[1] < 100 and 30 <= (s[2]-s[0]) <= 70 and 30 <= (s[3]-s[1]) <= 70]
        portrait_img = None
        if portraits:
            # Prendre le plus grand en surface
            portraits.sort(key=lambda s: (s[2]-s[0])*(s[3]-s[1]), reverse=True)
            p = portraits[0]
            portrait_img = img.crop((p[0], p[1], p[2]+1, p[3]+1))
            # Centrer dans 64x64
            p_final = Image.new("RGBA", (64, 64), (0,0,0,0))
            pw, ph = portrait_img.size
            p_final.paste(portrait_img, ((64-pw)//2, (64-ph)//2))
            portrait_img = p_final
            
        # Sprites de marche : taille moyenne
        walking = [s for s in sprites if 10 <= (s[2]-s[0]) <= 35 and 15 <= (s[3]-s[1]) <= 45]
        # Trier par Y (avec tolérance de 15px pour grouper par ligne) puis par X
        walking.sort(key=lambda s: (s[1] // 15, s[0]))
        
        frames_to_keep = 16
        if len(walking) < 16:
            print(f"Warning: only {len(walking)} frames found for {name}")
            frames_to_keep = len(walking)
            
        # Créer la composition 192x128
        final_img = Image.new("RGBA", (192, 128), (0,0,0,0))
        if portrait_img:
            final_img.paste(portrait_img, (0, 0))
            
        # Coller les 16 frames dans une grille 4x4 de 32x32, décalée de X=64
        for i in range(frames_to_keep):
            s = walking[i]
            frame_img = img.crop((s[0], s[1], s[2]+1, s[3]+1))
            fw, fh = frame_img.size
            
            row = i // 4
            col = i % 4
            
            # Centrer dans la case de 32x32
            # Attention, si fh > 32, on rogne ou on s'ajuste (certains font 34px). 
            # On colle par le bas pour que les pieds soient alignés
            fx = 64 + col * 32 + (32 - fw) // 2
            fy = row * 32 + (32 - fh) // 2
            if fh > 32:
                fy = row * 32  # align top
                
            final_img.paste(frame_img, (fx, fy), frame_img)
            
        # JSON
        anim_data = {
            "format": "WALKING_SPRITE_CHRONO",
            "description": desc,
            "animations": {
                "walk_down": { "frames": [{"x": 64, "y": 0, "w": 32, "h": 32}, {"x": 96, "y": 0, "w": 32, "h": 32}, {"x": 128, "y": 0, "w": 32, "h": 32}, {"x": 160, "y": 0, "w": 32, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_right": { "frames": [{"x": 64, "y": 32, "w": 32, "h": 32}, {"x": 96, "y": 32, "w": 32, "h": 32}, {"x": 128, "y": 32, "w": 32, "h": 32}, {"x": 160, "y": 32, "w": 32, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_up": { "frames": [{"x": 64, "y": 64, "w": 32, "h": 32}, {"x": 96, "y": 64, "w": 32, "h": 32}, {"x": 128, "y": 64, "w": 32, "h": 32}, {"x": 160, "y": 64, "w": 32, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_left": { "frames": [{"x": 64, "y": 96, "w": 32, "h": 32}, {"x": 96, "y": 96, "w": 32, "h": 32}, {"x": 128, "y": 96, "w": 32, "h": 32}, {"x": 160, "y": 96, "w": 32, "h": 32}], "timing": [200, 200, 200, 200] }
            }
        }
        
        safe_name = name.lower()
        final_img.save(os.path.join(out_dir, f"{safe_name}.png"))
        with open(os.path.join(out_dir, f"{safe_name}.json"), "w", encoding="utf-8") as jf:
            json.dump(anim_data, jf, indent=4, ensure_ascii=False)
            
        print(f"Extracted {name}")
        count += 1
        
    print(f"Total extracted: {count}")

if __name__ == "__main__":
    process_chrono_sprites()
