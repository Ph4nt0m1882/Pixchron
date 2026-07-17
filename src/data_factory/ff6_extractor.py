import os
import json
from PIL import Image

def process_ff6_sprites():
    in_dir = "scratch/ff6"
    out_dir = "datasets_ready/ff6"
    os.makedirs(out_dir, exist_ok=True)
    
    descriptions = {
        'Terra': 'Pixel art rétro (SNES). Terra Branford, jeune femme aux cheveux verts, portant une tenue rouge, violette et un justaucorps. Héros de Final Fantasy 6.',
        'Locke': 'Pixel art rétro (SNES). Locke Cole, aventurier et voleur avec un bandana bleu, une tenue bleue claire et un foulard. Héros de Final Fantasy 6.',
        'Celes': 'Pixel art rétro (SNES). Celes Chere, générale avec de longs cheveux blonds, portant une cape blanche et un justaucorps jaune/vert. Héros de Final Fantasy 6.',
        'Edgar': 'Pixel art rétro (SNES). Edgar Roni Figaro, roi avec de longs cheveux blonds attachés, portant une armure dorée majestueuse. Héros de Final Fantasy 6.',
        'Sabin': 'Pixel art rétro (SNES). Sabin Rene Figaro, moine musclé avec un bouc, portant des spalières cloutées et un pantalon sombre. Héros de Final Fantasy 6.',
        'Cyan': 'Pixel art rétro (SNES). Cyan Garamonde, samouraï imposant en armure bleue sombre avec une moustache noire. Héros de Final Fantasy 6.',
        'Gau': 'Pixel art rétro (SNES). Gau, enfant sauvage aux cheveux verts ébouriffés, vêtu de peaux de bêtes. Héros de Final Fantasy 6.',
        'Shadow': 'Pixel art rétro (SNES). Shadow, ninja mystérieux entièrement vêtu de noir avec un masque cachant son visage. Héros de Final Fantasy 6.'
    }
    
    count = 0
    for name, desc in descriptions.items():
        img_path = os.path.join(in_dir, f"{name}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(in_dir, f"{name}.gif")
            if not os.path.exists(img_path):
                print(f"Skipping {name}, file not found.")
                continue
            
        img = Image.open(img_path).convert("RGBA")
        data = img.load()
        w, h = img.size
        
        # Background is usually the color at (0,0)
        bg_color = data[0, 0][:3]
        for y in range(h):
            for x in range(w):
                if data[x, y][:3] == bg_color:
                    data[x, y] = (0, 0, 0, 0)
                    
        # Connected Components
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
                    
        # FF6 characters are 16x24 mostly
        walking = [s for s in sprites if 12 <= (s[2]-s[0]) <= 25 and 18 <= (s[3]-s[1]) <= 32]
        # Sort by Y (15px tolerance) then X
        walking.sort(key=lambda s: (s[1] // 15, s[0]))
        
        frames_to_keep = min(16, len(walking))
        if frames_to_keep == 0:
            print(f"No frames for {name}")
            continue
            
        # Compile into WALKING_SPRITE (16x32 format, 64x128 image)
        final_img = Image.new("RGBA", (64, 128), (0,0,0,0))
            
        for i in range(frames_to_keep):
            s = walking[i]
            frame_img = img.crop((s[0], s[1], s[2]+1, s[3]+1))
            fw, fh = frame_img.size
            
            row = i // 4
            col = i % 4
            
            fx = col * 16 + (16 - fw) // 2
            fy = row * 32 + (32 - fh) // 2
            if fh > 32: fy = row * 32
            
            final_img.paste(frame_img, (fx, fy), frame_img)
            
        anim_data = {
            "format": "WALKING_SPRITE",
            "description": desc,
            "animations": {
                "walk_down": { "frames": [{"x": 0, "y": 0, "w": 16, "h": 32}, {"x": 16, "y": 0, "w": 16, "h": 32}, {"x": 32, "y": 0, "w": 16, "h": 32}, {"x": 48, "y": 0, "w": 16, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_right": { "frames": [{"x": 0, "y": 32, "w": 16, "h": 32}, {"x": 16, "y": 32, "w": 16, "h": 32}, {"x": 32, "y": 32, "w": 16, "h": 32}, {"x": 48, "y": 32, "w": 16, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_up": { "frames": [{"x": 0, "y": 64, "w": 16, "h": 32}, {"x": 16, "y": 64, "w": 16, "h": 32}, {"x": 32, "y": 64, "w": 16, "h": 32}, {"x": 48, "y": 64, "w": 16, "h": 32}], "timing": [200, 200, 200, 200] },
                "walk_left": { "frames": [{"x": 0, "y": 96, "w": 16, "h": 32}, {"x": 16, "y": 96, "w": 16, "h": 32}, {"x": 32, "y": 96, "w": 16, "h": 32}, {"x": 48, "y": 96, "w": 16, "h": 32}], "timing": [200, 200, 200, 200] }
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
    process_ff6_sprites()
