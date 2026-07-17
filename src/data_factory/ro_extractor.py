import os
import json
from PIL import Image

def process_ro_sprites():
    in_dir = "scratch/ro"
    out_dir = "datasets_ready/ro"
    os.makedirs(out_dir, exist_ok=True)
    
    descriptions = {
        'Novice_M': 'Pixel art isométrique fluide (Ragnarok Online). Novice homme, cheveux courts, chemise blanche et short marron.',
        'Novice_F': 'Pixel art isométrique fluide (Ragnarok Online). Novice femme, cheveux blonds attachés, robe sans manches verte et blanche.',
        'Swordsman_M': 'Pixel art isométrique fluide (Ragnarok Online). Épéiste (Swordsman) homme, armure en cuir et plates, bandeau rouge.',
        'Swordsman_F': 'Pixel art isométrique fluide (Ragnarok Online). Épéiste (Swordsman) femme, armure en cuir et jupette, cheveux attachés.'
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
        
        bg_color = data[0, 0][:3]
        for y in range(h):
            for x in range(w):
                if data[x, y][:3] == bg_color:
                    data[x, y] = (0, 0, 0, 0)
                    
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
                            if 0 <= nx < w and 0 <= ny < h and data[nx, ny][3] > 0 and (nx, ny) not in visited:
                                visited.add((nx, ny)); q.append((nx, ny))
                    sprites.append((min_x, min_y, max_x, max_y))
                    
        # RO characters are larger (approx 30x40 to 60x80)
        valid = [s for s in sprites if 15 <= (s[2]-s[0]) <= 80 and 30 <= (s[3]-s[1]) <= 100]
        # Sort by Y with high tolerance (row by row)
        valid.sort(key=lambda s: (s[1] // 50, s[0]))
        
        # We need 64 frames. Skip the first 8 (usually idle) and take next 64 (usually walking)
        # If not enough, just take as many as possible up to 64
        skip = 8
        if len(valid) > skip + 64:
            walking = valid[skip:skip+64]
        else:
            walking = valid[:64]
            
        frames_to_keep = len(walking)
        if frames_to_keep == 0:
            print(f"No frames for {name}")
            continue
            
        # Compile into EIGHT_DIR_FLUID_WALK format
        # 8 directions * 8 frames = 64 frames. Let's make an image 8 * 64 = 512 width, 8 * 64 = 512 height
        # each frame 64x64
        final_img = Image.new("RGBA", (512, 512), (0,0,0,0))
            
        for i in range(frames_to_keep):
            s = walking[i]
            frame_img = img.crop((s[0], s[1], s[2]+1, s[3]+1))
            fw, fh = frame_img.size
            
            row = i // 8
            col = i % 8
            
            fx = col * 64 + (64 - fw) // 2
            fy = row * 64 + (64 - fh) // 2
            
            final_img.paste(frame_img, (fx, fy), frame_img)
            
        anim_data = {
            "format": "EIGHT_DIR_FLUID_WALK",
            "description": desc,
            "animations": {
                "walk_s": {"frames": [{"x": i*64, "y": 0, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_sw": {"frames": [{"x": i*64, "y": 64, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_w": {"frames": [{"x": i*64, "y": 128, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_nw": {"frames": [{"x": i*64, "y": 192, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_n": {"frames": [{"x": i*64, "y": 256, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_ne": {"frames": [{"x": i*64, "y": 320, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_e": {"frames": [{"x": i*64, "y": 384, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8},
                "walk_se": {"frames": [{"x": i*64, "y": 448, "w": 64, "h": 64} for i in range(8)], "timing": [100]*8}
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
    process_ro_sprites()
