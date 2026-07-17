import os
import json
from PIL import Image

def build_stardew_dataset():
    out_dir = "datasets_ready/stardew"
    os.makedirs(out_dir, exist_ok=True)
    
    with open("desc_stardew.json", "r", encoding="utf-8") as f:
        descriptions = json.load(f)
        
    chars_dir = "Content (unpacked)/Characters"
    ports_dir = "Content (unpacked)/Portraits"
    animals_dir = "Content (unpacked)/Animals"
    
    count = 0
    
    # 1. Characters
    if os.path.exists(chars_dir):
        for f in os.listdir(chars_dir):
            if not f.endswith(".png"): continue
            
            name = f.replace(".png", "")
            if name not in descriptions: continue
            
            img = Image.open(os.path.join(chars_dir, f)).convert("RGBA")
            # Filter standard villagers
            if img.size[0] == 64:
                # Need at least 4 rows (128px)
                if img.size[1] < 128: continue
                
                # Crop the 4 walking rows (64x128)
                walk_sheet = img.crop((0, 0, 64, 128))
                
                # Check for portrait
                port_path = os.path.join(ports_dir, f)
                portrait = None
                if os.path.exists(port_path):
                    p_img = Image.open(port_path).convert("RGBA")
                    # Portrait size is 64x64
                    portrait = p_img.crop((0, 0, 64, 64))
                    
                # Compose final image: 128x128. Portrait on left (0,0 to 64,64), Sprite on right (64,0 to 128,128)
                final_img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
                if portrait:
                    final_img.paste(portrait, (0, 0), portrait)
                final_img.paste(walk_sheet, (64, 0), walk_sheet)
                
                # Create JSON
                anim_data = {
                    "format": "WALKING_SPRITE",
                    "description": descriptions[name],
                    "animations": {
                        "walk_down": { "frames": [{"x": 64, "y": 0, "w": 16, "h": 32}, {"x": 80, "y": 0, "w": 16, "h": 32}, {"x": 96, "y": 0, "w": 16, "h": 32}, {"x": 112, "y": 0, "w": 16, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_right": { "frames": [{"x": 64, "y": 32, "w": 16, "h": 32}, {"x": 80, "y": 32, "w": 16, "h": 32}, {"x": 96, "y": 32, "w": 16, "h": 32}, {"x": 112, "y": 32, "w": 16, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_up": { "frames": [{"x": 64, "y": 64, "w": 16, "h": 32}, {"x": 80, "y": 64, "w": 16, "h": 32}, {"x": 96, "y": 64, "w": 16, "h": 32}, {"x": 112, "y": 64, "w": 16, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_left": { "frames": [{"x": 64, "y": 96, "w": 16, "h": 32}, {"x": 80, "y": 96, "w": 16, "h": 32}, {"x": 96, "y": 96, "w": 16, "h": 32}, {"x": 112, "y": 96, "w": 16, "h": 32}], "timing": [250, 250, 250, 250] }
                    }
                }
                
                final_img.save(os.path.join(out_dir, f"{name.lower()}.png"))
                with open(os.path.join(out_dir, f"{name.lower()}.json"), "w", encoding="utf-8") as jf:
                    json.dump(anim_data, jf, indent=4, ensure_ascii=False)
                count += 1

    # 2. Animals
    if os.path.exists(animals_dir):
        for f in os.listdir(animals_dir):
            if not f.endswith(".png"): continue
            
            name = f.replace(".png", "")
            if name not in descriptions: continue
            
            img = Image.open(os.path.join(animals_dir, f)).convert("RGBA")
            # Filter animals (width 128 = 4 frames of 32px)
            if img.size[0] == 128:
                # Need at least 4 rows (128px) for 4 directions
                if img.size[1] < 128: continue
                
                # Crop the 4 walking rows (128x128)
                walk_sheet = img.crop((0, 0, 128, 128))
                
                # Animals don't have portraits, we just save the 128x128 sprite sheet
                final_img = walk_sheet
                
                # Create JSON
                anim_data = {
                    "format": "WALKING_SPRITE_32",
                    "description": descriptions[name],
                    "animations": {
                        "walk_down": { "frames": [{"x": 0, "y": 0, "w": 32, "h": 32}, {"x": 32, "y": 0, "w": 32, "h": 32}, {"x": 64, "y": 0, "w": 32, "h": 32}, {"x": 96, "y": 0, "w": 32, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_right": { "frames": [{"x": 0, "y": 32, "w": 32, "h": 32}, {"x": 32, "y": 32, "w": 32, "h": 32}, {"x": 64, "y": 32, "w": 32, "h": 32}, {"x": 96, "y": 32, "w": 32, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_up": { "frames": [{"x": 0, "y": 64, "w": 32, "h": 32}, {"x": 32, "y": 64, "w": 32, "h": 32}, {"x": 64, "y": 64, "w": 32, "h": 32}, {"x": 96, "y": 64, "w": 32, "h": 32}], "timing": [250, 250, 250, 250] },
                        "walk_left": { "frames": [{"x": 0, "y": 96, "w": 32, "h": 32}, {"x": 32, "y": 96, "w": 32, "h": 32}, {"x": 64, "y": 96, "w": 32, "h": 32}, {"x": 96, "y": 96, "w": 32, "h": 32}], "timing": [250, 250, 250, 250] }
                    }
                }
                
                # Replace spaces in names (e.g. "White Cow" -> "white_cow")
                safe_name = name.lower().replace(" ", "_")
                final_img.save(os.path.join(out_dir, f"{safe_name}.png"))
                with open(os.path.join(out_dir, f"{safe_name}.json"), "w", encoding="utf-8") as jf:
                    json.dump(anim_data, jf, indent=4, ensure_ascii=False)
                count += 1
                
    print(f"Extracted {count} Stardew Valley entities successfully!")

if __name__ == "__main__":
    build_stardew_dataset()
