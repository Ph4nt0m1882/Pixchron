import os
import re
import json
from PIL import Image

class PokecrystalExtractor:
    def __init__(self, repo_path="pokecrystal", out_dir="datasets_ready/pokecrystal"):
        self.repo_path = repo_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.entities = {} # e.g. "ChrisSpriteGFX" -> {"png": "chris.png", "type": "WALKING_SPRITE"}
        
        self.descriptions = {}
        desc_path = "desc_pokecrystal.json"
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                self.descriptions = json.load(f)

    def parse_sprites(self):
        path = os.path.join(self.repo_path, "data", "sprites", "sprites.asm")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.readlines()
            
        for line in content:
            line = line.strip()
            # overworld_sprite ChrisSpriteGFX, 12, WALKING_SPRITE, PAL_OW_RED
            if line.startswith("overworld_sprite"):
                parts = line.replace("overworld_sprite ", "").split(",")
                if len(parts) >= 3:
                    sprite_var = parts[0].strip()
                    length = parts[1].strip()
                    sprite_type = parts[2].strip()
                    
                    if length == "12" and sprite_type == "WALKING_SPRITE":
                        # Guess the PNG name based on standard naming convention in pokecrystal
                        # e.g. ChrisSpriteGFX -> chris.png, CooltrainerMSpriteGFX -> cooltrainer_m.png
                        # Usually the macro points to the file directly in gfx/sprites.asm, let's parse that!
                        self.entities[sprite_var] = {"type": sprite_type}

    def parse_gfx_paths(self):
        path = os.path.join(self.repo_path, "gfx", "sprites.asm")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.readlines()
            
        for line in content:
            # ChrisSpriteGFX::               INCBIN "gfx/sprites/chris.2bpp"
            match = re.search(r"([A-Za-z0-9_]+)::\s*INCBIN\s*\"gfx/sprites/(.*?)\.2bpp\"", line)
            if match:
                var_name = match.group(1)
                png_name = match.group(2) + ".png"
                if var_name in self.entities:
                    self.entities[var_name]["png"] = png_name

    def process_image_augmentation(self, img):
        # In GBC, the background is usually pure white (255, 255, 255) or whatever the first palette color is.
        # Since images are indexed, let's convert to RGBA and make white transparent.
        img = img.convert("RGBA")
        pixels = img.load()
        width, height = img.size
        
        bg_color = pixels[0, 0] # Assume top-left is bg
        for y in range(height):
            for x in range(width):
                if pixels[x, y] == bg_color:
                    pixels[x, y] = (0, 0, 0, 0)
        return img

    def create_12_frame_spritesheet(self, img):
        # Takes a 16x96 (6 frames) img and returns a 16x192 (12 frames) with frames 6-11 flipped
        w, h = img.size
        if w != 16 or h != 96:
            return None
            
        new_img = Image.new("RGBA", (16, 192), (0, 0, 0, 0))
        new_img.paste(img, (0, 0))
        
        # Create flipped frames
        for i in range(6):
            frame = img.crop((0, i*16, 16, (i+1)*16))
            flipped = frame.transpose(Image.FLIP_LEFT_RIGHT)
            new_img.paste(flipped, (0, 96 + i*16))
            
        return new_img

    def extract(self):
        success = 0
        for entity_var, info in self.entities.items():
            if "png" not in info: continue
            
            png_name = info["png"]
            base_name = png_name.replace(".png", "")
            full_path = os.path.join(self.repo_path, "gfx", "sprites", png_name)
            
            if not os.path.exists(full_path): continue
            
            img = Image.open(full_path)
            if img.size != (16, 96): continue # Only process standard walking sprites
            
            img = self.process_image_augmentation(img)
            spritesheet = self.create_12_frame_spritesheet(img)
            
            output_dict = {
                "down": {"animation_type": "down", "frames": [0, 3, 0, 9], "timeline_ms": [133, 133, 133, 133]},
                "up": {"animation_type": "up", "frames": [1, 4, 1, 10], "timeline_ms": [133, 133, 133, 133]},
                "left": {"animation_type": "left", "frames": [2, 5, 2, 5], "timeline_ms": [133, 133, 133, 133]},
                "right": {"animation_type": "right", "frames": [8, 11, 8, 11], "timeline_ms": [133, 133, 133, 133]}
            }
            
            # Portrait (Trainers)
            front_pic_path = os.path.join(self.repo_path, "gfx", "trainers", png_name)
            if os.path.exists(front_pic_path):
                portrait = Image.open(front_pic_path)
                portrait = self.process_image_augmentation(portrait)
                
                # Pad from 56x56 to 64x64
                padded_portrait = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
                padded_portrait.paste(portrait, (4, 4))
                
                # Stitch
                final_img = Image.new("RGBA", (64 + 16, max(64, 192)), (0, 0, 0, 0))
                final_img.paste(padded_portrait, (0, 0))
                final_img.paste(spritesheet, (64, 0))
                
                output_dict["has_portrait"] = True
                output_dict["portrait_resolution"] = [64, 64]
            else:
                final_img = spritesheet
                output_dict["has_portrait"] = False
                
            # Description
            matched_desc = None
            for key in self.descriptions:
                if key == base_name:
                    matched_desc = self.descriptions[key]
                    break
            output_dict["description"] = matched_desc if matched_desc else ""
            
            out_png = os.path.join(self.out_dir, f"{base_name}.png")
            out_json = os.path.join(self.out_dir, f"{base_name}.json")
            
            final_img.save(out_png)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(output_dict, f, indent=4)
                
            success += 1
            
        print(f"Extracted {success} GBC sprite entities successfully!")

if __name__ == "__main__":
    ext = PokecrystalExtractor()
    ext.parse_sprites()
    ext.parse_gfx_paths()
    ext.extract()
