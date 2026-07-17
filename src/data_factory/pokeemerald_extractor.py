import os
import re
import json
import random
from PIL import Image

class PokeemeraldExtractor:
    def __init__(self, repo_path="pokeemerald", out_dir="datasets_ready/pokeemerald"):
        self.repo_path = repo_path
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.anims = {} # sAnim_GoSouth -> {"frames": [3, 0, 4, 0], "timeline_ms": [133, 133, ...]}
        self.anim_tables = {} # sAnimTable_Standard -> {"ANIM_STD_GO_SOUTH": "sAnim_GoSouth"}
        self.pic_tables = {} # sPicTable_NinjaBoy -> gObjectEventPic_NinjaBoy
        self.gfx_paths = {} # gObjectEventPic_NinjaBoy -> graphics/.../ninja_boy.png
        self.entities = {} # gObjectEventGraphicsInfo_NinjaBoy -> {"anims": sAnimTable_Standard, "images": sPicTable_NinjaBoy}
        
        # Load character descriptions if available
        self.descriptions = {}
        desc_path = "character_descriptions.json"
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                self.descriptions = json.load(f)
        
    def run(self):
        print("Parsing Pokeemerald C Headers...")
        self.parse_anims()
        self.parse_anim_tables()
        self.parse_pic_tables()
        self.parse_graphics()
        self.parse_graphics_info()
        
        print(f"Extraction ready. Found {len(self.entities)} entities.")
        self.extract_and_augment()

    def parse_anims(self):
        path = os.path.join(self.repo_path, "src", "data", "object_events", "object_event_anims.h")
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            return

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = re.finditer(r"static const union AnimCmd (sAnim_[a-zA-Z0-9_]+)\[\]\s*=\s*\{(.*?)\};", content, re.DOTALL)
        for b in blocks:
            anim_name = b.group(1)
            inner = b.group(2)
            
            frames = []
            timeline = []
            
            cmds = re.finditer(r"ANIMCMD_FRAME\(\s*(\d+)\s*,\s*(\d+)", inner)
            for c in cmds:
                frame_idx = int(c.group(1))
                duration_ticks = int(c.group(2))
                frames.append(frame_idx)
                timeline.append(int(duration_ticks * 16.666)) # 60 FPS -> ~16.66ms per tick
                
            if frames:
                self.anims[anim_name] = {
                    "frames": frames,
                    "timeline_ms": timeline
                }

    def parse_anim_tables(self):
        path = os.path.join(self.repo_path, "src", "data", "object_events", "object_event_anims.h")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = re.finditer(r"static const union AnimCmd \*const (sAnimTable_[a-zA-Z0-9_]+)\[\]\s*=\s*\{(.*?)\};", content, re.DOTALL)
        for b in blocks:
            table_name = b.group(1)
            inner = b.group(2)
            
            self.anim_tables[table_name] = {}
            entries = re.finditer(r"\[([A-Z0-9_]+)\]\s*=\s*(sAnim_[a-zA-Z0-9_]+)", inner)
            for e in entries:
                anim_type = e.group(1)
                anim_ptr = e.group(2)
                self.anim_tables[table_name][anim_type] = anim_ptr

    def parse_pic_tables(self):
        path = os.path.join(self.repo_path, "src", "data", "object_events", "object_event_pic_tables.h")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = re.finditer(r"static const struct SpriteFrameImage (sPicTable_[a-zA-Z0-9_]+)\[\]\s*=\s*\{(.*?)\};", content, re.DOTALL)
        for b in blocks:
            table_name = b.group(1)
            inner = b.group(2)
            
            match = re.search(r"(gObjectEventPic_[a-zA-Z0-9_]+)", inner)
            if match:
                self.pic_tables[table_name] = match.group(1)

    def parse_graphics(self):
        path = os.path.join(self.repo_path, "src", "data", "object_events", "object_event_graphics.h")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        entries = re.finditer(r"const u32 (gObjectEventPic_[a-zA-Z0-9_]+)\[\]\s*=\s*INCGFX_[A-Z0-9_]+\(\"(.*?\.png)\"", content)
        for e in entries:
            var_name = e.group(1)
            png_path = e.group(2)
            self.gfx_paths[var_name] = png_path

    def parse_graphics_info(self):
        path = os.path.join(self.repo_path, "src", "data", "object_events", "object_event_graphics_info.h")
        if not os.path.exists(path): return
        
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            
        blocks = re.finditer(r"const struct ObjectEventGraphicsInfo (gObjectEventGraphicsInfo_[a-zA-Z0-9_]+)\s*=\s*\{(.*?)\};", content, re.DOTALL)
        for b in blocks:
            entity_name = b.group(1)
            inner = b.group(2)
            
            anims_match = re.search(r"\.anims\s*=\s*(sAnimTable_[a-zA-Z0-9_]+)", inner)
            images_match = re.search(r"\.images\s*=\s*(sPicTable_[a-zA-Z0-9_]+)", inner)
            
            if anims_match and images_match:
                self.entities[entity_name] = {
                    "anims": anims_match.group(1),
                    "images": images_match.group(1)
                }

    def process_image_augmentation(self, img):
        """
        Data Augmentation for Background:
        - 33% Original color (Green/Magenta)
        - 33% Chroma Blue (0, 0, 255)
        - 33% True Alpha Transparency
        """
        img = img.convert("RGBA")
        pixels = img.load()
        width, height = img.size
        
        # Pokeemerald background is always the top-left pixel (Palette 0)
        bg_color = pixels[0, 0]
        
        choice = random.choice(["original", "blue", "alpha"])
        
        if choice == "original":
            return img
            
        for y in range(height):
            for x in range(width):
                if pixels[x, y] == bg_color:
                    if choice == "alpha":
                        pixels[x, y] = (0, 0, 0, 0)
                    elif choice == "blue":
                        pixels[x, y] = (0, 0, 255, 255)
                        
        return img

    def extract_and_augment(self):
        success_count = 0
        for entity, info in self.entities.items():
            pic_table = info["images"]
            anim_table = info["anims"]
            
            if pic_table not in self.pic_tables or anim_table not in self.anim_tables:
                continue
                
            gfx_var = self.pic_tables[pic_table]
            if gfx_var not in self.gfx_paths:
                continue
                
            png_path = self.gfx_paths[gfx_var]
            full_png_path = os.path.join(self.repo_path, png_path)
            
            if not os.path.exists(full_png_path):
                continue
                
            # Build the Animation Dictionary
            output_dict = {}
            for anim_type, anim_ptr in self.anim_tables[anim_table].items():
                if anim_ptr in self.anims:
                    anim_data = self.anims[anim_ptr]
                    output_dict[anim_type] = {
                        "animation_type": anim_type.lower().replace("anim_std_", ""),
                        "frames": anim_data["frames"],
                        "timeline_ms": anim_data["timeline_ms"]
                    }
            
            if not output_dict:
                continue
                
            # Check for Portrait (Front Pic)
            base_filename = os.path.basename(full_png_path)
            front_pic_path = os.path.join(self.repo_path, "graphics", "trainers", "front_pics", base_filename)
            has_portrait = os.path.exists(front_pic_path)
            
            # Process Image
            try:
                overworld_img = Image.open(full_png_path)
                overworld_img = self.process_image_augmentation(overworld_img)
                
                final_img = overworld_img
                
                # Stitch the Portrait if it exists
                if has_portrait:
                    portrait_img = Image.open(front_pic_path)
                    portrait_img = self.process_image_augmentation(portrait_img)
                    
                    # Create a new canvas side-by-side
                    ow_w, ow_h = overworld_img.size
                    pt_w, pt_h = portrait_img.size
                    
                    new_w = ow_w + pt_w
                    new_h = max(ow_h, pt_h)
                    
                    final_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                    final_img.paste(portrait_img, (0, 0)) # Portrait on the left
                    final_img.paste(overworld_img, (pt_w, 0)) # Overworld on the right
                    
                    output_dict["has_portrait"] = True
                    output_dict["portrait_resolution"] = [pt_w, pt_h]
                else:
                    output_dict["has_portrait"] = False
                
                # Save Final Assets
                base_name = entity.replace("gObjectEventGraphicsInfo_", "").lower()
                
                # Clean name to match description keys
                clean_name = base_name.replace("normal", "").replace("machbike", "").replace("acrobike", "").replace("surfing", "").replace("fieldmove", "").replace("underwater", "")
                # Some names might have underscores or minor typos in original files compared to code. Let's find best match.
                matched_desc = None
                for key in self.descriptions:
                    # check if the key matches clean_name (e.g. "gentleman" in "gentleman")
                    # replace underscores for comparison
                    if key.replace("_", "") == clean_name:
                        matched_desc = self.descriptions[key]
                        break
                        
                if matched_desc:
                    output_dict["description"] = matched_desc
                else:
                    output_dict["description"] = ""
                
                out_png = os.path.join(self.out_dir, f"{base_name}.png")
                out_json = os.path.join(self.out_dir, f"{base_name}.json")
                
                final_img.save(out_png)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(output_dict, f, indent=4)
                    
                success_count += 1
            except Exception as e:
                print(f"Error processing {entity}: {e}")
                
        print(f"Extracted {success_count} perfect sprite-timing pairs!")

if __name__ == "__main__":
    extractor = PokeemeraldExtractor()
    extractor.run()
