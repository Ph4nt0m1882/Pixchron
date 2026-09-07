import os
import json
import base64
import re
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image

CAPTION_SYSTEM_PROMPT = """You are an expert video game archivist and machine learning prompt engineer.
Generate a high-quality diffusion training prompt describing this pixel art sprite in rich visual detail.

Describe:
1. Subject & Character: Exactly what it is, species, armor/outfit, weapon or items held.
2. Colors & Palette: Key color scheme (e.g. golden shield, crimson cape, emerald eyes).
3. Pose & Angle: Direction facing (front view, side profile, isometric, action pose).
4. Artistic Style: Crisp retro pixel art, 16-bit / 32-bit video game sprite, clean isolated sprite on transparent background.

Output ONLY a concise, dense, comma-separated training caption in English (60 to 90 words max).
Do NOT include preamble, conversational filler, or quotes.
"""

def image_to_base64_jpeg(img: Image.Image, max_size: int = 512) -> str:
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    buffered = BytesIO()
    # Si RGBA, composer sur fond blanc ou conserver
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, img).convert("RGB")
    else:
        composite = img.convert("RGB")
        
    composite.save(buffered, format="JPEG", quality=88)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

class TextCaptioner:
    """
    Générateur de captions VLM pour le conditionnement Text-to-PixelArt.
    Supporte Transformers (Qwen2.5-VL / Llava) en local bfloat16 sur GPU DGX
    ou le serveur vLLM via API HTTP.
    """

    def __init__(
        self,
        backend: str = "transformers",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        cache_file: str = "datasets_captions_cache.json",
        device: str = "cuda"
    ):
        self.backend = backend
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.cache_file = cache_file
        self.device = device
        self.cache: Dict[str, str] = {}
        self._load_cache()
        
        self.model = None
        self.processor = None
        
        if self.backend == "transformers":
            self._init_transformers()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur de sauvegarde du cache de captions : {e}")

    def _init_transformers(self):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            if not torch.cuda.is_available() and self.device == "cuda":
                self.device = "cpu"
                
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            print(f"📝 Chargement du modèle de captioning {self.model_name} sur {self.device}...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            print("✅ Modèle de captioning VLM prêt !")
        except Exception as e:
            print(f"⚠️ Erreur chargement local VLM ({e}). Bascule en mode fallback prompt.")

    def generate_caption(self, image_path: str, category_name: str, img: Optional[Image.Image] = None) -> str:
        abs_key = os.path.abspath(image_path)
        if abs_key in self.cache:
            return self.cache[abs_key]
            
        if img is None:
            try:
                img = Image.open(image_path)
            except Exception:
                return f"Pixel art sprite of {category_name}, 16-bit video game asset, isolated on transparent background"

        caption = None
        if self.backend == "vllm":
            caption = self._caption_vllm(img, category_name)
        elif self.model is not None and self.processor is not None:
            caption = self._caption_transformers(img, category_name)
            
        if not caption:
            # Fallback structuré
            w, h = img.size
            caption = (
                f"Pixel art sprite of {category_name.replace('_', ' ')}, {w}x{h} resolution, "
                f"clean retro 16-bit video game character asset, detailed outline, isolated on transparent background"
            )
            
        self.cache[abs_key] = caption
        return caption

    def _caption_vllm(self, img: Image.Image, category_name: str) -> Optional[str]:
        import requests
        b64 = image_to_base64_jpeg(img)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Describe this pixel art sprite of '{category_name}':"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 150
        }
        try:
            resp = requests.post(self.vllm_url, json=payload, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            caption = data["choices"][0]["message"]["content"].strip()
            # Nettoyer les balises de réflexion
            caption = re.sub(r'<think>[\s\S]*?</think>', '', caption).strip()
            return caption
        except Exception as e:
            return None

    def _caption_transformers(self, img: Image.Image, category_name: str) -> Optional[str]:
        import torch
        rgb_img = img.convert("RGB")
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_img},
                    {"type": "text", "text": f"Describe this pixel art sprite of '{category_name}':"}
                ]
            }
        ]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[rgb_img], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
            trimmed = out_ids[0][len(inputs.input_ids[0]):]
            res = self.processor.decode(trimmed, skip_special_tokens=True).strip()
            return re.sub(r'<think>[\s\S]*?</think>', '', res).strip()
        except Exception:
            return None

if __name__ == "__main__":
    captioner = TextCaptioner(backend="transformers")
    print("Module TextCaptioner initialisé avec succès.")
