import os
import sys
import json
import re
import time
import base64
import subprocess
from io import BytesIO
from typing import Dict, Any, List, Optional
from PIL import Image

VLM_SYSTEM_PROMPT = """You are an expert Pixel Art Archivist and Machine Learning Dataset Curator.
Analyze this image and diagnose whether it is true digital pixel art or fake/unusable data.

Respond ONLY with a valid JSON object matching this exact schema:
{
  "type": "native_digital | presentation_card | screenshot | physical_photo | non_pixel_art",
  "degradation": "none | jpeg_artifacts | bicubic_blur | low_res_photo | mixed_resolution",
  "layout": "single_sprite | spritesheet | multi_sprite_card | full_scene",
  "has_mixels": false,
  "action_recommended": "clean_integer | slice_sprites | deblur_and_clean | crop_screenshot | reject",
  "estimated_macro_scale": 1,
  "confidence": 0.95,
  "notes": "brief explanation"
}

Classification Rules:
- "physical_photo": Hand-drawn marker/pencil on graph paper (cahier à carreaux), perler beads (perles Hama), embroidery, screen photograph with glare/moire. -> action: "reject"
- "non_pixel_art": Smooth digital illustration, 3D render, realistic photo, vector art. -> action: "reject"
- "presentation_card": Multiple pixel art sprites presented on a background with artist signature/logo/watermark (e.g. Aegislash card with SoulJun logo). -> action: "slice_sprites"
- "screenshot": Retro game screenshot with emulator borders, HUD, dialogue boxes. -> action: "crop_screenshot"
- "native_digital": True digital pixel art sprite or spritesheet, either at 1:1 resolution or upscaled by integer macro-pixels (2x, 3x, 4x, etc.). -> action: "clean_integer" (or "deblur_and_clean" if bicubic blurred)
"""

def image_to_base64(img: Image.Image, max_size: int = 768) -> str:
    """Redimensionne si nécessaire et encode l'image en base64 JPEG haute qualité."""
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        
    buffered = BytesIO()
    img.convert("RGB").save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def parse_vlm_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
    """Parse de manière ultra-robuste le JSON retourné par le VLM."""
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
    
    if "```json" in cleaned:
        parts = cleaned.split("```json")
        for p in parts[1:]:
            block = p.split("```")[0].strip()
            try:
                return json.loads(block)
            except Exception:
                pass
                
    matches = re.findall(r'\{[\s\S]*?\}', cleaned)
    for json_str in reversed(matches):
        try:
            parsed = json.loads(json_str)
            if "type" in parsed:
                return parsed
        except Exception:
            continue
            
    return None

class VLMCurator:
    """
    Classifieur et Tagger de Vision IA pour le dataset Pixchron.
    Supporte :
    1. Inférence locale via HuggingFace Transformers (bfloat16 direct sur GPU).
    2. Serveur vLLM Docker automatisé (avec auto-démarrage du conteneur si besoin).
    """

    def __init__(
        self,
        backend: str = "transformers",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        auto_docker_vllm: bool = True,
        cache_file: str = "datasets_tags_cache.json",
        device: str = "cuda"
    ):
        self.backend = backend
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.auto_docker_vllm = auto_docker_vllm
        self.cache_file = cache_file
        self.device = device
        self.cache: Dict[str, Any] = {}
        self._load_cache()
        
        self.model = None
        self.processor = None
        
        if self.backend == "vllm" and self.auto_docker_vllm:
            self._ensure_vllm_docker_running()
        elif self.backend == "transformers":
            self._init_transformers()

    def _ensure_vllm_docker_running(self):
        """Vérifie si vLLM répond, et le lance en Docker automatiquement si éteint."""
        import requests
        
        health_url = self.vllm_url.replace("/v1/chat/completions", "/health")
        try:
            r = requests.get(health_url, timeout=3)
            if r.status_code == 200:
                print("✅ Serveur vLLM actif et accessible !")
                return
        except Exception:
            pass
            
        print("⚡ Serveur vLLM inactif : Lancement automatique du conteneur Docker DGX...")
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "launch_vllm_dgx.sh"))
        
        if os.path.exists(script_path):
            subprocess.run(["bash", script_path, self.model_name], check=False)
        else:
            print("⚠️ Script launch_vllm_dgx.sh introuvable. Tentative directe...")

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"📦 Cache VLM chargé : {len(self.cache)} images déjà étiquetées.")
            except Exception as e:
                print(f"Avertissement : Impossible de charger le cache VLM ({e})")

    def save_cache(self):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Erreur de sauvegarde du cache VLM : {e}")

    def _init_transformers(self):
        """Initialise le modèle VLM local via HuggingFace en bfloat16."""
        print(f"🚀 Chargement du VLM {self.model_name} sur GPU DGX ({self.device})...")
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            print(f"✅ VLM {self.model_name} prêt en mémoire GPU !")
        except Exception as e:
            print(f"⚠️ Erreur de chargement Transformers local ({e}).")
            print("Astuce: Vous pouvez utiliser le backend vLLM si un serveur Docker est actif.")

    def classify_image(self, image_path: str, img: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Analyse une image et retourne le diagnostic sémantique complet.
        """
        abs_key = os.path.abspath(image_path)
        if abs_key in self.cache:
            return self.cache[abs_key]

        if img is None:
            try:
                img = Image.open(image_path)
            except Exception as e:
                return {
                    "type": "error",
                    "action_recommended": "reject",
                    "rejection_reason": f"Erreur de lecture: {e}"
                }

        if self.backend == "vllm":
            diagnosis = self._classify_vllm(img)
        else:
            diagnosis = self._classify_transformers(img)

        if diagnosis is None:
            diagnosis = {
                "type": "native_digital",
                "degradation": "none",
                "layout": "single_sprite",
                "action_recommended": "clean_integer",
                "estimated_macro_scale": 1,
                "confidence": 0.5,
                "notes": "Fallback par défaut"
            }

        self.cache[abs_key] = diagnosis
        return diagnosis

    def _classify_vllm(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """Interrogation via l'API vLLM locale."""
        import requests
        
        b64 = image_to_base64(img)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Diagnose this image according to the rules and return JSON only:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 512
        }
        
        try:
            resp = requests.post(self.vllm_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return parse_vlm_json_response(content)
        except Exception as e:
            print(f"Erreur requête vLLM : {e}")
            return None

    def _classify_transformers(self, img: Image.Image) -> Optional[Dict[str, Any]]:
        """Inférence directe en local sur GPU via transformers."""
        if self.model is None or self.processor is None:
            return None
            
        import torch
        
        messages = [
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Diagnose this image and return JSON:"}
                ]
            }
        ]
        
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return parse_vlm_json_response(response_text)
        except Exception as e:
            print(f"Erreur inférence VLM transformers : {e}")
            return None

if __name__ == "__main__":
    curator = VLMCurator(backend="vllm" if "VLLM_URL" in os.environ else "transformers")
    print("Module VLMCurator initialisé avec succès.")
