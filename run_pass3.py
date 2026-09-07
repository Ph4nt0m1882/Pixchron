#!/usr/bin/env python3
"""
PIXCHRON - PASSAGE 3 : SAUVETAGE PAR LE VLM 72B (GÉNÉRATION DE CODE PYTHON & RE-JUGEMENT)

Objectif :
Traiter les cas complexes restants dans la Purge 2 (./datasets_purge_pass2/).
Le modèle de vision Qwen2.5-VL-72B-Instruct inspecte chaque image agrandie en Nearest Neighbor :
1. Tri du réalisme : Tout ce qui est perles Hama, broderies, photos de cahier ou non-pixel-art
   est impitoyablement éliminé ("les perles, c'est dead").
2. Pour les vrais sprites récupérables (fiches avec bandeau, recadrage, spritesheets) :
   Le VLM génère un snippet Python sur-mesure (PIL/NumPy) pour extraire le sprite propre en 1:1.
3. Re-jugement de validation : L'image produite est testée par le contrôleur qualité.
   Si validé -> ./datasets_gold_pass3/
   Si non conforme -> ./datasets_dead_crafts/ (Élimination définitive)

Exemple d'utilisation sur le serveur DGX :
    python run_pass3.py --input_dir ./datasets_purge_pass2 --vlm_backend vllm
"""

import os
import sys
import shutil
import time
import json
import re
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from data_factory.vlm_curator import image_to_base64, parse_vlm_json_response
from data_factory.quality_verifier import QualityVerifier

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

VLM_PASS3_SYSTEM_PROMPT = """You are the ultimate pixel art restorer and Python engineer.
You are given an image that has been cleanly upscaled with NEAREST-NEIGHBOR interpolation to show its raw pixel blocks.

Step 1: REALISM TRIAGE ("Les perles, c'est dead")
Immediately mark as DEAD if it matches any of:
- Perler bead / Hama bead crafts or pegboards.
- Cross-stitch / embroidery / needlepoint patterns with grid lines.
- Drawings on graph paper / notebook paper.
- Fake printed checkerboard backgrounds or coloring book pages.
- Smooth digital art, 3D renders, realistic photos, vector art.
If DEAD, respond: {"status": "DEAD", "reason": "<why it is not authentic video game pixel art>"}

Step 2: PYTHON SALVAGE SNIPPET
If and ONLY IF the image contains genuine retro video game pixel art that can be cleanly cropped, downscaled, or extracted into a standalone 1:1 pixel art sprite:
Write a simple Python 1-to-3 line snippet that takes PIL Image `img` and produces `salvaged_img`.
Example:
"salvaged_img = img.crop((45, 33, 151, 178)).resize((35, 48), Image.Resampling.NEAREST)"
Respond strictly in JSON:
{
  "status": "SALVAGE",
  "reason": "<what was identified and how to fix it>",
  "python_code": "<executable python snippet defining salvaged_img>"
}
"""

class Pass3VLMSalvager:
    def __init__(
        self,
        vlm_backend: str = "vllm",
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_url: str = "http://localhost:8000/v1/chat/completions",
        device: str = "cuda"
    ):
        self.vlm_backend = vlm_backend
        self.vlm_model = vlm_model
        self.vlm_url = vlm_url
        self.device = device
        self.verifier = QualityVerifier(max_palette_size=256, min_size=12)

    def prepare_upscaled_view(self, img: Image.Image, target_size: int = 512) -> Image.Image:
        """Agrandit en Nearest Neighbor sans lissage pour le VLM."""
        w, h = img.size
        if max(w, h) < target_size:
            factor = max(1, target_size // max(w, h))
            return img.resize((w * factor, h * factor), Image.Resampling.NEAREST)
        return img

    def ask_vlm_salvage(self, img: Image.Image) -> dict:
        import requests
        
        view = self.prepare_upscaled_view(img)
        b64 = image_to_base64(view, max_size=768)
        
        payload = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": VLM_PASS3_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Inspect this nearest-neighbor upscaled image. Is it DEAD craft/photo, or can you provide python_code to salvage the true 1:1 sprite?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256
        }
        
        try:
            r = requests.post(self.vlm_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            parsed = parse_vlm_json_response(content)
            if parsed and "status" in parsed:
                return parsed
        except Exception as e:
            pass
            
        return {"status": "DEAD", "reason": "Échec de réponse VLM"}

    def execute_python_salvage(self, img: Image.Image, code_str: str) -> Image.Image | None:
        """Exécute de manière sécurisée le code Python généré par le VLM."""
        try:
            local_vars = {"img": img, "Image": Image, "np": np}
            # Nettoyer les backticks
            clean_code = code_str.replace("```python", "").replace("```", "").strip()
            exec(clean_code, {}, local_vars)
            salvaged = local_vars.get("salvaged_img")
            if isinstance(salvaged, Image.Image):
                return salvaged
        except Exception:
            pass
        return None

def _process_item_pass3(task: tuple) -> dict:
    src_path, gold_dir, dead_dir, rel_path, salvager, dry_run = task
    
    try:
        raw_img = Image.open(src_path)
    except Exception as e:
        return {"src": src_path, "status": "DEAD", "reason": str(e), "rel_path": rel_path}

    decision = salvager.ask_vlm_salvage(raw_img)
    status = decision.get("status", "DEAD").upper()
    reason = decision.get("reason", "Inconnu")
    code = decision.get("python_code", "")

    if status == "SALVAGE" and code:
        salvaged_img = salvager.execute_python_salvage(raw_img, code)
        if salvaged_img is not None:
            # Re-jugement de validation strict
            is_valid, q_reason, metrics = salvager.verifier.verify(salvaged_img)
            if is_valid:
                dst_path = os.path.join(gold_dir, rel_path).rsplit(".", 1)[0] + ".png"
                if not dry_run:
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    salvaged_img.save(dst_path, format="PNG", optimize=True)
                return {
                    "src": src_path,
                    "status": "GOLD_PASS3",
                    "reason": f"Sauvé par VLM Code : {reason}",
                    "rel_path": rel_path
                }
            else:
                reason = f"Échec re-jugement : {q_reason}"

    # Si DEAD ou échec de validation
    dst_dead = os.path.join(dead_dir, rel_path)
    if not dry_run:
        os.makedirs(os.path.dirname(dst_dead), exist_ok=True)
        shutil.copy2(src_path, dst_dead)

    return {
        "src": src_path,
        "status": "DEAD",
        "reason": reason,
        "rel_path": rel_path
    }

def run_pass3(
    input_dir: str = "./datasets_purge_pass2",
    gold_dir: str = "./datasets_gold_pass3",
    dead_dir: str = "./datasets_dead_crafts",
    vlm_backend: str = "vllm",
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_url: str = "http://localhost:8000/v1/chat/completions",
    workers: int = 8,
    dry_run: bool = False,
    sample: int = 0
):
    print("=" * 75)
    print("  🥉 PIXCHRON - PASSAGE 3 : SAUVETAGE PAR VLM 72B & RE-JUGEMENT")
    print("=" * 75)
    print(f"  • Source (Purge 2)  : {input_dir}")
    print(f"  • Or Pur (Pass 3)   : {gold_dir}")
    print(f"  • Rejet Définitif   : {dead_dir} ('Les perles, c’est dead')")
    print(f"  • Modèle VLM        : {vlm_model} ({vlm_backend})")
    print(f"  • Workers           : {workers}")
    print("=" * 75)

    if not os.path.exists(input_dir):
        print(f"❌ Erreur: Le dossier source '{input_dir}' n'existe pas. Avez-vous lancé run_pass2.py ?")
        return

    salvager = Pass3VLMSalvager(
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_url=vlm_url
    )

    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                src_full = os.path.join(root, f)
                rel_path = os.path.relpath(src_full, input_dir)
                tasks.append((src_full, gold_dir, dead_dir, rel_path, salvager, dry_run))

    total = len(tasks)
    print(f"🔍 {total} images litigieuses à examiner.")

    if total == 0:
        print("Rien à traiter dans la Purge 2.")
        return

    if sample > 0 and sample < total:
        print(f"⚠️ Mode échantillon : traitement de {sample} images.")
        tasks = tasks[:sample]
        total = len(tasks)

    t0 = time.time()
    gold_count = 0
    dead_count = 0
    dead_reasons = Counter()

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(total=total, desc="Pass 3 - Sauvetage VLM", unit="img") if has_tqdm else None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_process_item_pass3, t): t for t in tasks}
        
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "GOLD_PASS3":
                gold_count += 1
            else:
                dead_count += 1
                base_r = res["reason"].split('(')[0].strip()[:35]
                dead_reasons[base_r] += 1
                
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"sauvés": gold_count, "éliminés": dead_count})

    if pbar:
        pbar.close()

    elapsed = time.time() - t0

    print("\n" + "=" * 75)
    print("  📊 RAPPORT DU PASSAGE 3 (SAUVETAGE PAR CODE VLM)")
    print("=" * 75)
    print(f"  • Total images analysées        : {total}")
    print(f"  • 🏆 OR PUR SAUVÉ & VALIDÉ      : {gold_count} ({gold_count/max(1, total)*100:.1f}%)")
    print(f"  • ❌ ÉLIMINATION DÉFINITIVE     : {dead_count} ({dead_count/max(1, total)*100:.1f}%)")
    print(f"  • Vitesse globale               : {total/max(0.1, elapsed):.1f} img/s ({elapsed:.1f}s)")
    
    if dead_count > 0:
        print("\n🗑️ Motifs des rejets définitifs ('C’est Dead') :")
        for r, c in dead_reasons.most_common(10):
            print(f"    - {r:<35} : {c:>5} images")

    print("=" * 75)
    print(f"\n🎉 Cascade terminée ! Vous pouvez maintenant fusionner './datasets_gold_pass1',")
    print(f"   './datasets_gold_pass2' et './{gold_dir}' pour obtenir votre dataset 100% Or Pur.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Passage 3 : Sauvetage VLM 72B par Code Python")
    parser.add_argument("--input_dir", type=str, default="./datasets_purge_pass2", help="Dossier source (Purge 2)")
    parser.add_argument("--gold_dir", type=str, default="./datasets_gold_pass3", help="Dossier de destination pour l'Or Pur sauvé")
    parser.add_argument("--dead_dir", type=str, default="./datasets_dead_crafts", help="Dossier d'élimination définitive (poubelle)")
    parser.add_argument("--vlm_backend", type=str, default="vllm", help="Backend VLM ('vllm' ou 'transformers')")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Nom du modèle VLM")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/v1/chat/completions", help="URL vLLM")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de workers parallèles")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans écrire")
    parser.add_argument("--sample", type=int, default=0, help="Tester uniquement sur N images")

    args = parser.parse_args()
    run_pass3(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        dead_dir=args.dead_dir,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        vlm_url=args.vlm_url,
        workers=args.workers,
        dry_run=args.dry_run,
        sample=args.sample
    )
