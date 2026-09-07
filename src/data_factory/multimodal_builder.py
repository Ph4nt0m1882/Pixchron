import os
import sys
import time
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_factory.text_captioner import TextCaptioner
from data_factory.image_matcher import ImageReferenceMatcher
from data_factory.sketch_generator import AISketchGenerator

class MultimodalDatasetBuilder:
    """
    Orchestre la séparation 1/3 - 1/3 - 1/3 par répertoire et génère les conditionnements
    Texte (VLM), Image de Référence (CLIP) et Croquis (Sketch) pour l'entraînement Pixchron.
    """

    def __init__(
        self,
        input_dir: str = "datasets_cleaned",
        output_dir: str = "datasets_multimodal",
        raw_scraped_dir: str = "datasets_ready",
        vlm_backend: str = "transformers",
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_url: str = "http://localhost:8000/v1/chat/completions",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
        workers: int = 4
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.raw_scraped_dir = raw_scraped_dir
        self.device = device
        self.workers = workers

        # Initialisation des 3 moteurs d'IA
        print("\n🚀 Initialisation des 3 moteurs d'IA multimodaux pour DGX...")
        self.captioner = TextCaptioner(
            backend=vlm_backend,
            model_name=vlm_model,
            vllm_url=vlm_url,
            device=device
        )
        self.matcher = ImageReferenceMatcher(
            model_name=clip_model,
            raw_scraped_dir=raw_scraped_dir,
            device=device
        )
        self.sketcher = AISketchGenerator(output_size=256)

        # Création des dossiers cibles
        self.images_dir = os.path.join(self.output_dir, "images")
        self.conditions_dir = os.path.join(self.output_dir, "conditions")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.conditions_dir, exist_ok=True)
        self.metadata_path = os.path.join(self.output_dir, "metadata.jsonl")

    def build_split_plan(self, sample_limit: int = 0) -> List[Dict[str, Any]]:
        """
        Scanne tous les sous-répertoires et répartit chaque dossier en 3 tiers équitables.
        """
        plan = []
        categories = sorted([d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))])
        
        print(f"📂 Analyse de {len(categories)} catégories dans '{self.input_dir}'...")

        total_files = 0
        for cat in categories:
            cat_dir = os.path.join(self.input_dir, cat)
            files = sorted([f for f in os.listdir(cat_dir) if f.lower().endswith('.png')])
            if not files:
                continue

            # Mélange reproductible par catégorie
            seed_val = 42 + sum(ord(c) for c in cat)
            rng = random.Random(seed_val)
            rng.shuffle(files)

            n = len(files)
            total_files += n

            # Répartition 1/3
            if n == 1:
                splits = [("text", files[0])]
            elif n == 2:
                splits = [("text", files[0]), ("reference_image", files[1])]
            else:
                n1 = n // 3
                n2 = n // 3
                # Le reste va au croquis
                chunk_text = files[:n1]
                chunk_ref = files[n1:n1 + n2]
                chunk_sketch = files[n1 + n2:]
                
                splits = (
                    [("text", f) for f in chunk_text] +
                    [("reference_image", f) for f in chunk_ref] +
                    [("sketch", f) for f in chunk_sketch]
                )

            for modality, filename in splits:
                plan.append({
                    "category": cat,
                    "filename": filename,
                    "src_path": os.path.join(cat_dir, filename),
                    "modality": modality
                })

        if sample_limit > 0:
            print(f"⚠️ Mode échantillon activé : {sample_limit} images sélectionnées sur {len(plan)}.")
            plan = plan[:sample_limit]

        return plan

    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un élément individuel selon sa modalité assignée.
        """
        cat = item["category"]
        fname = item["filename"]
        src = item["src_path"]
        modality = item["modality"]
        base_name = os.path.splitext(fname)[0]

        img = Image.open(src)
        w, h = img.size
        has_alpha = img.mode == "RGBA"

        # Dossiers de sortie par catégorie
        cat_img_dir = os.path.join(self.images_dir, cat)
        cat_cond_dir = os.path.join(self.conditions_dir, cat)
        os.makedirs(cat_img_dir, exist_ok=True)
        os.makedirs(cat_cond_dir, exist_ok=True)

        # 1. Copier l'image cible nettoyée
        dst_img_path = os.path.join(cat_img_dir, fname)
        shutil.copy2(src, dst_img_path)
        rel_target = os.path.relpath(dst_img_path, self.output_dir)

        caption = ""
        rel_ref_path = None
        rel_sketch_path = None

        # 2. Générer le conditionnement selon la modalité
        if modality == "text":
            caption = self.captioner.generate_caption(src, cat, img)
            # Sauvegarder sidecar .txt
            txt_path = os.path.join(cat_img_dir, f"{base_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as tf:
                tf.write(caption)

        elif modality == "reference_image":
            ref_img = self.matcher.find_best_reference(img, cat, src)
            dst_ref_path = os.path.join(cat_cond_dir, f"{base_name}_ref.png")
            ref_img.save(dst_ref_path, format="PNG")
            rel_ref_path = os.path.relpath(dst_ref_path, self.output_dir)
            caption = f"Pixel art rendition of {cat.replace('_', ' ')} based on reference image"

        elif modality == "sketch":
            sketch_img = self.sketcher.generate_sketch(img)
            dst_sketch_path = os.path.join(cat_cond_dir, f"{base_name}_sketch.png")
            sketch_img.save(dst_sketch_path, format="PNG")
            rel_sketch_path = os.path.relpath(dst_sketch_path, self.output_dir)
            caption = f"Pixel art realization of {cat.replace('_', ' ')} from preparatory pencil sketch"

        # 3. Enregistrer l'entrée dans le JSONL
        entry = {
            "target_image": rel_target,
            "category": cat,
            "modality": modality,
            "caption": caption,
            "reference_image": rel_ref_path,
            "sketch_image": rel_sketch_path,
            "width": w,
            "height": h,
            "has_alpha": has_alpha
        }

        return entry

    def run(self, sample: int = 0):
        t0 = time.time()
        plan = self.build_split_plan(sample_limit=sample)
        total = len(plan)

        print("=" * 75)
        print("  🏭 GÉNÉRATION DU DATASET MULTIMODAL FINAL PIXCHRON")
        print("=" * 75)
        print(f"  • Total éléments à traiter : {total}")
        print(f"  • Dossier source           : {self.input_dir}")
        print(f"  • Dossier final            : {self.output_dir}")
        print(f"  • Workers parallèles       : {self.workers}")
        print("=" * 75)

        counts = {"text": 0, "reference_image": 0, "sketch": 0}
        for item in plan:
            counts[item["modality"]] += 1

        print(f"\n📊 Répartition planifiée :")
        print(f"    - Text-to-PixelArt   : {counts['text']} images ({counts['text']/max(1, total)*100:.1f}%)")
        print(f"    - Image-to-PixelArt  : {counts['reference_image']} images ({counts['reference_image']/max(1, total)*100:.1f}%)")
        print(f"    - Sketch-to-PixelArt : {counts['sketch']} images ({counts['sketch']/max(1, total)*100:.1f}%)")
        print("\n⚡ Traitement en cours...")

        try:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="Génération Multimodale", unit="img")
        except ImportError:
            pbar = None

        processed_entries = []

        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as executor:
            futures = {executor.submit(self.process_item, item): item for item in plan}
            
            for future in as_completed(futures):
                res = future.result()
                processed_entries.append(res)
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        # Sauvegarder le cache VLM
        self.captioner.save_cache()

        # Écriture du fichier metadata.jsonl
        print("\n📝 Écriture de metadata.jsonl...")
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            for entry in processed_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        elapsed = time.time() - t0
        print("\n" + "=" * 75)
        print("  ✅ DATASET MULTIMODAL FINAL GÉNÉRÉ AVEC SUCCÈS")
        print("=" * 75)
        print(f"  • Total images conditionnées : {len(processed_entries)}")
        print(f"  • Fichier index              : {self.metadata_path}")
        print(f"  • Dossier sprites            : {self.images_dir}")
        print(f"  • Dossier conditions         : {self.conditions_dir}")
        print(f"  • Vitesse globale            : {total / max(0.1, elapsed):.1f} img/s ({elapsed:.1f}s)")
        print("=" * 75)

if __name__ == "__main__":
    builder = MultimodalDatasetBuilder()
    builder.run(sample=10)
