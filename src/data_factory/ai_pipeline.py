import os
import sys
import time
import json
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

from data_factory.vlm_curator import VLMCurator
from data_factory.bg_remover import NeuralBackgroundRemover
from data_factory.sprite_slicer import SpriteSlicer
from data_factory.bicubic_repair import BicubicPixelRepairer
from data_factory.quality_verifier import QualityVerifier
from phase4.pixel_reconstructor import PixelReconstructor

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}

class AIPixelArtCuratorPipeline:
    """
    Pipeline complet de curation et de réparation par IA pour Pixchron.
    Exploite les GPUs DGX pour étiqueter, détourer, découper et réparer le dataset.
    """

    def __init__(
        self,
        vlm_backend: str = "transformers",
        vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_url: str = "http://localhost:8000/v1/chat/completions",
        auto_docker_vllm: bool = True,
        use_neural_bg: bool = False,
        remove_bg: bool = False,
        cache_file: str = "datasets_tags_cache.json",
        device: str = "cuda"
    ):
        self.remove_bg = remove_bg
        self.curator = VLMCurator(
            backend=vlm_backend,
            model_name=vlm_model,
            vllm_url=vlm_url,
            auto_docker_vllm=auto_docker_vllm,
            cache_file=cache_file,
            device=device
        )
        self.bg_remover = NeuralBackgroundRemover(use_neural=use_neural_bg, device=device)
        self.slicer = SpriteSlicer()
        self.repairer = BicubicPixelRepairer()
        self.verifier = QualityVerifier()
        self.reconstructor = PixelReconstructor(strict_mode=True, remove_background=remove_bg)

    def process_single_image(
        self,
        src_path: str,
        output_dir: str,
        quarantine_dir: str,
        rel_path: str
    ) -> Dict[str, Any]:
        """
        Traite une image individuelle selon son diagnostic VLM.
        """
        try:
            raw_img = Image.open(src_path)
        except Exception as e:
            return {"status": "error", "src": src_path, "reason": str(e)}

        # 1. Diagnostic sémantique par le VLM
        diag = self.curator.classify_image(src_path, raw_img)
        img_type = diag.get("type", "native_digital")
        degradation = diag.get("degradation", "none")
        action = diag.get("action_recommended", "clean_integer")
        macro_scale = diag.get("estimated_macro_scale", 1)
        if isinstance(macro_scale, (float, str)):
            try:
                macro_scale = int(round(float(macro_scale)))
            except Exception:
                macro_scale = 1
        macro_scale = max(1, macro_scale)

        # 2. Branche REJET : Photos physiques (cahier à carreaux, perles) et non-pixel-art
        if action == "reject" or img_type in ("physical_photo", "non_pixel_art"):
            reason = diag.get("notes") or f"Type rejeté ({img_type})"
            if quarantine_dir:
                q_sub = os.path.join(quarantine_dir, img_type)
                os.makedirs(q_sub, exist_ok=True)
                shutil.copy2(src_path, os.path.join(q_sub, os.path.basename(src_path)))
            return {
                "status": "rejected",
                "src": src_path,
                "type": img_type,
                "reason": reason
            }

        # 3. Branche DÉCOUPAGE : Fiches de présentation multi-sprites (ex: Exagide)
        if action == "slice_sprites" or img_type == "presentation_card":
            sprites = self.slicer.slice_presentation_card(raw_img, macro_scale=macro_scale)
            saved_paths = []
            
            base_dst = os.path.join(output_dir, rel_path).rsplit(".", 1)[0]
            os.makedirs(os.path.dirname(base_dst), exist_ok=True)
            
            for idx, (sp_img, box) in enumerate(sprites, 1):
                if self.remove_bg:
                    sp_img = self.bg_remover.remove_background(sp_img)
                is_valid, q_reason, metrics = self.verifier.verify(sp_img)
                if is_valid:
                    dst_file = f"{base_dst}_sprite_{idx:02d}.png"
                    sp_img.save(dst_file, format="PNG", optimize=True)
                    saved_paths.append(dst_file)
                    
            if saved_paths:
                return {
                    "status": "success",
                    "src": src_path,
                    "action": "sliced",
                    "sprites_count": len(saved_paths),
                    "saved": saved_paths
                }

        # 4. Branche DÉFLOUTAGE & RECALAGE : Pixel art lissé/interpolé sur le web (ex: Archange)
        if action == "deblur_and_clean" or degradation == "bicubic_blur":
            repaired_img, scale_used = self.repairer.repair(raw_img, estimated_scale=float(macro_scale))
            if self.remove_bg:
                repaired_img = self.bg_remover.remove_background(repaired_img)
                
            is_valid, q_reason, metrics = self.verifier.verify(repaired_img)
            if is_valid:
                dst_file = os.path.join(output_dir, rel_path).rsplit(".", 1)[0] + ".png"
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                repaired_img.save(dst_file, format="PNG", optimize=True)
                return {
                    "status": "success",
                    "src": src_path,
                    "action": "deblurred",
                    "scale": scale_used,
                    "clean_size": repaired_img.size,
                    "dst": dst_file
                }

        # 5. Branche NATIVE : Pixel art digital classique (reconstruction spectrale)
        cleaned_img, sc, stats = self.reconstructor.reconstruct(raw_img)
        if cleaned_img is not None and not stats.get("rejected", False):
            if self.remove_bg:
                cleaned_img = self.bg_remover.remove_background(cleaned_img)
                
            is_valid, q_reason, metrics = self.verifier.verify(cleaned_img)
            if is_valid:
                dst_file = os.path.join(output_dir, rel_path).rsplit(".", 1)[0] + ".png"
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                cleaned_img.save(dst_file, format="PNG", optimize=True)
                return {
                    "status": "success",
                    "src": src_path,
                    "action": "cleaned_native",
                    "scale": sc,
                    "clean_size": cleaned_img.size,
                    "dst": dst_file
                }

        # Fallback rejet si échec de validation qualité
        if quarantine_dir:
            q_sub = os.path.join(quarantine_dir, "failed_quality")
            os.makedirs(q_sub, exist_ok=True)
            shutil.copy2(src_path, os.path.join(q_sub, os.path.basename(src_path)))
            
        return {
            "status": "rejected",
            "src": src_path,
            "type": img_type,
            "reason": stats.get("reason", "Échec validation qualité") if 'stats' in locals() else "Validation échouée"
        }

def run_ai_curation_pipeline(
    input_dir: str = "datasets_ready",
    output_dir: str = "datasets_cleaned",
    quarantine_dir: str = "datasets_quarantine",
    vlm_backend: str = "transformers",
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    vlm_url: str = "http://localhost:8000/v1/chat/completions",
    auto_docker_vllm: bool = True,
    use_neural_bg: bool = False,
    remove_bg: bool = False,
    workers: int = 4,
    sample: int = 0
):
    """
    Point d'entrée pour lancer le pipeline IA complet sur le serveur DGX.
    """
    print("=" * 80)
    print("  🚀 PIXCHRON - PIPELINE IA DE CURATION & RECONSTRUCTION (DGX BLACKWELL)")
    print("=" * 80)
    print(f"  • Dossier Source      : {input_dir}")
    print(f"  • Dossier Propre      : {output_dir}")
    print(f"  • Dossier Quarantaine : {quarantine_dir}")
    print(f"  • VLM Tagger          : {vlm_model} ({vlm_backend})")
    print(f"  • Détourage Neural    : {'ACTIVÉ (RMBG)' if use_neural_bg else 'DÉSACTIVÉ (Algorithmique)'}")
    print(f"  • Détourage Fond      : {'OUI (Alpha transparent)' if remove_bg else 'NON'}")
    print("=" * 80)

    if not os.path.exists(input_dir):
        print(f"❌ Erreur : '{input_dir}' introuvable.")
        return

    pipeline = AIPixelArtCuratorPipeline(
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_url=vlm_url,
        auto_docker_vllm=auto_docker_vllm,
        use_neural_bg=use_neural_bg,
        remove_bg=remove_bg
    )

    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                src_full = os.path.join(root, f)
                rel_path = os.path.relpath(src_full, input_dir)
                tasks.append((src_full, output_dir, quarantine_dir, rel_path))

    total = len(tasks)
    print(f"🔍 {total} images détectées.")

    if sample > 0 and sample < total:
        print(f"⚠️ Mode échantillon : traitement des {sample} premières images.")
        tasks = tasks[:sample]
        total = len(tasks)

    print(f"\n⚡ Lancement de l'analyse IA...\n")
    start_time = time.time()
    
    cleaned_count = 0
    rejected_count = 0
    sprites_extracted = 0
    type_stats = {}

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(total=total, desc="Curation IA", unit="img") if has_tqdm else None

    # Exécution multithreadée (très performante pour les appels GPU batchés et I/O)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(pipeline.process_single_image, *task): task for task in tasks}
        
        for future in as_completed(futures):
            res = future.result()
            st = res.get("status")
            img_type = res.get("type", "inconnu")
            type_stats[img_type] = type_stats.get(img_type, 0) + 1
            
            if st == "success":
                cleaned_count += 1
                if res.get("action") == "sliced":
                    sprites_extracted += res.get("sprites_count", 1)
            elif st == "rejected":
                rejected_count += 1
                
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"clean": cleaned_count, "rejet": rejected_count})

    if pbar:
        pbar.close()

    pipeline.curator.save_cache()
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("  📊 RAPPORT DE SYNTHÈSE DE L'USINE IA PIXCHRON")
    print("=" * 80)
    print(f"  • Total images analysées    : {total}")
    print(f"  • Images nettoyées & sauvées: {cleaned_count}")
    if sprites_extracted > 0:
        print(f"  • Sprites isolés découpes   : {sprites_extracted} nouveaux sprites générés")
    print(f"  • Faux pixel art rejeté     : {rejected_count}")
    print(f"  • Vitesse globale           : {total / max(0.1, elapsed):.1f} img/s ({elapsed:.1f}s)")
    print("\n🏷️ Répartition par Types Diagnostiqués par l'IA :")
    for t_name, count in sorted(type_stats.items(), key=lambda x: -x[1]):
        print(f"    - {t_name:<20} : {count:>5} images")
    print("=" * 80)

if __name__ == "__main__":
    run_ai_curation_pipeline()
