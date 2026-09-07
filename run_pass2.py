#!/usr/bin/env python3
"""
PIXCHRON - PASSAGE 2 : CORRECTION DU RATIO SUR LA PURGE 1

Objectif :
Prendre les images refoulées lors du Passage 1 (dans ./datasets_purge_pass1/)
et leur appliquer un prétraitement mathématique de détection et de réduction de ratio
pour les ramener à l'échelle native 1:1.

Si la réduction produit un sprite pixel-perfect conforme (validé par test de résidu cyclique) :
-> Transféré dans ./datasets_gold_pass2/

Si le ratio ne peut pas être corrigé proprement (bruit de papier, perles, cartes complexes, mixels) :
-> Envoyé dans ./datasets_purge_pass2/ pour inspection manuelle et Passage 3.

Exemple d'utilisation sur le serveur DGX :
    python run_pass2.py --input_dir ./datasets_purge_pass1
"""

import os
import sys
import shutil
import time
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from phase4.pixel_reconstructor import PixelReconstructor

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def _worker_pass2(task: tuple) -> dict:
    src_path, gold_dir, purge_dir, rel_path, dry_run = task
    
    try:
        raw_img = Image.open(src_path)
        
        # Initialisation du PixelReconstructor avec tolérance calibrée
        reconstructor = PixelReconstructor(
            color_tolerance=12.0,
            max_colors=256,
            min_native_size=12,
            remove_background=False, # Conserver la géométrie pour inspection manuelle
            strict_mode=True,
            max_intra_block_std=20.0,
            min_psnr=17.0,
            max_mae=16.0
        )
        
        cleaned_img, scale, stats = reconstructor.reconstruct(raw_img)
        
        # Pour être admis au Pass 2, il DOIT avoir été un macro-pixel réduit (scale > 1)
        # et avoir passé avec succès le test de résidu
        is_success = (cleaned_img is not None) and (not stats.get("rejected", False)) and (scale >= 1.8)
        
        if is_success:
            dst_path = os.path.join(gold_dir, rel_path).rsplit(".", 1)[0] + ".png"
            if not dry_run:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                cleaned_img.save(dst_path, format="PNG", optimize=True)
            return {
                "src": src_path,
                "is_gold": True,
                "scale": scale,
                "clean_size": cleaned_img.size,
                "reason": f"Ratio corrigé ({scale:.1f}x -> 1:1)",
                "rel_path": rel_path
            }
        else:
            dst_path = os.path.join(purge_dir, rel_path)
            if not dry_run:
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
            reason = stats.get("reason") if stats else "Échec de recalage de ratio"
            if scale < 1.8:
                reason = f"Échelle non divisible ({scale:.1f}x < 2x)"
            return {
                "src": src_path,
                "is_gold": False,
                "scale": scale,
                "reason": reason,
                "rel_path": rel_path
            }
            
    except Exception as e:
        dst_path = os.path.join(purge_dir, rel_path)
        if not dry_run:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
        return {
            "src": src_path,
            "is_gold": False,
            "scale": 1,
            "reason": f"Erreur: {e}",
            "rel_path": rel_path
        }

def run_pass2(
    input_dir: str = "./datasets_purge_pass1",
    gold_dir: str = "./datasets_gold_pass2",
    purge_dir: str = "./datasets_purge_pass2",
    workers: int = 0,
    dry_run: bool = False,
    sample: int = 0
):
    if workers <= 0:
        workers = os.cpu_count() or 8

    print("=" * 75)
    print("  🥈 PIXCHRON - PASSAGE 2 : CORRECTION DU RATIO SUR LA PURGE 1")
    print("=" * 75)
    print(f"  • Source (Purge 1) : {input_dir}")
    print(f"  • Or Pur (Pass 2)  : {gold_dir}")
    print(f"  • Purge 2          : {purge_dir}")
    print(f"  • Mode Dry-Run     : {'ACTIVÉ (pas d’écriture)' if dry_run else 'ACTIF'}")
    print(f"  • Workers CPU      : {workers}")
    print("=" * 75)

    if not os.path.exists(input_dir):
        print(f"❌ Erreur: Le dossier source '{input_dir}' n'existe pas. Avez-vous lancé run_pass1.py ?")
        return

    print("🔍 Scan du répertoire Purge 1...")
    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                src_full = os.path.join(root, f)
                rel_path = os.path.relpath(src_full, input_dir)
                tasks.append((src_full, gold_dir, purge_dir, rel_path, dry_run))

    total = len(tasks)
    print(f"✅ {total} fichiers à traiter dans la Purge 1.")
    
    if total == 0:
        print("Aucun fichier dans la Purge 1. Fin.")
        return

    if sample > 0 and sample < total:
        print(f"⚠️ Mode échantillon : traitement de {sample} fichiers.")
        tasks = tasks[:sample]
        total = len(tasks)

    t0 = time.time()
    gold_count = 0
    purge_count = 0
    scale_hist = Counter()
    reasons_counter = Counter()

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(total=total, desc="Pass 2 - Ratio", unit="img") if has_tqdm else None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_pass2, t): t for t in tasks}
        
        for future in as_completed(futures):
            res = future.result()
            if res["is_gold"]:
                gold_count += 1
                sc_str = f"{res.get('scale', 1):.0f}x"
                scale_hist[sc_str] += 1
            else:
                purge_count += 1
                base_r = res["reason"].split('(')[0].strip()
                reasons_counter[base_r] += 1
                
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"or_pur_2": gold_count, "purge_2": purge_count})

    if pbar:
        pbar.close()

    elapsed = time.time() - t0

    print("\n" + "=" * 75)
    print("  📊 RAPPORT DU PASSAGE 2")
    print("=" * 75)
    print(f"  • Total images analysées      : {total}")
    print(f"  • 🏆 OR PUR RÉCUPÉRÉ (Pass 2) : {gold_count} ({gold_count/max(1, total)*100:.1f}%)")
    print(f"  • 📦 PURGE 2 (pour Pass 3)    : {purge_count} ({purge_count/max(1, total)*100:.1f}%)")
    print(f"  • Vitesse de traitement       : {total/max(0.1, elapsed):.1f} img/s ({elapsed:.1f}s)")
    
    if gold_count > 0:
        print("\n📈 Échelles de macro-pixels corrigées vers 1:1 :")
        for sc, c in sorted(scale_hist.items(), key=lambda x: -x[1]):
            print(f"    - Échelle {sc:<6} : {c:>5} sprites ramenés à 1:1 natif")

    if purge_count > 0:
        print("\n🔍 Motifs d'envoi en Purge 2 :")
        for r, c in reasons_counter.most_common():
            bar = "█" * int((c / max(1, purge_count)) * 25)
            print(f"    - {r:<32} : {c:>5} ({c/max(1, purge_count)*100:5.1f}%) | {bar}")

    print("=" * 75)
    print(f"\n👉 Étape suivante : Inspectez visuellement './{gold_dir}' et './{purge_dir}'.")
    print(f"   Quand vous serez prêt, vous pourrez lancer le Passage 3 (VLM 72B) sur './{purge_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Passage 2 : Correction du Ratio sur la Purge 1")
    parser.add_argument("--input_dir", type=str, default="./datasets_purge_pass1", help="Dossier source (Purge du Passage 1)")
    parser.add_argument("--gold_dir", type=str, default="./datasets_gold_pass2", help="Dossier de destination pour l'Or Pur corrigé 1:1")
    parser.add_argument("--purge_dir", type=str, default="./datasets_purge_pass2", help="Dossier de destination pour la Purge 2")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de workers CPU (0 = automatique)")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans copier/écrire")
    parser.add_argument("--sample", type=int, default=0, help="Tester uniquement sur N images")

    args = parser.parse_args()
    run_pass2(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        purge_dir=args.purge_dir,
        workers=args.workers,
        dry_run=args.dry_run,
        sample=args.sample
    )
