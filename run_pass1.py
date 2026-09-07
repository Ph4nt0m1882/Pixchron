#!/usr/bin/env python3
"""
PIXCHRON - PASSAGE 1 : L'OR PUR IMMÉDIAT (STRICTEMENT 1:1 NATIF)

Objectif :
Isoler immédiatement le pixel art 100% pur et intouché.
Règle d'or : 1 pixel de l'image = 1 pixel du sprite.
Toute image avec des macro-pixels (ex: sprite 16x16 dans un fichier 32x32) est REFUSÉE et envoyée en Purge 1.
Sont également refoulés en Purge 1 : les faux damiers peints, les bannières de texte et les grilles de patrons.

Sorties :
- ./datasets_gold_pass1/   : Sprites 1:1 certifiés intouchés.
- ./datasets_purge_pass1/  : Tout le reste (pour inspection manuelle et passage 2).

Exemple d'utilisation sur le serveur DGX :
    python run_pass1.py --input_dir ./datasets_ready
"""

import os
import sys
import shutil
import time
import json
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
import numpy as np

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}

def evaluate_image_pass1(img_path: str) -> tuple[bool, str, dict]:
    """
    Évalue si l'image est un pixel art 1:1 pur natif sans aucun agrandissement.
    """
    try:
        img = Image.open(img_path)
        w, h = img.size
        arr = np.array(img)
        c = arr.shape[-1] if arr.ndim == 3 else 1

        # 1. Vérification de la palette
        pixels = arr.reshape(-1, c)
        unique_colors = len(np.unique(pixels, axis=0))
        
        if unique_colors > 256:
            return False, f"Palette trop riche ({unique_colors} > 256 col)", {"colors": unique_colors}
        if unique_colors <= 2:
            return False, f"Image quasi-vide ou monochrome ({unique_colors} col)", {"colors": unique_colors}

        gray = np.mean(arr[:, :, :3], axis=-1)

        # 2. Détection de bannière / bandeau de texte (ex: en-tête OBÉLIX)
        top_h = max(8, int(h * 0.15))
        if top_h < h:
            top_band = gray[:top_h, :]
            if np.std(np.mean(top_band, axis=1)) < 12: # Bande de couleur uniforme
                top_diff = np.abs(np.diff(top_band, axis=1))
                if np.sum(top_diff > 20) > 40 and np.mean(top_diff > 0) > 0.04:
                    return False, "Bannière de texte détectée en haut", {"banner": True}

        # 3. Détection de faux damier de transparence peint ou grille de patron
        for step in [8, 10, 12, 14, 16]:
            if w > step * 4 and h > step * 4:
                # Test d'autocorrélation périodique sur les colonnes
                prof = np.abs(np.diff(gray, axis=1)).sum(axis=0)
                if len(prof) > step * 3:
                    folded = np.zeros(step)
                    for idx, v in enumerate(prof):
                        folded[idx % step] += v
                    if np.max(folded) > 3.8 * np.mean(folded):
                        return False, f"Grille de patron/faux damier peint ({step}px)", {"grid_step": step}

        # 4. Détection stricte de macro-pixels (Ratio 1:1 strict)
        # S'il y a un facteur d'échelle K >= 2, l'image n'est pas 1:1 native
        diff_h = np.abs(np.diff(gray, axis=1)) > 10
        diff_v = np.abs(np.diff(gray, axis=0)) > 10
        dens_x = np.sum(diff_h, axis=0).astype(float)
        dens_y = np.sum(diff_v, axis=1).astype(float)

        if len(dens_x) > 10 and len(dens_y) > 10:
            dens_x -= np.mean(dens_x)
            dens_y -= np.mean(dens_y)
            ac_x = np.correlate(dens_x, dens_x, mode='full')[len(dens_x)-1:]
            ac_y = np.correlate(dens_y, dens_y, mode='full')[len(dens_y)-1:]
            max_lag = min(24, len(ac_x)-1, len(ac_y)-1)
            
            if max_lag > 3:
                ac = ac_x[:max_lag] + ac_y[:max_lag]
                peaks = []
                for k in range(2, max_lag - 1):
                    if ac[k] > ac[k-1] and ac[k] > ac[k+1] and ac[k] > 0:
                        prominence = ac[k] - max(ac[k-1], ac[k+1])
                        peaks.append((k, ac[k], prominence))
                        
                if peaks:
                    max_val = max(p[1] for p in peaks)
                    for k, val, prom in peaks:
                        if val > 0.40 * max_val and prom > 0.05 * val:
                            return False, f"Macro-pixels détectés ({k}x)", {"scale": k}

        # 5. Test de flou anti-aliasing (gradients continus)
        diffs = np.abs(np.diff(gray, axis=1))
        non_zero = diffs[diffs > 0]
        if len(non_zero) > 50:
            micro_steps = np.mean(non_zero < 3.0)
            if micro_steps > 0.35:
                return False, f"Gradients flous/anti-aliasing ({micro_steps*100:.1f}%)", {"micro_steps": micro_steps}

        return True, "OR PUR 1:1 CERTIFIÉ", {
            "size": (w, h),
            "colors": unique_colors,
            "has_alpha": c == 4 and np.any(arr[:, :, 3] == 0)
        }

    except Exception as e:
        return False, f"Erreur lecture: {e}", {}

def _worker_pass1(task: tuple) -> dict:
    src_path, gold_dir, purge_dir, rel_path, dry_run = task
    is_gold, reason, meta = evaluate_image_pass1(src_path)
    
    dst_dir = gold_dir if is_gold else purge_dir
    dst_path = os.path.join(dst_dir, rel_path)
    
    if not dry_run:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

    return {
        "src": src_path,
        "is_gold": is_gold,
        "reason": reason,
        "meta": meta,
        "rel_path": rel_path
    }

def run_pass1(
    input_dir: str = "./datasets_ready",
    gold_dir: str = "./datasets_gold_pass1",
    purge_dir: str = "./datasets_purge_pass1",
    workers: int = 0,
    dry_run: bool = False,
    sample: int = 0
):
    if workers <= 0:
        workers = os.cpu_count() or 8

    print("=" * 75)
    print("  🥇 PIXCHRON - PASSAGE 1 : EXTRACTION DE L'OR PUR IMMÉDIAT (1:1 STRICT)")
    print("=" * 75)
    print(f"  • Source           : {input_dir}")
    print(f"  • Or Pur (Pass 1)  : {gold_dir}")
    print(f"  • Purge 1          : {purge_dir}")
    print(f"  • Mode Dry-Run     : {'ACTIVÉ (pas de copie)' if dry_run else 'ACTIF'}")
    print(f"  • Workers CPU      : {workers}")
    print("=" * 75)

    if not os.path.exists(input_dir):
        print(f"❌ Erreur: Le dossier source '{input_dir}' n'existe pas.")
        return

    print("🔍 Scan du répertoire source...")
    tasks = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTS:
                src_full = os.path.join(root, f)
                rel_path = os.path.relpath(src_full, input_dir)
                tasks.append((src_full, gold_dir, purge_dir, rel_path, dry_run))

    total = len(tasks)
    print(f"✅ {total} fichiers trouvés.")
    
    if total == 0:
        print("Rien à traiter.")
        return

    if sample > 0 and sample < total:
        print(f"⚠️ Mode échantillon : traitement de {sample} fichiers.")
        tasks = tasks[:sample]
        total = len(tasks)

    t0 = time.time()
    gold_count = 0
    purge_count = 0
    reasons_counter = Counter()

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(total=total, desc="Pass 1 - Or Pur", unit="img") if has_tqdm else None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker_pass1, t): t for t in tasks}
        
        for future in as_completed(futures):
            res = future.result()
            if res["is_gold"]:
                gold_count += 1
            else:
                purge_count += 1
                base_reason = res["reason"].split('(')[0].strip()
                reasons_counter[base_reason] += 1
                
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"or_pur": gold_count, "purge_1": purge_count})

    if pbar:
        pbar.close()

    elapsed = time.time() - t0

    print("\n" + "=" * 75)
    print("  📊 RAPPORT DU PASSAGE 1")
    print("=" * 75)
    print(f"  • Total images analysées      : {total}")
    print(f"  • 🏆 OR PUR 1:1 IMMÉDIAT     : {gold_count} ({gold_count/max(1, total)*100:.1f}%)")
    print(f"  • 📦 PURGE 1 (pour Pass 2)    : {purge_count} ({purge_count/max(1, total)*100:.1f}%)")
    print(f"  • Vitesse de traitement       : {total/max(0.1, elapsed):.1f} img/s ({elapsed:.1f}s)")
    print("\n🔍 Répartition des motifs d'envoi en Purge 1 :")
    for r, c in reasons_counter.most_common():
        bar = "█" * int((c / max(1, purge_count)) * 25)
        print(f"    - {r:<32} : {c:>5} ({c/max(1, purge_count)*100:5.1f}%) | {bar}")
    print("=" * 75)
    print(f"\n👉 Étape suivante : Inspectez visuellement './{gold_dir}' et './{purge_dir}'.")
    print(f"   Quand vous serez prêt, vous pourrez lancer le Passage 2 sur './{purge_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Passage 1 : Extraction de l'Or Pur Immédiat (1:1)")
    parser.add_argument("--input_dir", type=str, default="./datasets_ready", help="Dossier d'origine (~13 000 images brutes)")
    parser.add_argument("--gold_dir", type=str, default="./datasets_gold_pass1", help="Dossier de destination pour l'Or Pur 1:1")
    parser.add_argument("--purge_dir", type=str, default="./datasets_purge_pass1", help="Dossier de destination pour la Purge 1")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de workers CPU (0 = automatique)")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans copier les fichiers")
    parser.add_argument("--sample", type=int, default=0, help="Tester uniquement sur N images")

    args = parser.parse_args()
    run_pass1(
        input_dir=args.input_dir,
        gold_dir=args.gold_dir,
        purge_dir=args.purge_dir,
        workers=args.workers,
        dry_run=args.dry_run,
        sample=args.sample
    )
