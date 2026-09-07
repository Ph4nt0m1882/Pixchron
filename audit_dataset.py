import os
import sys
import time
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

dataset_dir = 'datasets_cleaned'

def audit_file(path_str):
    try:
        img = Image.open(path_str)
        w, h = img.size
        mode = img.mode
        arr = np.array(img)
        c = arr.shape[-1] if arr.ndim == 3 else 1
        
        # 1. Palette
        pixels = arr.reshape(-1, c)
        unique_colors = len(np.unique(pixels, axis=0))
        
        # 2. Alpha check
        has_alpha = False
        semi_trans_ratio = 0.0
        transparent_ratio = 0.0
        if mode == 'RGBA' and c == 4:
            alpha = arr[:, :, 3]
            transparent_mask = alpha == 0
            semi_mask = (alpha > 0) & (alpha < 250)
            transparent_ratio = float(np.mean(transparent_mask))
            semi_trans_ratio = float(np.mean(semi_mask))
            has_alpha = transparent_ratio > 0.01

        # 3. Macro-pixel check (is it still upscaled 2x, 3x, 4x?)
        diff_h = np.any(arr[:, 1:, :3] != arr[:, :-1, :3], axis=-1)
        runs_h = []
        for row in diff_h:
            changes = np.where(row)[0]
            if len(changes) > 1:
                runs_h.extend(np.diff(changes))
        
        top_runs = Counter(runs_h).most_common(3)
        has_macro_pixels = False
        detected_run = 1
        if top_runs and top_runs[0][0] > 1 and len(runs_h) > 50:
            count_1 = sum(1 for r in runs_h if r == 1)
            if count_1 < 0.04 * len(runs_h):
                has_macro_pixels = True
                detected_run = top_runs[0][0]

        # 4. Anomaly score & reasons
        anomaly_reasons = []
        score = 0.0
        if unique_colors > 256:
            score += 50
            anomaly_reasons.append(f"Trop de couleurs ({unique_colors})")
        elif unique_colors > 128:
            score += 15
            anomaly_reasons.append(f"Palette riche ({unique_colors} col)")
            
        if unique_colors <= 2:
            score += 30
            anomaly_reasons.append(f"Monochrome/quasi-vide ({unique_colors} col)")
            
        if w < 10 or h < 10:
            score += 30
            anomaly_reasons.append(f"Trop petit ({w}x{h})")
            
        if max(w, h) / max(1, min(w, h)) > 8:
            score += 20
            anomaly_reasons.append(f"Ratio d'aspect extrême ({w}x{h})")
            
        if semi_trans_ratio > 0.08:
            score += 25
            anomaly_reasons.append(f"Halo semi-transparent ({semi_trans_ratio*100:.1f}%)")
            
        if has_macro_pixels:
            score += 40
            anomaly_reasons.append(f"Reste de macro-pixels ({detected_run}x)")

        return {
            'path': path_str,
            'rel_path': os.path.relpath(path_str, dataset_dir),
            'w': w,
            'h': h,
            'palette': unique_colors,
            'has_alpha': has_alpha,
            'transparent_ratio': transparent_ratio,
            'semi_trans_ratio': semi_trans_ratio,
            'has_macro_pixels': has_macro_pixels,
            'score': score,
            'reasons': anomaly_reasons,
            'error': None
        }
    except Exception as e:
        return {'path': path_str, 'rel_path': path_str, 'error': str(e), 'score': 100, 'reasons': [f'Erreur: {e}']}

def main():
    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith('.png'):
                all_files.append(os.path.join(root, f))

    print(f"Lancement de l'audit sur {len(all_files)} fichiers avec multi-processing...")
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(audit_file, all_files, chunksize=100))
    t1 = time.time()
    print(f"Audit terminé en {t1-t0:.2f}s ({len(all_files)/(t1-t0):.1f} img/s) !")

    # Global Stats
    errors = [r for r in results if r.get('error')]
    widths = [r['w'] for r in results if not r.get('error')]
    heights = [r['h'] for r in results if not r.get('error')]
    palettes = [r['palette'] for r in results if not r.get('error')]
    alphas = [r for r in results if not r.get('error') and r['has_alpha']]
    macros = [r for r in results if not r.get('error') and r['has_macro_pixels']]
    outliers = sorted([r for r in results if r['score'] > 0], key=lambda x: -x['score'])

    print("\n" + "=" * 70)
    print("  📊 RAPPORT D'AUDIT GLOBAL DU DATASET (9 769 IMAGES)")
    print("=" * 70)
    print(f"  • Total fichiers vérifiés : {len(results)}")
    print(f"  • Images avec Alpha       : {len(alphas)} ({len(alphas)/len(results)*100:.1f}%)")
    print(f"  • Erreurs de lecture      : {len(errors)}")
    print(f"  • Résolution (Largeur)    : min={np.min(widths)} | médiane={np.median(widths):.0f} | p95={np.percentile(widths, 95):.0f} | max={np.max(widths)}")
    print(f"  • Résolution (Hauteur)    : min={np.min(heights)} | médiane={np.median(heights):.0f} | p95={np.percentile(heights, 95):.0f} | max={np.max(heights)}")
    print(f"  • Couleurs par Palette    : min={np.min(palettes)} | médiane={np.median(palettes):.0f} | p95={np.percentile(palettes, 95):.0f} | max={np.max(palettes)}")
    print(f"  • Macro-pixels résiduels  : {len(macros)} ({len(macros)/len(results)*100:.1f}%)")
    print(f"  • Images 100% conformes   : {len(results) - len(outliers)} ({(len(results) - len(outliers))/len(results)*100:.1f}%)")
    print(f"  • Images avec alertes     : {len(outliers)} ({len(outliers)/len(results)*100:.1f}%)")
    print("=" * 70)

    # Répartition des types d'alertes
    all_reasons = []
    for o in outliers:
        all_reasons.extend(o['reasons'])
    reason_counts = Counter([r.split('(')[0].strip() for r in all_reasons])
    
    print("\n⚠️ Répartition des Alertes Détectées :")
    for reason, count in reason_counts.most_common():
        print(f"    - {reason:<30} : {count:>5} occurrences ({count/len(results)*100:.1f}%)")

    print("\n🔍 TOP 30 DES ANOMALIES LES PLUS MARQUÉES :")
    print("-" * 100)
    for i, o in enumerate(outliers[:30], 1):
        print(f"#{i:02d} [Score {o['score']:>2.0f}] {o['rel_path']:<40} | {o['w']:>3}x{o['h']:<3} | {o['palette']:>3} col | {', '.join(o['reasons'])}")
    print("-" * 100)

if __name__ == '__main__':
    main()
