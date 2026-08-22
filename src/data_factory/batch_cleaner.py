import os
import sys
import time
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from PIL import Image

# Ajouter le chemin racine au sys.path pour les imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from phase4.pixel_reconstructor import PixelReconstructor

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'}

def _clean_worker(task: Tuple[str, str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fonction exécutée par chaque worker dans le pool multi-processus.
    """
    src_path, dst_path, config = task
    
    start_t = time.time()
    try:
        raw_img = Image.open(src_path)
        orig_bytes = os.path.getsize(src_path) if os.path.exists(src_path) else 0
        
        reconstructor = PixelReconstructor(
            color_tolerance=config.get("tolerance", 15.0),
            max_colors=config.get("max_colors", 256),
            min_native_size=config.get("min_size", 6),
            remove_background=config.get("remove_bg", False),
            bg_tolerance=config.get("bg_tolerance", 25.0)
        )
        
        cleaned_img, scale, stats = reconstructor.reconstruct(raw_img)
        
        if cleaned_img is None or stats.get("rejected", False):
            return {
                "status": "rejected",
                "src": src_path,
                "reason": stats.get("reason", "Inconnu"),
                "scale": scale,
                "orig_size": stats.get("original_size", (0, 0)),
                "elapsed": time.time() - start_t
            }
            
        # Si pas en dry-run, enregistrer
        if not config.get("dry_run", False):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            is_anim = stats.get("is_animated", False)
            if is_anim:
                # Sauvegarder en GIF
                dst_path_final = dst_path if dst_path.lower().endswith(".gif") else dst_path.rsplit(".", 1)[0] + ".gif"
                cleaned_img.save(dst_path_final, format="GIF", save_all=True)
            else:
                # Sauvegarder en PNG RGBA
                dst_path_final = dst_path if dst_path.lower().endswith(".png") else dst_path.rsplit(".", 1)[0] + ".png"
                cleaned_img.save(dst_path_final, format="PNG", optimize=True)
                
            if config.get("save_metadata", False):
                meta_path = dst_path_final.rsplit(".", 1)[0] + ".json"
                meta_data = {
                    "source_path": src_path,
                    "scale": scale,
                    "original_size": stats.get("original_size"),
                    "clean_size": stats.get("clean_size"),
                    "palette_size": stats.get("palette_size"),
                    "has_background_removed": stats.get("has_background_removed", False),
                    "is_animated": is_anim,
                    "frames": stats.get("frames", 1)
                }
                with open(meta_path, "w", encoding="utf-8") as mf:
                    json.dump(meta_data, mf, indent=2)
                    
            clean_bytes = os.path.getsize(dst_path_final) if os.path.exists(dst_path_final) else 0
        else:
            clean_bytes = 0

        return {
            "status": "success",
            "src": src_path,
            "dst": dst_path,
            "scale": scale,
            "orig_size": stats.get("original_size"),
            "clean_size": stats.get("clean_size"),
            "palette_size": stats.get("palette_size"),
            "orig_bytes": orig_bytes,
            "clean_bytes": clean_bytes,
            "elapsed": time.time() - start_t
        }

    except Exception as e:
        return {
            "status": "error",
            "src": src_path,
            "reason": str(e),
            "scale": 1,
            "elapsed": time.time() - start_t
        }

def run_batch_clean(
    input_dir: str = "datasets_ready",
    output_dir: str = "datasets_cleaned",
    tolerance: float = 15.0,
    max_colors: int = 256,
    min_size: int = 6,
    remove_bg: bool = False,
    bg_tolerance: float = 25.0,
    workers: int = 0,
    dry_run: bool = False,
    sample: int = 0,
    save_metadata: bool = False,
    report_file: str = ""
):
    """
    Parcourt récursivement input_dir et nettoie toutes les images en parallèle.
    """
    if workers <= 0:
        workers = os.cpu_count() or 4

    print("=" * 70)
    print("  🚀 PIXCHRON - BATCH DATASET CLEANER & PIXEL RECONSTRUCTOR")
    print("=" * 70)
    print(f"  • Dossier source    : {input_dir}")
    print(f"  • Dossier cible     : {output_dir}")
    print(f"  • Tolérance couleur : {tolerance} (distance RGB)")
    print(f"  • Palette max       : {max_colors} couleurs")
    print(f"  • Taille mini       : {min_size}x{min_size} px")
    print(f"  • Détourage fond    : {'OUI (transparence Alpha)' if remove_bg else 'NON (garder fond nettoyé)'}")
    print(f"  • Processus cœurs   : {workers} workers")
    print(f"  • Mode Dry-Run      : {'ACTIVÉ (pas d’écriture disque)' if dry_run else 'DÉSACTIVÉ'}")
    print("=" * 70)

    if not os.path.exists(input_dir):
        print(f"❌ Erreur : Le dossier source '{input_dir}' n'existe pas.")
        return

    # 1. Collecter tous les fichiers images
    print("🔍 Scan du répertoire en cours...")
    tasks = []
    config = {
        "tolerance": tolerance,
        "max_colors": max_colors,
        "min_size": min_size,
        "remove_bg": remove_bg,
        "bg_tolerance": bg_tolerance,
        "dry_run": dry_run,
        "save_metadata": save_metadata
    }

    input_path = Path(input_dir)
    for root, _, files in os.walk(input_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                src_full = os.path.join(root, f)
                rel_path = os.path.relpath(src_full, input_dir)
                dst_full = os.path.join(output_dir, rel_path)
                tasks.append((src_full, dst_full, config))

    total_found = len(tasks)
    print(f"✅ {total_found} images détectées.")

    if total_found == 0:
        print("Aucune image à traiter. Fin.")
        return

    if sample > 0 and sample < total_found:
        print(f"⚠️ Mode échantillon activé : traitement des {sample} premières images.")
        tasks = tasks[:sample]

    total_to_process = len(tasks)
    print(f"\n⚡ Lancement du traitement parallèle avec {workers} processus...\n")

    # 2. Exécution parallèle avec barre de progression
    start_global = time.time()
    results = []
    scale_histogram = {}
    rejection_reasons = {}
    total_cleaned = 0
    total_rejected = 0
    total_errors = 0
    total_bytes_orig = 0
    total_bytes_clean = 0

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    pbar = tqdm(total=total_to_process, desc="Nettoyage", unit="img") if has_tqdm else None

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_clean_worker, task): task for task in tasks}
        
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            results.append(res)
            
            st = res.get("status")
            sc = res.get("scale", 1)
            scale_str = f"{sc}x"
            
            if st == "success":
                total_cleaned += 1
                scale_histogram[scale_str] = scale_histogram.get(scale_str, 0) + 1
                total_bytes_orig += res.get("orig_bytes", 0)
                total_bytes_clean += res.get("clean_bytes", 0)
            elif st == "rejected":
                total_rejected += 1
                reason = res.get("reason", "Inconnu")
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            else:
                total_errors += 1
                reason = res.get("reason", "Erreur indéterminée")
                rejection_reasons[f"Erreur: {reason[:40]}"] = rejection_reasons.get(f"Erreur: {reason[:40]}", 0) + 1

            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    "clean": total_cleaned,
                    "rejet": total_rejected,
                    "err": total_errors
                })
            else:
                if i % max(1, (total_to_process // 20)) == 0 or i == total_to_process:
                    pct = (i / total_to_process) * 100
                    print(f"[{pct:5.1f}%] {i}/{total_to_process} images traitées | Clean: {total_cleaned} | Rejetées: {total_rejected}")

    if pbar:
        pbar.close()

    total_time = time.time() - start_global

    # 3. Rapport de synthèse
    print("\n" + "=" * 70)
    print("  📊 RAPPORT DE SYNTHÈSE DU NETTOYAGE PIXCHRON")
    print("=" * 70)
    print(f"  • Total images analysées    : {total_to_process}")
    print(f"  • Images nettoyées (Valides): {total_cleaned} ({total_cleaned / max(1, total_to_process) * 100:.1f}%)")
    print(f"  • Images rejetées (Qualité) : {total_rejected} ({total_rejected / max(1, total_to_process) * 100:.1f}%)")
    print(f"  • Erreurs de lecture        : {total_errors}")
    print(f"  • Temps d'exécution total   : {total_time:.2f} s ({total_to_process / max(0.01, total_time):.1f} img/s)")
    
    if total_cleaned > 0 and total_bytes_orig > 0 and not dry_run:
        saved_mb = (total_bytes_orig - total_bytes_clean) / (1024 * 1024)
        pct_saved = ((total_bytes_orig - total_bytes_clean) / total_bytes_orig) * 100
        print(f"  • Gain d'espace disque      : {saved_mb:.2f} Mo économisés (-{pct_saved:.1f}%)")

    print("\n📈 Répartition des Échelles de Macro-Pixels Détectées :")
    sorted_scales = sorted(scale_histogram.items(), key=lambda x: -x[1])
    for sc_name, count in sorted_scales:
        bar = "█" * int((count / max(1, total_cleaned)) * 30)
        print(f"    {sc_name:>5} : {count:>5} images ({count / max(1, total_cleaned) * 100:5.1f}%) | {bar}")

    if rejection_reasons:
        print("\n⚠️ Motifs de Rejet :")
        for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1]):
            print(f"    - {reason} : {count} occurrences")

    print("=" * 70)

    # 4. Sauvegarde du rapport optionnel
    if report_file:
        report_data = {
            "summary": {
                "total_scanned": total_to_process,
                "total_cleaned": total_cleaned,
                "total_rejected": total_rejected,
                "total_errors": total_errors,
                "duration_seconds": total_time,
                "scales": scale_histogram,
                "rejections": rejection_reasons
            },
            "results": results
        }
        with open(report_file, "w", encoding="utf-8") as rf:
            json.dump(report_data, rf, indent=2)
        print(f"📄 Rapport JSON détaillé sauvegardé dans : {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron Batch Dataset Cleaner")
    parser.add_argument("--input_dir", type=str, default="datasets_ready", help="Répertoire source des images scrappées")
    parser.add_argument("--output_dir", type=str, default="datasets_cleaned", help="Répertoire de sortie des images propres")
    parser.add_argument("--tolerance", type=float, default=15.0, help="Distance RGB pour fusionner les artefacts JPEG (défaut: 15.0)")
    parser.add_argument("--max_colors", type=int, default=256, help="Palette max autorisée avant rejet (défaut: 256)")
    parser.add_argument("--min_size", type=int, default=6, help="Dimension native minimale en pixels (défaut: 6)")
    parser.add_argument("--remove_bg", action="store_true", help="Convertir automatiquement le fond uni/bruité en canal Alpha transparent")
    parser.add_argument("--bg_tolerance", type=float, default=25.0, help="Tolérance de détection du fond (défaut: 25.0)")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de processus parallèles (0 = tous les cœurs CPU)")
    parser.add_argument("--dry_run", action="store_true", help="Analyser et afficher les stats sans écrire sur le disque")
    parser.add_argument("--sample", type=int, default=0, help="Tester uniquement sur les N premières images")
    parser.add_argument("--save_metadata", action="store_true", help="Générer un fichier .json de métadonnées pour chaque image")
    parser.add_argument("--report", type=str, default="", help="Chemin du fichier JSON de rapport global")
    
    args = parser.parse_args()
    
    run_batch_clean(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        max_colors=args.max_colors,
        min_size=args.min_size,
        remove_bg=args.remove_bg,
        bg_tolerance=args.bg_tolerance,
        workers=args.workers,
        dry_run=args.dry_run,
        sample=args.sample,
        save_metadata=args.save_metadata,
        report_file=args.report
    )
