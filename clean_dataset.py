#!/usr/bin/env python3
"""
Launcher principal pour le nettoyage du dataset Pixchron avec filtre de pureté anti-faux pixel art.

Exemples d'utilisation sur votre serveur :

1. Lancement standard avec filtrage strict et tous les cœurs CPU :
   python clean_dataset.py --input_dir datasets_ready --output_dir datasets_cleaned

2. Avec détourage automatique du fond (transparence Alpha) :
   python clean_dataset.py --input_dir datasets_ready --output_dir datasets_cleaned --remove_bg

3. Avec isolation des images rejetées (pour inspection manuelle facile) :
   python clean_dataset.py --input_dir datasets_ready --output_dir datasets_cleaned --quarantine_dir datasets_quarantine

4. Test rapide sur 50 images sans écriture (Dry-Run) :
   python clean_dataset.py --sample 50 --dry_run

5. Export du rapport statistique complet :
   python clean_dataset.py --report rapport_nettoyage.json
"""

import sys
import os
import argparse

# Ajouter src au chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_factory.batch_cleaner import run_batch_clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Nettoyeur Haute Précision & Filtre Strict Anti-Faux Pixel Art")
    parser.add_argument("--input_dir", type=str, default="datasets_ready", help="Dossier d'entrée (défaut: datasets_ready)")
    parser.add_argument("--output_dir", type=str, default="datasets_cleaned", help="Dossier de sortie (défaut: datasets_cleaned)")
    parser.add_argument("--quarantine_dir", type=str, default="", help="Dossier d'isolation des images rejetées (ex: datasets_quarantine)")
    parser.add_argument("--tolerance", type=float, default=15.0, help="Distance de couleur RGB pour fusionner le bruit JPEG (défaut: 15.0)")
    parser.add_argument("--max_colors", type=int, default=256, help="Palette max autorisée (défaut: 256)")
    parser.add_argument("--min_size", type=int, default=6, help="Taille native minimale en pixels (défaut: 6)")
    parser.add_argument("--remove_bg", action="store_true", help="Convertir le fond uni/bruité en canal Alpha transparent")
    parser.add_argument("--bg_tolerance", type=float, default=25.0, help="Tolérance pour la détection du fond (défaut: 25.0)")
    parser.add_argument("--strict", action="store_true", default=True, help="Activer le mode strict anti-faux pixel art (activé par défaut)")
    parser.add_argument("--no_strict", action="store_false", dest="strict", help="Désactiver le mode strict")
    parser.add_argument("--max_intra_std", type=float, default=18.0, help="Écart-type max dans les macro-pixels (défaut: 18.0)")
    parser.add_argument("--min_psnr", type=float, default=24.0, help="PSNR minimum pour valider une reconstruction (défaut: 24.0 dB)")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de processus CPU (0 = automatique)")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans sauvegarder les fichiers")
    parser.add_argument("--sample", type=int, default=0, help="Nombre d'images max à traiter pour un test rapide")
    parser.add_argument("--save_metadata", action="store_true", help="Générer un fichier .json de métadonnées par image")
    parser.add_argument("--report", type=str, default="", help="Chemin du fichier JSON de rapport global")
    
    args = parser.parse_args()
    
    run_batch_clean(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        quarantine_dir=args.quarantine_dir,
        tolerance=args.tolerance,
        max_colors=args.max_colors,
        min_size=args.min_size,
        remove_bg=args.remove_bg,
        bg_tolerance=args.bg_tolerance,
        strict=args.strict,
        max_intra_std=args.max_intra_std,
        min_psnr=args.min_psnr,
        workers=args.workers,
        dry_run=args.dry_run,
        sample=args.sample,
        save_metadata=args.save_metadata,
        report_file=args.report
    )
