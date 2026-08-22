#!/usr/bin/env python3
"""
Launcher principal pour le nettoyage du dataset Pixchron.

Exemples d'utilisation sur votre serveur :

1. Lancement standard avec tous les cœurs CPU :
   python clean_dataset.py --input_dir datasets_ready --output_dir datasets_cleaned

2. Avec détourage automatique du fond (transparence Alpha) :
   python clean_dataset.py --input_dir datasets_ready --output_dir datasets_cleaned --remove_bg

3. Test rapide sur 50 images sans écriture (Dry-Run) :
   python clean_dataset.py --sample 50 --dry_run

4. Export du rapport statistique complet :
   python clean_dataset.py --report rapport_nettoyage.json
"""

import sys
import os

# Ajouter src au chemin
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_factory.batch_cleaner import run_batch_clean
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Nettoyeur Haute Précision de Pixel Art")
    parser.add_argument("--input_dir", type=str, default="datasets_ready", help="Dossier d'entrée (défaut: datasets_ready)")
    parser.add_argument("--output_dir", type=str, default="datasets_cleaned", help="Dossier de sortie (défaut: datasets_cleaned)")
    parser.add_argument("--tolerance", type=float, default=15.0, help="Distance de couleur RGB pour fusionner le bruit JPEG (défaut: 15.0)")
    parser.add_argument("--max_colors", type=int, default=256, help="Palette max autorisée (défaut: 256)")
    parser.add_argument("--min_size", type=int, default=6, help="Taille native minimale en pixels (défaut: 6)")
    parser.add_argument("--remove_bg", action="store_true", help="Convertir le fond uni/bruité en canal Alpha transparent")
    parser.add_argument("--bg_tolerance", type=float, default=25.0, help="Tolérance pour la détection du fond (défaut: 25.0)")
    parser.add_argument("--workers", type=int, default=0, help="Nombre de processus CPU (0 = automatique)")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans sauvegarder les fichiers")
    parser.add_argument("--sample", type=int, default=0, help="Nombre d'images max à traiter pour un test rapide")
    parser.add_argument("--save_metadata", action="store_true", help="Générer un fichier .json de métadonnées par image")
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
