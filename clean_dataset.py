#!/usr/bin/env python3
"""
Launcher principal pour le nettoyage et la curation du dataset Pixchron.

Deux modes d'exécution disponibles :

1. MODE IA DGX (Recommandé sur votre serveur NVIDIA Grace Blackwell 120 Go) :
   Utilise un VLM (Qwen2.5-VL) pour étiqueter sémantiquement les images, découpe les fiches
   de présentation (Exagide), défloute le pixel art tramé (Archange), supprime les fonds
   par IA et rejette impitoyablement les photos de cahier quadrillé et dessins lisses :
   
   python clean_dataset.py --mode ai \
       --input_dir datasets_ready \
       --output_dir datasets_cleaned \
       --quarantine_dir datasets_quarantine \
       --remove_bg

2. MODE STANDARD (Mathématique & Multi-CPU rapide) :
   python clean_dataset.py --mode standard \
       --input_dir datasets_ready \
       --output_dir datasets_cleaned
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_factory.batch_cleaner import run_batch_clean
from data_factory.ai_pipeline import run_ai_curation_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixchron - Usine de Curation & Nettoyage de Pixel Art")
    parser.add_argument("--mode", type=str, choices=["ai", "standard"], default="ai", help="Mode de curation : 'ai' (VLM + DGX) ou 'standard' (CPU rapide)")
    parser.add_argument("--input_dir", type=str, default="datasets_ready", help="Dossier d'entrée")
    parser.add_argument("--output_dir", type=str, default="datasets_cleaned", help="Dossier de sortie propre")
    parser.add_argument("--quarantine_dir", type=str, default="datasets_quarantine", help="Dossier d'isolation des faux pixel art")
    
    # Options IA (Mode 'ai')
    parser.add_argument("--vlm_backend", type=str, choices=["transformers", "vllm"], default="transformers", help="Backend VLM : 'transformers' (local) ou 'vllm' (serveur API)")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Modèle VLM utilisé pour le diagnostic")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/v1/chat/completions", help="URL de l'API si backend vllm")
    parser.add_argument("--use_neural_bg", action="store_true", help="Activer le modèle neural RMBG-1.4 sur GPU pour détourer les sprites")
    
    parser.add_argument("--auto_docker_vllm", action="store_true", default=True, help="Démarrer automatiquement vLLM en conteneur Docker DGX si inactif")
    parser.add_argument("--no_auto_docker", action="store_false", dest="auto_docker_vllm", help="Ne pas démarrer automatiquement Docker")
    
    # Options communes
    parser.add_argument("--remove_bg", action="store_true", help="Convertir les fonds solides/dégradés en transparence Alpha (RGBA)")
    parser.add_argument("--workers", type=int, default=4, help="Nombre de workers parallèles")
    parser.add_argument("--sample", type=int, default=0, help="Nombre d'images à traiter pour un test rapide")
    parser.add_argument("--dry_run", action="store_true", help="Analyser sans sauvegarder les fichiers (mode standard)")
    
    # Options mode standard
    parser.add_argument("--tolerance", type=float, default=15.0, help="Distance RGB pour fusionner les artefacts JPEG")
    parser.add_argument("--max_colors", type=int, default=256, help="Palette max autorisée")
    parser.add_argument("--min_size", type=int, default=6, help="Taille native minimale en pixels")
    parser.add_argument("--strict", action="store_true", default=True, help="Activer le mode strict anti-faux pixel art")
    
    args = parser.parse_args()
    
    if args.mode == "ai":
        run_ai_curation_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            quarantine_dir=args.quarantine_dir,
            vlm_backend=args.vlm_backend,
            vlm_model=args.vlm_model,
            vlm_url=args.vlm_url,
            auto_docker_vllm=args.auto_docker_vllm,
            use_neural_bg=args.use_neural_bg,
            remove_bg=args.remove_bg,
            workers=args.workers,
            sample=args.sample
        )
    else:
        run_batch_clean(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            quarantine_dir=args.quarantine_dir,
            tolerance=args.tolerance,
            max_colors=args.max_colors,
            min_size=args.min_size,
            remove_bg=args.remove_bg,
            strict=args.strict,
            workers=args.workers,
            dry_run=args.dry_run,
            sample=args.sample
        )
