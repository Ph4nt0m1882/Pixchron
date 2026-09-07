#!/usr/bin/env python3
"""
Générateur Principal du Dataset Multimodal Pixchron.

Ce script sépare chaque répertoire de catégorie en 3 tiers équitables :
1. Tiers Texte (VLM)            : Description textuelle riche pour le Text-to-PixelArt.
2. Tiers Référence Image (CLIP) : Image réelle correspondante pour le Image-to-PixelArt.
3. Tiers Croquis (Sketch IA)    : Esquisse au crayon graphite pour le Sketch-to-PixelArt.

Exemple d'exécution sur votre serveur NVIDIA DGX Grace Blackwell (120 Go) :

1. Lancement complet sur les 9 003 images :
   python generate_final_dataset.py \
       --input_dir datasets_cleaned \
       --output_dir datasets_multimodal \
       --raw_scraped_dir datasets_ready \
       --vlm_backend vllm \
       --device cuda

2. Test rapide sur un échantillon de 30 images :
   python generate_final_dataset.py --sample 30
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data_factory.multimodal_builder import MultimodalDatasetBuilder

def main():
    parser = argparse.ArgumentParser(description="Pixchron - Générateur du Dataset Multimodal Final")
    parser.add_argument("--input_dir", type=str, default="datasets_cleaned", help="Dossier contenant les images pixel art propres")
    parser.add_argument("--output_dir", type=str, default="datasets_multimodal", help="Dossier de sortie du dataset multimodal")
    parser.add_argument("--raw_scraped_dir", type=str, default="datasets_ready", help="Dossier contenant les images brutes de référence")
    
    # Options IA
    parser.add_argument("--vlm_backend", type=str, choices=["transformers", "vllm"], default="transformers", help="Backend VLM pour le captioning")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Modèle VLM pour générer les prompts")
    parser.add_argument("--vlm_url", type=str, default="http://localhost:8000/v1/chat/completions", help="URL API vLLM")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Modèle CLIP ou SigLIP pour la similarité d'images")
    parser.add_argument("--device", type=str, default="cuda", help="Périphérique d'exécution ('cuda' ou 'cpu')")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de workers parallèles")
    parser.add_argument("--sample", type=int, default=0, help="Nombre d'images max à traiter pour un test rapide")

    args = parser.parse_args()

    builder = MultimodalDatasetBuilder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        raw_scraped_dir=args.raw_scraped_dir,
        vlm_backend=args.vlm_backend,
        vlm_model=args.vlm_model,
        vlm_url=args.vlm_url,
        clip_model=args.clip_model,
        device=args.device,
        workers=args.workers
    )

    builder.run(sample=args.sample)

if __name__ == "__main__":
    main()
