#!/usr/bin/env python3
"""
PIXCHRON - FUSION DU DATASET OR PUR FINAL

Copie les 3 étapes d'or pur validées manuellement vers ./datasets_gold_final/ :
- ./datasets_gold_pass1/  (Or Pur 1:1 natif)
- ./datasets_gold_pass2/  (Or Pur ratio corrigé)
- ./datasets_gold_pass3/  (Or Pur sauvé par VLM Code)
"""

import os
import shutil
import argparse
from pathlib import Path

def merge_gold(output_dir: str = "./datasets_gold_final"):
    gold_dirs = [
        "./datasets_gold_pass1",
        "./datasets_gold_pass2",
        "./datasets_gold_pass3"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    total_copied = 0

    print("=" * 65)
    print("  👑 FUSION DES 3 ÉTAPES VERS LE DATASET OR PUR FINAL")
    print("=" * 65)

    for g_dir in gold_dirs:
        if not os.path.exists(g_dir):
            print(f"⚠️ {g_dir} n'existe pas encore (étape non exécutée ?)")
            continue
            
        count = 0
        for root, _, files in os.walk(g_dir):
            for f in files:
                if f.lower().endswith('.png'):
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, g_dir)
                    dst = os.path.join(output_dir, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    count += 1
                    total_copied += 1
                    
        print(f"• {g_dir:<25} : {count:>5} images intégrées")

    print("=" * 65)
    print(f"🎉 DATASET OR PUR COMPLET : {total_copied} images dans '{output_dir}'.")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusionner les 3 étapes d'or pur")
    parser.add_argument("--output_dir", type=str, default="./datasets_gold_final")
    args = parser.parse_args()
    merge_gold(args.output_dir)
