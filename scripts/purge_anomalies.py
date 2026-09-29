import os
import shutil
from PIL import Image
import numpy as np

def purge_anomalies(
    dataset_dir: str = "datasets_cleaned",
    trash_dir: str = "datasets_cleaned_anomalies",
    min_dim: int = 14,
    max_aspect_ratio: float = 6.0,
    dry_run: bool = True
):
    """
    Isole les quelques anomalies restantes (images <= 12x12, vides, ou bandes étroites).
    """
    total = 0
    purged = 0
    reasons_count = {}

    print("=" * 65)
    print(f"🧹 FILTRAGE DES ANOMALIES DU DATASET ({'DRY-RUN' if dry_run else 'ACTIF'})")
    print("=" * 65)

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith('.png'):
                total += 1
                p = os.path.join(root, f)
                try:
                    im = Image.open(p)
                    w, h = im.size
                    arr = np.array(im)
                    
                    is_anomaly = False
                    reason = None
                    
                    # 1. Image 100% vide/transparente
                    if im.mode == "RGBA" and np.all(arr[:, :, 3] == 0):
                        is_anomaly = True
                        reason = "Entièrement transparente (vide)"
                    # 2. Dimensions trop faibles (micro-icônes/favicons)
                    elif w < min_dim or h < min_dim:
                        is_anomaly = True
                        reason = f"Trop petite ({w}x{h} < {min_dim})"
                    # 3. Ratio d'aspect extrême (bande/bordure de carte)
                    elif max(w, h) / max(1, min(w, h)) > max_aspect_ratio:
                        is_anomaly = True
                        reason = f"Ratio extrême ({w}x{h})"
                    # 4. Monochrome 1 seule couleur sur tout le rectangle
                    elif len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0)) <= 1:
                        is_anomaly = True
                        reason = "Monochrome uniforme (1 seule couleur)"
                        
                    if is_anomaly:
                        purged += 1
                        reasons_count[reason] = reasons_count.get(reason, 0) + 1
                        if not dry_run:
                            rel = os.path.relpath(p, dataset_dir)
                            dst = os.path.join(trash_dir, rel)
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.move(p, dst)
                except Exception as e:
                    pass

    print(f"• Total analysé : {total} images")
    print(f"• Anomalies identifiées : {purged} ({purged/max(1, total)*100:.1f}%)")
    print(f"• Dataset final propre restant : {total - purged} ({(total - purged)/max(1, total)*100:.1f}%)")
    print("\nDétail :")
    for r, c in sorted(reasons_count.items(), key=lambda x: -x[1]):
        print(f"  - {r:<35} : {c} fichiers")
    print("=" * 65)

if __name__ == "__main__":
    import sys
    dry = "--apply" not in sys.argv
    purge_anomalies(dry_run=dry)
