#!/usr/bin/env python3
"""
PIXCHRON - PIPELINE DE DOUBLE-CAPTIONING MULTIMODAL (GEMMA 4 31B)

Ce script prend en charge les 3 dossiers de référence :
1. datasets_gold_pass1/     : Sprites pixel art 1:1 vérifiés
2. raw_data_part_LospecAnim/: Animations Lospec (extraction de frame)
3. raw_data_part_OGAAnim/   : Sprites et icônes OpenGameArt

Pour chaque image, le modèle de vision (Gemma 4 31B Dense) génère 2 descriptions :
- Passe 1 (Conceptuelle / Nommée) :
    Image + Nom + Contexte dictionnaire (si disponible).
    -> Description identitaire, rôle, univers et attributs caractéristiques.
- Passe 2 (Visuelle Aveugle / Blind Visual) :
    Image SEULE sans aucun texte d'indice.
    -> Description purement visuelle et objective de la composition, silhouette, couleurs, pose.

Persistance :
- Écriture en streaming dans datasets_multimodal_captions.jsonl.
- Reprise automatique (--resume) sans recalculer les images déjà faites.

Exécution :
    python generate_double_captions.py --vllm_url http://localhost:8000/v1 --workers 8
"""

import os
import sys
import re
import io
import json
import time
import base64
import sqlite3
import argparse
import threading
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm

SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

DEFAULT_DIRS = [
    "datasets_gold_pass1",
    "raw_data_part_LospecAnim",
    "raw_data_part_OGAAnim"
]

def create_resilient_session(pool_size: int = 32) -> requests.Session:
    """Crée une session HTTP avec pool de connexions et retry exponentiel."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_local_knowledge(workspace_dir: str) -> Dict[str, str]:
    """
    Charge les dictionnaires locaux (JSON et SQLite) pour enrichir la Passe 1.
    """
    knowledge = {}
    
    # 1. Dictionnaires JSON (Pokédex, Stardew, etc.)
    for f in os.listdir(workspace_dir):
        if f.startswith("desc_") and f.endswith(".json"):
            fp = os.path.join(workspace_dir, f)
            try:
                with open(fp, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            norm_k = re.sub(r'[\s_\-]+', ' ', k).strip().lower()
                            knowledge[norm_k] = str(v).strip()
            except Exception as e:
                print(f"⚠️ Avertissement : échec de lecture de {f} : {e}")

    # 2. Base SQLite master_dictionary.db
    db_path = os.path.join(workspace_dir, "master_dictionary.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT english_word, french_translation, category, structural_needs FROM dictionary;")
            for row in c.fetchall():
                eng, fr, cat, needs = row
                if eng:
                    norm_k = eng.strip().lower()
                    desc = f"{fr} ({cat})"
                    if needs:
                        desc += f" - Structure: {needs}"
                    if norm_k not in knowledge:
                        knowledge[norm_k] = desc
            conn.close()
        except Exception as e:
            print(f"⚠️ Avertissement : échec de lecture de master_dictionary.db : {e}")

    print(f"📚 Base de connaissances chargée : {len(knowledge)} entrées trouvées.")
    return knowledge

def clean_entity_name(filename: str, category: str) -> str:
    """Nettoie le nom de l'entité à partir de la catégorie ou du nom de fichier."""
    base = os.path.splitext(filename)[0]
    
    # Retirer les suffixes de découpe (_crop_01, _crop_02, etc.)
    base = re.sub(r'_crop_\d+', '', base, flags=re.IGNORECASE)
    # Retirer les chiffres ou tirets de numérotation en fin de nom
    base = re.sub(r'[_\-\s]+\d+$', '', base)

    # Privilégier la catégorie si elle est plus explicite que la racine
    if category and category not in ("Racine", ".", "raw_data_part_OGAAnim", "raw_data_part_LospecAnim"):
        candidate = category
    else:
        candidate = base

    # Nettoyage des caractères de soulignement et tirets
    clean = re.sub(r'[\s_\-]+', ' ', candidate).strip()
    return clean

def extract_representative_frame(img_path: str) -> Tuple[Image.Image, Tuple[int, int], int, bool]:
    """
    Ouvre l'image (PNG ou GIF) et extrait une frame représentative avec ses métadonnées.
    Pour les GIFs, prend une frame intermédiaire pour éviter une frame de spawn vide.
    """
    with Image.open(img_path) as im:
        is_animated = getattr(im, "is_animated", False)
        if is_animated and im.n_frames > 1:
            frame_idx = min(im.n_frames // 2, im.n_frames - 1)
            im.seek(frame_idx)

        # Conversion standard RGBA
        frame = im.convert("RGBA")
        orig_w, orig_h = frame.size

        # Décompte des couleurs uniques et test alpha
        arr = frame.getdata()
        color_set = set(arr)
        num_colors = len(color_set)
        has_alpha = any(px[3] < 255 for px in arr)

        # Upscale au plus proche voisin (Nearest) si l'image est microscopique
        # Cela permet au ViT / vision transformer d'avoir des clusters de pixels nets
        min_dim = min(orig_w, orig_h)
        if min_dim < 128:
            scale = max(2, int(256 / max(orig_w, orig_h, 1)))
            scale = min(scale, 16)
            processed_img = frame.resize((orig_w * scale, orig_h * scale), Image.NEAREST)
        else:
            processed_img = frame.copy()

        return processed_img, (orig_w, orig_h), num_colors, has_alpha

def image_to_base64_data_uri(img: Image.Image) -> str:
    """Encode une image PIL en URI Base64 pour l'API vLLM."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def build_conceptual_prompt(entity_name: str, context: str) -> str:
    """Prompt 1 : Vision + Contexte identitaire."""
    ctx_info = f"\nInformations contextuelles : {context}" if context else ""
    return f"""Tu es un expert en analyse visuelle et en pixel art.
Voici une image en pixel art représentant : **{entity_name}**.{ctx_info}

Rédige une description précise, fluide et complète en français de cette entité en indiquant :
1. Qui ou ce que c'est (identité du sujet, type de créature, rôle, univers ou objet).
2. Son apparence globale et ses attributs caractéristiques (silhouette, posture, expressions).
3. Ses détails spécifiques visibles sur ce sprite (palette de couleurs, vêtements, accessoires, armure ou marques).
Reste factuel, naturel et descriptif. Ne fais aucune phrase d'introduction inutile."""

def build_blind_visual_prompt() -> str:
    """Prompt 2 : Vision pure à l'aveugle, sans mention de nom propre."""
    return """Tu es un expert en analyse visuelle.
Décris uniquement et précisément ce que tu vois sur cette image en pixel art, SANS JAMAIS présumer ni mentionner son nom propre, son identité ou son univers :
1. Le sujet principal (silhouette, anatomie, personnage, créature, objet ou décor).
2. La posture, l'orientation et l'action visible.
3. La palette de couleurs (couleurs dominantes, ombrages, contrastes et contours).
4. Les éléments distinctifs visibles (accessoires, vêtements, marques, textures géométriques).
Reste purement objectif et visuel."""

def query_vlm(session: requests.Session, vllm_url: str, model: str, prompt: str, data_uri: str, max_retries: int = 3) -> str:
    """Envoie une requête de vision à l'endpoint vLLM compatible OpenAI."""
    endpoint = f"{vllm_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }

    for attempt in range(max_retries):
        try:
            resp = session.post(endpoint, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            elif resp.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
            else:
                time.sleep(1.0)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.5 * (attempt + 1))

    return ""

def scan_dataset_files(root_dirs: List[str]) -> List[Dict[str, str]]:
    """Indexe toutes les images des répertoires spécifiés."""
    items = []
    for d in root_dirs:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_EXTS:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, d)
                    cat = os.path.basename(root) if root != d else "Racine"
                    items.append({
                        "source_dir": d,
                        "rel_path": rel_p,
                        "full_path": os.path.abspath(full_p),
                        "filename": f,
                        "category": cat
                    })
    return items

def process_single_item(
    item: Dict[str, str],
    knowledge: Dict[str, str],
    session: requests.Session,
    vllm_url: str,
    model: str
) -> Dict[str, Any]:
    """Traite une image : extrait la frame, génère les 2 descriptions et assemble le record."""
    # Extraction et prétraitement de l'image
    frame_img, orig_res, num_colors, has_alpha = extract_representative_frame(item["full_path"])
    data_uri = image_to_base64_data_uri(frame_img)

    # Résolution de l'entité et du contexte
    entity_name = clean_entity_name(item["filename"], item["category"])
    norm_name = re.sub(r'[\s_\-]+', ' ', entity_name).strip().lower()
    norm_cat = re.sub(r'[\s_\-]+', ' ', item["category"]).strip().lower()

    context = knowledge.get(norm_name) or knowledge.get(norm_cat) or ""

    # Passe 1 : Conceptuelle / Nommée
    p1 = build_conceptual_prompt(entity_name, context)
    cap_conceptual = query_vlm(session, vllm_url, model, p1, data_uri)

    # Passe 2 : Visuelle Aveugle
    p2 = build_blind_visual_prompt()
    cap_visual = query_vlm(session, vllm_url, model, p2, data_uri)

    return {
        "source_dir": item["source_dir"],
        "rel_path": item["rel_path"],
        "full_path": item["full_path"],
        "filename": item["filename"],
        "category": item["category"],
        "entity_name": entity_name,
        "context_injected": context,
        "resolution": list(orig_res),
        "num_colors": num_colors,
        "has_alpha": has_alpha,
        "caption_conceptual": cap_conceptual,
        "caption_visual": cap_visual,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

def main():
    parser = argparse.ArgumentParser(description="Pixchron - Double Captioning VLM Multimodal")
    parser.add_argument("--workspace", type=str, default=".", help="Racine du workspace Pixchron")
    parser.add_argument("--vllm_url", type=str, default="http://localhost:8000/v1", help="URL vLLM API")
    parser.add_argument("--model", type=str, default="google/gemma-4-31B-it", help="Modèle VLM à interroger")
    parser.add_argument("--output", type=str, default="datasets_multimodal_captions.jsonl", help="Fichier de sortie JSONL")
    parser.add_argument("--workers", type=int, default=8, help="Nombre de requêtes parallèles")
    parser.add_argument("--limit", type=int, default=0, help="Limiter le traitement à N images (test)")
    parser.add_argument("--test", action="store_true", help="Teste l'API sur 1 image et affiche les résultats")

    args = parser.parse_args()

    # 1. Vérification / Chargement de la base de connaissances
    knowledge = load_local_knowledge(args.workspace)

    # 2. Indexation des fichiers
    root_dirs = [os.path.join(args.workspace, d) for d in DEFAULT_DIRS]
    all_items = scan_dataset_files(root_dirs)
    print(f"📦 Total images trouvées dans les 3 répertoires : {len(all_items)}")

    if len(all_items) == 0:
        print("❌ Aucune image trouvée. Vérifiez les chemins des dossiers.")
        sys.exit(1)

    # 3. Test de connectivité
    session = create_resilient_session(pool_size=args.workers * 2)
    models_url = f"{args.vllm_url.rstrip('/')}/models"
    try:
        r = session.get(models_url, timeout=10)
        if r.status_code == 200:
            print(f"✅ Serveur vLLM connecté avec succès : {models_url}")
            avail_models = [m["id"] for m in r.json().get("data", [])]
            print(f"   Modèles disponibles : {avail_models}")
        else:
            print(f"⚠️ Serveur vLLM a répondu avec code {r.status_code}")
    except Exception as e:
        print(f"⚠️ Avertissement : Impossible de contacter vLLM à {models_url} : {e}")
        if not args.test:
            print("   Assurez-vous que le conteneur vLLM tourne ('docker compose -f docker-compose.vllm.yml up -d').")

    # Mode Test unitaire
    if args.test:
        sample = all_items[0]
        print(f"\n🔬 TEST UNITAIRE SUR : {sample['full_path']}")
        record = process_single_item(sample, knowledge, session, args.vllm_url, args.model)
        print("\n--- CAPTION 1 (CONCEPTUELLE / NOMMÉE) ---")
        print(record["caption_conceptual"])
        print("\n--- CAPTION 2 (VISUELLE AVEUGLE) ---")
        print(record["caption_visual"])
        return

    # 4. Reprise automatique (--resume)
    out_path = os.path.join(args.workspace, args.output)
    completed_paths = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        completed_paths.add(entry.get("full_path"))
                    except Exception:
                        pass
        print(f"🔄 Reprise : {len(completed_paths)} images déjà traitées dans '{args.output}'.")

    items_to_process = [it for it in all_items if it["full_path"] not in completed_paths]
    if args.limit > 0:
        items_to_process = items_to_process[:args.limit]

    print(f"🚀 Lancement de la génération pour {len(items_to_process)} images avec {args.workers} workers...")
    if len(items_to_process) == 0:
        print("🎉 Toutes les images sont déjà traitées !")
        return

    write_lock = threading.Lock()

    def worker_task(item):
        try:
            rec = process_single_item(item, knowledge, session, args.vllm_url, args.model)
            with write_lock:
                with open(out_path, "a", encoding="utf-8") as out_f:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
            return True, None
        except Exception as err:
            return False, f"Erreur sur {item['rel_path']}: {err}"

    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, item): item for item in items_to_process}
        with tqdm(total=len(items_to_process), desc="Double-Captioning", unit="img") as pbar:
            for fut in as_completed(futures):
                ok, err = fut.result()
                if ok:
                    success_count += 1
                else:
                    error_count += 1
                    tqdm.write(f"❌ {err}")
                pbar.update(1)

    print("\n" + "=" * 65)
    print(f"✨ TRAITEMENT TERMINÉ !")
    print(f"   • Succès : {success_count}")
    print(f"   • Erreurs: {error_count}")
    print(f"   • Fichier final : {out_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
