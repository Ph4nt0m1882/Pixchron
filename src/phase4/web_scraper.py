import json
import os
import requests
import time
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS
from .pixel_reconstructor import PixelReconstructor

def scrape_images():
    print("Démarrage du Scraper Web Pixchron (Phase 4)...")
    
    # 1. Charger le dictionnaire
    try:
        with open("target_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("Erreur: target_dataset.json introuvable. Exécutez la phase 3 d'abord.")
        return

    reconstructor = PixelReconstructor()
    os.makedirs("datasets_ready", exist_ok=True)
    
    # Pour ne pas surcharger les serveurs, on limite les requêtes
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    with DDGS() as ddgs:
        # Le fichier JSON a une clé "targets" qui contient la liste
        targets_list = dataset.get("targets", []) if isinstance(dataset, dict) else dataset
        
        for item in targets_list:
            if isinstance(item, dict):
                word = item.get("english_word", "")
                target_count = item.get("target_images_quota", 10) * 3 # On en cherche 3 fois plus
            elif isinstance(item, str):
                word = item
                target_count = 30 # Par défaut si c'est juste une liste de mots
            else:
                continue
                
            if not word:
                continue
                
            print(f"\n--- Recherche pour : '{word}' ---")
            # Ajout de filetype:png selon votre excellente suggestion
            query = f"{word} pixel art filetype:png"
            
            # Dossier de sauvegarde pour ce mot
            word_dir = os.path.join("datasets_ready", word.replace(" ", "_"))
            os.makedirs(word_dir, exist_ok=True)
            
            # Compteur de succès
            success_count = 0
            
            try:
                results = list(ddgs.images(query, max_results=target_count * 2))
                
                for i, r in enumerate(results):
                    if success_count >= target_count:
                        break
                        
                    img_url = r.get("image")
                    if not img_url:
                        continue
                        
                    print(f"[{success_count+1}/{target_count}] Téléchargement : {img_url[:60]}...")
                    
                    try:
                        # Téléchargement avec timeout court
                        response = session.get(img_url, timeout=5)
                        response.raise_for_status()
                        
                        img = Image.open(BytesIO(response.content))
                        
                        # Reconstruction
                        print("  -> Analyse et reconstruction...")
                        clean_img, scale = reconstructor.reconstruct(img)
                        
                        # Sauvegarde
                        save_path = os.path.join(word_dir, f"{word.replace(' ', '_')}_{success_count + 1}.png")
                        clean_img.save(save_path)
                        print(f"  -> Succès ! Échelle détectée: {scale}x. Sauvegardé dans {save_path}")
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"  -> Échec : {str(e)[:100]}")
                        
                    # Petite pause pour ne pas spammer les serveurs d'images
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"Erreur lors de la recherche DDG pour '{word}': {e}")
                
            print(f"Terminé pour '{word}'. {success_count} images sauvegardées.")
            # Pause entre les mots pour DDG
            time.sleep(2)

if __name__ == "__main__":
    scrape_images()
