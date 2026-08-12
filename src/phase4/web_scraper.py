import json
import os
import shutil
import time
from PIL import Image
from bing_image_downloader import downloader
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
    temp_dir = "temp_images"
    
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
        
        # On utilise "png" dans la requête pour favoriser les images propres
        query = f"{word} pixel art png"
        word_dir = os.path.join("datasets_ready", word.replace(" ", "_"))
        os.makedirs(word_dir, exist_ok=True)
        
        success_count = 0
        
        try:
            # Téléchargement via Bing (beaucoup moins strict sur les rate-limits)
            # Les images sont temporairement téléchargées dans temp_images/
            downloader.download(
                query, 
                limit=target_count, 
                output_dir=temp_dir, 
                adult_filter_off=False, 
                force_replace=True, 
                timeout=10, 
                verbose=False
            )
            
            # Nom du dossier créé par bing-image-downloader
            query_dir = os.path.join(temp_dir, query)
            
            if os.path.exists(query_dir):
                for img_name in os.listdir(query_dir):
                    if success_count >= target_count:
                        break
                        
                    img_path = os.path.join(query_dir, img_name)
                    
                    try:
                        print(f"[{success_count+1}/{target_count}] Traitement de : {img_name}")
                        img = Image.open(img_path)
                        
                        # Reconstruction
                        clean_img, scale = reconstructor.reconstruct(img)
                        
                        # Sauvegarde finale
                        save_path = os.path.join(word_dir, f"{word.replace(' ', '_')}_{success_count + 1}.png")
                        clean_img.save(save_path)
                        print(f"  -> Succès ! Échelle détectée: {scale}x. Sauvegardé.")
                        
                        success_count += 1
                        
                    except Exception as e:
                        print(f"  -> Échec lors du traitement : {str(e)[:100]}")
                        
                # Nettoyage du dossier temporaire pour ce mot
                shutil.rmtree(query_dir)
                
        except Exception as e:
            print(f"Erreur lors de la recherche pour '{word}': {e}")
            
        print(f"Terminé pour '{word}'. {success_count} images sauvegardées.")
        
    # Nettoyage final
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    scrape_images()
