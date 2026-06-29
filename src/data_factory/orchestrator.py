import os
from scraper import OpenGameArtScraper
from cleaner import PixelArtCleaner
from annotator import PixelArtAnnotator
from packer import WebDatasetPacker

def run_data_factory(start_page=0, end_page=5, shard_prefix="00"):
    print(f"=== DÉMARRAGE DE L'USINE À DONNÉES PIXCHRON (Pages {start_page} à {end_page}) ===")
    
    # Création d'un dossier brut unique pour ce processus parallèle
    raw_dir = f"raw_data_part_{shard_prefix}"
    
    # 1. Scraping
    scraper = OpenGameArtScraper(raw_dir=raw_dir)
    scraper.scrape_latest_2d_art(start_page=start_page, end_page=end_page)
    
    # Initialisation des composants lourds
    cleaner = PixelArtCleaner(max_colors_allowed=256)
    annotator = PixelArtAnnotator(model_id="llava-hf/llava-1.5-7b-hf", device="cuda")
    packer = WebDatasetPacker(output_dir="datasets_ready", max_size_mb=1000)
    
    # 2. Traitement en chaîne
    raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.png', '.gif'))]
    print(f"\n{len(raw_files)} fichiers bruts à traiter...")
    
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        print(f"\n--- Traitement de {filename} ---")
        
        # A. Nettoyage
        img, metadata = cleaner.process_image(filepath, source_url=filepath)
        if img is None or metadata is None:
            continue # Rejeté par le cleaner
            
        # B. Annotation (VLM)
        # On sauvegarde temporairement l'image propre pour que le VLM la lise
        temp_path = f"temp_{filename}"
        img.save(temp_path)
        metadata = annotator.annotate_image(temp_path, metadata)
        
        # C. Empaquetage WebDataset
        with open(temp_path, "rb") as f:
            img_bytes = f.read()
            
        packer.add_sample(
            image_id=filename.split('.')[0], 
            image_bytes=img_bytes, 
            metadata_json_str=metadata.to_json(),
            bucket=metadata.dataset_bucket
        )
        
        # Nettoyage fichier temporaire
        os.remove(temp_path)
        
    # Fin
    packer.close()
    print("\n=== USINE À DONNÉES TERMINÉE ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Orchestrateur Data Factory")
    parser.add_argument("--start", type=int, default=0, help="Page de départ du scraping")
    parser.add_argument("--end", type=int, default=5, help="Page de fin du scraping")
    parser.add_argument("--prefix", type=str, default="00", help="Préfixe pour ce worker (évite les conflits de dossiers)")
    args = parser.parse_args()
    
    # Lancement du pipeline complet
    run_data_factory(start_page=args.start, end_page=args.end, shard_prefix=args.prefix)
