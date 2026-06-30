import os
import argparse
from scraper import AdvancedOpenGameArtScraper, HuggingFaceScraper
from cleaner import PixelArtCleaner
from annotator import PixelArtAnnotator
from packer import WebDatasetPacker

def run_data_factory(source="web", start_page=0, end_page=5, hf_dataset="huggan/pokemon", max_hf_samples=1000, shard_prefix="00"):
    print(f"=== DÉMARRAGE DE L'USINE À DONNÉES PIXCHRON (Source: {source}) ===")
    
    raw_dir = f"raw_data_part_{shard_prefix}"
    
    # 1. Scraping
    if source == "web":
        scraper = AdvancedOpenGameArtScraper(raw_dir=raw_dir)
        scraper.scrape_latest_2d_art(start_page=start_page, end_page=end_page)
    elif source == "hf":
        scraper = HuggingFaceScraper(raw_dir=raw_dir)
        scraper.scrape_dataset(hf_dataset_name=hf_dataset, max_samples=max_hf_samples)
    else:
        print("Source inconnue. Utilisez 'web' ou 'hf'.")
        return
        
    # Vérification des fichiers avant de charger les IA lourdes
    raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith(('.png', '.gif', '.jpg', '.jpeg', '.webp'))]
    print(f"\n{len(raw_files)} fichiers bruts à traiter...")
    
    if len(raw_files) == 0:
        print("Aucun fichier à traiter. Arrêt de l'usine.")
        return
        
    # Initialisation des composants lourds
    cleaner = PixelArtCleaner(max_colors_allowed=256)
    annotator = PixelArtAnnotator(model_id="llava-hf/llava-1.5-7b-hf", device="cuda")
    packer = WebDatasetPacker(output_dir="datasets_ready", max_size_mb=1000)
    
    # 2. Traitement en chaîne
    
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        print(f"\n--- Traitement de {filename} ---")
        
        # A. Nettoyage
        img, metadata = cleaner.process_image(filepath, source_url=filepath)
        if img is None or metadata is None:
            continue # Rejeté par le cleaner
            
        # B. Annotation (VLM)
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
        
        os.remove(temp_path)
        
    packer.close()
    print("\n=== USINE À DONNÉES TERMINÉE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur Data Factory Multi-Sources")
    parser.add_argument("--source", type=str, choices=["web", "hf"], default="web", help="Source de données (web ou hf)")
    parser.add_argument("--start", type=int, default=0, help="Page de départ du scraping web")
    parser.add_argument("--end", type=int, default=5, help="Page de fin du scraping web")
    parser.add_argument("--hf_dataset", type=str, default="nerijs/pixel-art-xl", help="Nom du dataset HuggingFace")
    parser.add_argument("--hf_samples", type=int, default=1000, help="Nombre d'images à extraire de HF")
    parser.add_argument("--prefix", type=str, default="00", help="Préfixe pour ce worker (évite les conflits de dossiers)")
    args = parser.parse_args()
    
    run_data_factory(
        source=args.source, 
        start_page=args.start, 
        end_page=args.end, 
        hf_dataset=args.hf_dataset,
        max_hf_samples=args.hf_samples,
        shard_prefix=args.prefix
    )
