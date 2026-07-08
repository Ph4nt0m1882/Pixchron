import os
import argparse
from scraper import AdvancedOpenGameArtScraper, HuggingFaceScraper
from cleaner import PixelArtCleaner
from annotator import PixelArtAnnotator
from packer import WebDatasetPacker

def run_data_factory(source="web", start_page=0, end_page=5, hf_dataset="huggan/pokemon", max_hf_samples=1000, shard_prefix="00", tsr_console="snes", github_url="https://github.com/pret/pokeemerald", kaggle_dataset="ebrahimelgazar/pixel-art", gifs_only=False):
    print(f"=== DÉMARRAGE DE L'USINE À DONNÉES PIXCHRON (Source: {source}) ===")
    
    raw_dir = f"raw_data_part_{shard_prefix}"
    
    # 1. Scraping
    if source == "web":
        scraper = AdvancedOpenGameArtScraper(raw_dir=raw_dir, gifs_only=gifs_only)
        scraper.scrape_latest_2d_art(start_page=start_page, end_page=end_page)
    elif source == "hf":
        scraper = HuggingFaceScraper(raw_dir=raw_dir, gifs_only=gifs_only)
        scraper.scrape_dataset(hf_dataset_name=hf_dataset, max_samples=max_hf_samples)
    elif source == "tsr":
        # Import retardé pour éviter les erreurs circulaires ou importer si on ne l'utilise pas
        from scraper import TheSpritersResourceScraper
        scraper = TheSpritersResourceScraper(raw_dir=raw_dir, gifs_only=gifs_only)
        scraper.scrape_console(console_name=tsr_console, start_index=start_page, end_page=end_page)
    elif source == "lospec":
        from scraper import LospecScraper
        scraper = LospecScraper(raw_dir=raw_dir, gifs_only=gifs_only)
        scraper.scrape_gallery(start_page=start_page, end_page=end_page)
    elif source == "github":
        from scraper import GithubRepoScraper
        scraper = GithubRepoScraper(raw_dir=raw_dir, gifs_only=gifs_only)
        scraper.scrape_repo(repo_url=github_url)
    elif source == "kaggle":
        from scraper import KaggleScraper
        scraper = KaggleScraper(raw_dir=raw_dir) # Kaggle télécharge un ZIP complet, pas possible de filtrer avant
        scraper.scrape_dataset(dataset_name=kaggle_dataset)
    else:
        print("Source inconnue. Utilisez 'web', 'hf', 'tsr', 'lospec', 'github' ou 'kaggle'.")
        return
        
    # Vérification des fichiers (récursivement pour supporter les sous-dossiers comme ceux de Kaggle)
    raw_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if gifs_only and not f.lower().endswith('.gif'):
                # Suppression immédiate pour libérer l'espace disque si on ne veut que les GIFs
                os.remove(os.path.join(root, f))
                continue
                
            if f.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.webp')):
                raw_files.append(os.path.join(root, f))
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
        if metadata.is_animation and filepath.lower().endswith('.gif'):
            # 1. On lit les octets du GIF original brut
            with open(filepath, "rb") as f:
                img_bytes = f.read()
                
            packer.add_sample(
                image_id=filename.split('.')[0], 
                image_bytes=img_bytes, 
                metadata_json_str=metadata.to_json(),
                bucket=metadata.dataset_bucket,
                extension="gif"
            )
            
            # 2. Et on sauvegarde aussi la Frame 1 isolée (condition)
            with open(temp_path, "rb") as f:
                frame1_bytes = f.read()
                
            packer.add_sample(
                image_id=f"{filename.split('.')[0]}_frame1", 
                image_bytes=frame1_bytes, 
                metadata_json_str=metadata.to_json(),
                bucket=metadata.dataset_bucket,
                extension="png"
            )
            os.remove(temp_path)
            
        else:
            # Traitement normal pour les images fixes
            with open(temp_path, "rb") as f:
                img_bytes = f.read()
                
            packer.add_sample(
                image_id=filename.split('.')[0], 
                image_bytes=img_bytes, 
                metadata_json_str=metadata.to_json(),
                bucket=metadata.dataset_bucket,
                extension="png"
            )
            os.remove(temp_path)
        
    packer.close()
    print("\n=== USINE À DONNÉES TERMINÉE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrateur Data Factory Multi-Sources")
    parser.add_argument("--source", type=str, choices=["web", "hf", "tsr", "lospec", "github", "kaggle"], default="web", help="Source de données")
    parser.add_argument("--start", type=int, default=0, help="Index ou page de départ")
    parser.add_argument("--end", type=int, default=5, help="Index ou page de fin")
    parser.add_argument("--hf_dataset", type=str, default="nerijs/pixel-art-xl", help="Nom du dataset HuggingFace")
    parser.add_argument("--hf_samples", type=int, default=1000, help="Nombre d'images à extraire de HF")
    parser.add_argument("--tsr_console", type=str, default="snes", help="Console cible pour TSR (ex: snes, gba)")
    parser.add_argument("--github_url", type=str, default="https://github.com/pret/pokeemerald", help="URL du dépôt GitHub")
    parser.add_argument("--kaggle_dataset", type=str, default="ebrahimelgazar/pixel-art", help="Nom du dataset Kaggle")
    parser.add_argument("--prefix", type=str, default="00", help="Préfixe pour ce worker (évite les conflits de dossiers)")
    parser.add_argument("--gifs_only", action="store_true", help="Ne traiter et ne garder QUE les fichiers .gif animés")
    args = parser.parse_args()
    
    run_data_factory(
        source=args.source, 
        start_page=args.start, 
        end_page=args.end, 
        hf_dataset=args.hf_dataset,
        max_hf_samples=args.hf_samples,
        shard_prefix=args.prefix,
        tsr_console=args.tsr_console,
        github_url=args.github_url,
        kaggle_dataset=args.kaggle_dataset,
        gifs_only=args.gifs_only
    )
