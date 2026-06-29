import os
import requests
import re
from urllib.parse import urljoin
from typing import List

class OpenGameArtScraper:
    def __init__(self, raw_dir="raw_data"):
        self.base_url = "https://opengameart.org"
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def scrape_latest_2d_art(self, start_page=0, end_page=1):
        """Scrape la section 2D Art d'OpenGameArt à la recherche de pixel art (PNG/GIF)"""
        print(f"--- Lancement du Scraper OpenGameArt (Pages {start_page} à {end_page}) ---")
        
        # Regex pour trouver les liens (inclut jpg car OGA utilise des previews en jpg)
        file_regex = re.compile(r'(?:href|src)=["\']([^"\']+\.(?:png|gif|jpg|jpeg|webp))["\']')
        
        total_downloaded = 0
        
        for page in range(start_page, end_page):
            # URL de la catégorie 2D Art
            url = f"{self.base_url}/art-search-advanced?keys=&title=&field_art_tags_tid_op=or&field_art_tags_tid=pixel%20art&name=&Sort=created&page={page}"
            
            try:
                # OpenGameArt bloque les User-Agents inconnus (anti-bot), on simule un navigateur standard
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    print(f"Erreur HTTP {response.status_code} sur la page {page}")
                    continue
                
                # Cherche tous les fichiers png et gif dans le HTML
                matches = file_regex.findall(response.text)
                
                # Conversion en liens absolus
                unique_urls = list(set([urljoin(self.base_url, match) for match in matches]))
                
                for file_url in unique_urls:
                    if self.download_file(file_url):
                        total_downloaded += 1
                        
            except Exception as e:
                print(f"Erreur lors du scraping de {url}: {e}")
                
        print(f"--- Fin du scraping. {total_downloaded} fichiers téléchargés dans {self.raw_dir} ---")

    def download_file(self, url: str) -> bool:
        """Télécharge un fichier brut s'il n'existe pas déjà."""
        filename = url.split('/')[-1]
        
        # Sécurité : On ignore les très petits fichiers qui sont souvent des icones d'interface
        if "avatar" in filename.lower() or "icon" in filename.lower():
            return False
            
        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            return False # Déjà téléchargé
            
        try:
            print(f"Téléchargement : {filename}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            r = requests.get(url, stream=True, headers=headers)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                return True
        except Exception as e:
            print(f"Échec téléchargement {filename}: {e}")
        return False

if __name__ == "__main__":
    # Test local : On scrape juste la première page pour valider
    scraper = OpenGameArtScraper(raw_dir="src/data_factory/raw_safe_hf")
    scraper.scrape_latest_2d_art(max_pages=1)
