import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
from io import BytesIO
import shutil

class AdvancedOpenGameArtScraper:
    def __init__(self, raw_dir="raw_data"):
        self.base_url = "https://opengameart.org"
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
    def _extract_zip(self, zip_bytes: bytes, prefix: str):
        """Extrait uniquement les images d'une archive zip directement en mémoire."""
        try:
            with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
                for file_info in z.infolist():
                    if file_info.filename.lower().endswith(('.png', '.gif')) and not file_info.filename.startswith('__MACOSX'):
                        # On aplatit l'arborescence du zip pour éviter les sous-dossiers
                        target_filename = f"{prefix}_{os.path.basename(file_info.filename)}"
                        target_path = os.path.join(self.raw_dir, target_filename)
                        
                        with z.open(file_info) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
                        print(f"  -> Extrait : {target_filename}")
        except Exception as e:
            print(f"Erreur d'extraction ZIP : {e}")

    def scrape_latest_2d_art(self, start_page=0, end_page=1):
        """Navigue sur les pages de recherche, puis entre dans chaque fiche pour télécharger les vrais assets."""
        print(f"--- Lancement OGA Scraper (Pages {start_page} à {end_page}) ---")
        
        for page in range(start_page, end_page):
            url = f"{self.base_url}/art-search-advanced?keys=&title=&field_art_tags_tid_op=or&field_art_tags_tid=pixel%20art&name=&Sort=created&page={page}"
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Trouver tous les liens vers les fiches d'assets (les noeuds /content/...)
                # On cherche tous les liens 'a' dont le href commence par '/content/'
                node_links = set()
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith('/content/'):
                        node_links.add(urljoin(self.base_url, href))
                
                if not node_links:
                    print(f"Aucun lien /content/ trouvé sur la page {page}. Le site a peut-être bloqué la requête.")
                
                for node_url in node_links:
                    self._scrape_node(node_url)
                    
            except Exception as e:
                print(f"Erreur Page {page} : {e}")
                
    def _scrape_node(self, node_url: str):
        """Visite la page détaillée d'un asset et télécharge les fichiers sources joints."""
        try:
            print(f"Visite de la fiche : {node_url}")
            response = requests.get(node_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Les vrais fichiers à télécharger sont généralement dans la classe 'file'
            for a_tag in soup.select('span.file a'):
                file_url = a_tag.get('href')
                if not file_url:
                    continue
                
                file_url = urljoin(self.base_url, file_url)
                filename = file_url.split('/')[-1]
                
                # On ne prend que les PNG, GIF, et ZIP
                if filename.lower().endswith(('.png', '.gif', '.zip')):
                    self.download_asset(file_url, filename)
        except Exception as e:
            print(f"Erreur Noeud {node_url}: {e}")

    def download_asset(self, url: str, filename: str) -> bool:
        """Télécharge un fichier brut ou une archive."""
        # Eviter les icônes parasites
        if "avatar" in filename.lower() or "icon" in filename.lower():
            return False
            
        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            return False 
            
        try:
            print(f"  Téléchargement : {filename}")
            r = requests.get(url, headers=self.headers)
            if r.status_code == 200:
                if filename.lower().endswith('.zip'):
                    # Traitement de l'archive ZIP
                    prefix = filename.split('.')[0]
                    self._extract_zip(r.content, prefix)
                else:
                    # Enregistrement classique
                    with open(filepath, 'wb') as f:
                        f.write(r.content)
                return True
        except Exception as e:
            print(f"  Échec téléchargement {filename}: {e}")
        return False

class HuggingFaceScraper:
    def __init__(self, raw_dir="raw_data"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def scrape_dataset(self, hf_dataset_name: str, split="train", max_samples=1000):
        """
        Télécharge les images depuis un Dataset HuggingFace existant et les exporte en PNG bruts.
        """
        try:
            from datasets import load_dataset
            print(f"--- Lancement HF Scraper sur le dataset {hf_dataset_name} ---")
            dataset = load_dataset(hf_dataset_name, split=split)
            
            count = 0
            for i, item in enumerate(dataset):
                if count >= max_samples:
                    break
                    
                # Certains datasets appellent la colonne 'image', d'autres 'img'
                img = item.get('image') or item.get('img')
                if img is None:
                    continue
                
                # Le format de nommage inclut l'index pour éviter les conflits
                filename = f"hf_{hf_dataset_name.replace('/', '_')}_{i:06d}.png"
                filepath = os.path.join(self.raw_dir, filename)
                
                img.save(filepath, format="PNG")
                print(f"Extrait depuis HF : {filename}")
                count += 1
                
        except Exception as e:
            print(f"Erreur HuggingFace Scraper : {e}")

if __name__ == "__main__":
    # Test local
    scraper = AdvancedOpenGameArtScraper(raw_dir="src/data_factory/raw_safe_hf")
    scraper.scrape_latest_2d_art(0, 1)
