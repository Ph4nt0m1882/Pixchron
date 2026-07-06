import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import zipfile
from io import BytesIO
import shutil
import time

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
                
                # Si le dataset HF contient des URLs au lieu d'objets images intégrés (comme bghira/free-to-use-pixelart)
                if img is None:
                    img_url = item.get('full_image_url') or item.get('image_url') or item.get('url')
                    if img_url and isinstance(img_url, str) and img_url.startswith('http'):
                        try:
                            r = requests.get(img_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                            if r.status_code == 200:
                                from PIL import Image
                                from io import BytesIO
                                img = Image.open(BytesIO(r.content))
                        except Exception as e:
                            print(f"Erreur téléchargement URL HF {img_url} : {e}")

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

class TheSpritersResourceScraper:
    def __init__(self, raw_dir="raw_data"):
        self.base_url = "https://www.spriters-resource.com"
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
    def scrape_console(self, console_name: str, start_index=0, end_index=10):
        print(f"--- Lancement TSR Scraper ({console_name}, jeux {start_index} à {end_index}) ---")
        url = f"{self.base_url}/{console_name}/"
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                print(f"Erreur d'accès à la console {console_name} (HTTP {response.status_code})")
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            game_links = []
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                if href.startswith(f"/{console_name}/") and len(href.split('/')) == 4 and href.endswith('/'):
                    if len(href.split('/')[2]) > 1:
                        game_links.append(urljoin(self.base_url, href))
            
            seen = set()
            game_links = [x for x in game_links if not (x in seen or seen.add(x))]
            
            print(f"Trouvé {len(game_links)} jeux pour la console {console_name}.")
            
            target_games = game_links[start_index:end_index]
            for game_url in target_games:
                self._scrape_game(game_url)
                time.sleep(1) # Politesse envers le serveur
                
        except Exception as e:
            print(f"Erreur TSR Scraper : {e}")

    def _scrape_game(self, game_url: str):
        try:
            print(f"Exploration du jeu : {game_url}")
            response = requests.get(game_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            asset_links = set()
            for a_tag in soup.find_all('a', href=True):
                if '/asset/' in a_tag['href']:
                    asset_links.add(urljoin(self.base_url, a_tag['href']))
            
            for asset_url in asset_links:
                self._scrape_asset(asset_url)
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Erreur lors de l'exploration du jeu {game_url} : {e}")

    def _scrape_asset(self, asset_url: str):
        try:
            response = requests.get(asset_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            img_url = None
            for a_tag in soup.find_all('a', href=True):
                if '/media/assets/' in a_tag['href']:
                    img_url = urljoin(self.base_url, a_tag['href'])
                    break
            
            if img_url:
                filename = img_url.split('/')[-1].split('?')[0]
                filename = f"tsr_{filename}"
                filepath = os.path.join(self.raw_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"  Téléchargement sprite : {filename}")
                    r = requests.get(img_url, headers=self.headers)
                    if r.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(r.content)
                    else:
                        print(f"  Échec (HTTP {r.status_code})")
        except Exception as e:
            pass 

class LospecScraper:
    def __init__(self, raw_dir="raw_data"):
        self.base_url = "https://lospec.com"
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
    def scrape_gallery(self, start_page=1, end_page=5):
        print(f"--- Lancement Lospec Scraper (Pages {start_page} à {end_page}) ---")
        for page in range(start_page, end_page):
            url = f"{self.base_url}/gallery/?page={page}"
            try:
                print(f"Lospec: Exploration page {page}...")
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith('/gallery/') and len(href.split('/')) == 4:
                        img_page_url = urljoin(self.base_url, href)
                        self._scrape_image_page(img_page_url)
                        # DELAI TRES IMPORTANT (Serveur modeste)
                        time.sleep(3)
                
            except Exception as e:
                print(f"Erreur Lospec Scraper page {page} : {e}")

    def _scrape_image_page(self, url):
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            img_tag = soup.find('img', id='image') or soup.find('img', class_='image')
            if not img_tag:
                for img in soup.find_all('img'):
                    if 'src' in img.attrs and '/images/' in img['src']:
                        img_tag = img
                        break
                        
            if img_tag and 'src' in img_tag.attrs:
                img_url = urljoin(self.base_url, img_tag['src'])
                filename = f"lospec_{img_url.split('/')[-1].split('?')[0]}"
                filepath = os.path.join(self.raw_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"  Téléchargement Lospec : {filename}")
                    r = requests.get(img_url, headers=self.headers)
                    if r.status_code == 200:
                        with open(filepath, 'wb') as f:
                            f.write(r.content)
        except Exception as e:
            pass

class GithubRepoScraper:
    def __init__(self, raw_dir="raw_data"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def scrape_repo(self, repo_url: str):
        import subprocess
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        print(f"--- Lancement GitHub Scraper sur {repo_name} ---")
        
        temp_dir = f"temp_clone_{repo_name}"
        try:
            if not os.path.exists(temp_dir):
                subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            count = 0
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.gif')):
                        source_path = os.path.join(root, file)
                        target_filename = f"github_{repo_name}_{file}"
                        target_path = os.path.join(self.raw_dir, target_filename)
                        if not os.path.exists(target_path):
                            shutil.copy2(source_path, target_path)
                            count += 1
            print(f"  -> {count} assets pixel art extraits de {repo_name}")
        except Exception as e:
            print(f"Erreur GitHub Scraper : {e}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

class KaggleScraper:
    def __init__(self, raw_dir="raw_data"):
        self.raw_dir = raw_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        
    def scrape_dataset(self, dataset_name: str):
        print(f"--- Lancement Kaggle Scraper sur {dataset_name} ---")
        try:
            import kaggle
            kaggle.api.authenticate()
            print(f"Téléchargement de {dataset_name} depuis Kaggle...")
            kaggle.api.dataset_download_files(dataset_name, path=self.raw_dir, unzip=True)
            print("  -> Terminé. Les fichiers ont été extraits dans raw_data.")
        except Exception as e:
            print(f"Erreur Kaggle Scraper : {e}")
            print("Astuce: Avez-vous configuré ~/.kaggle/kaggle.json ?")

if __name__ == "__main__":
    # Test local
    scraper = AdvancedOpenGameArtScraper(raw_dir="src/data_factory/raw_safe_hf")
    scraper.scrape_latest_2d_art(0, 1)
