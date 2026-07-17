import urllib.request
import re
import os

def download_gs_sheets():
    os.makedirs('scratch/gs', exist_ok=True)
    
    # dict: name -> url
    assets = {
        'Isaac': 'https://www.spriters-resource.com/game_boy_advance/gs/asset/13253/',
        'Ivan': 'https://www.spriters-resource.com/game_boy_advance/gs/asset/184063/',
        'Mia': 'https://www.spriters-resource.com/game_boy_advance/gs/asset/13243/',
        'Jenna': 'https://www.spriters-resource.com/game_boy_advance/gs/asset/11505/',
        'Felix': 'https://www.spriters-resource.com/game_boy_advance/gs2/asset/46769/',
        'Garet': 'https://www.spriters-resource.com/game_boy_advance/gs2/asset/46768/',
        'Piers': 'https://www.spriters-resource.com/game_boy_advance/gs2/asset/46772/',
        'Sheba': 'https://www.spriters-resource.com/game_boy_advance/gs2/asset/46771/'
    }
    
    for name, url in assets.items():
        print(f"Fetching page for {name}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            # Look for data-file="/media/assets/..." or data-file="*.gif"
            match = re.search(r'data-file="([^"]+\.(?:png|gif)[^"]*)"', html)
            if match:
                ext = '.gif' if '.gif' in match.group(1) else '.png'
                img_url = 'https://www.spriters-resource.com' + match.group(1).replace('&amp;', '&')
                print(f"Downloading {name} from {img_url}...")
                
                img_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(img_req) as response:
                    img_data = response.read()
                    with open(f'scratch/gs/{name}{ext}', 'wb') as out_file:
                        out_file.write(img_data)
                print(f"Saved scratch/gs/{name}{ext}")
            else:
                print(f"Could not find download link for {name}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    download_gs_sheets()
