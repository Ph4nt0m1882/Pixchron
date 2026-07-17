import urllib.request
import re
import os

def download_ff6_sheets():
    os.makedirs('scratch/ff6', exist_ok=True)
    
    # dict: name -> url
    assets = {
        'Celes': 'https://www.spriters-resource.com/snes/ff6/asset/5830/',
        'Cyan': 'https://www.spriters-resource.com/snes/ff6/asset/5836/',
        'Edgar': 'https://www.spriters-resource.com/snes/ff6/asset/5844/',
        'Gau': 'https://www.spriters-resource.com/snes/ff6/asset/6646/',
        'Locke': 'https://www.spriters-resource.com/snes/ff6/asset/6666/',
        'Sabin': 'https://www.spriters-resource.com/snes/ff6/asset/6699/',
        'Shadow': 'https://www.spriters-resource.com/snes/ff6/asset/6702/',
        'Terra': 'https://www.spriters-resource.com/snes/ff6/asset/6707/'
    }
    
    for name, url in assets.items():
        print(f"Fetching page for {name}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            match = re.search(r'data-file="([^"]+\.(?:png|gif)[^"]*)"', html)
            if match:
                ext = '.gif' if '.gif' in match.group(1) else '.png'
                img_url = 'https://www.spriters-resource.com' + match.group(1).replace('&amp;', '&')
                print(f"Downloading {name} from {img_url}...")
                
                img_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(img_req) as response:
                    img_data = response.read()
                    with open(f'scratch/ff6/{name}{ext}', 'wb') as out_file:
                        out_file.write(img_data)
                print(f"Saved scratch/ff6/{name}{ext}")
            else:
                print(f"Could not find download link for {name}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    download_ff6_sheets()
