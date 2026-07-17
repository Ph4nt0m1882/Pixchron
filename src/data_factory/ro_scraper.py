import urllib.request
import re
import os

def download_ro_sheets():
    os.makedirs('scratch/ro', exist_ok=True)
    
    # dict: name -> url
    assets = {
        'Novice_M': 'https://www.spriters-resource.com/pc_computer/ragnarokonline/asset/141313/',
        'Novice_F': 'https://www.spriters-resource.com/pc_computer/ragnarokonline/asset/141222/',
        'Swordsman_M': 'https://www.spriters-resource.com/pc_computer/ragnarokonline/asset/141350/',
        'Swordsman_F': 'https://www.spriters-resource.com/pc_computer/ragnarokonline/asset/141255/'
    }
    
    for name, url in assets.items():
        print(f"Fetching page for {name}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            match = re.search(r'data-file="([^"]+\.zip[^"]*)"', html)
            if match:
                zip_url = 'https://www.spriters-resource.com' + match.group(1).replace('&amp;', '&')
                print(f"Downloading {name} from {zip_url}...")
                
                zip_req = urllib.request.Request(zip_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(zip_req) as response:
                    zip_data = response.read()
                    with open(f'scratch/ro/{name}.zip', 'wb') as out_file:
                        out_file.write(zip_data)
                print(f"Saved scratch/ro/{name}.zip")
            else:
                # sometimes it's PNG
                match = re.search(r'data-file="([^"]+\.png[^"]*)"', html)
                if match:
                    img_url = 'https://www.spriters-resource.com' + match.group(1).replace('&amp;', '&')
                    print(f"Downloading {name} from {img_url}...")
                    img_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(img_req) as response:
                        img_data = response.read()
                        with open(f'scratch/ro/{name}.png', 'wb') as out_file:
                            out_file.write(img_data)
                    print(f"Saved scratch/ro/{name}.png")
                else:
                    print(f"Could not find download link for {name}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    download_ro_sheets()
