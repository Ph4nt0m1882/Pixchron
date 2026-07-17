import urllib.request
import re
import os

def download_chrono_sheets():
    os.makedirs('scratch/ct', exist_ok=True)
    ids = {'Crono': '2514', 'Ayla': '2511', 'Frog': '3570', 'Lucca': '3614', 'Magus': '3615', 'Marle': '3617', 'Robo': '3651'}
    
    for name, asset_id in ids.items():
        url = f'https://www.spriters-resource.com/snes/chronotrigger/asset/{asset_id}/'
        print(f"Fetching page for {name}...")
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            # Look for data-file="/media/assets/..."
            match = re.search(r'data-file="([^"]+\.png[^"]*)"', html)
            if match:
                img_url = 'https://www.spriters-resource.com' + match.group(1).replace('&amp;', '&')
                print(f"Downloading {name} from {img_url}...")
                
                img_req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(img_req) as response:
                    img_data = response.read()
                    with open(f'scratch/ct/{name}.png', 'wb') as out_file:
                        out_file.write(img_data)
                print(f"Saved scratch/ct/{name}.png")
            else:
                print(f"Could not find download link for {name}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")

if __name__ == "__main__":
    download_chrono_sheets()
