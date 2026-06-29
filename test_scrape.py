import requests
import re

url = 'https://opengameart.org/art-search-advanced?keys=&title=&field_art_tags_tid_op=or&field_art_tags_tid=pixel%20art&name=&Sort=created&page=0'
r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
matches = re.findall(r'(?:href|src)=[\'"]([^\'"]+\.(?:png|gif|jpg|jpeg|webp))[\'"]', r.text)
print(f"Trouve: {len(matches)}")
print(matches[:10])
