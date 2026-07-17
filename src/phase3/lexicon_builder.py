import sqlite3
import os
import urllib.request
import json

DB_PATH = 'master_dictionary.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dictionary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            english_word TEXT UNIQUE,
            category TEXT,
            french_translation TEXT,
            pixel_art_score INTEGER,
            structural_needs TEXT,
            evaluated BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    return conn

def populate_modifiers(conn):
    cursor = conn.cursor()
    modifiers = [
        "abandoned", "futuristic", "rusty", "cyberpunk", "glowing", 
        "ruined", "shiny", "bloody", "damaged", "magical", "medieval",
        "steampunk", "neon", "ancient", "cursed", "holy", "dark"
    ]
    archetypes = [
        "knight", "mage", "ninja", "warrior", "archer", "cleric", 
        "paladin", "thief", "necromancer", "summoner", "monk", "pirate"
    ]
    
    for word in modifiers:
        cursor.execute('INSERT OR IGNORE INTO dictionary (english_word, category) VALUES (?, ?)', (word, 'Modifier'))
    for word in archetypes:
        cursor.execute('INSERT OR IGNORE INTO dictionary (english_word, category) VALUES (?, ?)', (word, 'Archetype'))
    conn.commit()
    print(f"Inserted {len(modifiers)} modifiers and {len(archetypes)} archetypes.")

def populate_common_nouns(conn):
    print("Downloading common nouns list...")
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req).read().decode('utf-8')
        words = response.split('\n')
        
        # We take a sample of the most common words that are likely nouns/objects
        # In a real scenario we would filter with NLP (spaCy/NLTK) to only keep nouns.
        # For Phase 3 setup, we insert them and the LLM will score them (verbs/adverbs will get score=1).
        cursor = conn.cursor()
        inserted = 0
        for word in words[:3000]: # Top 3000 words
            word = word.strip().lower()
            if len(word) > 2:
                cursor.execute('INSERT OR IGNORE INTO dictionary (english_word, category) VALUES (?, ?)', (word, 'Common'))
                inserted += 1
        conn.commit()
        print(f"Inserted {inserted} common English words.")
    except Exception as e:
        print(f"Error fetching common nouns: {e}")

def populate_pop_culture(conn):
    print("Downloading pop culture lists (Superheroes & Pokémon)...")
    cursor = conn.cursor()
    inserted = 0
    
    # 1. Superheroes
    try:
        url_heroes = "https://raw.githubusercontent.com/sindresorhus/superheroes/main/superheroes.json"
        req = urllib.request.Request(url_heroes, headers={'User-Agent': 'Mozilla/5.0'})
        heroes = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        for h in heroes:
            cursor.execute('INSERT OR IGNORE INTO dictionary (english_word, category) VALUES (?, ?)', (h.strip(), 'PopCulture'))
            inserted += 1
    except Exception as e:
        print(f"Error fetching superheroes: {e}")
        
    # 2. Pokémon
    try:
        url_poke = "https://raw.githubusercontent.com/sindresorhus/pokemon/main/data/en.json"
        req = urllib.request.Request(url_poke, headers={'User-Agent': 'Mozilla/5.0'})
        pokemon = json.loads(urllib.request.urlopen(req).read().decode('utf-8'))
        for p in pokemon:
            cursor.execute('INSERT OR IGNORE INTO dictionary (english_word, category) VALUES (?, ?)', (p.strip(), 'PopCulture'))
            inserted += 1
    except Exception as e:
        print(f"Error fetching pokemon: {e}")

    conn.commit()
    print(f"Inserted {inserted} pop culture entities dynamically.")

if __name__ == "__main__":
    os.makedirs('src/phase3', exist_ok=True)
    conn = init_db()
    populate_modifiers(conn)
    populate_pop_culture(conn)
    populate_common_nouns(conn)
    
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM dictionary')
    total = cursor.fetchone()[0]
    print(f"Total words in dictionary: {total}")
    conn.close()
