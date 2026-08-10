import sqlite3
import json
import os

DB_PATH = 'master_dictionary.db'
OUTPUT_FILE = 'target_dataset.json'

def generate_master_plan():
    if not os.path.exists(DB_PATH):
        print("Database not found. Please run lexicon_builder.py and llm_evaluator.py first.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # We select all evaluated words, regardless of score. 
    # Low score words will naturally have a lower quota (score * 3).
    cursor.execute('''
        SELECT english_word, french_translation, category, pixel_art_score, structural_needs
        FROM dictionary
        WHERE evaluated = 1
        ORDER BY pixel_art_score DESC
    ''')
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No evaluated words found. Run llm_evaluator.py to score some words.")
        return
        
    master_plan = {
        "version": "1.0",
        "description": "Master Dictionary Target Dataset for Pixchron",
        "total_targets": len(rows),
        "targets": []
    }
    
    for row in rows:
        eng, fr, cat, score, needs = row
        
        # Calculate target quota based on score
        # A score of 10 means we want many images (e.g., 30 targets, so scraper fetches 90)
        # A score of 5 means fewer images (e.g., 5 targets, so scraper fetches 15)
        base_quota = score * 3
        
        needs_list = [n.strip() for n in needs.split(',')] if needs else []
        needs_list = [n for n in needs_list if n.lower() != 'none']
        
        target = {
            "english_word": eng,
            "french_translation": fr,
            "category": cat,
            "priority_score": score,
            "structural_needs": needs_list,
            "target_images_quota": base_quota
        }
        master_plan["targets"].append(target)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(master_plan, f, indent=4, ensure_ascii=False)
        
    print(f"Master plan successfully generated: {OUTPUT_FILE} with {len(rows)} targets.")

if __name__ == "__main__":
    generate_master_plan()
