import sqlite3
import json
import time
import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = 'master_dictionary.db'
MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct'
VLLM_API_URL = 'http://localhost:8000/v1/chat/completions'

SYSTEM_PROMPT = """You are a Pixel Art Art Director.
You will be given an English word. Your task is to evaluate its importance and relevance for training a Pixel Art Video Game Machine Learning Model.
Respond ONLY with a valid JSON object in the exact following format. DO NOT include any reasoning, thoughts, or extra text outside the JSON:
{
  "french_translation": "The direct French translation of the word",
  "pixel_art_score": 1 to 10,
  "structural_needs": "Comma-separated list of formats needed (e.g., 'Tileset, Sprite, Background', or 'Sprite' or 'None')"
}

Guidelines for score:
- 9-10: Extremely important for games/pixel art (e.g., house, tree, knight, sword, magical, grass)
- 5-8: Useful objects, animals, or characters (e.g., chair, dog, car, cyberpunk)
- 1-4: Abstract concepts, verbs, adverbs, or things rarely drawn in games (e.g., the, jump, quickly, philosophy)
"""

def evaluate_word(row_id, word):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Word: {word}\nOutput JSON:"}
    ]
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
    }
    
    try:
        response = requests.post(VLLM_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        text = data['choices'][0]['message']['content'].strip()
        
        text_no_think = re.sub(r'<think>[\s\S]*?</think>', '', text)
        matches = re.findall(r'\{[\s\S]*?\}', text_no_think)
        
        for json_str in reversed(matches):
            try:
                result = json.loads(json_str)
                if 'pixel_art_score' in result:
                    return row_id, word, result
            except Exception:
                continue
    except Exception as e:
        print(f"Error evaluating {word}: {e}")
        
    return row_id, word, None

def run_evaluation(limit=10000):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, english_word FROM dictionary WHERE evaluated = 0 LIMIT ?', (limit,))
    words = cursor.fetchall()
    
    if not words:
        print("All words evaluated!")
        return

    print(f"🚀 Starting parallel evaluation of {len(words)} words via vLLM API...")
    start_time = time.time()
    
    # We send 100 requests simultaneously to the vLLM server.
    # vLLM's PagedAttention engine will automatically batch them dynamically on the GPU.
    max_workers = 100
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_word, row_id, word): (row_id, word) for row_id, word in words}
        
        for i, future in enumerate(as_completed(futures), 1):
            row_id, word, result = future.result()
            
            if result:
                french = result.get('french_translation', '')
                score = result.get('pixel_art_score', 1)
                needs = result.get('structural_needs', '')
                
                cursor.execute('''
                    UPDATE dictionary 
                    SET french_translation = ?, pixel_art_score = ?, structural_needs = ?, evaluated = 1 
                    WHERE id = ?
                ''', (french, score, needs, row_id))
                
                success_count += 1
                # Print progress every 50 words
                if success_count % 50 == 0:
                    conn.commit()
                    elapsed = time.time() - start_time
                    rate = success_count / elapsed
                    print(f"  -> Progress: {success_count}/{len(words)} | Speed: {rate:.1f} words/sec")
            else:
                print(f"  -> Failed to evaluate: {word}")

    # Final commit
    conn.commit()
    conn.close()
    
    total_time = time.time() - start_time
    print(f"\n✅ Evaluation complete! Processed {len(words)} words in {total_time:.1f}s (~{total_time/len(words):.2f}s per word).")

if __name__ == "__main__":
    run_evaluation(limit=10000)
