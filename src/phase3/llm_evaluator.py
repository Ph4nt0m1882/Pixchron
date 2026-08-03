import sqlite3
import json
import time
import os
import re
import torch
from transformers import pipeline

DB_PATH = 'master_dictionary.db'
MODEL_NAME = 'Qwen/Qwen3.6-35B-A3B' # Latest version, needs ~70GB VRAM in bf16/fp16

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

def get_pipeline():
    print(f"Loading {MODEL_NAME} on GPU...")
    return pipeline(
        "text-generation",
        model=MODEL_NAME,
        model_kwargs={"torch_dtype": torch.float16},
        device_map="auto"
    )

def evaluate_batch(words_batch, pipe):
    prompts = []
    for row_id, word in words_batch:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Word: {word}\nOutput JSON:"}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    print(f"Running inference on batch of {len(prompts)} words...")
    outputs = pipe(prompts, max_new_tokens=512, do_sample=False, return_full_text=False)
    
    results = []
    for out in outputs:
        text = out[0]['generated_text'].strip()
        
        # Robustly extract JSON block even if there is reasoning text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                results.append(result)
            except Exception as e:
                print(f"Failed to parse JSON: {json_str} | Error: {e}")
                results.append(None)
        else:
            print(f"No JSON object found in output: {text}")
            results.append(None)
            
    return results

def run_evaluation(limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, english_word FROM dictionary WHERE evaluated = 0 LIMIT ?', (limit,))
    words = cursor.fetchall()
    
    if not words:
        print("All words evaluated!")
        return

    pipe = get_pipeline()
    
    # Process in smaller batches
    batch_size = 10
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        results = evaluate_batch(batch, pipe)
        
        for (row_id, word), result in zip(batch, results):
            if result:
                french = result.get('french_translation', '')
                score = result.get('pixel_art_score', 1)
                needs = result.get('structural_needs', '')
                
                cursor.execute('''
                    UPDATE dictionary 
                    SET french_translation = ?, pixel_art_score = ?, structural_needs = ?, evaluated = 1 
                    WHERE id = ?
                ''', (french, score, needs, row_id))
                print(f"  [{word}] -> {french} | Score: {score}")
            else:
                print(f"  [{word}] -> Failed.")
        conn.commit()
        
    conn.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation(limit=50) # Just test with 50 for now
