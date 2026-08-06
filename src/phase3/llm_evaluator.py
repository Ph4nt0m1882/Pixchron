import sqlite3
import json
import time
import os
import re
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DB_PATH = 'master_dictionary.db'
MODEL_NAME = 'Qwen/Qwen2.5-32B-Instruct' # Dense model, highly intelligent

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

def get_vllm_engine():
    num_gpus = torch.cuda.device_count()
    print(f"Loading {MODEL_NAME} across {num_gpus} GPUs using vLLM...")
    
    # vLLM automatically handles memory and tensor parallelism across all available GPUs
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=num_gpus,
        dtype="float16",
        max_model_len=4096, # Keep it reasonable to save VRAM for large batch sizes
        gpu_memory_utilization=0.90 # Use 90% of VRAM per GPU
    )
    return llm

def evaluate_chunk(words_chunk, llm, tokenizer):
    prompts = []
    for row_id, word in words_chunk:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Word: {word}\nOutput JSON:"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["<|endoftext|>", "<|im_end|>"]
    )
    
    print(f"\n🚀 Running vLLM on chunk of {len(prompts)} words (Super-fast Parallel Generation)...")
    start_time = time.time()
    
    # vLLM generate() processes the entire list using PagedAttention (infinitely faster than transformers pipeline)
    outputs = llm.generate(prompts, sampling_params)
    
    elapsed = time.time() - start_time
    print(f"⚡ Chunk processed in {elapsed:.1f} seconds! (~{elapsed/len(prompts):.2f}s per word)")
    
    results = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        
        # Remove thinking blocks if present
        text_no_think = re.sub(r'<think>[\s\S]*?</think>', '', text)
        
        matches = re.findall(r'\{[\s\S]*?\}', text_no_think)
        parsed = False
        
        for json_str in reversed(matches):
            try:
                result = json.loads(json_str)
                if 'pixel_art_score' in result:
                    results.append(result)
                    parsed = True
                    break
            except Exception:
                continue
                
        if not parsed:
            results.append(None)
            
    return results

def run_evaluation(limit=10000):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, english_word FROM dictionary WHERE evaluated = 0 LIMIT ?', (limit,))
    words = cursor.fetchall()
    
    if not words:
        print("All words evaluated!")
        return

    llm = get_vllm_engine()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Process in massive chunks of 500 (vLLM excels at huge parallel batches)
    chunk_size = 500
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i+chunk_size]
        results = evaluate_chunk(chunk, llm, tokenizer)
        
        for (row_id, word), result in zip(chunk, results):
            if result:
                french = result.get('french_translation', '')
                score = result.get('pixel_art_score', 1)
                needs = result.get('structural_needs', '')
                
                cursor.execute('''
                    UPDATE dictionary 
                    SET french_translation = ?, pixel_art_score = ?, structural_needs = ?, evaluated = 1 
                    WHERE id = ?
                ''', (french, score, needs, row_id))
                print(f"    [{word}] -> {french} (Score: {score})")
            else:
                print(f"    [{word}] -> Failed.")
                
        conn.commit()
        print(f"💾 Saved chunk to database.\n")
        
    conn.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    run_evaluation(limit=10000)
