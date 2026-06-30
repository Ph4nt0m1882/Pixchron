import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from schema import PixelArtMetadata

class PixelArtAnnotator:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initialise le modèle de vision. 
        Llava-7B est un excellent compromis : très intelligent pour l'annotation,
        mais assez léger (~15 Go VRAM) pour que vous puissiez faire tourner 
        plusieurs instances en parallèle sur votre machine de 120 Go.
        """
        self.device = device
        print(f"Chargement du VLM {model_id} sur {device}...")
        
        # Pour contourner le bug 'image_token' d'AutoProcessor avec transformers 4.40
        from transformers import CLIPImageProcessor, AutoTokenizer, LlavaProcessor
        
        image_processor = CLIPImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True
        ).to(self.device)
        
        print("Modèle d'annotation prêt !")

    def annotate_image(self, image_path: str, metadata: PixelArtMetadata) -> PixelArtMetadata:
        """
        Analyse l'image avec le VLM et met à jour les métadonnées (description, frames).
        """
        try:
            image = Image.open(image_path).convert("RGB") # Llava s'attend à du RGB
        except Exception as e:
            print(f"Erreur de lecture image: {e}")
            return metadata

        # Prompt d'instruction stricte pour forcer le modèle à répondre en JSON
        prompt = (
            "USER: <image>\n"
            "You are an expert video game pixel artist. Analyze this image. "
            "1. Describe the character, object, or scene in high detail (colors, style, action). "
            "2. Determine if this image is a sprite sheet (a grid of multiple animation frames). "
            "3. If it is a sprite sheet, count the exact number of frames. "
            "Respond strictly with a valid JSON containing three keys: 'description' (string), 'is_animation' (boolean), and 'frames' (integer).\n"
            "ASSISTANT:"
        )

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.bfloat16)

        # Génération
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs, 
                max_new_tokens=200, 
                temperature=0.2, # Température basse pour avoir des réponses JSON déterministes
                do_sample=True
            )
            
        response_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Le modèle renvoie tout le prompt, on coupe pour ne garder que la réponse de l'assistant
        response_text = response_text.split("ASSISTANT:")[-1].strip()
        
        # Parsing du JSON renvoyé par l'IA
        try:
            # Sécurité basique pour extraire le JSON s'il a mis des backticks markdown
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response_text
                
            json_str = json_str.replace("\\_", "_") # Fix pour is\_animation (LLaVA markdown bug)
                
            ai_data = json.loads(json_str)
            
            # Mise à jour des métadonnées avec l'intelligence du VLM
            metadata.description = ai_data.get("description", "No description")
            metadata.is_animation = ai_data.get("is_animation", False)
            metadata.frames = ai_data.get("frames", 1)
            
            print(f"[{image_path}] Annoté avec succès : {metadata.is_animation} | {metadata.frames} frames")
        except json.JSONDecodeError:
            print(f"[{image_path}] Échec du parsing JSON du VLM. Réponse brute : {response_text}")
            metadata.description = response_text # On sauvegarde quand même le texte brut au cas où
            
        return metadata

if __name__ == "__main__":
    print("Module Annotator prêt. (Test désactivé pour éviter le téléchargement de 15Go de poids sans GPU)")
