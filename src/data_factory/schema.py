from dataclasses import dataclass, asdict
from typing import Optional
import json

@dataclass
class PixelArtMetadata:
    """
    Schéma de métadonnées standardisé pour chaque image/animation du dataset.
    Ce JSON sera utilisé par le modèle de diffusion pour le conditionnement.
    """
    # Catégorisation de la licence
    license_type: str  # "public_domain", "cc-by", "copyrighted_scrap", etc.
    dataset_bucket: str # "safe_hf" ou "private_train"
    
    # Informations Visuelles
    description: str = ""
    width: int = 0
    height: int = 0
    has_background: bool = False
    palette_size: int = 0
    
    # Informations Temporelles (Animation)
    # L'annotateur IA (Florence-2) se chargera de détecter si c'est une spritesheet PNG
    is_animation: bool = False
    frames: int = 1
    fps: Optional[float] = None  # Rempli uniquement si la source est un GIF/Vidéo, sinon None (null en JSON)
    
    # Source
    source_url: str = ""
    
    def to_json(self):
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

if __name__ == "__main__":
    # Test du schéma
    sample = PixelArtMetadata(
        license_type="public_domain",
        dataset_bucket="safe_hf",
        description="Un chevalier bleu en armure qui court, style 16-bit",
        width=64,
        height=64,
        has_background=False,
        palette_size=16,
        is_animation=True,
        frames=6,
        source_url="https://opengameart.org/example"
    )
    print("Exemple de JSON généré :")
    print(sample.to_json())
