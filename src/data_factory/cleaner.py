import os
import sys
from typing import Tuple, Optional
from PIL import Image
import numpy as np

# Gestion des imports relatifs / absolus
try:
    from schema import PixelArtMetadata
except ImportError:
    from .schema import PixelArtMetadata

try:
    from ..phase4.pixel_reconstructor import PixelReconstructor
except ImportError:
    try:
        from phase4.pixel_reconstructor import PixelReconstructor
    except ImportError:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from phase4.pixel_reconstructor import PixelReconstructor

class PixelArtCleaner:
    """
    Façade de nettoyage de Pixel Art intégrant le moteur PixelReconstructor et le schéma de métadonnées Pixchron.
    """
    def __init__(
        self,
        max_colors_allowed: int = 256,
        color_tolerance: float = 15.0,
        min_native_size: int = 6,
        remove_background: bool = False,
        bg_tolerance: float = 25.0
    ):
        self.max_colors = max_colors_allowed
        self.reconstructor = PixelReconstructor(
            color_tolerance=color_tolerance,
            max_colors=max_colors_allowed,
            min_native_size=min_native_size,
            remove_background=remove_background,
            bg_tolerance=bg_tolerance
        )

    def process_image(
        self, 
        filepath: str, 
        source_url: str = "", 
        license_type: str = "private_train"
    ) -> Tuple[Optional[Image.Image], Optional[PixelArtMetadata], dict]:
        """
        Traite une image ou un GIF et renvoie (image_nettoyée, métadonnées, statistiques).
        Si rejetée, renvoie (None, None, stats_rejet).
        """
        try:
            img_raw = Image.open(filepath)
        except Exception as e:
            return None, None, {"rejected": True, "reason": f"Erreur de lecture: {e}"}

        cleaned_img, scale, stats = self.reconstructor.reconstruct(img_raw)
        
        if cleaned_img is None or stats.get("rejected", False):
            return None, None, stats

        is_animated = stats.get("is_animated", False)
        n_frames = stats.get("frames", 1)
        w, h = cleaned_img.size
        has_bg_removed = stats.get("has_background_removed", False)
        palette_size = stats.get("palette_size", 0)

        metadata = PixelArtMetadata(
            license_type=license_type,
            dataset_bucket="safe_hf" if license_type in ["public_domain", "cc-by"] else "private_train",
            description="", 
            width=w,
            height=h,
            has_background=not has_bg_removed, 
            palette_size=palette_size,
            is_animation=is_animated,
            frames=n_frames,
            source_url=source_url or filepath
        )

        return cleaned_img, metadata, stats

if __name__ == "__main__":
    cleaner = PixelArtCleaner()
    print("Module PixelArtCleaner initialisé avec succès et connecté à PixelReconstructor.")
