import unittest
import os
import sys
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np

# Ajouter src au sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from phase4.pixel_reconstructor import PixelReconstructor

class TestPixelReconstructor(unittest.TestCase):
    def setUp(self):
        self.reconstructor = PixelReconstructor(
            color_tolerance=15.0,
            max_colors=256,
            min_native_size=4,
            remove_background=True
        )

    def test_pure_native_pixel_art(self):
        """Vérifie qu'une image 16x16 native reste en 16x16 (échelle 1x) sans altération."""
        arr = np.zeros((16, 16, 4), dtype=np.uint8)
        arr[4:12, 4:12] = [255, 0, 0, 255] # Carré rouge
        arr[0:4, 0:4] = [0, 255, 0, 255]   # Coin vert
        img = Image.fromarray(arr)

        res, scale, stats = self.reconstructor.reconstruct(img)
        self.assertIsNotNone(res)
        self.assertEqual(scale, 1)
        self.assertEqual(stats["clean_size"], (16, 16))

    def test_upscaled_macro_pixel_with_noise(self):
        """Vérifie qu'une image 16x16 upscalée 4x (64x64) avec du bruit JPEG est ramenée à 16x16."""
        native = np.zeros((16, 16, 3), dtype=np.uint8)
        # Dessin d'un petit bonhomme / motif pixel art
        native[2:6, 6:10] = [220, 180, 140] # Tête
        native[6:12, 5:11] = [30, 100, 200]  # Corps
        native[12:15, 6:8] = [50, 50, 50]    # Jambe G
        native[12:15, 9:11] = [50, 50, 50]   # Jambe D

        # Upscale 4x
        native_img = Image.fromarray(native)
        upscaled_img = native_img.resize((64, 64), Image.Resampling.NEAREST)
        upscaled_arr = np.array(upscaled_img).astype(float)

        # Ajout de bruit JPEG simulé (+/- 8 sur les pixels)
        np.random.seed(42)
        noise = np.random.randint(-8, 8, upscaled_arr.shape)
        noisy_arr = np.clip(upscaled_arr + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_arr)

        res, scale, stats = self.reconstructor.reconstruct(noisy_img)
        self.assertIsNotNone(res)
        self.assertEqual(scale, 4)
        self.assertEqual(stats["clean_size"], (16, 16))
        # Palette nettoyée (peu de couleurs)
        self.assertLess(stats["palette_size"], 10)

    def test_background_removal(self):
        """Vérifie la conversion du fond uni/bruité en canal Alpha transparent."""
        rec_bg = PixelReconstructor(remove_background=True, bg_tolerance=20.0)
        
        arr = np.full((32, 32, 3), [74, 103, 107], dtype=np.uint8) # Fond teal (comme Zombie_29)
        arr[8:24, 8:24] = [200, 50, 50] # Sujet rouge

        img = Image.fromarray(arr)
        res, scale, stats = rec_bg.reconstruct(img)

        self.assertIsNotNone(res)
        res_arr = np.array(res)
        # Vérifier que les coins sont maintenant transparents (Alpha = 0)
        self.assertEqual(res_arr[0, 0, 3], 0)
        self.assertEqual(res_arr[0, 31, 3], 0)
        # Vérifier que le centre rouge est opaque (Alpha = 255)
        self.assertEqual(res_arr[16, 16, 3], 255)

    def test_rejection_of_photo(self):
        """Vérifie qu'un dégradé photo / bruit continu est rejeté."""
        # Image avec gradient lisse (trop de couleurs)
        x = np.linspace(0, 255, 64)
        y = np.linspace(0, 255, 64)
        xx, yy = np.meshgrid(x, y)
        grad = np.stack([xx, yy, 255 - xx], axis=-1).astype(np.uint8)
        photo_img = Image.fromarray(grad)

        rec_strict = PixelReconstructor(max_colors=64)
        res, scale, stats = rec_strict.reconstruct(photo_img)
        self.assertIsNone(res)
        self.assertTrue(stats.get("rejected"))

if __name__ == "__main__":
    unittest.main()
