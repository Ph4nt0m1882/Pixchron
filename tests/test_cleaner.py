import unittest
import os
import sys
from PIL import Image
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
            remove_background=True,
            strict_mode=True
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

    def test_upscaled_macro_pixel_with_jpeg_noise(self):
        """Vérifie qu'un vrai pixel art 16x16 upscalé 4x avec du bruit JPEG est validé et nettoyé."""
        native = np.zeros((16, 16, 3), dtype=np.uint8)
        native[2:6, 6:10] = [220, 180, 140] # Tête
        native[6:12, 5:11] = [30, 100, 200]  # Corps
        native[12:15, 6:8] = [50, 50, 50]    # Jambe G
        native[12:15, 9:11] = [50, 50, 50]   # Jambe D

        native_img = Image.fromarray(native)
        upscaled_img = native_img.resize((64, 64), Image.Resampling.NEAREST)
        upscaled_arr = np.array(upscaled_img).astype(float)

        np.random.seed(42)
        noise = np.random.randint(-4, 4, upscaled_arr.shape)
        noisy_arr = np.clip(upscaled_arr + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy_arr)

        res, scale, stats = self.reconstructor.reconstruct(noisy_img)
        self.assertIsNotNone(res)
        self.assertEqual(scale, 4)
        self.assertEqual(stats["clean_size"], (16, 16))

    def test_rejection_of_graph_paper_photo(self):
        """Vérifie qu'une photo de dessin sur papier quadrillé (bruit lourd de papier/feutre) est rejetée."""
        # Simulation d'un dessin sur papier avec lignes de cahier et grain
        arr = np.full((128, 128, 3), 230, dtype=float)
        # Ajouter le quadrillage imprimé du cahier
        arr[::8, :] = 180
        arr[:, ::8] = 180
        # Ajouter le grain de papier
        np.random.seed(42)
        arr += np.random.normal(0, 15, arr.shape)
        # Dessin au feutre avec bavures
        arr[32:96, 32:96] = np.clip(arr[32:96, 32:96] - 150 + np.random.normal(0, 20, (64, 64, 3)), 0, 255)
        paper_img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        res, scale, stats = self.reconstructor.reconstruct(paper_img)
        self.assertIsNone(res)
        self.assertTrue(stats.get("rejected"))
        self.assertIn("Faux pixel art", stats.get("reason"))

    def test_background_removal(self):
        """Vérifie la conversion du fond uni en canal Alpha transparent."""
        rec_bg = PixelReconstructor(remove_background=True, bg_tolerance=20.0)
        
        arr = np.full((32, 32, 3), [74, 103, 107], dtype=np.uint8)
        arr[8:24, 8:24] = [200, 50, 50]

        img = Image.fromarray(arr)
        res, scale, stats = rec_bg.reconstruct(img)

        self.assertIsNotNone(res)
        res_arr = np.array(res)
        self.assertEqual(res_arr[0, 0, 3], 0)
        self.assertEqual(res_arr[16, 16, 3], 255)

if __name__ == "__main__":
    unittest.main()
