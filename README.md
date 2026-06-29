![Header](.github/assets/images/header.jpeg)

# Pixchron: Modèle Fondateur de Génération de Pixel Art

Un modèle d'intelligence artificielle de pointe conçu pour générer du véritable pixel art (pixel-perfect) sans aucun anti-aliasing. Pixchron est un modèle de diffusion latente hybride couplé à un VQ-VAE.

## Spécifications Techniques (Specs)

- **Architecture Core** : Modèle de Diffusion Latente (UNet) + Vector Quantized Autoencoder (VQ-VAE).
- **Espace Colorimétrique** : Support natif de la transparence via le canal Alpha (4 canaux : RGBA).
- **Rendu "Pixel-Perfect"** : Quantification de l'espace latent (VQ) et interpolation Nearest-Neighbor stricte pour prévenir l'anti-aliasing et la "dérive" des pixels.
- **Color Indexing** : Le VQ-VAE contraint mathématiquement les couleurs de sortie pour utiliser des palettes indexées.
- **Génération Massive** : Support futur de résolutions extrêmes (jusqu'à 2000x1000) grâce à la nature convolutionnelle et aux techniques de "Latent Tiling".
- **Animations (I2V / T2V)** : L'architecture de diffusion est conçue pour intégrer des "Motion Adapters" (modules temporels) afin de générer des séquences d'animation et des sprite sheets.

## Démarrage Rapide (PoC)

```bash
# Activation de l'environnement uv
.venv\Scripts\activate

# Test de compilation des tenseurs RGBA du VQ-VAE et de la boucle UNet
python src/train.py
```