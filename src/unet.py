import torch
from diffusers import UNet2DModel

def get_latent_unet(in_channels=64, out_channels=64):
    """
    Crée un UNet configuré pour opérer sur l'espace latent du VQ-VAE.
    Nous utilisons la bibliothèque `diffusers` pour avoir une implémentation
    robuste et optimisée.
    
    in_channels: doit correspondre à embedding_dim du VQ-VAE.
    """
    model = UNet2DModel(
        sample_size=16,           # Taille de l'espace latent (ex: 64x64 img -> 16x16 latent)
        in_channels=in_channels,  # Nombre de canaux en entrée (embedding_dim du VAE)
        out_channels=out_channels,# Nombre de canaux en sortie (prédiction du bruit)
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512), # Réseau assez profond pour le PoC
        down_block_types=(
            "DownBlock2D",        
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",    # Attention sur la résolution la plus basse
        ),
        up_block_types=(
            "AttnUpBlock2D",      # Attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

if __name__ == "__main__":
    # Test local
    unet = get_latent_unet(in_channels=64, out_channels=64)
    # L'entrée du UNet en diffusion: (batch_size, channels, height, width) et le timestep t
    dummy_latent = torch.randn(2, 64, 16, 16) 
    timesteps = torch.tensor([10, 10])
    
    # La sortie du UNet est de la même dimension que l'entrée (c'est le bruit prédit)
    prediction = unet(dummy_latent, timesteps).sample
    print(f"Shape du latent en entrée : {dummy_latent.shape}")
    print(f"Shape du bruit prédit : {prediction.shape}")
