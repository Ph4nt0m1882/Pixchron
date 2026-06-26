import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from dataset import get_dataloader
from vq_vae import PixelVQVAE
from unet import get_latent_unet
import os

def train_vqvae(vqvae, dataloader, optimizer, device, epochs=1):
    print("--- Démarrage de l'entraînement du VQ-VAE ---")
    vqvae.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad()
            
            reconstructed, vq_loss = vqvae(images)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, images)
            
            # Total loss
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    print("--- Fin de l'entraînement du VQ-VAE ---")

def train_diffusion(unet, vqvae, noise_scheduler, dataloader, optimizer, device, epochs=1):
    print("--- Démarrage de l'entraînement du UNet (Diffusion Latente) ---")
    unet.train()
    vqvae.eval() # On fige le VQ-VAE
    
    for epoch in range(epochs):
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # 1. Obtenir la représentation latente (sans gradient pour le VAE)
            with torch.no_grad():
                # On encode les images
                z = vqvae._encoder(images)
                z = vqvae._pre_vq_conv(z)
                # On quantifie pour forcer l'espace discret
                latents, _, _ = vqvae._vq_vae(z)
            
            # 2. Générer du bruit aléatoire à ajouter aux latents
            noise = torch.randn_like(latents)
            
            # 3. Échantillonner un timestep aléatoire pour chaque image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            
            # 4. Ajouter le bruit selon le timestep (forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # 5. Prédire le bruit avec le UNet
            optimizer.zero_grad()
            noise_pred = unet(noisy_latents, timesteps).sample
            
            # 6. Calculer la perte (MSE entre le bruit ajouté et le bruit prédit)
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | UNet Loss: {loss.item():.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")
    
    # Hyperparamètres
    batch_size = 8
    image_size = 64
    embedding_dim = 64
    
    # 1. Charger le Dataset
    dataloader = get_dataloader(batch_size=batch_size, image_size=image_size)
    
    # 2. Initialiser les Modèles
    vqvae = PixelVQVAE(in_channels=4, embedding_dim=embedding_dim).to(device)
    unet = get_latent_unet(in_channels=embedding_dim, out_channels=embedding_dim).to(device)
    
    # 3. Optimiseurs
    optimizer_vqvae = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=1e-4)
    
    # 4. Scheduler de bruit pour la diffusion
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    
    # Normalement, on entraîne le VQ-VAE de manière extensive d'abord.
    # Pour le PoC, on fait un run court des deux.
    train_vqvae(vqvae, dataloader, optimizer_vqvae, device, epochs=1)
    train_diffusion(unet, vqvae, noise_scheduler, dataloader, optimizer_unet, device, epochs=1)
    
    print("Test local du PoC réussi ! Les modèles ont compilé et la boucle a tourné.")

if __name__ == "__main__":
    main()
