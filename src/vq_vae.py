import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # Convertir inputs de BCHW à BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Aplatir les inputs
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculer les distances avec le dictionnaire
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Trouver les vecteurs les plus proches (encoding)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantifier
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss (perte) : s'assurer que l'encodeur produit des valeurs proches du dictionnaire
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Trick pour propager le gradient (Straight Through Estimator)
        quantized = inputs + (quantized - inputs).detach()
        
        # Reconvertir en BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(num_hiddens//2, num_hiddens, kernel_size=4, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels, num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = nn.Sequential(
            *[ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens) for _ in range(num_residual_layers)]
        )
        self._conv_trans_1 = nn.ConvTranspose2d(num_hiddens, num_hiddens//2, kernel_size=4, stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(num_hiddens//2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        # Sortie sans activation car les tenseurs de diffusion/images sont souvent entre -1 et 1
        return self._conv_trans_2(x)

class PixelVQVAE(nn.Module):
    """
    Autoencodeur quantifié pour forcer la création de blocs discrets,
    optimisé pour les images à 4 canaux (RGBA).
    """
    def __init__(self, in_channels=4, num_hiddens=128, num_residual_layers=2, 
                 num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, 
                 commitment_cost=0.25):
        super(PixelVQVAE, self).__init__()
        
        self._encoder = Encoder(in_channels, num_hiddens, num_residual_layers, num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)
        self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, in_channels)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        quantized, vq_loss, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        
        # Pour forcer le style "pixel art", on pourrait ajouter un nearest neighbor ou clip
        # Mais le réseau apprendra la structure en grille si le VQ_loss est fort.
        return x_recon, vq_loss

if __name__ == "__main__":
    # Test local
    model = PixelVQVAE(in_channels=4)
    dummy_input = torch.randn(2, 4, 64, 64) # Batch de 2, RGBA, 64x64
    reconstructed, loss = model(dummy_input)
    print(f"Entrée : {dummy_input.shape}")
    print(f"Sortie reconstruite : {reconstructed.shape}")
    print(f"Loss VQ : {loss.item()}")
