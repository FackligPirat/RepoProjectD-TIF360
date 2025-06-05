#%%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from data_to_sound import data_to_sound
from torch.utils.data import random_split


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),  # (128,160) → (64,80)
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # → (32,40)
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # → (16,20)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), # → (8,10)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1), # → (4,5)
            nn.LeakyReLU(0.1),
        )
        self.flatten = nn.Flatten()  # (64,4,5) → 1280
        self.fc_mu = nn.Linear(64 * 4 * 5, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 5, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 5),
            nn.LeakyReLU(0.1)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # → (8,10)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # → (16,20)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # → (32,40)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # → (64,80)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # → (128,160)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 4, 5)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def train_vae(model, train_loader, val_loader, optimizer, device, num_epochs=50, verbose=True):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # === Training ===
        model.train()
        total_train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                val_loss = vae_loss(recon, batch, mu, logvar)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if verbose:
            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

class SpectrogramDataset(Dataset):
    def __init__(self, directory):
        self.min_db = -100.0
        self.max_db = 42.68
        self.files = []

        for f in os.listdir(directory):
            if f.endswith('.npy'):
                path = os.path.join(directory, f)
                try:
                    spec = np.load(path)
                    if spec.shape == (128, 160):  # expected shape
                        self.files.append(path)
                    else:
                        print(f"Skipping {f} with shape {spec.shape}")
                except Exception as e:
                    print(f"Error loading {f}: {e}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spec = np.load(self.files[idx])
        spec = np.clip(spec, self.min_db, self.max_db)
        spec = (spec - self.min_db) / (self.max_db - self.min_db)  # Normalize to [0, 1]
        spec = np.expand_dims(spec, axis=0)
        filename = os.path.basename(self.files[idx])
        return spec.astype(np.float32), filename

def create_dataloaders(dataset, batch_size=16, val_split=0.2):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

class NewConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(NewConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder with progressively increasing channels: 1 → 4 → 8 → 16 → 32 → 64
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 4, 4, stride=2, padding=1),    # (128,160) → (64,80)
            nn.LeakyReLU(0.1),
            nn.Conv2d(4, 8, 4, stride=2, padding=1),    # → (32,40)
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 16, 4, stride=2, padding=1),   # → (16,20)
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),  # → (8,10)
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # → (4,5)
            nn.LeakyReLU(0.1),
        )

        self.flatten = nn.Flatten()  # (64,4,5) → 1280
        self.fc_mu = nn.Linear(64 * 4 * 5, latent_dim)
        self.fc_logvar = nn.Linear(64 * 4 * 5, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 4 * 5),
            nn.LeakyReLU(0.1)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # → (8,10)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # → (16,20)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # → (32,40)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # → (64,80)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # → (128,160)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 4, 5)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
class BigConvVAE(nn.Module):
    def __init__(self, latent_dim=2048):
        super(BigConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: compress less aggressively, output shape will be (128, 8, 4)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),    # (128,160) → (64,80)
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),   # → (32,40)
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # → (16,20)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # → (8,10)
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, stride=1, padding=0), # → (6,8)
            nn.LeakyReLU(0.1),
        )

        # Now output shape = (128, 6, 8) → 128 * 6 * 8 = **6144 features**
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 6 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 6 * 8, latent_dim)

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 6 * 8),
            nn.LeakyReLU(0.1)
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=0),  # (6,8) → (8,10)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → (16,20)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → (32,40)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # → (64,80)
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),     # → (128,160)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 6, 8)  # Match encoder's output shape
        return self.decoder_conv(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar