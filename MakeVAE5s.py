#%% Imports
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

#%% Dataset
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

#%% Autoencoder

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
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

def plot_train_val_losses(all_losses):
    plt.figure(figsize=(10, 6))
    for latent_dim, (train_losses, val_losses) in all_losses.items():
        plt.plot(train_losses, label=f"Train (dim={latent_dim})", linestyle='-')
        plt.plot(val_losses, label=f"Val (dim={latent_dim})", linestyle='--')
    
    plt.title("VAE Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#%% Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
dataset = SpectrogramDataset('spectrogramsFull_5s/')
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ConvVAE(latent_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#%% Batch training
latent_dims = [8, 16, 32, 64, 128, 256, 1024]
all_losses = {}

train_loader, val_loader = create_dataloaders(dataset, batch_size=16)

for dim in latent_dims:
    print(f"\nTraining VAE with latent_dim = {dim}")
    model = ConvVAE(latent_dim=dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses = train_vae(model, train_loader, val_loader, optimizer, device, num_epochs=100)
    all_losses[dim] = (train_losses, val_losses)

    torch.save(model.state_dict(), f"vae_latent{dim}Full5s.pth")

plot_train_val_losses(all_losses)

import pickle

with open("vae_all_losses_new.pkl", "wb") as f:
    pickle.dump(all_losses, f)

#%% Plot
spectrogram_path = 'spectrogramsFull_5s' 
filename = '006358.npy' 
full_path = os.path.join(spectrogram_path, filename)

min_db = -100.0
max_db = 42.68

spec = np.load(full_path)
spec = np.clip(spec, min_db, max_db)
spec = (spec - min_db) / (max_db - min_db)  # normalize to [0, 1]
spec = np.expand_dims(spec, axis=0)
spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device) 


model.eval()
with torch.no_grad():
    recon, _, _ = model(spec)  


original_norm = spec[0][0].cpu().numpy()
reconstructed_norm = recon[0][0].cpu().numpy()


plt.subplot(1, 2, 1)
plt.imshow(original_norm, aspect='auto', origin='lower', vmin=0, vmax=1)
plt.title("Original (Normalized)")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_norm, aspect='auto', origin='lower', vmin=0, vmax=1)
plt.title("Reconstructed (Normalized)")

plt.tight_layout()
plt.show()

reconstructed_db = reconstructed_norm * (max_db - min_db) + min_db
output_name = f"reconstructed_5s_VAE_{os.path.splitext(filename)[0]}"
data_to_sound(reconstructed_db, output_name)

print(f"Saved audio to: new_audio/{output_name}.wav")
# %%
