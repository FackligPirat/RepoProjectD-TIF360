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
from torch.utils.data import DataLoader, random_split
import pickle
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

#%% Autoencoder

class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=128):
        super(ConvAutoencoder, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),     # (128,160) → (64,80)
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),    # → (32,40)
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),    # → (16,20)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),    # → (8,10)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),    # → (4,5)
            nn.LeakyReLU(0.1),
        )

        self.encoder_fc = nn.Sequential(
            nn.Flatten(),                                # (64,4,5) → 256
            nn.Linear(64 * 4 * 5, bottleneck_dim),
            nn.LeakyReLU(0.1)
        )

        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(bottleneck_dim, 64 * 4 * 5),
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

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = x.view(-1, 64, 4, 5)
        x = self.decoder_conv(x)
        return x


#%% Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataset
dataset = SpectrogramDataset('spectrogramsFull_5s/')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Loss function
criterion = nn.MSELoss()
#%% Training
# Bottleneck values to test
bottleneck_dims = [16, 32, 64, 128, 256, 512, 1024]
num_epochs = 100
loss_history = {}

for bottleneck_dim in bottleneck_dims:
    print(f"\nTraining with bottleneck_dim = {bottleneck_dim}")
    
    # Init model and optimizer
    model = ConvAutoencoder(bottleneck_dim=bottleneck_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch, _ in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, _ in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save weights
    torch.save(model.state_dict(), f"models/autoencoder_bottleneck{bottleneck_dim}.pth")
    print(f"Model saved: autoencoder_bottleneck{bottleneck_dim}.pth")

    # Store loss history
    loss_history[bottleneck_dim] = {
        'train': train_losses,
        'val': val_losses
    }

# Save loss history
with open("loss_history.pkl", "wb") as f:
    pickle.dump(loss_history, f)
print("Loss history saved to loss_history.pkl")

# Plot
for bottleneck_dim in bottleneck_dims:
    plt.plot(loss_history[bottleneck_dim]['train'], label=f"Train {bottleneck_dim}")
    plt.plot(loss_history[bottleneck_dim]['val'], linestyle='--', label=f"Val {bottleneck_dim}")

plt.title("Train vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Plot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(bottleneck_dim=256).to(device)
model.load_state_dict(torch.load("models/autoencoder_bottleneck256.pth", map_location=device))


spectrogram_path = 'spectrogramsFull_5s' 
filename = '111335.npy'        
full_path = os.path.join(spectrogram_path, filename)

min_db = -100.0
max_db = 42.68

spec = np.load(full_path)
spec = np.clip(spec, min_db, max_db)
spec = (spec - min_db) / (max_db - min_db)  
spec = np.expand_dims(spec, axis=0) 
spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device)


model.eval()
with torch.no_grad():
    recon = model(spec)


original_norm = spec[0][0].cpu().numpy()
reconstructed_norm = recon[0][0].cpu().numpy()

# === Plot ===
plt.subplot(1, 2, 1)
plt.imshow(original_norm, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='inferno')
plt.title("Original (Normalized)")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_norm, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='inferno')
plt.title("Reconstructed (Normalized)")

plt.tight_layout()
plt.show()

original_db = original_norm * (max_db - min_db) + min_db
reconstructed_db = reconstructed_norm * (max_db - min_db) + min_db

plt.figure(figsize=(8, 4)) 

plt.subplot(1, 2, 1)
plt.imshow(original_db, aspect='auto', origin='lower', vmin=min_db, vmax=max_db, cmap='inferno')
plt.title("Original (dB)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_db, aspect='auto', origin='lower', vmin=min_db, vmax=max_db, cmap='inferno')
plt.title("Reconstructed (dB)")
plt.colorbar()

plt.tight_layout()
plt.show()

reconstructed_db = reconstructed_norm * (max_db - min_db) + min_db
output_name = f"reconstructed_30s_{os.path.splitext(filename)[0]}"
data_to_sound(reconstructed_db, output_name)

print(f"Saved audio to: new_audio/{output_name}.wav")

#%% make wav file from npy
def convert_npy_to_audio(npy_path):
    if not os.path.exists(npy_path):
        print(f"File not found: {npy_path}")
        return

    data_db = np.load(npy_path)

    filename = f"5s_{os.path.splitext(os.path.basename(npy_path))[0]}"

    data_to_sound(data_db, filename)

npy_file = f"spectrograms/{111335}.npy"
convert_npy_to_audio(npy_file)
# %%