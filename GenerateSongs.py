#%%
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
from modelVAE5s import ConvVAE, BigConvVAE
import matplotlib.pyplot as plt
import math

SAMPLE_RATE = 22050
MIN_DB = -100.0
MAX_DB = 42.68
OUTPUT_DIR = "new_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def data_to_sound(spec_db, filename, sr=SAMPLE_RATE):
    """Convert dB-scaled mel spec to audio and save both audio + image."""
    power_spec = librosa.db_to_power(spec_db)
    audio = librosa.feature.inverse.mel_to_audio(power_spec, sr=sr)
    
    wav_path = os.path.join(OUTPUT_DIR, f"{filename}.wav")
    sf.write(wav_path, audio, samplerate=sr)
    print(f"Saved: {wav_path}")

    # Also plot and save spectrogram image
    plot_spectrogram(spec_db, filename)

def load_spec(filename, min_db=MIN_DB, max_db=MAX_DB, device='cpu'):
    """Load and normalize a .npy spectrogram file for model input."""
    spec = np.load(filename)
    spec = np.clip(spec, min_db, max_db)
    spec = (spec - min_db) / (max_db - min_db)
    spec = np.expand_dims(spec, axis=0)  # (1, 128, 160)
    spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, 128, 160)
    return spec

def plot_spectrogram(spec_db, filename, cmap="inferno"):
    """Plot and save a mel spectrogram in dB scale."""
    plt.figure(figsize=(6, 4))
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap=cmap)
    plt.title(f"Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Mel Bin")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"), dpi=300)
    plt.close()


def generate_latent_samples(model, num_samples=4, latent_dim=1024, device='cpu', prefix="sample"):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device) * 0.5
        generated = model.decode(z).cpu().numpy()

    for i, gen in enumerate(generated):
        gen = np.squeeze(gen, axis=0)
        gen_db = gen * (MAX_DB - MIN_DB) + MIN_DB
        data_to_sound(gen_db, f"{prefix}_{i}")


def interpolate_and_generate(model, file_a, file_b, num_steps=1, device='cpu', prefix="interp"):
    model.eval()
    with torch.no_grad():
        spec_a = load_spec(file_a, device=device)
        spec_b = load_spec(file_b, device=device)

        mu_a, _ = model.encode(spec_a)
        mu_b, _ = model.encode(spec_b)

        alphas = torch.linspace(0, 1, num_steps).to(device)

        for i, alpha in enumerate(alphas):
            z = (1 - alpha) * mu_a + alpha * mu_b
            recon = model.decode(z)
            spec = recon[0][0].detach().cpu().numpy()
            spec_db = spec * (MAX_DB - MIN_DB) + MIN_DB
            data_to_sound(spec_db, f"{prefix}_{i}")

def linear_timbre_walk(model, file_list, steps_per_pair=50, device='cpu', prefix='linear_walk'):
    model.eval()
    with torch.no_grad():
        latents = []
        for file in file_list:
            spec = load_spec(file, device=device)
            mu, _ = model.encode(spec)
            latents.append(mu)

        z_walk = []
        for i in range(len(latents) - 1):
            a, b = latents[i], latents[i + 1]
            for alpha in torch.linspace(0, 1, steps_per_pair):
                z = (1 - alpha) * a + alpha * b
                z_walk.append(z)

        for i, z in enumerate(z_walk):
            recon = model.decode(z.unsqueeze(0))
            spec = recon[0][0].detach().cpu().numpy()
            spec_db = spec * (MAX_DB - MIN_DB) + MIN_DB
            data_to_sound(spec_db, f"{prefix}_{i}")

#%% Make songs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE(latent_dim=256).to(device)
model.load_state_dict(torch.load("vae5s256BestModel.pth", map_location=device))

file_a = "spectrogramsFull_5s/097041.npy"
file_b = "spectrogramsFull_5s/145555.npy"
file_c = "spectrogramsFull_5s/097043.npy"
file_d = "spectrogramsFull_5s/073342.npy"

interpolate_and_generate(model, file_a, file_b, num_steps=3, device=device, prefix="interp_song")

generate_latent_samples(model, num_samples=6, latent_dim=256, device=device, prefix="random_song")

# %% Make timbre walk
files = [file_a, file_b, file_c]
linear_timbre_walk(model, files, steps_per_pair=3, device=device)
#%%
# %%
