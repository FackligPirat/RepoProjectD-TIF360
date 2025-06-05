#%%
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from modelVAE5s import ConvVAE  # Your VAE model
from modelVAE5s import SpectrogramDataset  # Your dataset class
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import seaborn as sns

# === Load genre labels ===
def load_genre_labels(metadata_path='fma_metadata'):
    tracks = pd.read_csv(os.path.join(metadata_path, 'tracks.csv'), index_col=0, header=[0, 1])
    return tracks[('track', 'genre_top')]

# Convert filename (e.g., '000123.npy') to track ID (int)
def filename_to_track_id(filename):
    return int(os.path.splitext(filename)[0])



def compute_confusion_matrix(y_true, y_pred, genre_names):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Match predicted labels to true labels
    row_ind, col_ind = linear_sum_assignment(-cm)
    new_pred = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        new_pred[y_pred == j] = i
    
    # Recalculate confusion matrix with aligned labels
    cm_aligned = confusion_matrix(y_true, new_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_aligned,
        annot=True,
        fmt="d",
        cmap="Blues",
        annot_kws={"size": 16},  # <-- Larger numbers
        xticklabels=genre_names,
        yticklabels=genre_names
    )
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Genre", fontsize=18)
    plt.title("Confusion Matrix", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.show()

    return cm_aligned

#%% Setup


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE(latent_dim=256).to(device)
model.load_state_dict(torch.load("models/vae5s256BestModel.pth", map_location=device))
model.eval()

data_dir = "spectrograms_5s"
dataset = SpectrogramDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

# === Load genre labels ===
genre_labels = load_genre_labels("fma_metadata")

used_genres_set = set()
for _, fname in tqdm(dataset, desc="Scanning genres"):
    track_id = filename_to_track_id(fname)
    genre = genre_labels.get(track_id)
    if pd.notna(genre):
        used_genres_set.add(genre)

genre_names = sorted(used_genres_set)
genre_to_idx = {genre: idx for idx, genre in enumerate(genre_names)}
print(f"Detected genres used in dataset: {genre_names}")

#%% VAE Latent Space Inference & Clustering

vae_latents = []
vae_labels = []

with torch.no_grad():
    for batch, filenames in dataloader:
        batch = batch.to(device)
        mu, _ = model.encode(batch)
        latents = mu.cpu().numpy()

        for i, fname in enumerate(filenames):
            track_id = filename_to_track_id(fname)
            genre = genre_labels.get(track_id)
            if pd.notna(genre):
                vae_latents.append(latents[i])
                vae_labels.append(genre_to_idx[genre])

vae_latents = np.array(vae_latents)
vae_labels = np.array(vae_labels)

print(f"Latents shape: {vae_latents.shape}, Labels shape: {vae_labels.shape}")

n_clusters = len(genre_to_idx)
print(f"Number of genres: {n_clusters}")

# Clustering in latent space
kmeans_latent = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_latent_labels = kmeans_latent.fit_predict(vae_latents)

gmm_latent = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
gmm_latent_labels = gmm_latent.fit_predict(vae_latents)

ari_kmeans_latent = adjusted_rand_score(vae_labels, kmeans_latent_labels)
ari_gmm_latent = adjusted_rand_score(vae_labels, gmm_latent_labels)

print(f"VAE Latent Space ARI (KMeans): {ari_kmeans_latent:.4f}")
print(f"VAE Latent Space ARI (GMM): {ari_gmm_latent:.4f}")

#%% Raw Spectrogram Clustering 
spec_vectors = []
spec_labels = []

for batch, filenames in tqdm(dataloader, desc="Processing raw spectrograms"):
    batch = batch.numpy().squeeze(1).reshape(batch.shape[0], -1)  # (B, 20480)

    for i, fname in enumerate(filenames):
        track_id = filename_to_track_id(fname)
        genre = genre_labels.get(track_id)
        if pd.notna(genre):
            spec_vectors.append(batch[i])
            spec_labels.append(genre_to_idx[genre])

spec_vectors = np.stack(spec_vectors)
spec_labels = np.array(spec_labels)

print(f"Spectrograms shape: {spec_vectors.shape}, Labels shape: {spec_labels.shape}")

# Clustering on raw spectrograms
kmeans_spec = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_spec_labels = kmeans_spec.fit_predict(spec_vectors)

gmm_spec = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=0)
gmm_spec_labels = gmm_spec.fit_predict(spec_vectors)

ari_kmeans_spec = adjusted_rand_score(spec_labels, kmeans_spec_labels)
ari_gmm_spec = adjusted_rand_score(spec_labels, gmm_spec_labels)

print(f"Raw Spectrogram ARI (KMeans): {ari_kmeans_spec:.4f}")
print(f"Raw Spectrogram ARI (GMM): {ari_gmm_spec:.4f}")

#%% Summary and Confusion Matrix for Latent Space
print("\n=== ARI Summary ===")
print(f"VAE Latent Space -> KMeans: {ari_kmeans_latent:.4f}, GMM: {ari_gmm_latent:.4f}")
print(f"Raw Spectrograms -> KMeans: {ari_kmeans_spec:.4f}, GMM: {ari_gmm_spec:.4f}")
#%%
# Plot confusion matrix for latent space clustering
compute_confusion_matrix(vae_labels, gmm_latent_labels, genre_names)

# %%
