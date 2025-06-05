#%%
import matplotlib.pyplot as plt
import numpy as np

import librosa

import pandas as pd
import os

#%%
def read_audio(conf, pathname):
    y, _ = librosa.load(pathname, sr=conf.sampling_rate)
    
    if len(y) < conf.sampling_rate * 20:
        raise ValueError("Audio too short")
    
    return y

def audio_to_spectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(y=audio)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)

    return spectrogram

def read_as_spectrogram(conf, pathname):
    x = read_audio(conf, pathname)
    mels = audio_to_spectrogram(conf, x)
    return mels

class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 2
    hop_length = 347*duration # to make time steps 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20

def save_data_from_sound(pathname, filename, foldername):
    data_name = filename[:-4] + ".npy"

    try:
        x = read_as_spectrogram(conf, pathname)
        np.save(foldername + data_name, x[:, :160])
    except Exception as e:
        print(f"Error ({e}). Skipping {filename}")
        with open("problems.txt", "a") as f:
            f.write(filename + " - " + str(e) + "\n")

#%% Generate data small
tracks = pd.read_csv('tracks.csv', header=[0,1])
small = tracks[tracks['set', 'subset'] == 'small']

names = small['Unnamed: 0_level_0', 'Unnamed: 0_level_1'].values

folder_name = 'spectrograms5s'
os.makedirs(folder_name, exist_ok=True)



for n in names:
    # GET FOLDER AND FILE NAME
    folder = '000'
    name = str(n)

    while len(name) < 3:
        name = '0' + name
        folder = '000'
    if len(name) > 3:
        folder = name[:-3]
        name = name[-3:]

        while len(folder) < 3:
            folder = '0' + folder
    
    # CHECK IF SPECTOGRAM EXISTS
    exists = os.path.exists('spectrograms5s/'+ folder + name + '.npy')

    # CREATE SPECTOGRAM
    if not exists: 
        filename = folder + name + '.mp3'
        path = 'fma_small/' + folder + '/' + filename

        print(filename)

        save_data_from_sound(path, filename, 'spectrograms5s/')

# %% Generate data big
tracks = pd.read_csv('tracks.csv', header=[0,1])

names = tracks['Unnamed: 0_level_0', 'Unnamed: 0_level_1'].values

folder_name = 'spectrogramsFull5s'
os.makedirs(folder_name, exist_ok=True)



for n in names:
    # GET FOLDER AND FILE NAME
    folder = '000'
    name = str(n)

    while len(name) < 3:
        name = '0' + name
        folder = '000'
    if len(name) > 3:
        folder = name[:-3]
        name = name[-3:]

        while len(folder) < 3:
            folder = '0' + folder
    
    # CHECK IF SPECTOGRAM EXISTS
    exists = os.path.exists('spectrogramsFull5s/'+ folder + name + '.npy')

    # CREATE SPECTOGRAM
    if not exists: 
        filename = folder + name + '.mp3'
        path = 'fma_large/' + folder + '/' + filename

        print(filename)

        save_data_from_sound(path, filename, 'spectrogramsFull5s/')