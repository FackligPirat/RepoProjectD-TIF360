#%%
import numpy as np
import librosa
import matplotlib.pyplot as plt

import soundfile as sf
import subprocess

import os

def data_to_sound(data, filename):
    data = librosa.db_to_power(data)
    audio = librosa.feature.inverse.mel_to_audio(data)

    temp_wav = "new_audio/" + filename + ".wav"
    sf.write(temp_wav, audio, samplerate=22050)

# %%

# %%
