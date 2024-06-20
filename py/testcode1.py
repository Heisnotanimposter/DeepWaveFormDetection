# Import necessary libraries
import os
import pandas as pd
import torch
import torchaudio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import librosa
import librosa.display
import IPython.display as ipd
import sklearn
import warnings
warnings.filterwarnings('ignore')

# Load metadata
train_csv = pd.read_csv('path_to_your_voice_metadata.csv')
train_csv.head()

# Update base directory
base_dir = 'path_to_your_voice_data'
train_csv['full_path'] = base_dir + '/' + train_csv['speaker_label'] + '/' + train_csv['filename']

# Sample audio files from the dataset
speakers = train_csv['speaker_label'].unique()
samples = {speaker: train_csv[train_csv['speaker_label'] == speaker].sample(1, random_state=33)['full_path'].values[0] for speaker in speakers}

# Play audio samples
for speaker, file_path in samples.items():
    ipd.Audio(file_path)

# Load and preprocess audio files
audio_data = {}
for speaker, file_path in samples.items():
    y, sr = librosa.load(file_path)
    audio_data[speaker] = librosa.effects.trim(y)[0]

# Visualize waveforms
fig, ax = plt.subplots(len(speakers), figsize=(16, 12))
fig.suptitle('Sound Waves', fontsize=16)
for i, (speaker, audio) in enumerate(audio_data.items()):
    librosa.display.waveplot(y=audio, sr=sr, ax=ax[i])
    ax[i].set_ylabel(speaker, fontsize=13)
plt.show()

# Extract and visualize features (Mel spectrograms)
for speaker, audio in audio_data.items():
    S = librosa.feature.melspectrogram(audio, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'Mel Spectrogram - {speaker}', fontsize=16)
    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='log', cmap='cool', ax=ax)
    plt.colorbar(img, ax=ax)
    plt.show()

# Example of calculating and plotting Spectral Centroids
for speaker, audio in audio_data.items():
    spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sr)[0]
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    plt.figure(figsize=(16, 6))
    librosa.display.waveplot(audio, sr=sr, alpha=0.4)
    plt.plot(t, sklearn.preprocessing.minmax_scale(spectral_centroids), color='r')
    plt.title(f'Spectral Centroid - {speaker}', fontsize=16)
    plt.legend(['Spectral Centroid', 'Wave'])
    plt.show()

# Extracting MFCCs and plotting them
for speaker, audio in audio_data.items():
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    fig, ax = plt.subplots(1, figsize=(12, 6))
    img = librosa.display.specshow(mfcc, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=f'MFCC - {speaker}')
    plt.show()