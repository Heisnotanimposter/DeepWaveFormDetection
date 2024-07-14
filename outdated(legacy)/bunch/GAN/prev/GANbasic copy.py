
!pip install torch librosa matplotlib numpy soundfile

from google.colab import drive
drive.mount('/content/drive')

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/content/drive/MyDrive/voice_dataset'
audio_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]

def load_audio(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# Load and plot sample spectrogram
sample_file = audio_files[0]
sample_spectrogram = load_audio(sample_file)
plt.figure(figsize=(10, 4))
librosa.display.specshow(sample_spectrogram, sr=22050, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram of Sample Audio')
plt.tight_layout()
plt.show()

spectrograms = [load_audio(file) for file in audio_files]
spectrograms = np.array(spectrograms)
print(spectrograms.shape)

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 128 * 128),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 1, 128, 128)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128 * 128, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

import torch.optim as optim

def train_gan(generator, discriminator, data, epochs=100, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        for real_images in data:
            batch_size = real_images.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train Discriminator
            optimizer_D.zero_grad()
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)
            d_loss_real.backward()

            z = torch.randn(batch_size, 100)
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            d_loss_fake.backward()
            optimizer_D.step()

            d_loss = d_loss_real + d_loss_fake
            d_losses.append(d_loss.item())

            # Train Generator
            optimizer_G.zero_grad()
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            g_losses.append(g_loss.item())

        print(f"Epoch [{epoch+1}/{epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.title('Training Losses')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Convert spectrograms to tensor and create DataLoader
spectrograms_tensor = torch.tensor(spectrograms, dtype=torch.float32).unsqueeze(1)
dataloader = torch.utils.data.DataLoader(spectrograms_tensor, batch_size=64, shuffle=True)

# Initialize models
generator = Generator()
discriminator = Discriminator()

# Train GAN
train_gan(generator, discriminator, dataloader)

import soundfile as sf

def generate_voice_samples(generator, num_samples=10, save_path='/content/drive/MyDrive/voice_dataset/generated'):
    os.makedirs(save_path, exist_ok=True)
    z = torch.randn(num_samples, 100)
    generated_spectrograms = generator(z)
    
    for i, spectrogram in enumerate(generated_spectrograms):
        spectrogram = spectrogram.detach().numpy().reshape(128, 128)
        y_inv = librosa.feature.inverse.mel_to_audio(spectrogram)
        file_path = os.path.join(save_path, f"generated_{i+1}.wav")
        sf.write(file_path, y_inv, 22050)
        print(f"Saved {file_path}")

generate_voice_samples(generator)

# Visualize generated spectrograms
z = torch.randn(5, 100)
generated_spectrograms = generator(z).detach().numpy()

plt.figure(figsize=(15, 10))
for i, spectrogram in enumerate(generated_spectrograms, 1):
    plt.subplot(2, 3, i)
    librosa.display.specshow(spectrogram.reshape(128, 128), sr=22050, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Generated Spectrogram {i}')
plt.tight_layout()
plt.show()

def plot_spectrograms(spectrograms, title):
    fig, axes = plt.subplots(1, len(spectrograms), figsize=(15, 5))
    for i, spectrogram in enumerate(spectrograms):
        axes[i].imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
        axes[i].set_title(f"{title} {i+1}")
    plt.show()

# Load and plot real spectrograms
real_spectrograms = [load_audio(file) for file in audio_files[:5]]
plot_spectrograms(real_spectrograms, "Real Spectrogram")

# Generate and plot fake spectrograms
fake_spectrograms = [generator(torch.randn(1, 100)).detach().numpy().reshape(128, 128) for _ in range(5)]
plot_spectrograms(fake_spectrograms, "Generated Spectrogram")
