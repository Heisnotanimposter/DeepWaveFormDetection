from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import glob
import torch.nn.functional as F

# Configuration
class Config:
    SR = 32000
    N_MELS = 128
    N_MFCC = 13
    MAX_SEQ_LEN = 200
    ROOT_FOLDER = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/preprocessed/'
    BATCH_SIZE = 64
    N_EPOCHS = 20
    LR = 1e-4
    SUBSET_SIZE = 1000

CONFIG = Config()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Custom Dataset for Mel-spectrogram images
class CustomDataset(Dataset):
    def __init__(self, mel_files, labels=None, transform=None):
        self.mel_files = mel_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel_image = Image.open(self.mel_files[idx]).convert('RGB')
        if self.transform:
            mel_image = self.transform(mel_image)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return mel_image, label
        return mel_image

# Load file paths and labels for the datasets
def load_file_paths_and_labels(root_folder, subset_size=CONFIG.SUBSET_SIZE, mode='train'):
    if mode == 'train':
        real_mel_files = glob.glob(os.path.join(root_folder, 'train', 'real', '*.png'))[:subset_size]
        fake_mel_files = glob.glob(os.path.join(root_folder, 'train', 'fake', '*.png'))[:subset_size]

        mel_files = real_mel_files + fake_mel_files
        labels = [[0, 1]] * len(real_mel_files) + [[1, 0]] * len(fake_mel_files)

        print(f"Real Mel Files: {len(real_mel_files)}, Fake Mel Files: {len(fake_mel_files)}")
    else:
        mel_files = glob.glob(os.path.join(root_folder, 'test', '*.png'))[:subset_size]
        labels = None

    print(f"Mode: {mode}")
    print(f"Mel Files: {mel_files[:5]}")  # Print first 5 file paths to verify
    if mode == 'train':
        print(f"Labels: {labels[:5]}")      # Print first 5 labels to verify

    return mel_files, labels

# Verify the directory structure and files
def verify_directories(root_folder):
    try:
        train_real_mel_dir = os.path.join(root_folder, 'train', 'real')
        train_fake_mel_dir = os.path.join(root_folder, 'train', 'fake')

        print(f"Train Real Mel Directory: {train_real_mel_dir}")
        print(f"Train Fake Mel Directory: {train_fake_mel_dir}")

        print("Contents of Train Real Mel Directory:", os.listdir(train_real_mel_dir)[:5])
        print("Contents of Train Fake Mel Directory:", os.listdir(train_fake_mel_dir)[:5])
    except (OSError, IOError) as e:
        print(f"Error accessing directories: {e}")

verify_directories(CONFIG.ROOT_FOLDER)

# Load file paths and labels
train_mel_files, train_labels = load_file_paths_and_labels(CONFIG.ROOT_FOLDER, mode='train')
test_mel_files, _ = load_file_paths_and_labels(CONFIG.ROOT_FOLDER, mode='test')

# Print dataset sizes
print(f'Training samples: {len(train_mel_files)}')
print(f'Test samples: {len(test_mel_files)}')

# Ensure non-empty loaders
assert len(train_mel_files) > 0, "Training dataset is empty!"
assert len(test_mel_files) > 0, "Test dataset is empty!"

# Data transformations for Mel-spectrogram images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create dataset
full_dataset = CustomDataset(train_mel_files, train_labels, transform=transform)
test_dataset = CustomDataset(test_mel_files, transform=transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # hidden_dim * 2 because it's bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(device)
        c_0 = torch.zeros(self.lstm.num_layers * num_directions, x.size(0), self.lstm.hidden_size).to(device)

        x = self.dropout(x)
        lstm_out, _ = self.lstm(x, (h_0, c_0))
        x = self.fc(self.dropout(lstm_out[:, -1, :]))
        return x

# Define the CNN model for Mel-spectrogram images
class CNN(nn.Module):
    def __init__(self, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)  # Updated dimensions after additional conv layer
        self.fc2 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.gelu(self.conv1(x)))
        x = self.pool(F.gelu(self.conv2(x)))
        x = self.pool(F.gelu(self.conv3(x)))  # Additional convolutional layer
        x = x.view(-1, 128 * 16 * 16)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Combine both models
class CombinedModel(nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, lstm_output_dim, lstm_n_layers, lstm_bidirectional, lstm_dropout, cnn_output_dim):
        super(CombinedModel, self).__init__()
        self.lstm = BiLSTM(lstm_input_dim, lstm_hidden_dim, lstm_output_dim, lstm_n_layers, lstm_bidirectional, lstm_dropout)
        self.cnn = CNN(cnn_output_dim)
        self.fc = nn.Linear(lstm_output_dim + cnn_output_dim, 2)

    def forward(self, mfcc, mel):
        if mfcc is not None:
            mfcc = mfcc.permute(0, 2, 1)  # Change from (batch, channels, seq_len) to (batch, seq_len, input_dim)
            lstm_out = self.lstm(mfcc)
        else:
            lstm_out = torch.zeros(mel.size(0), 128).to(device)  # Dummy LSTM output if no MFCC is provided

        cnn_out = self.cnn(mel)
        combined = torch.cat((lstm_out, cnn_out), dim=1)
        out = self.fc(combined)
        return out

# Model initialization
model = CombinedModel(
    lstm_input_dim=CONFIG.N_MFCC,
    lstm_hidden_dim=128,
    lstm_output_dim=128,
    lstm_n_layers=2,
    lstm_bidirectional=True,
    lstm_dropout=0.5,
    cnn_output_dim=256  # Updated to match the new CNN architecture
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=CONFIG.LR)

# Training and validation functions
def train(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for mel, labels in tqdm(loader, desc="Training", leave=False):
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(None, mel)  # Only use Mel-spectrogram
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        _, labels = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    return epoch_loss / len(loader) if len(loader) > 0 else 0, accuracy

def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, labels in tqdm(loader, desc="Evaluating", leave=False):
            mel, labels = mel.to(device), labels.to(device)
            outputs = model(None, mel)  # Only use Mel-spectrogram
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    return epoch_loss / len(loader) if len(loader) > 0 else 0, accuracy

# Training loop
best_valid_loss = float('inf')
for epoch in range(CONFIG.N_EPOCHS):
    print(f'Epoch {epoch+1}/{CONFIG.N_EPOCHS}')
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-model.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

# Load the best model
model.load_state_dict(torch.load('best-model.pt'))

# Final evaluation on the validation set
final_loss, final_acc = evaluate(model, val_loader, criterion, device)
print(f'Final Loss: {final_loss:.3f} | Final Acc: {final_acc*100:.2f}%')

# Prediction on test dataset
def predict(model, loader, device):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for mel in tqdm(loader, desc="Predicting", leave=False):
            mel = mel.to(device)
            outputs = model(None, mel)  # Only use Mel-spectrogram
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_predictions.extend(probs.cpu().numpy())
    return np.array(all_predictions)

# Predict on test data
test_predictions = predict(model, test_loader, device)

# Create a DataFrame for submission
submission_df = pd.DataFrame(test_predictions, columns=['fake', 'real'])

# Extracting IDs from test file paths
test_ids = [os.path.basename(f).replace('.png', '') for f in test_mel_files]
submission_df.insert(0, 'id', test_ids)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print('Submission file created successfully!')