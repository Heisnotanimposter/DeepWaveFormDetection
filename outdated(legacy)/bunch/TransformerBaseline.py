import librosa
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './open'
    N_CLASSES = 2
    BATCH_SIZE = 64
    N_EPOCHS = 10
    LR = 3e-4
    SEED = 42
    MAX_SEQ_LEN = 200  # Maximum sequence length for transformer input

CONFIG = Config()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)

df = pd.read_csv('./.csv')
train_df, val_df = train_test_split(df, test_size=0.2, random_state=CONFIG.SEED)

def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        pad_width = CONFIG.MAX_SEQ_LEN - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :CONFIG.MAX_SEQ_LEN]
        features.append(mfcc)
        
        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)
    
    if train_mode:
        return np.array(features), np.array(labels)
    return np.array(features)

train_mfcc, train_labels = get_mfcc_feature(train_df, True)
val_mfcc, val_labels = get_mfcc_feature(val_df, True)

class CustomDataset(Dataset):
    def __init__(self, mfcc, labels=None):
        self.mfcc = mfcc
        self.labels = labels

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.mfcc[index], self.labels[index]
        return self.mfcc[index]

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, output_dim=CONFIG.N_CLASSES, num_heads=4, num_layers=2):
        super(TransformerClassifier, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, CONFIG.MAX_SEQ_LEN, input_dim))
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x + self.positional_encoding
        x = self.transformer_encoder(x.permute(1, 0, 2))
        x = self.fc(x.mean(dim=0))
        return self.sigmoid(x)

transformer_model = TransformerClassifier()
transformer_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=CONFIG.LR)

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, CONFIG.N_EPOCHS+1):
        model.train()
        train_loss = []
        for features, labels in tqdm(train_loader):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
                    
        val_loss, val_score = validate(model, criterion, val_loader, device)
        train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss: [{train_loss:.5f}] Val Loss: [{val_loss:.5f}] Val AUC: [{val_score:.5f}]')
            
        if best_val_score < val_score:
            best_val_score = val_score
            best_model = model
    
    return best_model

def validate(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            loss = criterion(probs, labels)
            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        val_loss = np.mean(val_loss)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        val_score = multiLabel_AUC(all_labels, all_probs)
    
    return val_loss, val_score

def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    return np.mean(auc_scores)

# Train the model
best_transformer_model = train(transformer_model, transformer_optimizer, train_loader, val_loader, device)

test_df = pd.read_csv('./test.csv')
test_mfcc = get_mfcc_feature(test_df, False)

test_dataset = CustomDataset(test_mfcc)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader):
            features = features.float().to(device)
            probs = model(features)
            predictions.append(probs.cpu().numpy())
    return np.concatenate(predictions, axis=0)

test_predictions = predict(best_transformer_model, test_loader, device)

# Save predictions to CSV
test_df['prediction'] = test_predictions[:, 1]  # Assuming the second column is the 'real' class probability
test_df.to_csv('test_predictions.csv', index=False)