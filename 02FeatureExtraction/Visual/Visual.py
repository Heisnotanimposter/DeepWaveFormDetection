# Import necessary libraries
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import TensorBoard
import datetime
import IPython

# Define paths
real_dataset_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/KaggleDataset/real'
fake_dataset_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/KaggleDataset/fake'
metadata_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/cv-corpus-17.0-delta-2024-03-15/en'

# Load and extract features
def load_and_extract_features(file_path, max_pad_len=128):
    y, sr = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    if mel_spectrogram_db.shape[1] > max_pad_len:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mel_spectrogram_db.shape[1]
        mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mel_spectrogram_db

# Get file paths and labels
real_files = [os.path.join(real_dataset_path, f) for f in os.listdir(real_dataset_path) if f.endswith('.wav')]
fake_files = [os.path.join(fake_dataset_path, f) for f in os.listdir(fake_dataset_path) if f.endswith('.wav')]
real_labels = [0] * len(real_files)  # Label 0 for real
fake_labels = [1] * len(fake_files)  # Label 1 for fake

# Combine datasets
file_paths = real_files + fake_files
labels = real_labels + fake_labels

# Split the data
X_train_paths, X_test_paths, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Prepare the data
def prepare_data(file_paths):
    data = []
    for file_path in file_paths:
        features = load_and_extract_features(file_path)
        data.append(features)
    return np.array(data)

X_train = prepare_data(X_train_paths)
X_test = prepare_data(X_test_paths)

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape the data for the Conv2D layer
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the CNN model
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Evaluate the CNN model
y_pred_prob = cnn_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n {conf_matrix}")

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Define paths for the metadata
metadata_files = glob.glob(metadata_path + '/*.tsv')
dataframes = []

# Combine metadata files
for filename in metadata_files:
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    dataframes.append(df)
metadata_csv = pd.concat(dataframes)
metadata_csv.to_csv('/content/drive/MyDrive/dataset/TeamDeepwave/dataset/metadata_file.csv', index=False, encoding='utf-8-sig')

# Preprocess the metadata
dropna_csv = metadata_csv.dropna()
cleansed_csv = metadata_csv.dropna(how='all')

# Extract features and labels from metadata
X = metadata_csv.drop(columns=['clips_count'])
y = metadata_csv['client_id']

# Train-test split
X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(X, y, test_size=0.2, random_state=42)

# Metric function
def print_metrics(y_true, y_pred):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

# Initialize extended label encoder
class ExtendedLabelEncoder(LabelEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_ = None

    def fit(self, y):
        super().fit(y)
        self.classes_ = super().classes_

    def transform(self, y):
        try:
            return super().transform(y)
        except NotFittedError:
            raise
        except ValueError as e:
            unseen_label = max(self.classes_) + 1
            self.classes_ = np.append(self.classes_, unseen_label)
            return np.where(y == e.args[0], unseen_label, super().transform(y))

extended_label_encoder = ExtendedLabelEncoder()
X_train_meta['sentence_id'] = extended_label_encoder.fit_transform(X_train_meta['sentence_id'])
X_test_meta['sentence_id'] = extended_label_encoder.transform(X_test_meta['sentence_id'])

# Evaluate different models on metadata
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

for model_name, model in models.items():
    model.fit(X_train_meta, y_train_meta)
    y_pred_meta = model.predict(X_test_meta)
    print(f"\n{model_name} Metrics:")
    print_metrics(y_test_meta, y_pred_meta)

    conf_matrix_meta = confusion_matrix(y_test_meta, y_pred_meta)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_meta, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png")
    plt.show()

# Define the parameter grid for Grid Search and Random Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# XGBoost Classifier with Grid Search
xgb = XGBClassifier()
grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)
grid_search_xgb.fit(X_train_meta, y_train_meta)

# LightGBM Classifier with Random Search
lgbm = LGBMClassifier()
random_search_lgbm = RandomizedSearchCV(estimator=lgbm, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', verbose=2)
random_search_lgbm.fit(X_train_meta, y_train_meta)

# Best parameters
print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
print("Best parameters for LightGBM:", random_search_lgbm.best_params_)

# Evaluate the best models
best_xgb = grid_search_xgb.best_estimator_
best_lgbm = random_search_lgbm.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n{label} Model Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {label}")
    plt.savefig(f"confusion_matrix_{label.replace(' ', '_').lower()}.png")
    plt.show()

# Evaluate XGBoost and LightGBM models
evaluate_model(best_xgb, X_train_meta, X_test_meta, y_train_meta, y_test_meta, "XGBoost")
evaluate_model(best_lgbm, X_train_meta, X_test_meta, y_train_meta, y_test_meta, "LightGBM")

# Define TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Modify the model training to include TensorBoard callback
cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

#%load_ext tensorboard
#%tensorboard --logdir logs/fit