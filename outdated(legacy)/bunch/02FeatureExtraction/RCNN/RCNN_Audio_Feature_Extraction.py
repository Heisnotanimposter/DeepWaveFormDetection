!pip install opencv-python-headless tensorflow keras

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import TensorBoard
import datetime

# Define paths
real_dataset_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/KaggleDataset/real'
fake_dataset_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/KaggleDataset/fake'

# Function to load Faster R-CNN model
def load_model():
    model = tf.saved_model.load('path_to_faster_rcnn_model')
    return model

# Function to extract features using Faster R-CNN
def extract_features(image_path, model):
    image = cv2.imread(image_path)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = model(input_tensor)
    return detections['detection_features'][0].numpy()

# Load Faster R-CNN model
faster_rcnn_model = load_model()

# Prepare data
def prepare_data(file_paths, model):
    data = []
    for file_path in file_paths:
        features = extract_features(file_path, model)
        data.append(features)
    return np.array(data)

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
X_train = prepare_data(X_train_paths, faster_rcnn_model)
X_test = prepare_data(X_test_paths, faster_rcnn_model)

# Convert labels to numpy arrays
y_train = np.array(y_train)
y_test = np.array(y_test)

# Reshape the data for the Conv2D layer
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[tensorboard_callback])

# Evaluate the model
y_pred_prob = model.predict(X_test)
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
grid_search_xgb.fit(X_train, y_train)

# LightGBM Classifier with Random Search
lgbm = LGBMClassifier()
random_search_lgbm = RandomizedSearchCV(estimator=lgbm, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy', verbose=2)
random_search_lgbm.fit(X_train, y_train)

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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
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
    plt.show()

# Evaluate XGBoost and LightGBM models
evaluate_model(best_xgb, X_train, X_test, y_train, y_test, "XGBoost")
evaluate_model(best_lgbm, X_train, X_test, y_train, y_test, "LightGBM")

