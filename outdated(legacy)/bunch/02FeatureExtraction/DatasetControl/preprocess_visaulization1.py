# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
import librosa
import librosa.display
import IPython.display as ipd

# Define a custom label encoder to handle unseen labels
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

# Load and preprocess dataset
chunk_size = 50000
chunks = pd.read_csv('/content/drive/MyDrive/dataset/TeamDeepwave/dataset/combined_file.csv', low_memory=False, chunksize=chunk_size)

# Save chunks to separate files
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'chunks_{i}.csv', index=False)

# Combine TSV files into a single dataframe
dataset_path = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/cv-corpus-17.0-delta-2024-03-15/en'
all_filenames = glob.glob(dataset_path + '/*.tsv')
dataframes = [pd.read_csv(filename, sep='\t', encoding='utf-8') for filename in all_filenames]
combined_csv = pd.concat(dataframes)
combined_csv.to_csv('/content/drive/MyDrive/dataset/TeamDeepwave/dataset/combined_file.csv', index=False, encoding='utf-8-sig')

# Inspect the combined dataset
print(combined_csv.head(10))

# Handle missing values
dropna_csv = combined_csv.dropna()
cleansed_csv = combined_csv.dropna(how='all')
print("DataFrame with rows dropped where any column has NaN:")
print(dropna_csv.head(10))
print("\nDataFrame with rows dropped where all columns are NaN:")
print(cleansed_csv.head(10))

# Load a specific TSV file for further processing
tsv_read = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/cv-corpus-17.0-delta-2024-03-15/en/validated.tsv'
df = pd.read_csv(tsv_read, sep='\t', encoding='utf-8')
print(f"Inspecting {tsv_read}:")
print(df.head(10))
print(df.shape)
print(df.info())

# Fill missing values in 'sentence_domain' and drop rows with other NaNs
df['sentence_domain'] = df['sentence_domain'].fillna('Unknown')
df_cleaned = df.dropna()

# Visualizing the dataset
# Distribution of 'client_id' (assuming it's the label we are interested in)
plt.figure(figsize=(10, 6))
sns.countplot(y='client_id', data=combined_csv, order=combined_csv['client_id'].value_counts().index)
plt.title('Distribution of Client IDs')
plt.show()

# Histogram of some numeric features (you can adjust based on your dataset)
numeric_features = combined_csv.select_dtypes(include=[np.number]).columns
combined_csv[numeric_features].hist(bins=30, figsize=(20, 15))
plt.show()

# Correlation heatmap
plt.figure(figsize=(16, 6))
sns.heatmap(combined_csv.corr(), annot=True, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Example of a waveplot and spectrogram of one audio file
# Load a sample audio file
sample_audio_path = combined_csv['path_to_audio_file_column'].iloc[0]  # Update column name as needed
y, sr = librosa.load(sample_audio_path)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveplot of a Sample Audio File')
plt.show()

# Plotting the spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure(figsize=(14, 5))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of a Sample Audio File')
plt.show()

# Pairplot for selected features
selected_features = combined_csv.columns[:5]  # Adjust based on your dataset
sns.pairplot(combined_csv[selected_features])
plt.show()

# Preprocessing for modeling
X = combined_csv.drop(columns=['clips_count'])  # Assuming 'clips_count' is not a feature
y = combined_csv['client_id']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the extended label encoder
extended_label_encoder = ExtendedLabelEncoder()
X_train['sentence_id'] = extended_label_encoder.fit_transform(X_train['sentence_id'])
X_test['sentence_id'] = extended_label_encoder.transform(X_test['sentence_id'])

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate models
results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# Display results
for model_name, metrics in results.items():
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print(f"Confusion Matrix:\n{metrics['Confusion Matrix']}\n")

# Visualizing Confusion Matrices
for model_name, metrics in results.items():
    plt.figure(figsize=(10, 7))
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Visualize performance metrics
performance_df = pd.DataFrame.from_dict(results, orient='index').drop(columns='Confusion Matrix')
performance_df.plot(kind='bar', figsize=(14, 7))
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()