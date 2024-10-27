import numpy as np
import librosa
from tqdm import tqdm

# Function to extract Mel-spectrogram features
def get_mel_spectrogram_feature(file_paths):
    features = []
    labels = []
    for file_path in tqdm(file_paths):
        try:
            y, sr = librosa.load(file_path, sr=CONFIG.SR)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            pad_width = CONFIG.MAX_SEQ_LEN - mel_spectrogram_db.shape[1]
            if pad_width > 0:
                mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_spectrogram_db = mel_spectrogram_db[:, :CONFIG.MAX_SEQ_LEN]
            features.append(mel_spectrogram_db)

            # Assuming labels are derived from file names (e.g., 'fake' or 'real')
            if 'fake' in file_path:
                label_vector = [1, 0]
            else:
                label_vector = [0, 1]
            labels.append(label_vector)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return np.array(features), np.array(labels)

# Extract Mel-spectrogram features from the training and test datasets
train_mel_spectrogram, train_labels = get_mel_spectrogram_feature(train_files, train_labels)
test_mel_spectrogram, test_labels = get_mel_spectrogram_feature(test_files, test_labels)

# Save the data using numpy
np.save('train_mel_spectrogram.npy', train_mel_spectrogram)
np.save('test_mel_spectrogram.npy', test_mel_spectrogram)

print(f'Training and testing Mel-spectrogram features and labels saved successfully.')