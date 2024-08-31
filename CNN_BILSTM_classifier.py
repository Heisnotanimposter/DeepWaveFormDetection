# Auto-load and install required libraries
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import librosa
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    from google.colab import drive
except ImportError as e:
    missing_package = str(e).split(' ')[-1]
    print(f"Installing missing package: {missing_package}")
    install(missing_package)
    # Reload after installation
    import librosa
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    from google.colab import drive

# Mount Google Drive automatically if not mounted
try:
    drive.mount('/content/drive')
except:
    print("Drive already mounted")

import os

# Configuration
class Config:
    SR = 32000
    N_MELS = 128
    MAX_SEQ_LEN = 200
    ROOT_FOLDER = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/'
    PREPROCESSED_FOLDER = '/content/drive/MyDrive/dataset/TeamDeepwave/dataset/preprocessed/'
    SUBSET_SIZE = 50000

CONFIG = Config()

# Ensure output directory for test data exists
os.makedirs(os.path.join(CONFIG.PREPROCESSED_FOLDER, 'test'), exist_ok=True)

# Function to load file paths from CSV
def load_test_file_paths(csv_path):
    df = pd.read_csv(csv_path)
    file_paths = df['path'].apply(lambda x: os.path.join(CONFIG.ROOT_FOLDER, x)).tolist()
    return file_paths

# Function to save Mel-spectrogram as grayscale PNG with visualization
def save_mel_spectrogram(file_path, output_folder, show_image=False):
    filename = os.path.basename(file_path).replace('.ogg', '.png')
    output_path = os.path.join(output_folder, filename)

    if os.path.exists(output_path):
        print(f"Skipping {filename} (already exists)")
        return

    try:
        y, sr = librosa.load(file_path, sr=CONFIG.SR)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CONFIG.N_MELS)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Visualize the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis='time', y_axis='mel', cmap='gray')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram (Grayscale)')
    plt.tight_layout()

    # Show the image if requested
    if show_image:
        plt.show()

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load test file paths from CSV
test_files = load_test_file_paths('/content/drive/MyDrive/dataset/TeamDeepwave/dataset/open/test.csv')

# Get existing preprocessed test filenames
preprocessed_test_folder = os.path.join(CONFIG.PREPROCESSED_FOLDER, 'test')
existing_test_files = set(os.listdir(preprocessed_test_folder))

# Find files that need preprocessing
remaining_files_to_preprocess = [
    file_path for file_path in tqdm(test_files, desc="Checking existing files")
    if os.path.basename(file_path).replace('.ogg', '.png') not in existing_test_files
]

print(f"Found {len(remaining_files_to_preprocess)} files that need preprocessing.")

# Preprocess the remaining files with visualization
for file_path in tqdm(remaining_files_to_preprocess, desc="Preprocessing remaining files"):
    save_mel_spectrogram(file_path, preprocessed_test_folder, show_image=False)

print('Mel-spectrogram images saved successfully.')