import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_mfcc = np.load('/mnt/data/train_mfcc.npy')
train_labels = np.load('/mnt/data/train_labels.npy')
test_mfcc = np.load('/mnt/data/test_mfcc.npy')
test_labels = np.load('/mnt/data/test_labels.npy')

# Print shapes of the arrays
print(f'train_mfcc shape: {train_mfcc.shape}')
print(f'train_labels shape: {train_labels.shape}')
print(f'test_mfcc shape: {test_mfcc.shape}')
print(f'test_labels shape: {test_labels.shape}')

# Print a few samples
print('train_mfcc sample:', train_mfcc[0])
print('train_labels sample:', train_labels[0])
print('test_mfcc sample:', test_mfcc[0])
print('test_labels sample:', test_labels[0])

# Function to plot MFCC features
def plot_mfcc(mfcc, title):
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc, aspect='auto', origin='lower')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar()
    plt.show()

# Plot a few samples
plot_mfcc(train_mfcc[0], 'Train MFCC Sample 1')
plot_mfcc(train_mfcc[1], 'Train MFCC Sample 2')
plot_mfcc(test_mfcc[0], 'Test MFCC Sample 1')
plot_mfcc(test_mfcc[1], 'Test MFCC Sample 2')

# Count the occurrences of each label
train_labels_count = np.sum(train_labels, axis=0)
test_labels_count = np.sum(test_labels, axis=0)

print(f'Train Labels Count: {train_labels_count}')
print(f'Test Labels Count: {test_labels_count}')