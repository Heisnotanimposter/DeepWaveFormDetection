import matplotlib.pyplot as plt

# Data
epochs = list(range(1, 21))
train_losses = [0.689, 0.628, 0.558, 0.456, 0.392, 0.335, 0.284, 0.263, 0.274, 0.239, 0.212, 0.196, 0.189, 0.193, 0.153, 0.153, 0.142, 0.134, 0.118, 0.127]
train_accuracies = [51.56, 68.62, 71.62, 78.69, 82.19, 85.62, 88.50, 89.56, 87.50, 90.75, 91.31, 92.56, 92.69, 92.38, 94.88, 94.00, 95.00, 95.88, 95.62, 94.94]
val_losses = [0.679, 0.598, 0.515, 0.447, 0.401, 0.363, 0.341, 0.328, 0.326, 0.305, 0.272, 0.300, 0.260, 0.250, 0.229, 0.232, 0.265, 0.218, 0.221, 0.239]
val_accuracies = [52.00, 72.25, 77.25, 84.00, 84.50, 86.00, 86.75, 88.50, 89.00, 89.25, 90.75, 89.75, 92.00, 92.00, 92.25, 93.75, 91.25, 93.50, 93.50, 92.75]

# Plot training and validation loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')

plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()