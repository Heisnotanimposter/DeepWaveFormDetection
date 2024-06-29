import torch
import torch.nn as nn
import torch.optim as optim

class RCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(RCNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.rnn_layers = nn.LSTM(128*32, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        x = self.cnn_layers(x)
        x = x.view(batch_size, height // 4, -1)
        x, _ = self.rnn_layers(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

# Initialize model, loss function, optimizer
model = RCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)