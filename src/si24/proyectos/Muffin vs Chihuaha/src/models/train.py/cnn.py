import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Definir las capas convolucionales y de pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Definir la capa completamente conectada (fully connected)
        self.fc = nn.Linear(64 * 56 * 56, num_classes)  # Ajustar el tamaño de entrada según la salida de la última capa convolucional
        
    def forward(self, x):
        # Aplicar las capas convolucionales y de pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Aplanar la salida para pasarla a la capa completamente conectada
        x = torch.flatten(x, 1)
        # Aplicar la capa completamente conectada
        x = self.fc(x)
        return x
