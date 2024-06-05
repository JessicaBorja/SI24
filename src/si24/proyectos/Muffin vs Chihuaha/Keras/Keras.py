import torch
import torch.nn as nn
from torchviz import make_dot

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Define convolutional layers and pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Define fully connected layer
        self.fc = nn.Linear(64 * 56 * 56, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Initialize your model
model = CNN(num_classes=2)

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Visualize the model architecture
visualize_model = make_dot(model(dummy_input), params=dict(model.named_parameters()))
visualize_model.render("model_architecture", format="png")
