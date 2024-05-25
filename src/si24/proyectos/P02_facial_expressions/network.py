import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Calcular la dimensión de salida
        out_dim = self.calc_out_dim(input_dim, kernel_size=3)

        # Definir las capas de la red
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3)
        self.max_pool3 = nn.MaxPool2d(2, stride=2)

        # Definir las capas completamente conectadas
        self.fc1 = nn.Linear(out_dim * out_dim * 64, 128)  # Ajustar el tamaño de entrada de fc1
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, n_classes)

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = (in_dim + 2 * padding - (kernel_size - 1) - 1) // stride + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Propagación hacia adelante de la red
        feature_map = self.conv1(x)
        feature_map = self.max_pool1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.max_pool2(feature_map)
        feature_map = self.conv3(feature_map)
        feature_map = self.max_pool3(feature_map)

        # Aplanar el tensor para las capas completamente conectadas
        features = torch.flatten(feature_map, start_dim=1)

        # Capas completamente conectadas
        features = self.fc1(features)
        features = F.relu(features)
        features = self.fc2(features)
        features = F.relu(features)
        logits = self.fc3(features)
        
        proba = F.softmax(logits,dim=-1)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        # Guardar los pesos de la red neuronal
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        # Cargar los pesos de la red neuronal
        self.load_state_dict(torch.load(model_name))