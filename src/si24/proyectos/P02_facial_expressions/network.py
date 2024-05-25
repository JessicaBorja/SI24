import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights
from pathlib import Path

file_path = pathlib.Path(__file__).parent.absolute()

def build_backbone(model='resnet18', weights='imagenet', freeze=True, last_n_layers=2):
    if model == 'resnet18':
        backbone = resnet18(pretrained=weights == 'imagenet')
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f'Model {model} not supported')

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
 # TODO: Calcular dimension de salida
        out_dim = n_classes
        # TODO: Define las capas de tu red

        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3) #46x46
        self.conv2 = nn.Conv2d(64, 128, 3) #44x44  
        #22x22
        self.conv3 = nn.Conv2d(128, 128, 3) #20x20
        #10x10
        self.fc1 = nn.Linear(10 * 10 * 128,1024 )
        self.fc2 = nn.Linear(1024, out_dim)
        self.to(self.device)
   
    
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x =x.cuda()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = torch.flatten(x) #poner (x,1) para entrenar y (x) para inferencia
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x
        

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name

        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(self.state_dict(), models_path)

    def load_model(self,model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name

        self.load_state_dict(torch.load(models_path))
