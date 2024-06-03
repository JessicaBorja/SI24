import torch
import torch.nn as nn
import torchvision
import torchivison.transforms as transforms
import matplotlib.pyplot as plt

#coniguracion  cpu :/
device = torch.device('cuda' if torch.cuda is available() else 'cpu')
#device = torch.device('cpu')