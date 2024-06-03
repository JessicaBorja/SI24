import torch
import matplotlib.pyplot as plt

# Cargar las pérdidas de entrenamiento y validación
train_losses = torch.load('train_losses.pt')
valid_losses = torch.load('valid_losses.pt')

# Graficar pérdidas durante el entrenamiento
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, valid_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
