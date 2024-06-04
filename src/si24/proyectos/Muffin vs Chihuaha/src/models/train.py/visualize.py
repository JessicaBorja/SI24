import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from cnn import CNN  # Asegúrate de que el modelo CNN esté definido en cnn.py
import matplotlib.pyplot as plt
import numpy as np

# Definir transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Cargar el modelo entrenado
model = CNN(num_classes=2)  # Ajusta num_classes según tu problema
model.load_state_dict(torch.load('model.ckpt'))  # Ajusta el nombre del archivo según donde hayas guardado tu modelo
model.eval()

# Cargar datos preprocesados de prueba
test_set = torch.load("C:/Users/jfros/OneDrive/OneDriveDocs/GitHub/Maravilla-/src/si24/proyectos/Muffin vs Chihuaha/Dataset/processed/test.pt")  # Ajusta la ruta según donde tengas tus datos
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

# Obtener clases
classes = ['muffin', 'chihuahua']  # Ajusta las clases según tu problema

# Función para mostrar imágenes con sus clases predichas
def visualize_predictions():
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_class = classes[predicted.item()]
            actual_class = classes[labels.item()]
            image = np.transpose(images.squeeze().numpy(), (1, 2, 0))
            
            # Mostrar la imagen
            plt.imshow(image)
            plt.title(f'Predicted: {predicted_class}, Actual: {actual_class}')
            plt.axis('off')
            plt.show()
            break  # Solo mostramos una imagen por simplicidad

# Visualizar predicciones
visualize_predictions()

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
