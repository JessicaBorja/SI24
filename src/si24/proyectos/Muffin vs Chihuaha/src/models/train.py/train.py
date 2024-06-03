import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn import CNN  # Importa el modelo definido en cnn.py

# Definir transformaciones de datos (ajusta según sea necesario)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar datos preprocesados
train_set = torch.load("Dataset/processed/train.pt")
test_set = torch.load("Dataset/processed/test.pt")

# Definir DataLoader para cargar los datos en lotes
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Inicializar modelo y función de pérdida
model = CNN(num_classes=2)  # Crea una instancia del modelo CNN
criterion = nn.CrossEntropyLoss()  # Utiliza la pérdida de entropía cruzada como función de pérdida

# Definir hiperparámetros y configuración de entrenamiento
num_epochs = 10
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Utiliza el optimizador Adam

# Entrenamiento del modelo
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward y optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.ckpt')
print("Modelo entrenado guardado correctamente.")
