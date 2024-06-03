import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cnn import CNN  # Importa el modelo definido en cnn.py
import time
from tqdm import tqdm  # Importa tqdm para la barra de progreso

# Definir transformaciones de datos
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

# Listas para almacenar pérdidas
train_losses = []
valid_losses = []

# Función para calcular la precisión
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

# Registrar tiempo de inicio
start_time = time.time()

# Entrenamiento del modelo
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward y optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_corrects += accuracy(outputs, labels)
    
    # Calcular pérdida y precisión promedio por época
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects / len(train_loader)
    train_losses.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')
    
    # Validación del modelo
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_corrects += accuracy(outputs, labels)
    
    # Calcular pérdida y precisión promedio por época
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = running_corrects / len(test_loader)
    valid_losses.append(epoch_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.4f}')

# Registrar tiempo de finalización
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'model.ckpt')

# Guardar las pérdidas
torch.save(train_losses, 'train_losses.pt')
torch.save(valid_losses, 'valid_losses.pt')

print("Modelo entrenado y pérdidas guardadas correctamente.")
