import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN
from preprocess import get_data_loaders

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(data_dir, epochs=10, batch_size=32, learning_rate=0.001):
    train_loader, test_loader = get_data_loaders(data_dir, batch_size)
    
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    torch.save(model.state_dict(), 'reports/results/muffin_vs_chihuahua_cnn.pth')
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model('data/raw')

