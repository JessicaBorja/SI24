import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

def preprocess_data(data_dir, train_ratio=0.7, val_ratio=0.15, random_state=42):
    # Definir transformaciones de imagen
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Cargar datos desde los directorios
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Dividir en conjuntos de entrenamiento, validación y prueba
    train_indices, test_val_indices = train_test_split(
        list(range(len(dataset))),
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True
    )

    val_size = int(len(test_val_indices) * (val_ratio / (1 - train_ratio)))
    val_indices, test_indices = train_test_split(
        test_val_indices,
        test_size=val_size,
        random_state=random_state,
        shuffle=True
    )

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    return train_set, val_set, test_set

if __name__ == "__main__":
    # Directorio de datos sin procesar
    data_dir = "C:/Users/jfros/OneDrive/OneDriveDocs/GitHub/Maravilla-/src/si24/proyectos/Muffin vs Chihuaha/Dataset/raw"

    # Directorio de salida para datos preprocesados
    processed_dir = "C:/Users/jfros/OneDrive/OneDriveDocs/GitHub/Maravilla-/src/si24/proyectos/Muffin vs Chihuaha/Dataset/processed"

    # Preprocesar datos
    train_set, val_set, test_set = preprocess_data(data_dir)

    # Guardar datos preprocesados
    os.makedirs(processed_dir, exist_ok=True)
    torch.save(train_set, os.path.join(processed_dir, "train.pt"))
    torch.save(val_set, os.path.join(processed_dir, "val.pt"))
    torch.save(test_set, os.path.join(processed_dir, "test.pt"))
