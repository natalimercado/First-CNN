import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

#device = torch.device('cuda') #if torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformaciones de datos
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Descargar el conjunto de datos MNIST
train_dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transform, download=True)

#Tamaño del batch 
batch_size = 128

# Cargar los datos (proporciona imagen,etiqueta)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Carga del modelo ResNet18 preentrenado 
model = models.resnet18(pretrained=True)
#Extracción del vector de caracteristicas 
num_features = model.fc.in_features
#model.fc ->  capa de clasificación final de un modelo pre-entrenado en PyTorch
model.fc = nn.Linear(num_features, 10)  # Reemplazar la capa final para adaptarla al número de clases de MNIST

model = model.to(device)

#Función de pérdida y optimizador 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función de entrenamiento
def train(model, loader, criterion, optimizer):
    model.train() #se establece el modelo en modo entrenamiento
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) #predicciones del modelo
        loss = criterion(outputs, labels) #función de pérdida (compara el output con el label original)
        loss.backward() #retropropagación 
        optimizer.step() #Actualización de parámetros 

        train_loss += loss.item()
        _, predicted = outputs.max(1) #tupla (puntuaciones de cada clase para cada imagen en el batch, predicciones de clase)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item() #predicted.eq ->  comparación elemento a elemento entre las predicciones de clase y las etiquetas 0-> diferentes, 1->iguales

    accuracy = 100.0 * correct / total
    avg_loss = train_loss / len(loader)

    return avg_loss, accuracy


# Función de evaluación
def evaluate(model, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(loader)

    return avg_loss, accuracy

# Entrenamiento y evaluación del modelo
num_epochs = 10

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, criterion, optimizer)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}')
