import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta absoluta requerida por el enunciado
csv_path = "/workspace/dataset.csv"

# Cargar datos
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    # Respaldo para poder probar localmente si hiciera falta
    df = pd.read_csv("dataset.csv")

print("Primeras filas del dataset:")
print(df.head())

# Limpiar nombres de columnas
(df.columns) = df.columns.str.strip()
print("\nColumnas del dataset:")
print(df.columns.tolist())

# Seleccionar columnas clave y eliminar nulos
feature_columns = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "grade", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated", "zipcode",
    "lat", "long", "sqft_living15", "sqft_lot15"
]
required_columns = feature_columns + ["price"]

df = df[required_columns].dropna().copy()

print("\nValores nulos por columna:")
print(df.isnull().sum())
print(f"\nFilas después de limpieza: {len(df)}")

# Variables de entrada y objetivo
X = df[feature_columns]
y = df["price"]

# División 65% entrenamiento, 20% validación, 15% prueba
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.35, random_state=42
)

# De ese 35% restante, tomar 15/35 para test y 20/35 para validación
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(15/35), random_state=42
)

print(f"\nTamaño entrenamiento: {len(X_train)}")
print(f"Tamaño validación: {len(X_val)}")
print(f"Tamaño prueba: {len(X_test)}")

# Normalizar características de entrada, sin tocar la variable objetivo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convertir a tensores
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Modelo feed-forward
class HousePriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

model = HousePriceModel(X_train_tensor.shape[1])

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Entrenamiento
num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float("inf")
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# Cargar el mejor modelo según validación
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Graficar curva de pérdida
sns.set()
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de pérdida")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# Evaluación final en test
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

final_test_loss = test_loss / len(test_loader)
print(f"\nPérdida final en test (MSE): {final_test_loss:.4f}")
