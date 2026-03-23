import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Ruta absoluta pedida en el enunciado
DATA_PATH = "/workspace/dataset.csv"

# Si se ejecuta fuera del evaluador, usamos una ruta alternativa local
if not os.path.exists(DATA_PATH):
    local_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if os.path.exists(local_path):
        DATA_PATH = local_path


# Cargar datos
_df = pd.read_csv(DATA_PATH)
print("Primeras filas del dataset:")
print(_df.head())

# Limpiar nombres de columnas
_df.columns = _df.columns.str.strip()
print("\nColumnas limpias:")
print(_df.columns.tolist())

# Eliminar filas con nulos en columnas clave
key_columns = [
    "BMI",
    "GDP",
    "Schooling",
    "Adult Mortality",
    "Income composition of resources",
    "Life expectancy",
]

df = _df.dropna(subset=key_columns).copy()

print("\nValores nulos tras la limpieza en columnas clave:")
print(df[key_columns].isnull().sum())

# Usar solo columnas numéricas
numeric_df = df.select_dtypes(include=[np.number]).copy()

# Variable objetivo y variables de entrada
target_column = "Life expectancy"
X = numeric_df.drop(columns=[target_column])
y = numeric_df[target_column]

print("\nColumnas usadas como entrada:")
print(X.columns.tolist())
print("\nVariable objetivo:", target_column)

# División 65% entrenamiento, 20% validación, 15% prueba
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

val_size_relative = 0.20 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_size_relative, random_state=42
)

print("\nTamaños de los conjuntos:")
print("Entrenamiento:", X_train.shape, y_train.shape)
print("Validación:", X_val.shape, y_val.shape)
print("Prueba:", X_test.shape, y_test.shape)

# Normalizar características de entrada sin tocar la variable objetivo
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
class LifeExpectancyModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


input_size = X_train_tensor.shape[1]
model = LifeExpectancyModel(input_size)

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Entrenamiento
num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch_X.size(0)

    epoch_train_loss = total_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_val_loss += loss.item() * batch_X.size(0)

    epoch_val_loss = total_val_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] - "
        f"Train Loss: {epoch_train_loss:.4f} - "
        f"Val Loss: {epoch_val_loss:.4f}"
    )

# Curva de pérdida
sns.set()
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Curva de pérdida")
plt.legend()
plt.tight_layout()
plot_path = os.path.join(os.path.dirname(__file__), "curva_perdida.png")
plt.savefig(plot_path)
print(f"\nGráfica guardada en: {plot_path}")

# Evaluación final en prueba
model.eval()
all_predictions = []
all_targets = []

test_loss_total = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        test_loss_total += loss.item() * batch_X.size(0)
        all_predictions.extend(predictions.squeeze(1).cpu().numpy())
        all_targets.extend(batch_y.squeeze(1).cpu().numpy())

final_test_loss = test_loss_total / len(test_loader.dataset)
mae = mean_absolute_error(all_targets, all_predictions)
mse = mean_squared_error(all_targets, all_predictions)

print("\nResultados en test:")
print(f"Test Loss (MSE): {final_test_loss:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
