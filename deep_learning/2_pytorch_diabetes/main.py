import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


# Ruta absoluta pedida en el enunciado
CSV_PATH = "/workspace/diabetes.csv"


class DiabetesNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# Cargar el dataset usando la ruta absoluta.
# Si no existe en ese entorno, se usa la copia local de apoyo.
def load_dataset():
    path_to_use = CSV_PATH
    if not os.path.exists(path_to_use):
        local_path = os.path.join(os.path.dirname(__file__), "diabetes.csv")
        path_to_use = local_path

    dataset = pd.read_csv(path_to_use)
    print("Primeras filas del dataset:")
    print(dataset.head())
    return dataset


# En este dataset algunos ceros pueden representar valores no válidos.
# Se sustituyen por la mediana de la columna.
def clean_dataset(dataset):
    columns_with_possible_invalid_zeros = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    cleaned = dataset.copy()

    for column in columns_with_possible_invalid_zeros:
        cleaned[column] = cleaned[column].replace(0, np.nan)
        median_value = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(median_value)

    return cleaned


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = load_dataset()
    dataset = clean_dataset(dataset)

    # Separar características y variable objetivo
    X = dataset.drop("Outcome", axis=1)
    y = dataset["Outcome"]

    # Partición 75% entrenamiento y 25% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Normalización de las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir a tensores
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

    # Crear el modelo
    input_size = X_train.shape[1]
    model = DiabetesNet(input_size)

    # Función de pérdida y optimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    epochs = 200
    for epoch in range(epochs):
        model.train()

        predictions = model(X_train_tensor)
        loss = criterion(predictions, y_train_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluación
    model.eval()
    with torch.no_grad():
        test_probabilities = model(X_test_tensor)
        test_predictions = (test_probabilities >= 0.5).float()

    y_true = y_test_tensor.numpy().flatten()
    y_pred = test_predictions.numpy().flatten()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print("\nResultados en el conjunto de prueba:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
