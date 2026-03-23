import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "/workspace/student_scores.csv"

if os.path.exists(DATA_PATH):
    csv_path = DATA_PATH
else:
    csv_path = os.path.join(os.path.dirname(__file__), "student_scores.csv")

dataset = pd.read_csv(csv_path)
print("Primeras filas del dataset:")
print(dataset.head())

X = dataset[["Hours"]].values.astype(np.float32)
y = dataset[["Scores"]].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class FeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x

model = FeedForwardNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200
loss_history = []

for epoch in range(num_epochs):
    model.train()

    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

y_pred = y_pred_tensor.numpy()
y_test_np = y_test_tensor.numpy()

r2 = r2_score(y_test_np, y_pred)
mae = mean_absolute_error(y_test_np, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))

print("\nResultados en test:")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

results = pd.DataFrame({
    "Hours_scaled": X_test.flatten(),
    "Real_Score": y_test_np.flatten(),
    "Predicted_Score": y_pred.flatten()
})
results.to_csv(os.path.join(os.path.dirname(__file__), "predicciones.csv"), index=False)

loss_df = pd.DataFrame({
    "epoch": list(range(1, num_epochs + 1)),
    "loss": loss_history
})
loss_df.to_csv(os.path.join(os.path.dirname(__file__), "loss_history.csv"), index=False)

print("\nArchivos generados:")
print("- predicciones.csv")
print("- loss_history.csv")
