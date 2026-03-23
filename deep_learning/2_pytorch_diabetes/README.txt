Proyecto 2 - Red neuronal feed-forward con PyTorch

Archivos incluidos:
- main.py
- diabetes.csv

Qué hace el script:
1. Carga el dataset diabetes.csv.
2. Limpia valores no válidos simples en algunas columnas.
3. Separa características y variable objetivo.
4. Divide los datos en entrenamiento y prueba con 75% - 25%.
5. Normaliza las características.
6. Convierte los datos a tensores de PyTorch.
7. Construye una red neuronal feed-forward con capas 64 y 32.
8. Usa Sigmoid en la salida para clasificación binaria.
9. Entrena con Binary Cross-Entropy Loss y Adam.
10. Evalúa con Accuracy, F1 Score y classification report.

Ruta absoluta pedida por el enunciado:
- /workspace/diabetes.csv

Si esa ruta no existe, el script usa la copia local del archivo dentro de esta carpeta.
