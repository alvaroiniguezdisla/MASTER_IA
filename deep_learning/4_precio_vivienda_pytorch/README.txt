Proyecto de regresión con PyTorch para predecir el precio de la vivienda.

Archivos:
- main.py: código principal
- dataset.csv: dataset del ejercicio

Cómo ejecutar:
1. Coloca el archivo dataset.csv en /workspace/dataset.csv si trabajas en VS Code web.
2. Ejecuta: python main.py

Qué hace el script:
- Carga y limpia los datos
- Divide en train, validación y test
- Normaliza las variables de entrada
- Crea una red neuronal feed-forward en PyTorch
- Entrena durante 100 épocas
- Guarda la gráfica loss_curve.png
- Evalúa el modelo en test
