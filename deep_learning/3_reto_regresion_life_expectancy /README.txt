Proyecto de regresión con red neuronal feed-forward en PyTorch.

Archivos:
- main.py: código principal del ejercicio.
- dataset.csv: dataset usado en la práctica.
- curva_perdida.png: se genera al ejecutar el script.

El código:
- carga el dataset con ruta absoluta /workspace/dataset.csv
- limpia espacios en columnas
- elimina nulos en columnas clave
- divide los datos en train, validation y test
- normaliza las características
- convierte los datos a tensores
- usa DataLoader con batch_size=32 y num_workers=4
- construye una red neuronal feed-forward con dos capas ocultas
- entrena durante 100 épocas
- grafica la curva de pérdida
- evalúa el modelo en test con MSE y MAE
