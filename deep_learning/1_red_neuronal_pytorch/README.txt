Proyecto 1 - Red neuronal feed-forward con PyTorch

Contenido:
- main.py
- student_scores.csv

Qué hace el código:
1. Carga el dataset student_scores.csv
2. Usa Hours como entrada y Scores como salida
3. Divide los datos en train y test con 75%-25%
4. Convierte los datos a tensores de PyTorch
5. Crea una red neuronal con:
   - entrada de 1 neurona
   - capa oculta de 64 neuronas
   - capa oculta de 32 neuronas
   - salida de 1 neurona
6. Entrena con Adam y MSELoss durante 200 épocas
7. Evalúa con R2, MAE y RMSE

Para ejecutar:
python main.py

Nota:
El código intenta usar la ruta absoluta /workspace/student_scores.csv, que es la que pide el enunciado.
Si no existe, usa el CSV local de esta carpeta.
