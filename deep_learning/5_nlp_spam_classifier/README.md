# 5_nlp_spam_classifier

Proyecto de clasificación de SMS spam con NLP.

## Archivos

- `reto_nlp_spam_classifier.ipynb`: notebook principal.
- `solution.py`: versión script del proyecto.
- `data/spam.csv`: copia local del dataset para ejecutar fuera del entorno del campus.
- `requirements.txt`: librerías usadas.

## Ruta del dataset

La solución usa como ruta principal:

`/workspace/spam.csv`

Si esa ruta no existe, usa la copia local `data/spam.csv`.

## Qué hace

1. Carga y limpia el dataset.
2. Renombra las columnas a `label` y `message`.
3. Preprocesa el texto:
   - minúsculas
   - eliminación de caracteres especiales y números
   - tokenización
   - eliminación de stopwords
4. Convierte el texto con TF-IDF (`max_features=5000`).
5. Divide los datos en entrenamiento y validación (80/20).
6. Entrena una regresión logística.
7. Evalúa con accuracy, precision, recall, F1-score y matriz de confusión.

## Ejecución

Notebook:
- Abrir `reto_nlp_spam_classifier.ipynb` y ejecutar todas las celdas.

Script:
```bash
python solution.py
```
