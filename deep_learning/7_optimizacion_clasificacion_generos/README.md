# Reto proyecto - Optimiza un modelo de clasificación

Este proyecto predice el género principal de una película a partir de la sinopsis usando el dataset `TMDB 5000 Movie Dataset`.

## Qué incluye

- `reto_optimizacion_clasificacion_generos.ipynb`: notebook principal.
- `solution.py`: versión en script del proyecto.
- `requirements.txt`: dependencias.

## Requisitos cubiertos

- Carga del archivo desde ruta absoluta.
- Limpieza y preprocesamiento del texto.
- Extracción del género principal desde la columna `genres`.
- Uso de embeddings preentrenados con `glove-wiki-gigaword-100` mediante `gensim.downloader`.
- Modelo de clasificación con LSTM, capas densas, dropout y salida softmax.
- Optimización con Adam, dropout, ajuste de batch size y ponderación de clases.
- Evaluación con accuracy, precision, recall, F1-score y reporte por clase.
- Comparación entre un modelo base y un modelo optimizado.

## Nota importante sobre los embeddings

La ruta principal del código usa:

```python
api.load("glove-wiki-gigaword-100")
```

Si el entorno no puede descargar GloVe, el código activa un respaldo local para que el notebook siga siendo ejecutable. Esto se ha dejado así para que funcione tanto dentro como fuera del portal.

## Ruta del dataset

El proyecto busca el archivo en estas rutas:

- `/workspace/tmdb_5000_movies.csv`
- `/workspace/tmdb/tmdb_5000_movies.csv`

También tiene rutas de respaldo para pruebas locales.

## Ejecución

En notebook, ejecuta todas las celdas en orden.

En script:

```bash
python solution.py
```
