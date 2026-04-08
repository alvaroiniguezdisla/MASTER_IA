# Reto proyecto 2 - Recomendador de películas con word embeddings

Este proyecto implementa un sistema de recomendación de películas usando las sinopsis del dataset `tmdb_5000_movies.csv`.

## Archivos

- `reto_nlp_word_embeddings_movies.ipynb`: notebook con todo el desarrollo.
- `solution.py`: versión en script del proyecto.
- `requirements.txt`: dependencias usadas.

## Qué hace la solución

1. Carga el dataset desde una ruta absoluta.
2. Selecciona las columnas `title` y `overview`.
3. Elimina filas con sinopsis vacías o nulas.
4. Preprocesa el texto:
   - minúsculas
   - eliminación de caracteres especiales
   - tokenización
   - eliminación de stopwords
5. Entrena un modelo Word2Vec.
6. Convierte cada sinopsis en un vector usando el promedio de los embeddings de sus palabras.
7. Calcula similitud del coseno entre películas.
8. Devuelve las 10 películas más similares a partir de un título.

## Ruta de datos

La ruta principal usada es:

`/workspace/tmdb_5000_movies.csv`

Si no existe, el código intenta usar otras rutas de respaldo para facilitar la ejecución en otros entornos.
