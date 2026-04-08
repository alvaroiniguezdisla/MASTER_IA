from pathlib import Path
import re
import difflib

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = set(ENGLISH_STOP_WORDS)
DATA_CANDIDATES = [
    Path('/workspace/tmdb_5000_movies.csv'),
    Path('tmdb_5000_movies.csv'),
    Path('/mnt/data/tmdb_5000_movies.csv'),
]


def find_data_path():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError('No se encontró tmdb_5000_movies.csv')


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]
    return tokens


def load_and_prepare_data():
    data_path = find_data_path()
    movies = pd.read_csv(data_path)
    movies = movies[['title', 'overview']].copy()
    movies = movies.dropna(subset=['overview'])
    movies = movies[movies['overview'].str.strip() != '']
    movies = movies.drop_duplicates(subset=['title']).reset_index(drop=True)
    movies['tokens'] = movies['overview'].apply(preprocess_text)
    movies = movies[movies['tokens'].apply(len) > 0].reset_index(drop=True)
    return movies, data_path


def train_word2vec(tokenized_texts):
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=200,
        window=8,
        min_count=2,
        workers=1,
        sg=1,
        epochs=20,
        seed=42,
    )
    return model


def overview_to_vector(tokens, embedding_model):
    vectors = [embedding_model.wv[word] for word in tokens if word in embedding_model.wv]
    if not vectors:
        return np.zeros(embedding_model.vector_size, dtype=float)
    return np.mean(vectors, axis=0)


def build_movie_vectors(movies, embedding_model):
    matrix = np.vstack(movies['tokens'].apply(lambda tokens: overview_to_vector(tokens, embedding_model)).values)
    return matrix


def find_title_index(movies, title):
    matches = movies.index[movies['title'].str.lower() == title.lower()].tolist()
    if matches:
        return matches[0]

    suggestions = difflib.get_close_matches(title, movies['title'].tolist(), n=5, cutoff=0.5)
    if suggestions:
        raise ValueError(f"Película no encontrada. Prueba con una de estas: {', '.join(suggestions)}")
    raise ValueError('Película no encontrada en el dataset.')


def recommend_movies(title, movies, movie_vectors, top_n=10):
    movie_index = find_title_index(movies, title)
    similarities = cosine_similarity([movie_vectors[movie_index]], movie_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    sorted_indices = [index for index in sorted_indices if index != movie_index][:top_n]

    recommendations = movies.loc[sorted_indices, ['title']].copy()
    recommendations['similarity'] = similarities[sorted_indices]
    recommendations = recommendations.reset_index(drop=True)
    return recommendations


def vocabulary_report(tokens, embedding_model):
    known_words = [word for word in tokens if word in embedding_model.wv]
    unknown_words = [word for word in tokens if word not in embedding_model.wv]
    return {
        'total_words': len(tokens),
        'known_words': len(known_words),
        'unknown_words': len(unknown_words),
    }


def main():
    movies, data_path = load_and_prepare_data()
    embedding_model = train_word2vec(movies['tokens'].tolist())
    movie_vectors = build_movie_vectors(movies, embedding_model)

    print('Ruta de datos usada:', data_path)
    print('Películas usadas:', len(movies))
    print('Tamaño del vocabulario:', len(embedding_model.wv))
    print('Dimensión de los vectores:', embedding_model.vector_size)
    print()

    sample_title = 'Star Wars'
    sample_index = find_title_index(movies, sample_title)
    report = vocabulary_report(movies.loc[sample_index, 'tokens'], embedding_model)
    print('Control de palabras fuera de vocabulario para:', sample_title)
    print(report)
    print()

    example_titles = [
        'Star Wars',
        'Batman',
        'Shrek 2',
    ]

    for title in example_titles:
        print(f'Recomendaciones para {title}:')
        print(recommend_movies(title, movies, movie_vectors, top_n=10))
        print()


if __name__ == '__main__':
    main()
