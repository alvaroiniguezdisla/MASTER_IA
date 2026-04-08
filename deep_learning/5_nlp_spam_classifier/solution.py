import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path('/workspace/spam.csv')
if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / 'data' / 'spam.csv'

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return ' '.join(tokens)


def load_data(path):
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']].copy()
    df.columns = ['label', 'message']
    df['message'] = df['message'].fillna('')
    return df


def prepare_data(df):
    df = df.copy()
    df['clean_message'] = df['message'].apply(preprocess_text)
    return df


def train_model(df):
    X = df['clean_message']
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)

    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, pos_label='spam'),
        'recall': recall_score(y_val, y_pred, pos_label='spam'),
        'f1_score': f1_score(y_val, y_pred, pos_label='spam'),
        'classification_report': classification_report(y_val, y_pred),
        'confusion_matrix': confusion_matrix(y_val, y_pred, labels=['ham', 'spam']),
    }

    return vectorizer, model, metrics


def save_confusion_matrix(matrix):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['ham', 'spam'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Matriz de confusión')
    output_path = BASE_DIR / 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def predict_examples(vectorizer, model):
    examples = [
        'Congratulations, you have won a free ticket. Call now.',
        'Hi, are we still meeting for lunch today?',
        'Free entry in a weekly competition. Text WIN to 80086 now.',
    ]

    clean_examples = [preprocess_text(text) for text in examples]
    example_vectors = vectorizer.transform(clean_examples)
    predictions = model.predict(example_vectors)

    result = pd.DataFrame({
        'message': examples,
        'prediction': predictions,
    })
    return result


def main():
    print(f'Dataset usado: {DATA_PATH}')
    df = load_data(DATA_PATH)
    df = prepare_data(df)

    vectorizer, model, metrics = train_model(df)

    print('\nPrimeras filas del dataset limpio:')
    print(df[['label', 'message', 'clean_message']].head())

    print('\nMétricas:')
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")

    print('\nClassification report:')
    print(metrics['classification_report'])

    print('Matriz de confusión:')
    print(metrics['confusion_matrix'])

    output_path = save_confusion_matrix(metrics['confusion_matrix'])
    print(f'Imagen guardada en: {output_path}')

    print('\nPredicciones de ejemplo:')
    print(predict_examples(vectorizer, model))


if __name__ == '__main__':
    main()
