import ast
import os
import random
import re
from collections import Counter

import gensim.downloader as api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
MAX_VOCAB_SIZE = 8000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATHS = [
    "/workspace/tmdb_5000_movies.csv",
    "/workspace/tmdb/tmdb_5000_movies.csv",
    "/mnt/data/tmdb_5000_movies.csv",
    os.path.join(BASE_DIR, "tmdb_5000_movies.csv"),
    os.path.join(BASE_DIR, "data", "tmdb_5000_movies.csv"),
    "tmdb_5000_movies.csv",
    "data/tmdb_5000_movies.csv",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


class MovieDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


class LSTMGenreClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        num_classes: int,
        hidden_dim: int = 32,
        dense_dim: int = 64,
        dropout_rate: float = 0.2,
        freeze_embeddings: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze_embeddings,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_matrix.shape[1],
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden_state, _) = self.lstm(embedded)
        features = self.dense(hidden_state[-1])
        features = self.relu(features)
        features = self.dropout(features)
        return self.output(features)


def find_dataset_path() -> str:
    for path in DEFAULT_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No se encontró tmdb_5000_movies.csv en las rutas esperadas."
    )


def extract_primary_genre(value: str):
    try:
        genres = ast.literal_eval(value) if isinstance(value, str) else value
        if isinstance(genres, list) and genres:
            return genres[0].get("name")
    except (ValueError, SyntaxError, TypeError):
        return None
    return None


def preprocess_text(text: str, stopwords: set[str]) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
    return tokens


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)
    df["primary_genre"] = df["genres"].apply(extract_primary_genre)
    df = df.dropna(subset=["overview", "primary_genre"]).copy()
    df["overview"] = df["overview"].astype(str).str.strip()
    df = df[df["overview"] != ""].copy()

    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update({"film", "movie", "story"})

    df["tokens"] = df["overview"].apply(lambda text: preprocess_text(text, stopwords))
    df = df[df["tokens"].map(len) > 0].reset_index(drop=True)
    return df


def build_vocabulary(token_lists: list[list[str]], max_vocab_size: int = MAX_VOCAB_SIZE):
    token_counter = Counter()
    for tokens in token_lists:
        token_counter.update(tokens)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in token_counter.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    return vocab


def load_embedding_model(train_tokens: list[list[str]], vector_size: int = EMBEDDING_DIM):
    try:
        print("Intentando cargar glove-wiki-gigaword-100...")
        model = api.load("glove-wiki-gigaword-100")
        print("Embeddings cargados con GloVe.")
        return model, "GloVe"
    except Exception as error:
        print("No fue posible descargar GloVe en este entorno.")
        print(f"Se usa un respaldo local para mantener el notebook ejecutable: {error}")
        fallback_model = Word2Vec(
            sentences=train_tokens,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=1,
            epochs=5,
            seed=SEED,
        )
        return fallback_model.wv, "Word2Vec de respaldo"


def build_embedding_matrix(vocab: dict[str, int], embedding_model, embedding_dim: int = EMBEDDING_DIM):
    embedding_matrix = np.random.normal(
        loc=0.0,
        scale=0.6,
        size=(len(vocab), embedding_dim),
    ).astype(np.float32)
    embedding_matrix[0] = np.zeros(embedding_dim, dtype=np.float32)

    covered_words = 0
    for word, index in vocab.items():
        if index in (0, 1):
            continue
        if word in embedding_model:
            embedding_matrix[index] = np.asarray(embedding_model[word], dtype=np.float32)
            covered_words += 1
    coverage = covered_words / max(len(vocab) - 2, 1)
    return embedding_matrix, coverage


def tokens_to_sequences(token_lists: list[list[str]], vocab: dict[str, int], max_len: int = MAX_SEQUENCE_LENGTH):
    sequences = []
    for tokens in token_lists:
        token_ids = [vocab.get(token, 1) for token in tokens][:max_len]
        if len(token_ids) < max_len:
            token_ids += [0] * (max_len - len(token_ids))
        sequences.append(token_ids)
    return np.array(sequences, dtype=np.int64)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    epochs: int,
    device: torch.device,
    early_stopping_patience: int | None = None,
):
    history = []
    best_state = None
    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        metrics = evaluate_model(model, val_loader, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(train_losses))
        history.append(metrics)

        print(
            f"Epoch {epoch:02d} | loss={metrics['train_loss']:.4f} "
            f"| accuracy={metrics['accuracy']:.4f} "
            f"| macro_f1={metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print("Se activa early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate_model(model, val_loader, device)
    return history, final_metrics


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(batch_y.numpy())

    report = classification_report(
        all_targets,
        all_predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(all_targets, all_predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "predictions": np.array(all_predictions),
        "targets": np.array(all_targets),
        "report": report,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, output_path: str = "confusion_matrix.png"):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def print_metrics(title: str, metrics: dict, label_encoder: LabelEncoder):
    print(f"\n{title}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(
        classification_report(
            metrics["targets"],
            metrics["predictions"],
            labels=list(range(len(label_encoder.classes_))),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )


def main() -> None:
    set_seed(SEED)
    dataset_path = find_dataset_path()
    print(f"Dataset usado: {dataset_path}")

    df = load_and_prepare_data(dataset_path)
    print(f"Películas válidas: {len(df)}")
    print(f"Clases detectadas: {df['primary_genre'].nunique()}")

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["primary_genre"])

    train_df, val_df, y_train, y_val = train_test_split(
        df,
        labels,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    vocab = build_vocabulary(train_df["tokens"].tolist())
    print(f"Tamaño del vocabulario: {len(vocab)}")

    embedding_model, embedding_source = load_embedding_model(train_df["tokens"].tolist())
    embedding_matrix, coverage = build_embedding_matrix(vocab, embedding_model)
    print(f"Fuente de embeddings: {embedding_source}")
    print(f"Cobertura del vocabulario: {coverage:.2%}")

    X_train = tokens_to_sequences(train_df["tokens"].tolist(), vocab)
    X_val = tokens_to_sequences(val_df["tokens"].tolist(), vocab)

    train_dataset = MovieDataset(X_train, y_train)
    val_dataset = MovieDataset(X_val, y_val)

    device = torch.device("cpu")

    baseline_model = LSTMGenreClassifier(
        embedding_matrix=embedding_matrix,
        num_classes=len(label_encoder.classes_),
        hidden_dim=32,
        dense_dim=64,
        dropout_rate=0.2,
        freeze_embeddings=True,
    ).to(device)

    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    baseline_criterion = nn.CrossEntropyLoss()
    baseline_train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    baseline_val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("\nEntrenando modelo base...")
    baseline_history, baseline_metrics = train_model(
        baseline_model,
        baseline_train_loader,
        baseline_val_loader,
        baseline_criterion,
        baseline_optimizer,
        epochs=3,
        device=device,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimized_model = LSTMGenreClassifier(
        embedding_matrix=embedding_matrix,
        num_classes=len(label_encoder.classes_),
        hidden_dim=64,
        dense_dim=64,
        dropout_rate=0.5,
        freeze_embeddings=False,
    ).to(device)

    optimized_optimizer = torch.optim.Adam(optimized_model.parameters(), lr=0.001)
    optimized_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimized_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    optimized_val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    print("\nEntrenando modelo optimizado...")
    optimized_history, optimized_metrics = train_model(
        optimized_model,
        optimized_train_loader,
        optimized_val_loader,
        optimized_criterion,
        optimized_optimizer,
        epochs=5,
        device=device,
        early_stopping_patience=2,
    )

    print_metrics("Resultados del modelo base", baseline_metrics, label_encoder)
    print_metrics("Resultados del modelo optimizado", optimized_metrics, label_encoder)

    comparison = pd.DataFrame(
        [
            {
                "modelo": "Base",
                "accuracy": baseline_metrics["accuracy"],
                "macro_precision": baseline_metrics["macro_precision"],
                "macro_recall": baseline_metrics["macro_recall"],
                "macro_f1": baseline_metrics["macro_f1"],
                "weighted_f1": baseline_metrics["weighted_f1"],
            },
            {
                "modelo": "Optimizado",
                "accuracy": optimized_metrics["accuracy"],
                "macro_precision": optimized_metrics["macro_precision"],
                "macro_recall": optimized_metrics["macro_recall"],
                "macro_f1": optimized_metrics["macro_f1"],
                "weighted_f1": optimized_metrics["weighted_f1"],
            },
        ]
    )

    print("\nComparación de resultados")
    print(comparison)

    plot_confusion_matrix(
        optimized_metrics["targets"],
        optimized_metrics["predictions"],
        label_encoder.classes_,
    )
    comparison.to_csv("comparison_metrics.csv", index=False)
    pd.DataFrame(optimized_history).to_csv("optimized_history.csv", index=False)
    pd.DataFrame(baseline_history).to_csv("baseline_history.csv", index=False)


if __name__ == "__main__":
    main()
