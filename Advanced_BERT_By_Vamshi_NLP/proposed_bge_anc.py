"""
Proposed BGE-Base + Attention Nearest Centroid (ANC) model.

Follows the function-oriented structure used in the project's BERT notebooks:
    upload_dataset()
    preprocess_data()
    feature_extraction()
    train_proposed_model()
    evaluate_model()
    Prediction()

Current benchmark configuration:
    Target: Execution_Success
    Inputs: every other column in Dataset/benchmark_results.csv

The report describes Level and Confidence_Score as additional prediction targets.
They are not trained as targets here because the current notebook configuration
explicitly uses Execution_Success as its sole target and retains every other
column as an input. Do not report multi-task outputs until those targets are
separated from the input feature set and trained/evaluated independently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


# ===============================
# Configuration & Model Directory
# ===============================

RANDOM_STATE = 42
TARGET_COLUMN = "Execution_Success"
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_DIR = Path(__file__).resolve().parent / "BGE_ANC_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("BGE_ANC")


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_file(relative_path: str | Path) -> Path:
    """Resolve a repository file independently of the notebook launch folder."""
    relative_path = Path(relative_path)
    roots = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    roots.extend(Path.cwd().resolve().parents)
    roots.extend(Path(__file__).resolve().parents)

    checked = set()
    for root in roots:
        if root in checked:
            continue
        checked.add(root)

        candidate = root / relative_path
        if candidate.is_file():
            return candidate.resolve()

        # Support launching from inside Advanced_BERT_By_Vamshi_NLP.
        candidate = root / "Advanced_BERT_By_Vamshi_NLP" / relative_path
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        f"Could not locate '{relative_path}'. Expected it relative to the "
        "repository root. Check that Dataset/benchmark_results.csv exists."
    )


# ===============================
# Dataset Reading
# ===============================

def resolve_dataset_path(dataset_path: Optional[str | Path] = None) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
        return path.resolve()
    return resolve_project_file("Dataset/benchmark_results.csv")


def upload_dataset(dataset_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the benchmark dataset and validate the configured target."""
    file_path = resolve_dataset_path(dataset_path)
    dataframe = pd.read_csv(file_path)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET_COLUMN}'.")
    if dataframe.empty:
        raise ValueError("Dataset contains no records.")

    logger.info("Loaded dataset: %s", file_path)
    logger.info("Dataset shape: %s", dataframe.shape)
    logger.info(
        "Input columns (%d): %s",
        len(dataframe.columns) - 1,
        [column for column in dataframe.columns if column != TARGET_COLUMN],
    )
    logger.info("Target distribution:\n%s", dataframe[TARGET_COLUMN].value_counts(dropna=False))
    return dataframe


# ===============================
# Dataset Preprocessing
# ===============================

def _normalize_execution_labels(values: pd.Series) -> pd.Series:
    mapping = {
        "true": "TRUE",
        "1": "TRUE",
        "1.0": "TRUE",
        "yes": "TRUE",
        "false": "FALSE",
        "0": "FALSE",
        "0.0": "FALSE",
        "no": "FALSE",
    }
    normalized = values.astype("string").str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        invalid = values[normalized.isna()].drop_duplicates().tolist()
        raise ValueError(f"Unrecognized {TARGET_COLUMN} labels: {invalid[:10]}")
    return normalized


def preprocess_data(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[List[str], np.ndarray, List[str], LabelEncoder]:
    """
    Convert every non-target column into a column-prefixed text record.

    This intentionally mirrors the existing notebook's preprocessing strategy.
    The target column is never included in the embedding input.
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Missing target column: {target_column}")

    input_dataframe = dataframe.drop(columns=[target_column]).copy()
    target_series = dataframe[target_column].copy()
    input_columns = input_dataframe.columns.tolist()

    if not input_columns:
        raise ValueError("No input columns remain after removing the target.")

    target_series = _normalize_execution_labels(target_series)
    valid_rows = target_series.notna() & input_dataframe.notna().any(axis=1)
    input_dataframe = input_dataframe.loc[valid_rows].reset_index(drop=True)
    target_series = target_series.loc[valid_rows].reset_index(drop=True)

    if input_dataframe.empty:
        raise ValueError("No valid rows remain after preprocessing.")

    # Preserve the same column-name-prefixed input representation as the notebooks.
    text_parts = []
    for column in input_columns:
        values = input_dataframe[column].fillna("").astype(str).str.strip()
        text_parts.append(column + ": " + values)

    input_text = (
        pd.concat(text_parts, axis=1)
        .agg(" | ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )

    label_encoder = LabelEncoder()
    encoded_targets = label_encoder.fit_transform(target_series.to_numpy())

    logger.info("Preprocessing completed: %d records, %d input columns.",
                len(input_text), len(input_columns))
    logger.info("Encoded target classes: %s", label_encoder.classes_.tolist())

    return input_text, encoded_targets.astype(np.int64), input_columns, label_encoder


# ===============================
# BGE-Base Semantic Feature Extraction
# ===============================

class BGEBaseEmbedder:
    """BGE-Base encoder using CLS pooling and L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = BGE_MODEL_NAME,
        device: torch.device = DEVICE,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("Cannot extract embeddings from an empty text collection.")

        batches = []
        for start in tqdm(range(0, len(texts), batch_size), desc="BGE-Base embeddings"):
            batch_texts = list(texts[start:start + batch_size])
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)

            # BGE model card recommends CLS pooling for this encoder family.
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            batches.append(embeddings.cpu().numpy().astype(np.float32))

        return np.concatenate(batches, axis=0)


def feature_extraction(
    input_text: Sequence[str],
    model_name: str = BGE_MODEL_NAME,
    model_dir: str | Path = MODEL_DIR,
    is_train: bool = True,
    batch_size: int = 16,
    local_files_only: bool = False,
) -> np.ndarray:
    """Extract BGE-Base embeddings and cache training embeddings."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_dir / "X_BGE_Base_embeddings.npy"
    cache_metadata_path = model_dir / "X_BGE_Base_embeddings.meta.json"
    fingerprint = hashlib.sha256(
        (model_name + "\\n" + "\\n".join(input_text)).encode("utf-8")
    ).hexdigest()

    if is_train and cache_path.is_file() and cache_metadata_path.is_file():
        try:
            cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
            cached = np.load(cache_path, mmap_mode="r")
            if (
                cache_metadata.get("fingerprint") == fingerprint
                and cached.shape[0] == len(input_text)
            ):
                logger.info("Loading matching cached BGE embeddings: %s", cache_path)
                return np.asarray(cached, dtype=np.float32)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Embedding cache metadata is invalid; recomputing embeddings.")
        logger.info("Embedding cache does not match current data/model; recomputing.")

    logger.info("Feature extraction method: BGE-Base | Train mode: %s", is_train)
    embedder = BGEBaseEmbedder(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    embeddings = embedder.encode(input_text, batch_size=batch_size)

    if is_train:
        np.save(cache_path, embeddings)
        cache_metadata_path.write_text(
            json.dumps({
                "fingerprint": fingerprint,
                "model_name": model_name,
                "rows": len(input_text),
                "embedding_dim": int(embeddings.shape[1]),
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("BGE embeddings cached at: %s", cache_path)

    return embeddings


# ===============================
# Proposed ANC Architecture
# ===============================

class AttentionNearestCentroid(nn.Module):
    """
    Attention Nearest Centroid classifier.

    A learnable feature-wise attention gate reweights BGE embedding dimensions.
    Class centroids are calculated only from the training embeddings. Prediction
    uses scaled cosine similarity to those centroids.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        attention_hidden_dim: int = 256,
        temperature: float = 12.0,
    ) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, embedding_dim),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.num_classes = num_classes

    def attend(self, embeddings: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(embeddings)
        attended = embeddings * attention_weights
        return F.normalize(attended, p=2, dim=1)

    def calculate_centroids(
        self,
        attended_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        centroids = []
        for class_index in range(self.num_classes):
            class_vectors = attended_embeddings[labels == class_index]
            if class_vectors.shape[0] == 0:
                raise ValueError(
                    f"Training split has no samples for class index {class_index}."
                )
            centroids.append(class_vectors.mean(dim=0))
        return F.normalize(torch.stack(centroids), p=2, dim=1)

    def forward(
        self,
        embeddings: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attend(embeddings)
        return self.temperature.clamp(min=1.0, max=50.0) * (
            attended @ centroids.transpose(0, 1)
        )


# ===============================
# Proposed Model Training
# ===============================

def train_proposed_model(
    embeddings: np.ndarray,
    labels: np.ndarray,
    model_dir: str | Path = MODEL_DIR,
    validation_size: float = 0.2,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Train the attention gate with centroid-based classification loss.

    Centroids are recomputed from the training partition only. Validation data
    is never used to construct class centroids.
    """
    set_seed()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("Embeddings must be 2-D and match the label count.")

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        raise ValueError("ANC classification requires at least two classes.")
    if counts.min() < 2:
        raise ValueError(
            "Each class needs at least two records for a stratified train/validation split."
        )

    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    X_train = torch.tensor(embeddings[train_indices], dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(labels[train_indices], dtype=torch.long, device=DEVICE)
    X_validation = torch.tensor(
        embeddings[validation_indices], dtype=torch.float32, device=DEVICE
    )
    y_validation = torch.tensor(
        labels[validation_indices], dtype=torch.long, device=DEVICE
    )

    model = AttentionNearestCentroid(
        embedding_dim=embeddings.shape[1],
        num_classes=len(classes),
    ).to(DEVICE)

    # Inverse-frequency weights reduce majority-class dominance.
    train_counts = torch.bincount(y_train, minlength=len(classes)).float()
    class_weights = train_counts.sum() / (
        len(classes) * train_counts.clamp(min=1.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "validation_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_attended = model.attend(X_train)
        train_centroids = model.calculate_centroids(train_attended, y_train)
        train_logits = model(X_train, train_centroids)
        train_loss = F.cross_entropy(
            train_logits,
            y_train,
            weight=class_weights,
        )
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Recompute centroids after the optimizer update using training rows only.
            attended_train = model.attend(X_train)
            centroids = model.calculate_centroids(attended_train, y_train)
            validation_logits = model(X_validation, centroids)
            validation_loss = F.cross_entropy(
                validation_logits,
                y_validation,
            ).item()

        history["train_loss"].append(float(train_loss.item()))
        history["validation_loss"].append(float(validation_loss))

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            logger.info(
                "Epoch %03d/%03d | train_loss=%.5f | val_loss=%.5f",
                epoch, epochs, train_loss.item(), validation_loss,
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d.", epoch)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_train_attended = model.attend(X_train)
        final_centroids = model.calculate_centroids(final_train_attended, y_train)

    artifact = {
        "state_dict": model.state_dict(),
        "centroids": final_centroids.detach().cpu(),
        "embedding_dim": int(embeddings.shape[1]),
        "num_classes": int(len(classes)),
        "class_indices": classes.tolist(),
        "model_name": BGE_MODEL_NAME,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
    }

    artifact_path = model_dir / "BGE_ANC_Execution_Success.pt"
    torch.save(artifact, artifact_path)
    joblib.dump(history, model_dir / "BGE_ANC_training_history.pkl")

    logger.info("Proposed BGE-ANC model saved: %s", artifact_path)
    return {
        "model": model,
        "centroids": final_centroids,
        "history": history,
        "artifact_path": artifact_path,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


# ===============================
# Model Loading & Evaluation
# ===============================

def load_proposed_model(
    model_dir: str | Path = MODEL_DIR,
) -> Tuple[AttentionNearestCentroid, torch.Tensor, Dict[str, Any]]:
    model_dir = Path(model_dir)
    artifact_path = model_dir / "BGE_ANC_Execution_Success.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Trained BGE-ANC artifact not found: {artifact_path}. "
            "Run train_proposed_model() first."
        )

    artifact = torch.load(artifact_path, map_location=DEVICE, weights_only=False)
    model = AttentionNearestCentroid(
        embedding_dim=artifact["embedding_dim"],
        num_classes=artifact["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    centroids = artifact["centroids"].to(DEVICE)
    return model, centroids, artifact


@torch.inference_mode()
def evaluate_model(
    model: AttentionNearestCentroid,
    centroids: torch.Tensor,
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_encoder: LabelEncoder,
) -> Dict[str, Any]:
    """Evaluate a supplied held-out split and return standard classification metrics."""
    model.eval()
    X = torch.tensor(np.asarray(embeddings), dtype=torch.float32, device=DEVICE)
    y_true = np.asarray(labels, dtype=np.int64)

    logits = model(X, centroids)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)

    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=np.arange(len(label_encoder.classes_))
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probabilities[:, 1]))

    logger.info("BGE-ANC evaluation:\n%s", json.dumps(result, indent=2))
    return result


# ===============================
# Prediction
# ===============================

@torch.inference_mode()
def predict_embeddings(
    embeddings: np.ndarray,
    model_dir: str | Path = MODEL_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    model, centroids, artifact = load_proposed_model(model_dir)
    X = torch.tensor(np.asarray(embeddings), dtype=torch.float32, device=DEVICE)
    logits = model(X, centroids)
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    class_indices = probabilities.argmax(axis=1)
    return np.asarray(artifact["class_indices"])[class_indices], probabilities


def Prediction(
    test_filename: Optional[str | Path] = None,
    output_path: Optional[str | Path] = None,
    model_dir: str | Path = MODEL_DIR,
    batch_size: int = 16,
) -> pd.DataFrame:
    """
    Predict Execution_Success for a CSV.

    The inference CSV may omit Execution_Success. Its non-target columns must
    match the input columns saved during training.
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "BGE_ANC_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. Train the model first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_columns = metadata["input_columns"]

    test_path = resolve_dataset_path(test_filename) if test_filename else resolve_dataset_path()
    dataframe = pd.read_csv(test_path)
    inference_frame = dataframe.drop(columns=[TARGET_COLUMN], errors="ignore").copy()

    missing = [column for column in input_columns if column not in inference_frame.columns]
    if missing:
        raise ValueError(f"Inference CSV is missing required input columns: {missing}")

    inference_frame = inference_frame.loc[:, input_columns]
    if inference_frame.empty:
        raise ValueError("Inference CSV contains no records.")

    text_parts = []
    for column in input_columns:
        values = inference_frame[column].fillna("").astype(str).str.strip()
        text_parts.append(column + ": " + values)
    input_text = (
        pd.concat(text_parts, axis=1)
        .agg(" | ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )

    embeddings = feature_extraction(
        input_text,
        model_name=metadata["embedding_model"],
        model_dir=model_dir,
        is_train=False,
        batch_size=batch_size,
    )
    predicted_labels, probabilities = predict_embeddings(embeddings, model_dir)

    result = dataframe.copy()
    result[f"Predicted_{TARGET_COLUMN}"] = predicted_labels
    result[f"{TARGET_COLUMN}_Probability"] = probabilities.max(axis=1)

    labels = metadata["label_classes"]
    for class_index, label in enumerate(labels):
        result[f"Probability_{label}"] = probabilities[:, class_index]

    destination = (
        Path(output_path)
        if output_path
        else model_dir / "BGE_ANC_prediction_results.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    logger.info("Prediction results saved: %s", destination.resolve())
    return result


# ===============================
# End-to-End Training Workflow
# ===============================

def run_training_pipeline(
    dataset_path: Optional[str | Path] = None,
    model_dir: str | Path = MODEL_DIR,
    embedding_batch_size: int = 16,
    epochs: int = 100,
) -> Dict[str, Any]:
    """
    Run dataset loading, preprocessing, BGE embeddings, ANC training and evaluation.

    The returned validation metrics are computed on the held-out validation split.
    For a final unbiased test result, keep a separate untouched test dataset.
    """
    set_seed()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataframe = upload_dataset(dataset_path)
    input_text, labels, input_columns, label_encoder = preprocess_data(dataframe)

    # Persist inference schema and label mapping.
    metadata = {
        "target_column": TARGET_COLUMN,
        "input_columns": input_columns,
        "label_classes": label_encoder.classes_.tolist(),
        "embedding_model": BGE_MODEL_NAME,
        "pooling": "CLS",
        "architecture": "BGE-Base + feature-wise attention + nearest-centroid classifier",
    }
    (model_dir / "BGE_ANC_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    joblib.dump(label_encoder, model_dir / "BGE_ANC_label_encoder.pkl")

    embeddings = feature_extraction(
        input_text,
        model_name=BGE_MODEL_NAME,
        model_dir=model_dir,
        is_train=True,
        batch_size=embedding_batch_size,
    )

    training = train_proposed_model(
        embeddings,
        labels,
        model_dir=model_dir,
        epochs=epochs,
    )

    validation_indices = training["validation_indices"]
    validation_metrics = evaluate_model(
        training["model"],
        training["centroids"],
        embeddings[validation_indices],
        labels[validation_indices],
        label_encoder,
    )

    metrics_path = model_dir / "BGE_ANC_validation_metrics.json"
    metrics_path.write_text(
        json.dumps(validation_metrics, indent=2),
        encoding="utf-8",
    )

    logger.info("Training pipeline completed.")
    return {
        "metrics": validation_metrics,
        "artifact_path": str(training["artifact_path"]),
        "metrics_path": str(metrics_path),
        "model_dir": str(model_dir.resolve()),
    }


if __name__ == "__main__":
    # Run: python proposed_bge_anc.py
    run_training_pipeline()
