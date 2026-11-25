"""Allow-listed model catalog and inference boundary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from app.core.config import settings


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    family: str
    target_column: str
    description: str
    artifact_path: str
    required_files: tuple[str, ...]
    adapter_key: str
    is_proposed: bool = False


class InferenceAdapter(Protocol):
    def readiness(self, *, artifact_root: Path, model: ModelSpec) -> str | None:
        """Return a human-readable blocker, or None when the adapter can run."""

    def predict(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        artifact_root: Path,
        model: ModelSpec,
    ) -> list[dict[str, Any]]:
        """Run inference using a trusted, application-owned implementation."""


class ModelNotFound(LookupError):
    pass


class ModelUnavailable(RuntimeError):
    pass


_ADAPTERS: dict[str, InferenceAdapter] = {}


def register_adapter(adapter_key: str, adapter: InferenceAdapter) -> None:
    if not adapter_key or adapter_key in _ADAPTERS:
        raise ValueError(f"Invalid or duplicate model adapter: {adapter_key!r}")
    _ADAPTERS[adapter_key] = adapter


def _build_catalog() -> tuple[ModelSpec, ...]:
    specs: list[ModelSpec] = []

    traditional_models = (
        "LogisticRegression", "SVM", "RandomForest", "ExtraTrees",
        "GradientBoosting", "AdaBoost", "KNN", "NaiveBayes", "XGBoost",
        "LightGBM",
    )
    for name in traditional_models:
        artifact = f"models/{name}.npz"
        specs.append(ModelSpec(
            key=f"ml-{name.lower().replace('_', '-')}",
            display_name=name,
            family="traditional_ml",
            target_column="Execution_Success",
            description="Traditional ML classifier trained on benchmark features.",
            artifact_path=artifact,
            required_files=(
                artifact,
                "models/vectorizer_Bag_of_Words.pkl",
                "models/scaler.pkl",
                "models/inference_manifest.json",
            ),
            adapter_key="sklearn_tfidf",
        ))

    deep_models = (
        ("DNN", "Dense neural network"),
        ("CNN", "Convolutional neural network"),
        ("RNN_LSTM", "LSTM recurrent neural network"),
        ("RNN_GRU", "GRU recurrent neural network"),
        ("RNN_BIGRU", "Bidirectional GRU recurrent neural network"),
    )
    for name, description in deep_models:
        artifact = f"TF_IDF_NN/{name}_model.keras"
        specs.append(ModelSpec(
            key=f"tfidf-{name.lower().replace('_', '-')}",
            display_name=name.replace("_", "-"),
            family="tfidf_dl",
            target_column="Execution_Success",
            description=description + " trained on Bag-of-Words features.",
            artifact_path=artifact,
            required_files=(
                artifact,
                "TF_IDF_NN/vectorizer_Bag_of_Words.pkl",
                "TF_IDF_NN/label_encoder.pkl",
                "TF_IDF_NN/inference_manifest.json",
            ),
            adapter_key="keras_tfidf",
        ))

    embedding_algorithms = ("GBM", "LR", "MLP", "Proposed", "QDA")
    for target in ("Execution_Success", "Level"):
        for algorithm in embedding_algorithms:
            artifact = f"modelbert/Transformer_Embeddings_{target}_{algorithm}_model.pkl"
            specs.append(ModelSpec(
                key=f"transformer-{target.lower()}-{algorithm.lower()}",
                display_name=f"Transformer Embeddings · {target} · {algorithm}",
                family="transformer_classifier",
                target_column=target,
                description="Classifier trained on cached transformer embeddings.",
                artifact_path=artifact,
                required_files=(
                    artifact,
                    "modelbert/inference_manifest.json",
                    "modelbert/encoder/config.json",
                ),
                adapter_key="transformer_embedding_classifier",
            ))

    bge_artifact = (
        "Advanced_BERT_By_Vamshi_NLP/BGE_ANC_model/"
        "BGE_ANC_Execution_Success.pt"
    )
    specs.append(ModelSpec(
        key="proposed-bge-anc",
        display_name="Proposed BGE-ANC",
        family="embedding_classifier",
        target_column="Execution_Success",
        description="BGE-Base semantic embeddings with Attention Nearest Centroid.",
        artifact_path=bge_artifact,
        required_files=(
            bge_artifact,
            "Advanced_BERT_By_Vamshi_NLP/BGE_ANC_model/BGE_ANC_metadata.json",
            "Advanced_BERT_By_Vamshi_NLP/BGE_ANC_model/BGE_ANC_label_encoder.pkl",
            "huggingface/bge-base-en-v1.5/config.json",
            "huggingface/bge-base-en-v1.5/model.safetensors",
        ),
        adapter_key="bge_anc",
        is_proposed=True,
    ))
    return tuple(specs)


CATALOG = _build_catalog()


def _artifact_root() -> Path:
    return settings.model_artifact_root.expanduser().resolve()


def resolve_artifact_path(relative_path: str) -> Path:
    """Resolve a catalog-owned path and reject traversal or symlink escape."""
    root = _artifact_root()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Model artifact path escapes the configured artifact root")
    return candidate


def _availability(spec: ModelSpec) -> tuple[bool, str | None]:
    root = _artifact_root()
    missing = [
        relative
        for relative in spec.required_files
        if not (root / relative).resolve().is_file()
    ]
    if missing:
        return False, "Missing required inference files: " + ", ".join(missing)

    adapter = _ADAPTERS.get(spec.adapter_key)
    if adapter is None:
        return False, f"No trusted inference adapter is registered for '{spec.adapter_key}'."

    blocker = adapter.readiness(artifact_root=root, model=spec)
    if blocker:
        return False, blocker
    return True, None


def get_model_catalog() -> list[dict[str, Any]]:
    result = []
    for spec in CATALOG:
        available, reason = _availability(spec)
        result.append({
            "key": spec.key,
            "display_name": spec.display_name,
            "family": spec.family,
            "target_column": spec.target_column,
            "description": spec.description,
            "is_proposed": spec.is_proposed,
            "is_available": available,
            "unavailable_reason": reason,
        })
    return result


def get_model_spec(model_key: str) -> ModelSpec:
    for spec in CATALOG:
        if spec.key == model_key:
            return spec
    raise ModelNotFound(model_key)


def run_prediction(
    model_key: str,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    spec = get_model_spec(model_key)
    available, reason = _availability(spec)
    if not available:
        raise ModelUnavailable(reason or "Model is not ready for inference.")

    adapter = _ADAPTERS[spec.adapter_key]
    return adapter.predict(records, artifact_root=_artifact_root(), model=spec)
