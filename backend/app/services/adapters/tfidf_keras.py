"""Leakage-aware Keras adapter for the project's TF-IDF neural networks.

The adapter intentionally accepts only NL_Query as a pre-execution feature.
The existing notebook trained with additional post-execution fields, so its
checkpoint will fail the feature-dimension check until a leakage-free model
bundle is produced. No training or artifact files are modified here.
"""
from __future__ import annotations

import importlib.util
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.models.model_definition import ModelDefinition  # type-only project import avoided at runtime
from app.services.model_registry import ModelSpec, ModelUnavailable


MANIFEST_NAME = "inference_manifest.json"
TEXT_FIELDS = ("NL_Query",)
SAFE_NUMERIC_FIELDS: frozenset[str] = frozenset()
FORBIDDEN_FEATURE_TOKENS = (
    "generated", "expected", "execution", "exact_match", "value_match",
    "row_match", "col_match", "confidence", "generation_time",
    "actual_rows", "result_rows", "match_pct", "sql", "queryid",
)
MAX_BATCH_SIZE = 128


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def validate_manifest(data: Mapping[str, Any], model: ModelSpec) -> None:
    if data.get("schema_version") != 1:
        raise ModelUnavailable("Unsupported inference manifest schema_version.")
    if data.get("model_key") != model.key:
        raise ModelUnavailable("Inference manifest model_key does not match the selected model.")
    if data.get("target_column") != model.target_column:
        raise ModelUnavailable("Inference manifest target_column does not match the model catalog.")
    if data.get("prediction_stage") != "pre_execution":
        raise ModelUnavailable("Only pre_execution inference manifests are accepted.")

    text_fields = data.get("text_fields")
    numeric_fields = data.get("numeric_fields")
    if not isinstance(text_fields, list) or tuple(text_fields) != TEXT_FIELDS:
        raise ModelUnavailable("Only the raw NL_Query field is approved as a text input.")
    if not isinstance(numeric_fields, list):
        raise ModelUnavailable("numeric_fields must be a JSON array.")
    if numeric_fields:
        raise ModelUnavailable(
            "No numeric benchmark columns are approved for pre-execution prediction."
        )

    all_fields = [*text_fields, *numeric_fields]
    for field in all_fields:
        if not isinstance(field, str) or not field.strip():
            raise ModelUnavailable("Feature names must be non-empty strings.")
        normalised = _normalise_name(field)
        if any(token in normalised for token in FORBIDDEN_FEATURE_TOKENS):
            raise ModelUnavailable(f"Post-execution or identifier feature is forbidden: {field}")

    preprocessing = data.get("preprocessing", {})
    expected_preprocessing = {
        "lowercase": True,
        "remove_non_alphabetic": True,
        "remove_english_stopwords": True,
        "wordnet_lemmatization": True,
        "min_token_length": 3,
    }
    if preprocessing != expected_preprocessing:
        raise ModelUnavailable(
            "Preprocessing settings must exactly match the documented notebook pipeline."
        )

    vectorizer = data.get("vectorizer", {})
    if vectorizer != {
        "type": "CountVectorizer",
        "max_features": 7000,
        "stop_words": "english",
    }:
        raise ModelUnavailable("Vectorizer settings do not match the saved Bag-of-Words pipeline.")


def _load_manifest(path: Path, model: ModelSpec) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailable(f"Unable to read inference manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelUnavailable("Inference manifest must contain a JSON object.")
    validate_manifest(data, model)
    return data


def _missing_runtime_packages() -> list[str]:
    packages = ("numpy", "scipy", "sklearn", "joblib", "nltk", "tensorflow")
    return [name for name in packages if importlib.util.find_spec(name) is None]


def _check_nltk_data() -> str | None:
    try:
        import nltk
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
    except LookupError:
        return "NLTK stopwords and wordnet corpora must be provisioned before inference."
    except ImportError:
        return "NLTK is not installed in the inference runtime."
    return None


@lru_cache(maxsize=8)
def _load_bundle(
    model_path: str,
    vectorizer_path: str,
    label_encoder_path: str,
    model_mtime_ns: int,
    vectorizer_mtime_ns: int,
    label_encoder_mtime_ns: int,
):
    # These are trusted, repository-managed artifacts. Never load user uploads here.
    import joblib
    from tensorflow.keras.models import load_model

    keras_model = load_model(model_path, compile=False)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(label_encoder_path)
    return keras_model, vectorizer, label_encoder


def _preprocess_text(value: str, stop_words: set[str], lemmatizer) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", value.lower())
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)


def _native(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


class TfidfKerasAdapter:
    def readiness(self, *, artifact_root: Path, model: ModelSpec) -> str | None:
        manifest_path = artifact_root / "TF_IDF_NN" / MANIFEST_NAME
        try:
            _load_manifest(manifest_path, model)
        except ModelUnavailable as exc:
            return str(exc)

        missing_packages = _missing_runtime_packages()
        if missing_packages:
            return "Install backend/requirements-inference.txt; missing: " + ", ".join(missing_packages)
        nltk_blocker = _check_nltk_data()
        if nltk_blocker:
            return nltk_blocker
        return None

    def predict(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        artifact_root: Path,
        model: ModelSpec,
    ) -> list[dict[str, Any]]:
        if not records or len(records) > MAX_BATCH_SIZE:
            raise ModelUnavailable(f"Batch size must be between 1 and {MAX_BATCH_SIZE}.")
        manifest = _load_manifest(artifact_root / "TF_IDF_NN" / MANIFEST_NAME, model)

        expected_fields = set(manifest["text_fields"])
        merged_text: list[str] = []
        for index, record in enumerate(records):
            if set(record) != expected_fields:
                raise ModelUnavailable(
                    f"Record {index} must contain exactly these fields: "
                    + ", ".join(sorted(expected_fields))
                )
            raw_query = record.get("NL_Query")
            if not isinstance(raw_query, str) or not raw_query.strip():
                raise ModelUnavailable(f"Record {index} has an empty or invalid NL_Query.")
            merged_text.append(raw_query)

        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer

            stop_words = set(stopwords.words("english"))
            lemmatizer = WordNetLemmatizer()
            processed = [
                _preprocess_text(text, stop_words, lemmatizer)
                for text in merged_text
            ]

            model_path = artifact_root / model.artifact_path
            vectorizer_path = artifact_root / "TF_IDF_NN" / "vectorizer_Bag_of_Words.pkl"
            label_encoder_path = artifact_root / "TF_IDF_NN" / "label_encoder.pkl"
            bundle = _load_bundle(
                str(model_path),
                str(vectorizer_path),
                str(label_encoder_path),
                model_path.stat().st_mtime_ns,
                vectorizer_path.stat().st_mtime_ns,
                label_encoder_path.stat().st_mtime_ns,
            )
            keras_model, vectorizer, label_encoder = bundle
            if vectorizer.__class__.__name__ != "CountVectorizer":
                raise ModelUnavailable("Saved vectorizer is not a CountVectorizer.")
            if vectorizer.get_params(deep=False).get("max_features") != 7000:
                raise ModelUnavailable("Saved CountVectorizer max_features differs from training.")
            if vectorizer.get_params(deep=False).get("stop_words") != "english":
                raise ModelUnavailable("Saved CountVectorizer stop_words differs from training.")

            features = vectorizer.transform(processed)
            expected_dim = len(vectorizer.vocabulary_)
            input_shape = keras_model.input_shape
            if isinstance(input_shape, list):
                raise ModelUnavailable("Multi-input Keras models are not supported by this adapter.")
            if len(input_shape) == 2:
                model_input = features.toarray().astype("float32", copy=False)
                model_dim = input_shape[-1]
            elif len(input_shape) == 3 and input_shape[-1] == 1:
                model_input = features.toarray().astype("float32", copy=False)[..., None]
                model_dim = input_shape[-2]
            else:
                raise ModelUnavailable(f"Unsupported Keras input shape: {input_shape}")

            if model_dim is not None and int(model_dim) != expected_dim:
                raise ModelUnavailable(
                    "Saved checkpoint expects "
                    f"{model_dim} features, but the approved NL_Query-only pipeline produces "
                    f"{expected_dim}. The existing notebook checkpoint includes additional "
                    "benchmark columns; retrain with the leakage-free feature contract."
                )

            probabilities = keras_model.predict(model_input, verbose=0)
            if getattr(probabilities, "ndim", 0) != 2:
                raise ModelUnavailable("Classifier output must be a 2D class-probability matrix.")
            classes = list(label_encoder.classes_)
            if probabilities.shape[1] != len(classes):
                raise ModelUnavailable("Model output classes do not match the saved label encoder.")

            predictions = []
            for index, row in enumerate(probabilities):
                class_index = int(row.argmax())
                predictions.append({
                    "record_index": index,
                    "prediction": _native(classes[class_index]),
                    "probability": float(row[class_index]),
                })
            return predictions
        except ModelUnavailable:
            raise
        except (ImportError, LookupError, OSError, ValueError, TypeError) as exc:
            raise ModelUnavailable(f"TF-IDF Keras inference failed safely: {exc}") from exc
