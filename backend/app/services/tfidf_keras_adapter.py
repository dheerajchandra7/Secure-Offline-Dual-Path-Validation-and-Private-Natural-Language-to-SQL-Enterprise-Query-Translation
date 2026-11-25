"""Trusted inference adapter for the existing TF-IDF/Keras models.

The notebook trained these classifiers on benchmark rows after SQL generation
and execution. This adapter deliberately preserves that prediction stage and
the saved feature pipeline; it is not a pre-execution SQL-success predictor.
"""
from __future__ import annotations

import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from app.services.model_registry import (
    InvalidPredictionInput,
    ModelInferenceError,
    ModelSpec,
    resolve_artifact_path,
)

_ASSET_LOCK = threading.RLock()
_PREDICT_LOCK = threading.RLock()
_MAX_TEXT_LENGTH = 50_000


def _validate_records(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[str], np.ndarray]:
    text_columns = manifest["text_columns"]
    numeric_columns = manifest["numeric_columns"]
    required = set(manifest["required_input_columns"])
    ignored = set(manifest.get("ignored_columns", []))
    target = manifest["target_column"]

    merged_text: list[str] = []
    numeric_rows: list[list[float]] = []

    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise InvalidPredictionInput(f"Record {row_index} must be a JSON object.")

        missing = sorted(required.difference(record.keys()))
        if missing:
            raise InvalidPredictionInput(
                f"Record {row_index} is missing required fields: {', '.join(missing)}"
            )

        allowed = required | ignored | {target}
        extra = sorted(set(record.keys()).difference(allowed))
        if extra:
            raise InvalidPredictionInput(
                f"Record {row_index} contains unsupported fields: {', '.join(extra)}"
            )

        text_parts: list[str] = []
        for column in text_columns:
            value = record[column]
            if not isinstance(value, str):
                raise InvalidPredictionInput(
                    f"Record {row_index}: '{column}' must be a string."
                )
            if len(value) > _MAX_TEXT_LENGTH:
                raise InvalidPredictionInput(
                    f"Record {row_index}: '{column}' exceeds {_MAX_TEXT_LENGTH} characters."
                )
            text_parts.append(value)

        numeric_values: list[float] = []
        for column in numeric_columns:
            value = record[column]
            if isinstance(value, bool) or value is None:
                raise InvalidPredictionInput(
                    f"Record {row_index}: '{column}' must be a finite number."
                )
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise InvalidPredictionInput(
                    f"Record {row_index}: '{column}' must be a finite number."
                ) from exc
            if not math.isfinite(number):
                raise InvalidPredictionInput(
                    f"Record {row_index}: '{column}' must be a finite number."
                )
            numeric_values.append(number)

        # Execution_Success is accepted for labelled evaluation uploads but is
        # never included in the feature matrix. Exact_Match was not a feature
        # in the original notebook and is intentionally ignored as well.
        merged_text.append(" ".join(text_parts))
        numeric_rows.append(numeric_values)

    if not records:
        raise InvalidPredictionInput("At least one record is required.")

    return merged_text, np.asarray(numeric_rows, dtype=np.float64)


def _preprocess_texts(texts: Sequence[str]) -> list[str]:
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        processed: list[str] = []
        for value in texts:
            text = re.sub(r"[^a-zA-Z\s]", "", value.lower())
            words = [
                lemmatizer.lemmatize(word)
                for word in text.split()
                if word not in stop_words and len(word) > 2
            ]
            processed.append(" ".join(words))
        return processed
    except LookupError as exc:
        raise ModelInferenceError(
            "NLTK stopwords/wordnet data is missing. Install the backend NLP resources."
        ) from exc


@lru_cache(maxsize=5)
def _load_assets(
    model_path: str,
    vectorizer_path: str,
    scaler_path: str,
    label_encoder_path: str,
    manifest_path: str,
):
    """Load trusted, repository-owned artifacts once per worker process."""
    try:
        import joblib
        from tensorflow.keras.models import load_model

        with _ASSET_LOCK:
            with Path(manifest_path).open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            model = load_model(model_path, compile=False)
            vectorizer = joblib.load(vectorizer_path)
            scaler = joblib.load(scaler_path)
            label_encoder = joblib.load(label_encoder_path)

        if manifest.get("schema_version") != 1:
            raise ModelInferenceError("Unsupported TF-IDF inference manifest version.")
        if manifest.get("prediction_stage") != "post_execution":
            raise ModelInferenceError("Unexpected prediction stage in inference manifest.")
        if len(manifest.get("text_columns", [])) == 0:
            raise ModelInferenceError("Inference manifest has no text columns.")
        if len(manifest.get("numeric_columns", [])) == 0:
            raise ModelInferenceError("Inference manifest has no numeric columns.")
        if not hasattr(label_encoder, "classes_"):
            raise ModelInferenceError("Saved label encoder is invalid.")

        return model, vectorizer, scaler, label_encoder, manifest
    except ModelInferenceError:
        raise
    except Exception as exc:
        raise ModelInferenceError(
            "Could not load the TF-IDF/Keras inference bundle. Check artifact integrity "
            "and compatibility with the training versions of TensorFlow and scikit-learn."
        ) from exc


class TfidfKerasAdapter:
    def predict(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        artifact_root: Path,
        model: ModelSpec,
    ) -> list[dict[str, Any]]:
        if model.family != "tfidf_dl":
            raise ModelInferenceError("The TF-IDF/Keras adapter received an incompatible model.")

        model_path = resolve_artifact_path(model.artifact_path)
        vectorizer_path = resolve_artifact_path("TF_IDF_NN/vectorizer_Bag_of_Words.pkl")
        scaler_path = resolve_artifact_path("TF_IDF_NN/scaler.pkl")
        encoder_path = resolve_artifact_path("TF_IDF_NN/label_encoder.pkl")
        manifest_path = resolve_artifact_path("TF_IDF_NN/inference_manifest.json")

        loaded_model, vectorizer, scaler, label_encoder, manifest = _load_assets(
            str(model_path),
            str(vectorizer_path),
            str(scaler_path),
            str(encoder_path),
            str(manifest_path),
        )

        merged_text, numeric_values = _validate_records(records, manifest)
        processed_text = _preprocess_texts(merged_text)

        try:
            text_matrix = vectorizer.transform(processed_text)
            scaled_numeric = scaler.transform(numeric_values)
            from scipy.sparse import csr_matrix, hstack

            combined = hstack(
                [text_matrix, csr_matrix(scaled_numeric)],
                format="csr",
            )
            dense = combined.toarray().astype(np.float32, copy=False)

            input_shape = loaded_model.input_shape
            if isinstance(input_shape, list):
                if len(input_shape) != 1:
                    raise ModelInferenceError("Multi-input Keras models are not supported.")
                input_shape = input_shape[0]

            if len(input_shape) == 2:
                model_input = dense
            elif len(input_shape) == 3 and input_shape[-1] == 1:
                model_input = dense.reshape(dense.shape[0], dense.shape[1], 1)
            else:
                raise ModelInferenceError(
                    f"Unsupported Keras input shape: {input_shape!r}"
                )

            expected_features = input_shape[1]
            if expected_features is not None and dense.shape[1] != expected_features:
                raise ModelInferenceError(
                    "Feature count mismatch between saved preprocessing artifacts "
                    f"({dense.shape[1]}) and model input ({expected_features})."
                )

            with _PREDICT_LOCK:
                probabilities = np.asarray(
                    loaded_model.predict(
                        model_input,
                        batch_size=min(64, len(records)),
                        verbose=0,
                    ),
                    dtype=np.float64,
                )

            if probabilities.ndim != 2:
                raise ModelInferenceError("The Keras model did not return class probabilities.")
            if probabilities.shape[1] != len(label_encoder.classes_):
                raise ModelInferenceError(
                    "Model output classes do not match the saved label encoder."
                )
            if not np.isfinite(probabilities).all():
                raise ModelInferenceError("The model returned non-finite probabilities.")

            class_indices = np.argmax(probabilities, axis=1)
            labels = label_encoder.inverse_transform(class_indices)
            classes = [
                value.item() if hasattr(value, "item") else value
                for value in label_encoder.classes_
            ]

            results: list[dict[str, Any]] = []
            for index, (label, row_probabilities) in enumerate(zip(labels, probabilities)):
                scalar_label = label.item() if hasattr(label, "item") else label
                results.append({
                    "record_index": index,
                    "predicted_label": scalar_label,
                    "confidence": float(np.max(row_probabilities)),
                    "class_probabilities": {
                        str(class_name): float(probability)
                        for class_name, probability in zip(classes, row_probabilities)
                    },
                    "target_column": manifest["target_column"],
                    "prediction_stage": manifest["prediction_stage"],
                })
            return results
        except (InvalidPredictionInput, ModelInferenceError):
            raise
        except Exception as exc:
            raise ModelInferenceError(
                "TF-IDF/Keras inference failed. Verify the uploaded feature values and "
                "that the saved model bundle matches its training manifest."
            ) from exc
