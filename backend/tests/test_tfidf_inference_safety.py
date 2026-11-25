from pathlib import Path

import pytest

from app.services.adapters.tfidf_keras import (
    TfidfKerasAdapter,
    validate_manifest,
)
from app.services.model_registry import ModelSpec, ModelUnavailable


@pytest.fixture
def model_spec():
    return ModelSpec(
        key="tfidf-dnn",
        display_name="DNN",
        family="tfidf_dl",
        target_column="Execution_Success",
        description="test",
        artifact_path="TF_IDF_NN/DNN_model.keras",
        required_files=(),
        adapter_key="keras_tfidf",
    )


@pytest.fixture
def safe_manifest():
    return {
        "schema_version": 1,
        "model_key": "tfidf-dnn",
        "target_column": "Execution_Success",
        "prediction_stage": "pre_execution",
        "text_fields": ["NL_Query"],
        "numeric_fields": [],
        "preprocessing": {
            "lowercase": True,
            "remove_non_alphabetic": True,
            "remove_english_stopwords": True,
            "wordnet_lemmatization": True,
            "min_token_length": 3,
        },
        "vectorizer": {
            "type": "CountVectorizer",
            "max_features": 7000,
            "stop_words": "english",
        },
    }


def test_accepts_only_the_declared_pre_execution_contract(model_spec, safe_manifest):
    validate_manifest(safe_manifest, model_spec)


@pytest.mark.parametrize(
    "field",
    [
        "Generated_SQL",
        "Expected_Rows",
        "Exact_Match",
        "Value_Match_Pct",
        "Confidence_Score",
        "Generation_Time_s",
        "QueryID",
    ],
)
def test_rejects_post_execution_or_identifier_features(model_spec, safe_manifest, field):
    unsafe = {**safe_manifest, "text_fields": ["NL_Query", field]}
    with pytest.raises(ModelUnavailable):
        validate_manifest(unsafe, model_spec)


def test_rejects_non_pre_execution_stage(model_spec, safe_manifest):
    unsafe = {**safe_manifest, "prediction_stage": "post_execution"}
    with pytest.raises(ModelUnavailable):
        validate_manifest(unsafe, model_spec)


def test_rejects_numeric_benchmark_features(model_spec, safe_manifest):
    unsafe = {**safe_manifest, "numeric_fields": ["Generated_Rows"]}
    with pytest.raises(ModelUnavailable):
        validate_manifest(unsafe, model_spec)


def test_rejects_vectorizer_drift(model_spec, safe_manifest):
    unsafe = {
        **safe_manifest,
        "vectorizer": {"type": "TfidfVectorizer", "max_features": 7000, "stop_words": "english"},
    }
    with pytest.raises(ModelUnavailable):
        validate_manifest(unsafe, model_spec)


def test_missing_manifest_keeps_model_unavailable(tmp_path: Path, model_spec):
    blocker = TfidfKerasAdapter().readiness(artifact_root=tmp_path, model=model_spec)
    assert blocker is not None
    assert "manifest" in blocker.lower()
