import os

# The settings module validates required secrets at import time. These are
# isolated test-only values and never used by the application runtime.
os.environ.setdefault("JWT_SECRET_KEY", "unit-test-secret-key-that-is-long-enough-123456")
os.environ.setdefault("INITIAL_ADMIN_PASSWORD", "unit-test-admin-password")

from app.services.tfidf_keras_adapter import _validate_records
from app.services.model_registry import InvalidPredictionInput


MANIFEST = {
    "target_column": "Execution_Success",
    "text_columns": ["Level", "NL_Query", "Expected_SQL", "Generated_SQL"],
    "numeric_columns": [
        "QueryID", "Generated_Rows", "Expected_Rows", "Value_Match_Pct",
        "Row_Match_Pct", "Col_Match_Pct", "Confidence_Score",
        "Operation_Match", "Column_Match", "Generation_Time_s",
    ],
    "ignored_columns": ["Exact_Match"],
    "required_input_columns": [
        "Level", "NL_Query", "Expected_SQL", "Generated_SQL",
        "QueryID", "Generated_Rows", "Expected_Rows", "Value_Match_Pct",
        "Row_Match_Pct", "Col_Match_Pct", "Confidence_Score",
        "Operation_Match", "Column_Match", "Generation_Time_s",
    ],
}


def valid_record():
    return {
        "Level": "Simple",
        "NL_Query": "Show all sales from the North region",
        "Expected_SQL": "SELECT * FROM data WHERE Region = 'North'",
        "Generated_SQL": "SELECT * FROM data WHERE Region = 'North'",
        "QueryID": 1,
        "Generated_Rows": 206,
        "Expected_Rows": 206,
        "Value_Match_Pct": 100,
        "Row_Match_Pct": 100,
        "Col_Match_Pct": 100,
        "Confidence_Score": 100,
        "Operation_Match": 100,
        "Column_Match": 100,
        "Generation_Time_s": 28.68,
        "Execution_Success": True,
        "Exact_Match": True,
    }


def test_target_and_ignored_columns_are_not_features():
    text, numeric = _validate_records([valid_record()], MANIFEST)
    assert len(text) == 1
    assert text[0].startswith("Simple Show all sales")
    assert numeric.shape == (1, 10)


def test_missing_feature_is_rejected():
    record = valid_record()
    del record["Generated_SQL"]
    try:
        _validate_records([record], MANIFEST)
    except InvalidPredictionInput as exc:
        assert "Generated_SQL" in str(exc)
    else:
        raise AssertionError("Expected missing feature validation to fail")


def test_non_finite_numeric_value_is_rejected():
    record = valid_record()
    record["Confidence_Score"] = float("nan")
    try:
        _validate_records([record], MANIFEST)
    except InvalidPredictionInput as exc:
        assert "Confidence_Score" in str(exc)
    else:
        raise AssertionError("Expected non-finite numeric validation to fail")


def test_unsupported_fields_are_rejected():
    record = valid_record()
    record["untrusted_feature"] = "unexpected"
    try:
        _validate_records([record], MANIFEST)
    except InvalidPredictionInput as exc:
        assert "untrusted_feature" in str(exc)
    else:
        raise AssertionError("Expected unsupported feature validation to fail")
