# TF-IDF / Keras inference adapter

## Implemented models

The adapter supports the five saved Keras checkpoints produced by
`NLP_by_Vamshi_DL_updated.ipynb`:

- `tfidf-dnn`
- `tfidf-cnn`
- `tfidf-rnn-lstm`
- `tfidf-rnn-gru`
- `tfidf-rnn-bigru`

It reuses the existing `vectorizer_Bag_of_Words.pkl`, `scaler.pkl`,
`label_encoder.pkl` and each existing `.keras` checkpoint. It does not
retrain, rewrite or modify the notebooks or checkpoints.

## Important: prediction timing

The source notebook builds features from these text columns:
`Level`, `NL_Query`, `Expected_SQL`, `Generated_SQL`.

It also scales these numeric columns in this exact order:
`QueryID`, `Generated_Rows`, `Expected_Rows`, `Value_Match_Pct`,
`Row_Match_Pct`, `Col_Match_Pct`, `Confidence_Score`,
`Operation_Match`, `Column_Match`, `Generation_Time_s`.

The `Execution_Success` target is excluded from the feature matrix.
`Exact_Match` is not included by the notebook's preprocessing logic.

Several inputs are only known after SQL generation or execution. Therefore
this adapter is a **post-execution benchmark classifier**. It is not a
pre-execution predictor and it cannot accept only an NL query. A future
pre-execution model requires a separately trained checkpoint using only
features available before execution.

The manifest is based on the notebook's preprocessing code and the
`Dataset/benchmark_results.csv` schema. The saved scaler/vectorizer are the
source of truth for transformations. Do not reorder, rename or independently
recompute their features.

## API

`POST /api/v1/models/{model_key}/predict`

Example keys: `tfidf-dnn`, `tfidf-cnn`, `tfidf-rnn-lstm`,
`tfidf-rnn-gru`, `tfidf-rnn-bigru`.

Request envelope:

```json
{
  "records": [
    {
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
      "Generation_Time_s": 28.68
    }
  ]
}
```

The target field `Execution_Success` may be present in a labelled input row,
but is ignored. `Exact_Match` is also ignored. Unsupported feature names,
missing fields, non-finite numeric values and malformed rows return HTTP 422.

The response includes the predicted class, confidence and per-class
probabilities. Confidence is the maximum softmax output; it is not a
calibrated probability unless separately calibrated and evaluated.

## Runtime

The backend image installs CPU TensorFlow, SciPy, scikit-learn, joblib and NLTK.
The Docker build downloads the NLTK stopwords, WordNet and OMW corpora. Model
artifacts remain read-only mounts and are loaded lazily once per API worker.

Pickle/joblib files can execute code during deserialization. Only load the
trusted artifacts committed with this repository; never point the artifact
root at user uploads.

## Validation

Unit tests cover required features, ignored target fields, numeric validation
and unknown input fields. Run from `backend/` after installing requirements:

    pytest -q

An end-to-end prediction still needs to be run in the target environment
against the actual saved artifacts. Compatibility with the exact training
versions of TensorFlow/scikit-learn must be confirmed before production use.
