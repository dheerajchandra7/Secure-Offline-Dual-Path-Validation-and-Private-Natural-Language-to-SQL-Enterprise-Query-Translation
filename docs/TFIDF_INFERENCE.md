# TF-IDF Keras inference adapter

## What is implemented

- Lazy-loading adapter for the existing Keras DNN, CNN, LSTM, GRU and BiGRU checkpoint formats.
- Cached model/vectorizer/label-encoder loading, with cache invalidation when artifact modification times change.
- The notebook-compatible text pipeline: lowercase, remove non-alphabetic characters, English stopword removal, WordNet noun lemmatization, then the saved CountVectorizer.
- Strict manifest validation, batch limits, exact input-field validation and model input/output dimension checks.
- Optional ML runtime dependencies are isolated in `backend/requirements-inference.txt`.

## Why the existing checkpoints are still blocked

The notebook's `get_preprocessed_data` function creates processed columns for every object column other than the target. Its `get_vectorized_data` function concatenates all processed text columns and appends all numeric columns. The benchmark dataset contains fields including:

- `processed_Expected_SQL` and `processed_Generated_SQL`
- `Generated_Rows` and `Expected_Rows`
- `Exact_Match`, `Value_Match_Pct`, `Row_Match_Pct`, `Col_Match_Pct`
- `Confidence_Score`, `Operation_Match`, `Column_Match`, `Generation_Time_s`

These are unavailable or outcome-derived at the time a user submits an NL query. They leak information about the generated SQL or its execution into a prediction of `Execution_Success`. The saved scaler and Keras input dimension also reflect the training feature matrix. The current checkpoints therefore cannot be treated as valid pre-execution models.

The adapter only accepts `NL_Query`, and verifies that the Keras input dimension matches the vectorizer output. With the current checkpoints, this check is expected to fail because the saved model was trained with additional columns. Do not add a manifest beside the old checkpoint to bypass this check.

## Next model-training change required

Create a separate, versioned training run with a leakage-free contract:

1. Input features: raw `NL_Query` only (or other explicitly available pre-execution fields).
2. Target: `Execution_Success`.
3. Exclude generated SQL, expected SQL, row counts, match metrics, confidence scores, execution timings and identifiers.
4. Fit the preprocessing/vectorizer on training data only; persist the vectorizer, label encoder, model and manifest together.
5. Record the exact input dimension and dataset version in the manifest.
6. Evaluate using a held-out split and report class-wise precision, recall and F1; compare against a majority-class baseline.

The example manifest is at `docs/templates/tfidf_inference_manifest.example.json`. It is a template for a newly trained model, not a declaration that the current checkpoint is compatible.

## Local setup

Install the base API dependencies, then the optional inference dependencies:

```bash
cd backend
pip install -r requirements.txt
pip install -r requirements-inference.txt
python -m nltk.downloader -d ./nltk_data stopwords wordnet
set NLTK_DATA=./nltk_data
```

For Linux/macOS, export `NLTK_DATA=./nltk_data`. Provision NLTK corpora during image build or mount a verified read-only corpus directory in production; the API does not download NLP data at request time.

Docker Compose enables the optional inference runtime using the `INSTALL_INFERENCE=true` build argument. This increases the image size significantly because TensorFlow is included.

## API contract

`POST /api/v1/models/tfidf-dnn/predict`

```json
{
  "records": [
    {"NL_Query": "Show all sales from the North region"}
  ]
}
```

The endpoint returns HTTP 409 while the manifest, runtime, corpora or model dimensions are incompatible. This is intentional; it will not silently guess missing features or return a misleading prediction.
