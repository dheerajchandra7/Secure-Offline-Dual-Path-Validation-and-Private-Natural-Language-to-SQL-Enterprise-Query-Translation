# Model registry and inference integration

## What this phase adds

- An allow-listed catalog for the traditional ML, TF-IDF deep-learning,
  transformer-embedding and proposed BGE-ANC model families.
- Authenticated model-list and model-detail endpoints.
- A bounded batch-prediction request contract (1–1,000 records).
- A trusted adapter interface and artifact-path confinement. The API never
  imports Python modules, executes files, or accepts artifact paths from users.
- Explicit readiness checks for every required model and preprocessing file.

## Current readiness

The catalog intentionally reports a model as unavailable until both its full
inference bundle and a trusted adapter are present.

| Family | Repository artifacts | Current blocker |
|---|---|---|
| Traditional ML | Vectorizer/scaler artifacts are present; classifier checkpoints are not in the root `models/` directory | Classifier checkpoint and inference manifest |
| TF-IDF deep learning | Keras checkpoints, vectorizer, scaler and label encoder are present under `TF_IDF_NN/` | Inference manifest describing exact input columns, text preprocessing and numeric feature order |
| Transformer embedding classifiers | Classifier pickle files and cached embeddings are present under `modelbert/` | Local transformer encoder bundle and inference manifest |
| Proposed BGE-ANC | Training source is present | Trained checkpoint, metadata, label encoder and local BGE encoder bundle |

No prediction endpoint is marked ready in this commit. This is deliberate:
the training notebooks do not yet package a complete, reproducible inference
pipeline for each model. Returning a prediction with guessed preprocessing
would be misleading.

## Important feature-timing constraint

The benchmark contains fields such as `Generated_SQL`, `Expected_SQL`,
execution row counts, exact/value/row match scores and confidence metrics.
Some are only known after SQL generation or execution. A model intended to
predict execution success before those steps must not consume post-execution
fields. Each future adapter must declare and validate its feature schema and
prediction timing before it can be registered.

## Artifact root

Set `MODEL_ARTIFACT_ROOT` to the directory containing the repository's
model folders. For local development from `backend/`, the default is `../`.
Docker Compose mounts the artifact folders read-only under
`/app/model-artifacts`.

## API

- `GET /api/v1/models` — authenticated model catalog and readiness.
- `GET /api/v1/models/{model_key}` — model details and readiness.
- `POST /api/v1/models/{model_key}/predict` — authenticated batch prediction.
  Returns HTTP 409 while the model's artifacts or trusted adapter are missing.

Prediction request:

```json
{
  "records": [
    {
      "NL_Query": "Show all sales from the North region"
    }
  ]
}
```

The request example documents the envelope only; it is not a claim that the
current models accept that partial feature set.
