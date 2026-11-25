# DualPath frontend

React + TypeScript + Vite application for the Secure Offline Dual-Path Validation platform.

## Features

- Registration with administrator approval and cookie-based login.
- Protected workspace routes and administrator-only access control.
- Live model catalog populated from the FastAPI model registry.
- CSV upload, preview and inference against ready models.
- Prediction distribution and confidence visualizations with CSV export.
- CSRF header support for cookie-authenticated unsafe requests.
- Responsive dark interface with reduced-motion-friendly CSS transitions.

## Run locally

1. Start PostgreSQL and the backend API using the repository's backend instructions.
2. Ensure `backend/.env` includes `CORS_ORIGINS=["http://localhost:3000"]` and `COOKIE_SECURE=false` for local HTTP development.
3. From this directory:

```bash
npm install
npm run dev
```

Open http://localhost:3000.

Set `VITE_API_BASE_URL` if the API is not available at `http://localhost:8000/api/v1`.

## CSV inference contract

The frontend accepts CSV only and currently sends up to 1,000 rows per request. Required columns are model-specific. The existing TF-IDF/Keras models use the complete post-execution benchmark feature schema documented in `../docs/TFIDF_KERAS_INFERENCE.md`; an NL query by itself is insufficient.

The frontend never uploads model files or executes user-provided code. CSV contents are sent to the authenticated API for inference.
