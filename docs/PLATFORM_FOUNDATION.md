# Platform foundation

This branch adds application scaffolding without editing existing research notebooks, datasets or model artifacts.

## Database entities
- users: password hash, role, approval status, approval actor and timestamps.
- model_definitions: allow-listed model metadata, adapter key, target and artifact location.
- inference_jobs: job lifecycle, ownership, input/result paths and summary.
- audit_logs: administrator actions and security events.

## Registration and access
Public registration creates a regular user with pending status. Pending, rejected and disabled accounts cannot log in. Active administrators approve or reject users. Public registration never accepts a role. The initial administrator is provisioned explicitly from environment variables.

## Authentication
Passwords use Argon2. Signed access tokens use an environment-provided secret. The API issues an HttpOnly access-token cookie and a readable CSRF cookie. Unsafe cookie-authenticated requests must send the CSRF value in X-CSRF-Token. Use HTTPS, COOKIE_SECURE=true, rotate secrets and apply ingress rate limiting in production.

## Model integration boundary
The registry stores metadata and adapter identifiers. It must not dynamically execute arbitrary user-supplied Python or paths. Adapters will be allow-listed. BGE-ANC remains unavailable until its trained checkpoint, metadata and label encoder are present and verified.

## Dataset safety
The benchmark contains post-execution fields such as generated/expected SQL, execution results and exact/value/row match metrics. These must not be accepted as features for predictions intended before SQL execution. Each adapter must declare its input schema and target timing.

## Local development
Copy backend/.env.example to backend/.env, configure secrets and PostgreSQL, install backend/requirements.txt, apply Alembic migrations, then run from backend:

    uvicorn app.main:app --reload

Health endpoint: /health. Interactive docs are enabled outside production.
