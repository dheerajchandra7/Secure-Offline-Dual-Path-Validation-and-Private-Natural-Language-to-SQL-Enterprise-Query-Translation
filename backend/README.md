# Backend quick start

## 1. Configure environment

From the repository root:

    cp .env.example .env
    cp backend/.env.example backend/.env

Replace every placeholder. Use a unique random value for POSTGRES_PASSWORD and JWT_SECRET_KEY. Set INITIAL_ADMIN_PASSWORD to a strong password of at least 12 characters. Never commit either .env file.

## 2. Start PostgreSQL

    docker compose up -d db

## 3. Install and migrate

    cd backend
    python -m venv .venv
    # Activate the virtual environment, then:
    pip install -r requirements.txt
    alembic upgrade head

## 4. Create the first administrator

    python -m app.cli bootstrap-admin

The command is idempotent: if an administrator already exists, it does not create another one. Set the initial credentials in backend/.env before running it.

## 5. Start the API

    uvicorn app.main:app --reload

Health: http://localhost:8000/health
API docs: http://localhost:8000/docs

## Current endpoints

- POST /api/v1/auth/register — create a pending user account.
- POST /api/v1/auth/login — authenticate an active user and set session cookies.
- GET /api/v1/auth/me — return the current user.
- POST /api/v1/auth/logout — clear session cookies.
- GET /api/v1/admin/users/pending — list pending accounts (admin only).
- POST /api/v1/admin/users/{id}/approve — approve an account (admin only).
- POST /api/v1/admin/users/{id}/reject — reject an account (admin only).

The frontend must read the non-HttpOnly csrf_token cookie and send its value as X-CSRF-Token on cookie-authenticated POST/PATCH/DELETE requests. Use HTTPS and COOKIE_SECURE=true in production.
