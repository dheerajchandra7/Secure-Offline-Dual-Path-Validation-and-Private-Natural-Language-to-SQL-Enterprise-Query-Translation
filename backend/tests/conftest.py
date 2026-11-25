import os

# Settings are constructed at import time by the application.
os.environ.setdefault("JWT_SECRET_KEY", "test-only-secret-key-with-at-least-32-characters")
os.environ.setdefault("INITIAL_ADMIN_PASSWORD", "test-only-admin-password")
