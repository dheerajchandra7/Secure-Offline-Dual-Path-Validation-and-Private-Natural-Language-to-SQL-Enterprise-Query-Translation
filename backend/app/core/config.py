from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    app_name: str = "Dual-Path Validation Platform API"
    api_v1_prefix: str = "/api/v1"
    database_url: str = Field(
        default="postgresql+psycopg://app_user:change_me@localhost:5432/dualpath"
    )
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = Field(default=30, ge=5, le=1440)
    cookie_secure: bool = True
    cookie_samesite: str = "lax"
    cors_origins: list[str] = ["http://localhost:3000"]
    initial_admin_email: str = "admin@example.com"
    initial_admin_username: str = "platform-admin"
    initial_admin_password: str = Field(min_length=12)
    upload_root: Path = Path("./var/uploads")
    max_upload_mb: int = Field(default=100, ge=1, le=2048)

    @field_validator("cookie_samesite")
    @classmethod
    def validate_samesite(cls, value: str) -> str:
        value = value.lower()
        if value not in {"lax", "strict", "none"}:
            raise ValueError("COOKIE_SAMESITE must be lax, strict, or none")
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value):
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
