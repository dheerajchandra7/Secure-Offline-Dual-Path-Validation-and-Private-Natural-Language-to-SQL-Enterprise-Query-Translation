"""Application-owned inference adapter registration."""
from app.services.model_registry import register_adapter
from app.services.tfidf_keras_adapter import TfidfKerasAdapter


def register_builtin_adapters() -> None:
    register_adapter("keras_tfidf", TfidfKerasAdapter())
