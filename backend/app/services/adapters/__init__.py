from app.services.adapters.tfidf_keras import TfidfKerasAdapter
from app.services.model_registry import register_adapter


def register_builtin_adapters() -> None:
    # Called once during API startup. Runtime libraries are imported lazily.
    try:
        register_adapter("keras_tfidf", TfidfKerasAdapter())
    except ValueError as exc:
        if "duplicate model adapter" not in str(exc):
            raise
