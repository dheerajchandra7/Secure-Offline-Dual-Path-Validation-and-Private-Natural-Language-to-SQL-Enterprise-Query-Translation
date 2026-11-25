import enum
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelFamily(str, enum.Enum):
    TRADITIONAL_ML = "traditional_ml"
    TFIDF_DL = "tfidf_dl"
    TRANSFORMER_CLASSIFIER = "transformer_classifier"
    EMBEDDING_CLASSIFIER = "embedding_classifier"


class ModelDefinition(Base):
    __tablename__ = "model_definitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(160))
    family: Mapped[ModelFamily] = mapped_column(Enum(ModelFamily, name="model_family"))
    description: Mapped[str] = mapped_column(Text, default="")
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    adapter_key: Mapped[str] = mapped_column(String(100))
    target_column: Mapped[str] = mapped_column(String(100), default="Execution_Success")
    input_schema: Mapped[dict] = mapped_column(JSON, default=dict)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_proposed: Mapped[bool] = mapped_column(Boolean, default=False)
    version: Mapped[str] = mapped_column(String(40), default="1.0.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
