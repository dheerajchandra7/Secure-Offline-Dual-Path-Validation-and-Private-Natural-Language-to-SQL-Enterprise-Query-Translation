from typing import Any

from pydantic import BaseModel, Field


class ModelCatalogItem(BaseModel):
    key: str
    display_name: str
    family: str
    target_column: str
    description: str
    is_proposed: bool
    is_available: bool
    unavailable_reason: str | None = None


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=128)


class PredictionResponse(BaseModel):
    model_key: str
    predictions: list[dict[str, Any]]
