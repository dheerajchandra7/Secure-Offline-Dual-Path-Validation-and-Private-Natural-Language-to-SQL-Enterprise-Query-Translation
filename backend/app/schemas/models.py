from typing import Any

from pydantic import BaseModel, Field, field_validator


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
    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)

    @field_validator("records")
    @classmethod
    def validate_record_objects(cls, records):
        if any(not isinstance(record, dict) for record in records):
            raise ValueError("Every prediction record must be a JSON object.")
        return records


class PredictionResponse(BaseModel):
    model_key: str
    predictions: list[dict[str, Any]]
