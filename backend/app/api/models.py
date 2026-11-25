from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.models import (
    ModelCatalogItem,
    PredictionRequest,
    PredictionResponse,
)
from app.services.model_registry import (
    InvalidPredictionInput,
    ModelInferenceError,
    ModelNotFound,
    ModelUnavailable,
    get_model_catalog,
    get_model_spec,
    run_prediction,
)

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelCatalogItem])
def list_models(_: User = Depends(get_current_user)):
    """List known models and their current inference readiness."""
    return get_model_catalog()


@router.get("/{model_key}", response_model=ModelCatalogItem)
def model_details(model_key: str, _: User = Depends(get_current_user)):
    try:
        get_model_spec(model_key)
    except ModelNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    item = next(item for item in get_model_catalog() if item["key"] == model_key)
    return item


@router.post("/{model_key}/predict", response_model=PredictionResponse)
def predict(
    model_key: str,
    payload: PredictionRequest,
    _: User = Depends(get_current_user),
):
    try:
        predictions = run_prediction(model_key, payload.records)
    except ModelNotFound:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    except InvalidPredictionInput as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except ModelUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Model is not ready for inference", "reason": str(exc)},
        )
    except ModelInferenceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": "Model inference is temporarily unavailable", "reason": str(exc)},
        )
    return PredictionResponse(model_key=model_key, predictions=predictions)
