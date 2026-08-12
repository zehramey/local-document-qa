from fastapi import APIRouter, Depends

from app.api.dependencies import get_app_settings
from app.api.schemas import ModelInfoResponse, ModelListResponse
from app.core.config import Settings
from app.services.lm_studio_provider import list_available_models

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelListResponse)
def get_models(settings: Settings = Depends(get_app_settings)) -> ModelListResponse:
    models = list_available_models(base_url=settings.llm_base_url)
    return ModelListResponse(
        models=[
            ModelInfoResponse(
                model_id=model.model_id,
                model_type=model.model_type,
                quantization=model.quantization,
                state=model.state,
            )
            for model in models
        ]
    )
