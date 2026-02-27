import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from schemas.models import (
    CalculatorRequest,
    CalculatorResult,
    ModelDetectRequest,
    ModelInfo,
)
from services.calculator import calculate_compression_targets
from services.detector import detect_and_validate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class DetectResponse(BaseModel):
    model_info: ModelInfo
    calculator_result: CalculatorResult


@router.post("/detect", response_model=DetectResponse)
def detect_model(body: ModelDetectRequest) -> DetectResponse:
    try:
        config = detect_and_validate(body.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    architecture = config.get("architectures", [None])[0]
    calc = calculate_compression_targets(config, 0.0, 0.0)

    model_info = ModelInfo(
        architecture=architecture,
        total_params_B=calc["original"]["total_params_B"],
        supported=True,  # detect_and_validate raises if unsupported
    )
    result = CalculatorResult(**calc)

    logger.info(f"[detect] {body.model_id} → {architecture}, {model_info.total_params_B}B params")
    return DetectResponse(model_info=model_info, calculator_result=result)


@router.post("/calculate", response_model=CalculatorResult)
def calculate(body: CalculatorRequest) -> CalculatorResult:
    try:
        calc = calculate_compression_targets(
            body.config,
            body.width_pruning_pct,
            body.depth_pruning_pct,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return CalculatorResult(**calc)
