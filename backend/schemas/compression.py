from datetime import datetime

from pydantic import BaseModel

from schemas.models import CalculatorResult


class CompressionRequest(BaseModel):
    model_id: str
    width_pruning_pct: float = 0.0
    depth_pruning_pct: float = 0.0
    do_pruning: bool = True
    do_distillation: bool = False
    do_quantization: bool = False
    dataset_path: str | None = None
    enable_mmlu: bool = False


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime


class CompressionResult(BaseModel):
    job_id: str
    download_url: str
    compression_info: CalculatorResult
