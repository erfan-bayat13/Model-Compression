import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import get_settings
from core.exceptions import CompressionJobError
from schemas.compression import CompressionRequest, JobStatusResponse
from services.sagemaker import SageMakerHandler
from services.storage import StorageClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["compression"])


def _make_sagemaker() -> SageMakerHandler:
    s = get_settings()
    return SageMakerHandler(
        bucket=s.s3_bucket,
        role_arn=s.sagemaker_role_arn,
        image_uri=s.sagemaker_image_uri,
        instance_type=s.sagemaker_instance_type,
        region=s.aws_region,
    )


class JobLaunchResponse(BaseModel):
    job_id: str
    status: str


class DownloadResponse(BaseModel):
    download_url: str


class JobResultResponse(BaseModel):
    job_id: str
    download_url: str


@router.post("/compress", response_model=JobLaunchResponse)
def compress(body: CompressionRequest) -> JobLaunchResponse:
    try:
        job_id = _make_sagemaker().launch_job(
            model_id=body.model_id,
            width_pruning_pct=body.width_pruning_pct or 0.0,
            depth_pruning_pct=body.depth_pruning_pct or 0.0,
            do_pruning=body.do_pruning if body.do_pruning is not None else True,
            do_distillation=body.do_distillation or False,
            do_quantization=body.do_quantization or False,
            dataset_path=body.dataset_path,
            enable_mmlu=body.enable_mmlu or False,
        )
    except Exception as exc:
        logger.error(f"[compress] Failed to launch job: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(f"[compress] Launched job {job_id} for {body.model_id}")
    return JobLaunchResponse(job_id=job_id, status="InProgress")


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    try:
        info = _make_sagemaker().get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return JobStatusResponse(
        job_id=job_id,
        status=info["status"],
        created_at=info["created_at"],
    )


@router.get("/jobs/{job_id}/result", response_model=JobResultResponse)
def job_result(job_id: str) -> JobResultResponse:
    sm = _make_sagemaker()

    try:
        info = sm.get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if info["status"] != "Completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {info['status']}).",
        )

    settings = get_settings()
    url = StorageClient(region=settings.aws_region).generate_presigned_url(
        bucket=settings.s3_bucket,
        job_id=job_id,
    )
    return JobResultResponse(job_id=job_id, download_url=url)


@router.get("/jobs/{job_id}/download", response_model=DownloadResponse)
def job_download(job_id: str) -> DownloadResponse:
    sm = _make_sagemaker()

    try:
        info = sm.get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if info["status"] != "Completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed (status: {info['status']}).",
        )

    settings = get_settings()
    url = StorageClient(region=settings.aws_region).generate_presigned_url(
        bucket=settings.s3_bucket,
        job_id=job_id,
    )
    return DownloadResponse(download_url=url)
