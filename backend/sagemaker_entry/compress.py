"""
SageMaker container entrypoint.

SageMaker Training Jobs pass hyperparameters as environment variables
prefixed with SM_HP_. This script reads those variables and calls
CompressionEngine.run() with the right arguments.

SageMaker also sets:
  SM_CHANNEL_MODEL = /opt/ml/input/data/model  (our input HF checkpoint)
  SM_OUTPUT_DATA_DIR = /opt/ml/output/data     (where we write the final HF model)
  SM_OUTPUT_DIR = /opt/ml/output               (general output dir)
"""

import logging
import os
import sys
import json

from compression_engine import CompressionEngine

# ---------------------------------------------------------------------------
# Logging — goes to CloudWatch automatically when running in SageMaker
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


HP_PATH = "/opt/ml/input/config/hyperparameters.json"


def parse_bool(value: str) -> bool:
    """SageMaker passes booleans as strings — 'True'/'False'."""
    return str(value).lower() in ("true", "1", "yes")


def load_hyperparameters() -> dict:
    """
    SageMaker Training writes hyperparameters to a JSON file.
    For local runs, this file may not exist; in that case we fall back to SM_HP_* env vars.
    """
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            return json.load(f)

    params: dict[str, str] = {}
    prefix = "SM_HP_"
    for k, v in os.environ.items():
        if k.startswith(prefix):
            params[k[len(prefix) :].lower()] = v
    return params

def get_env(key: str, params: dict, default=None, required: bool = False):
    value = params.get(key, default)
    if required and value is None:
        raise ValueError(f"Required hyperparameter '{key}' not set.")
    return value


def require_gpu() -> None:
    """
    Fail fast if the container is not running with NVIDIA GPU access.
    Set ALLOW_CPU=1 to bypass (not recommended for NeMo LLM compression).
    """
    if os.environ.get("ALLOW_CPU", "").lower() in {"1", "true", "yes"}:
        logger.warning("ALLOW_CPU is set; skipping GPU check.")
        return

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is not available in the container; cannot run compression."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected inside the container.\n"
            "- Local Docker: install NVIDIA Container Toolkit and run with `--gpus all`.\n"
            "- SageMaker: use a GPU instance type and ensure the job is scheduled on GPU capacity.\n"
            "If you *really* want to run without GPU, set ALLOW_CPU=1."
        )


def main():
    logger.info("Starting compression entrypoint")
    
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    nemo_script_dir = "/opt/NeMo/scripts/llm"
    
    # Load hyperparameters from file (SageMaker writes them here)
    params = load_hyperparameters()
    logger.info(f"Hyperparameters: {params}")
    
    model_id               = get_env("model_id",               params, required=True)
    width_pruning_pct      = float(get_env("width_pruning_pct",      params, "0.0"))
    depth_pruning_pct      = float(get_env("depth_pruning_pct",      params, "0.0"))
    do_pruning             = parse_bool(get_env("do_pruning",         params, "True"))
    do_distillation        = parse_bool(get_env("do_distillation",    params, "False"))
    do_quantization        = parse_bool(get_env("do_quantization",    params, "False"))
    distillation_steps     = int(get_env("distillation_steps",        params, "1000"))
    quantization_algorithm = get_env("quantization_algorithm",        params, "fp8")
    seq_length             = int(get_env("seq_length",                params, "2048"))
    dataset_path           = get_env("dataset_path",                  params, None)

    logger.info(
        f"Job config: model={model_id}, "
        f"width={width_pruning_pct}, depth={depth_pruning_pct}, "
        f"pruning={do_pruning}, distillation={do_distillation}, "
        f"quantization={do_quantization}"
    )

    require_gpu()

    engine = CompressionEngine(
        work_dir=output_dir,
        nemo_script_dir=nemo_script_dir,
        device_count=int(os.environ.get("SM_NUM_GPUS", "1")),
    )

    result = engine.run(
        model_id=model_id,
        width_pruning_pct=width_pruning_pct,
        depth_pruning_pct=depth_pruning_pct,
        do_pruning=do_pruning,
        do_distillation=do_distillation,
        do_quantization=do_quantization,
        dataset_path=dataset_path,
        distillation_steps=distillation_steps,
        quantization_algorithm=quantization_algorithm,
        seq_length=seq_length,
    )

    logger.info(f"Compression complete. Output: {result['hf_output_path']}")


if __name__ == "__main__":
    main()