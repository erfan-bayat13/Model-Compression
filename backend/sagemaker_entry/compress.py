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


def get_env(key: str, default=None, required: bool = False):
    """
    Read an environment variable. SageMaker prefixes hyperparameters with SM_HP_.
    Falls back to default if not set. Raises if required and missing.
    """
    value = os.environ.get(f"SM_HP_{key.upper()}", default)
    if required and value is None:
        raise ValueError(f"Required hyperparameter '{key}' not set.")
    return value


def parse_bool(value: str) -> bool:
    """SageMaker passes booleans as strings — 'True'/'False'."""
    return str(value).lower() in ("true", "1", "yes")


def main():
    logger.info("Starting compression entrypoint")

    # --- SageMaker standard paths ---
    # SageMaker copies S3 input to SM_CHANNEL_MODEL before the job starts
    # We write output to SM_OUTPUT_DATA_DIR and SageMaker syncs it to S3 after
    output_dir     = os.environ.get("SM_OUTPUT_DATA_DIR",  "/opt/ml/output/data")
    nemo_script_dir = "/opt/NeMo/scripts/llm"

    logger.info(f"Output dir: {output_dir}")

    # --- Read hyperparameters (set by sagemaker_handler.py) ---
    model_id               = get_env("model_id",               required=True)
    width_pruning_pct      = float(get_env("width_pruning_pct",      "0.0"))
    depth_pruning_pct      = float(get_env("depth_pruning_pct",      "0.0"))
    do_pruning             = parse_bool(get_env("do_pruning",         "True"))
    do_distillation        = parse_bool(get_env("do_distillation",    "False"))
    do_quantization        = parse_bool(get_env("do_quantization",    "False"))
    distillation_steps     = int(get_env("distillation_steps",        "1000"))
    quantization_algorithm = get_env("quantization_algorithm",        "fp8")
    seq_length             = int(get_env("seq_length",                "2048"))
    dataset_path           = get_env("dataset_path",                  None)

    logger.info(
        f"Job config: model={model_id}, "
        f"width={width_pruning_pct}, depth={depth_pruning_pct}, "
        f"pruning={do_pruning}, distillation={do_distillation}, "
        f"quantization={do_quantization}"
    )

    # --- Run compression pipeline ---
    # work_dir uses /tmp for intermediate files (nemo conversions, pruned checkpoints)
    # final HF output goes to output_dir which SageMaker syncs to S3
    engine = CompressionEngine(
        work_dir=output_dir,
        nemo_script_dir=nemo_script_dir,
        device_count=int(os.environ.get("SM_NUM_GPUS", "1")),
    )

    # Override hf_input_dir to point at SageMaker's input channel
    # (SageMaker already downloaded the HF model here from S3)

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