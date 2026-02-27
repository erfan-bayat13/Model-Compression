import logging
import time
import uuid
from pathlib import Path

import boto3
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How often to poll SageMaker for job status (seconds)
POLL_INTERVAL = 30

# SageMaker job statuses
TERMINAL_STATUSES  = {"Completed", "Failed", "Stopped"}
SUCCESS_STATUS     = "Completed"


class SageMakerHandler:
    """
    Runs the compression pipeline as a SageMaker Training Job.

    Responsibilities:
      1. Download HF weights locally and upload to S3 input prefix
      2. Launch a SageMaker Training Job pointing at that prefix
      3. Poll until the job completes or fails
      4. Return the S3 output path of the final HF checkpoint

    The actual compression logic lives in CompressionEngine (compression_engine.py)
    which runs INSIDE the SageMaker container. This class only orchestrates
    from the outside — it never touches model weights directly beyond the
    initial upload.
    """

    def __init__(
        self,
        bucket: str,
        role_arn: str,
        image_uri: str,
        instance_type: str = "ml.p3.2xlarge",
        region: str = "eu-west-1",
    ):
        """
        Args:
            bucket:        S3 bucket name for all job data.
            role_arn:      IAM role ARN SageMaker assumes to access S3, ECR, etc.
            image_uri:     ECR URI of the Docker image containing NeMo + our code.
            instance_type: SageMaker instance type. p3.2xlarge = 1x V100 GPU.
            region:        AWS region.
        """
        self.bucket        = bucket
        self.role_arn      = role_arn
        self.image_uri     = image_uri
        self.instance_type = instance_type
        self.region        = region

        self.sm_client = boto3.client("sagemaker", region_name=region)
        self.s3_client = boto3.client("s3",        region_name=region)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _generate_job_id(self) -> str:
        """
        Generate a unique job ID used as the S3 prefix and SageMaker job name.
        SageMaker job names must be unique and max 63 chars.
        Format: compression-{8 char uuid}
        """
        return f"compression-{uuid.uuid4().hex[:8]}"

    def _s3_input_prefix(self, job_id: str) -> str:
        return f"jobs/{job_id}/input"

    def _s3_output_prefix(self, job_id: str) -> str:
        return f"jobs/{job_id}/output"

    def _s3_uri(self, prefix: str) -> str:
        return f"s3://{self.bucket}/{prefix}"

    def _download_hf_model(self, model_id: str, local_dir: str) -> str:
        """
        Download HF model weights to a local directory using snapshot_download.
        This pulls the full model repo (weights, tokenizer, config) — everything
        needed for llm.import_ckpt inside the container.

        Returns the local path where weights were saved.
        """
        logger.info(f"[s3-upload] Downloading {model_id} from HuggingFace Hub")

        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            ignore_patterns=["*.md", "*.txt"],  # skip docs, only weights + config
        )

        logger.info(f"[s3-upload] Downloaded to {local_path}")
        return local_path

    def _upload_dir_to_s3(self, local_dir: str, s3_prefix: str) -> None:
        """
        Recursively upload a local directory to an S3 prefix.
        Walks the directory tree and uploads each file individually.
        SageMaker will then sync this prefix to /opt/ml/input/data/ in the container.
        """
        local_path = Path(local_dir)
        files      = list(local_path.rglob("*"))
        files      = [f for f in files if f.is_file()]

        logger.info(
            f"[s3-upload] Uploading {len(files)} files to "
            f"s3://{self.bucket}/{s3_prefix}"
        )

        for file_path in files:
            # preserve directory structure under the S3 prefix
            relative    = file_path.relative_to(local_path)
            s3_key      = f"{s3_prefix}/{relative}"

            self.s3_client.upload_file(
                Filename=str(file_path),
                Bucket=self.bucket,
                Key=s3_key,
            )

        logger.info("[s3-upload] Upload complete.")

    def _launch_training_job(
        self,
        job_id: str,
        s3_input_uri: str,
        s3_output_uri: str,
        hyperparameters: dict,
    ) -> str:
        """
        Launch a SageMaker Training Job.

        The container entrypoint runs compression_engine.py with the
        hyperparameters passed here as CLI args. SageMaker automatically:
          - Copies s3_input_uri → /opt/ml/input/data/ before the job starts
          - Copies /opt/ml/output/data/ → s3_output_uri after the job ends

        Returns the job name (same as job_id).
        """
        logger.info(f"[sagemaker] Launching training job: {job_id}")

        # SageMaker expects hyperparameter values as strings
        str_hyperparams = {k: str(v) for k, v in hyperparameters.items()}

        self.sm_client.create_training_job(
            TrainingJobName=job_id,
            RoleArn=self.role_arn,
            AlgorithmSpecification={
                "TrainingImage":     self.image_uri,
                "TrainingInputMode": "File",  # SageMaker copies S3 → local before start
            },
            InputDataConfig=[
                {
                    "ChannelName":     "model",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType":             "S3Prefix",
                            "S3Uri":                  s3_input_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            OutputDataConfig={
                "S3OutputPath": s3_output_uri,
            },
            ResourceConfig={
                "InstanceType":   self.instance_type,
                "InstanceCount":  1,
                "VolumeSizeInGB": 100,  # scratch space for intermediate checkpoints
            },
            HyperParameters=str_hyperparams,
            StoppingCondition={
                "MaxRuntimeInSeconds": 86400,  # 24h max — large models can take a while
            },
            EnableManagedSpotTraining=False,  # on-demand for reliability
        )

        logger.info(f"[sagemaker] Job launched: {job_id}")
        return job_id

    def _poll_job(self, job_name: str) -> str:
        """
        Poll SageMaker every POLL_INTERVAL seconds until the job reaches
        a terminal status (Completed, Failed, Stopped).

        Logs status updates so CloudWatch has a live view.
        Raises RuntimeError if the job fails or is stopped.

        Returns the final status string.
        """
        logger.info(f"[sagemaker] Polling job {job_name} every {POLL_INTERVAL}s")

        last_status = None

        while True:
            response = self.sm_client.describe_training_job(
                TrainingJobName=job_name
            )
            status = response["TrainingJobStatus"]

            # only log when status changes to avoid log spam
            if status != last_status:
                logger.info(f"[sagemaker] Job {job_name} status: {status}")
                last_status = status

            if status in TERMINAL_STATUSES:
                if status != SUCCESS_STATUS:
                    # pull failure reason from SageMaker response if available
                    reason = response.get("FailureReason", "No reason provided.")
                    raise RuntimeError(
                        f"SageMaker job {job_name} ended with status "
                        f"'{status}'. Reason: {reason}"
                    )
                break

            time.sleep(POLL_INTERVAL)

        logger.info(f"[sagemaker] Job {job_name} completed successfully.")
        return status

    def _get_output_path(self, job_name: str) -> str:
        """
        After a successful job, retrieve the S3 path of the output artifact.
        SageMaker stores output at: {S3OutputPath}/{job_name}/output/output.tar.gz
        But since we write directly to /opt/ml/output/data/ our output is at:
        {s3_output_prefix}/{job_name}/output/
        """
        response = self.sm_client.describe_training_job(
            TrainingJobName=job_name
        )
        # SageMaker appends job_name/output to whatever S3OutputPath we gave it
        return response["OutputDataConfig"]["S3OutputPath"] + f"/{job_name}/output/"

    def _cleanup_s3_input(self, job_id: str) -> None:
        """
        Delete the S3 input prefix once the job is done.
        The HF weights in S3 are no longer needed — the final output
        is the compressed HF model in the output prefix.
        Logs but never raises — cleanup failure should not surface to the user.
        """
        prefix = self._s3_input_prefix(job_id)
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages     = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

            objects = [
                {"Key": obj["Key"]}
                for page in pages
                for obj in page.get("Contents", [])
            ]

            if objects:
                self.s3_client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": objects},
                )
                logger.info(
                    f"[cleanup] Deleted {len(objects)} objects from "
                    f"s3://{self.bucket}/{prefix}"
                )
        except Exception as exc:
            logger.warning(f"[cleanup] Failed to clean up S3 input: {exc}")

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run_compression_job(
        self,
        model_id: str,
        local_download_dir: str = "/tmp/hf_download",
        width_pruning_pct: float = 0.0,
        depth_pruning_pct: float = 0.0,
        do_pruning: bool = True,
        do_distillation: bool = False,
        do_quantization: bool = False,
        dataset_path: str | None = None,
        distillation_steps: int = 1000,
        quantization_algorithm: str = "fp8",
        seq_length: int = 2048,
    ) -> dict:
        """
        Full orchestration from model ID to compressed HF checkpoint in S3.

        Flow:
          1. Generate unique job ID
          2. Download HF weights locally
          3. Upload to S3 input prefix
          4. Launch SageMaker Training Job
          5. Poll until complete
          6. Clean up S3 input
          7. Return output S3 path + job metadata

        Args:
            model_id:              HuggingFace model ID e.g. "meta-llama/Llama-3.1-8B"
            local_download_dir:    Temp dir for downloading HF weights before S3 upload.
            width_pruning_pct:     Fraction of layer params to remove via width pruning.
            depth_pruning_pct:     Fraction of layers to remove via depth pruning.
            do_pruning:            Whether to run pruning.
            do_distillation:       Whether to run distillation after pruning.
            do_quantization:       Whether to run quantization last.
            dataset_path:          S3 path to dataset. Uses default wikitext if None.
            distillation_steps:    Training steps for distillation.
            quantization_algorithm: Quantization algorithm (fp8, int8_sq, etc.)
            seq_length:            Sequence length for pruning/distillation.

        Returns:
            {
                "job_id":        str,   # SageMaker job name
                "s3_output_uri": str,   # S3 path to final compressed HF model
                "status":        str,   # "Completed"
            }
        """
        # Step 1: unique job ID — used for both SageMaker job name and S3 prefix
        job_id = self._generate_job_id()
        logger.info(f"[handler] Starting compression job: {job_id}")

        # Step 2: download HF weights to local disk
        local_model_path = self._download_hf_model(
            model_id=model_id,
            local_dir=f"{local_download_dir}/{job_id}",
        )

        # Step 3: upload to S3 so SageMaker can pull them into the container
        s3_input_prefix = self._s3_input_prefix(job_id)
        self._upload_dir_to_s3(
            local_dir=local_model_path,
            s3_prefix=s3_input_prefix,
        )

        # Step 4: launch the SageMaker Training Job
        # Hyperparameters are passed to the container entrypoint as CLI args
        # compression_engine.py reads these to configure its run() call
        hyperparameters = {
            "model_id":               model_id,
            "width_pruning_pct":      width_pruning_pct,
            "depth_pruning_pct":      depth_pruning_pct,
            "do_pruning":             do_pruning,
            "do_distillation":        do_distillation,
            "do_quantization":        do_quantization,
            "distillation_steps":     distillation_steps,
            "quantization_algorithm": quantization_algorithm,
            "seq_length":             seq_length,
        }

        if dataset_path:
            hyperparameters["dataset_path"] = dataset_path

        s3_input_uri  = self._s3_uri(s3_input_prefix)
        s3_output_uri = self._s3_uri(self._s3_output_prefix(job_id))

        self._launch_training_job(
            job_id=job_id,
            s3_input_uri=s3_input_uri,
            s3_output_uri=s3_output_uri,
            hyperparameters=hyperparameters,
        )

        # Step 5: poll until done
        status = self._poll_job(job_name=job_id)

        # Step 6: clean up S3 input — compressed output is all we need now
        self._cleanup_s3_input(job_id)

        # Step 7: return output location
        output_uri = self._get_output_path(job_name=job_id)

        logger.info(
            f"[handler] Job {job_id} complete. "
            f"Output at: {output_uri}"
        )

        return {
            "job_id":        job_id,
            "s3_output_uri": output_uri,
            "status":        status,
        }