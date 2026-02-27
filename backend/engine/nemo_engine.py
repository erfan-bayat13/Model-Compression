import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default datasets pre-loaded in S3 for users without their own data.
# Wikitext is NVIDIA's own recommendation for pruning calibration.
# Users can override by passing dataset_path explicitly.
# ---------------------------------------------------------------------------
DEFAULT_DATASETS = {
    ## TODO: replace these with actual S3 paths to tokenized datasets.
    "wikitext": "s3://your-bucket/datasets/wikitext/wikitext_text_document",
    "pubmed":   "s3://your-bucket/datasets/pubmed/pubmed_text_document",
}

DEFAULT_DATASET = "wikitext"  # used when user provides nothing


class NemoCompressionEngine:
    """
    Orchestrates NeMo compression scripts (gpt_prune.py, gpt_train.py, ptq.py)
    as subprocesses. Each step reads from self.current_checkpoint and writes
    to a new path, updating self.current_checkpoint so the next step
    automatically picks up where the last one left off.

    The user-facing entry point is run() — everything else is internal.
    """

    def __init__(
        self,
        nemo_script_dir: str,
        input_checkpoint: str,
        output_dir: str,
        device_count: int = 1,
    ):
        """
        Args:
            nemo_script_dir:  Path to NeMo's scripts/llm/ directory.
                              e.g. /opt/NeMo/scripts/llm
            input_checkpoint: Path to the input .nemo checkpoint.
            output_dir:       Directory where all output checkpoints are written.
            device_count:     Number of GPUs to use (passed to torchrun).
        """
        self.nemo_script_dir    = Path(nemo_script_dir)
        self.output_dir         = Path(output_dir)
        self.device_count       = device_count

        # current_checkpoint starts as the input and advances after each step.
        # output of step N automatically becomes input of step N+1.
        self.current_checkpoint = Path(input_checkpoint)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _run_command(self, cmd: list[str], step_name: str) -> None:
        """
        Single place where ALL subprocess calls happen.
        Streams stdout/stderr live (important for CloudWatch log visibility).
        Raises RuntimeError if the process exits non-zero — stops the pipeline
        immediately rather than silently continuing with a broken checkpoint.
        """
        logger.info(f"[{step_name}] Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout — one unified stream
            text=True,
        )

        for line in result.stdout.splitlines():
            logger.info(f"[{step_name}] {line}")

        if result.returncode != 0:
            raise RuntimeError(
                f"[{step_name}] NeMo script failed with exit code "
                f"{result.returncode}. Check logs above for details."
            )

        logger.info(f"[{step_name}] Completed successfully.")

    def _script(self, name: str) -> str:
        """Returns the full path to a NeMo script by filename."""
        return str(self.nemo_script_dir / name)

    def _resolve_dataset(self, dataset_path: str | None) -> str:
        """
        Resolves dataset path. If user provides one, use it.
        Otherwise fall back to our pre-loaded default in S3.
        """
        if dataset_path:
            return dataset_path
        logger.info(
            f"No dataset provided, using default: {DEFAULT_DATASETS[DEFAULT_DATASET]}"
        )
        return DEFAULT_DATASETS[DEFAULT_DATASET]

    # -----------------------------------------------------------------------
    # Compression steps
    # -----------------------------------------------------------------------

    def run_pruning(
        self,
        targets: dict,
        dataset_path: str | None = None,
        seq_length: int = 2048,
    ) -> None:
        """
        Runs gpt_prune.py with the targets dict produced by calculator.py.
        Supports width pruning, depth pruning, or both simultaneously.

        Dataset is used to compute activation-based importance scores for
        deciding which layers/heads/neurons to remove — more representative
        data = better pruning decisions. Defaults to wikitext if not provided.

        Args:
            targets:      The 'targets' sub-dict from calculate_compression_targets().
            dataset_path: Path to tokenized dataset. Falls back to default if None.
            seq_length:   Sequence length for importance scoring.
        """
        output_path   = self.output_dir / "pruned"
        resolved_data = self._resolve_dataset(dataset_path)

        cmd = [
            "python",
            self._script("gpt_prune.py"),
            "--restore_path",               str(self.current_checkpoint),
            "--save_path",                  str(output_path),
            "--data_paths",                 "1.0", resolved_data,
            "--seq_length",                 str(seq_length),
            "--target_num_layers",          str(targets["target_num_layers"]),
            "--target_hidden_size",         str(targets["target_hidden_size"]),
            "--target_ffn_hidden_size",     str(targets["target_ffn_hidden_size"]),
            "--target_num_attention_heads", str(targets["target_num_attention_heads"]),
            "--target_num_query_groups",    str(targets["target_num_query_groups"]),
        ]

        self._run_command(cmd, step_name="pruning")
        self.current_checkpoint = output_path

    def run_distillation(
        self,
        teacher_checkpoint: str,
        dataset_path: str | None = None,
        seq_length: int = 2048,
        max_steps: int = 1000,
        warmup_steps: int = 50,
        learning_rate: float = 1e-4,
        min_lr: float = 1e-5,
        global_batch_size: int = 32,
        micro_batch_size: int = 4,
        kd_config: str | None = None,
        experiment_name: str = "distillation",
    ) -> None:
        """
        Runs gpt_train.py to distill knowledge from teacher into the
        pruned student (self.current_checkpoint).

        The teacher checkpoint is the ORIGINAL unpruned model — it must stay
        alive throughout this step. This is the distillation exception to the
        cascade cleanup rule documented in the architecture doc.

        Args:
            teacher_checkpoint: Path to the original (teacher) .nemo checkpoint.
            dataset_path:       Path to tokenized dataset. Falls back to default.
            seq_length:         Sequence length for training.
            max_steps:          Training steps. More = better recovery, more cost.
            warmup_steps:       LR warmup steps.
            learning_rate:      Peak LR for distillation training.
            min_lr:             Minimum LR at end of cosine schedule.
            global_batch_size:  Global batch size across all GPUs.
            micro_batch_size:   Per-GPU micro batch size.
            kd_config:          Optional path to custom distillation config YAML.
            experiment_name:    Name tag for logs.
        """
        output_path   = self.output_dir / "distilled"
        resolved_data = self._resolve_dataset(dataset_path)

        cmd = [
            "torchrun",
            f"--nproc_per_node={self.device_count}",
            self._script("gpt_train.py"),
            "--name",             experiment_name,
            "--model_path",       str(self.current_checkpoint),  # pruned student
            "--teacher_path",     str(teacher_checkpoint),        # original teacher
            "--data_paths",       "1.0", resolved_data,
            "--seq_length",       str(seq_length),
            "--max_steps",        str(max_steps),
            "--warmup_steps",     str(warmup_steps),
            "--lr",               str(learning_rate),
            "--min_lr",           str(min_lr),
            "--gbs",              str(global_batch_size),
            "--mbs",              str(micro_batch_size),
            "--save_path",        str(output_path),
            "--limit_val_batches", "0",   # skip validation during distillation
            "--legacy_ckpt",              # required for NeMo 2.0 checkpoint format
        ]

        # optional distillation config (controls loss type etc.)
        if kd_config:
            cmd += ["--kd_config", kd_config]

        self._run_command(cmd, step_name="distillation")
        self.current_checkpoint = output_path

    def run_quantization(
        self,
        algorithm: str = "fp8",
        enable_kv_cache: bool = True,
        kv_cache_qformat: str = "fp8",
        calibration_dataset: str = "wikitext",
        generate_sample: bool = True,
    ) -> None:
        """
        Runs ptq.py (post-training quantization) on self.current_checkpoint.
        Always runs last — expects a fully trained/distilled checkpoint.
        Only needs a small calibration set to compute scaling factors,
        no full dataset required.

        Args:
            algorithm:           Quantization algorithm. fp8 is the best
                                 accuracy/speed tradeoff on modern NVIDIA GPUs.
            enable_kv_cache:     Whether to quantize the KV cache too.
                                 Reduces memory at inference time.
            kv_cache_qformat:    KV cache quantization format.
            calibration_dataset: Small dataset for computing scaling factors.
            generate_sample:     Whether to generate a sample output after
                                 quantization to verify the model still works.
        """
        output_path = self.output_dir / "quantized"

        cmd = [
            "python",
            self._script("ptq.py"),
            f"--algorithm={algorithm}",
            f"--nemo_checkpoint={str(self.current_checkpoint)}",
            f"--export_path={str(output_path)}",
            f"--calibration_dataset={calibration_dataset}",
        ]

        if enable_kv_cache:
            cmd += [
                "--enable_kv_cache",
                "--kv_cache_qformat", kv_cache_qformat,
            ]

        if generate_sample:
            cmd.append("--generate_sample")

        self._run_command(cmd, step_name="quantization")
        self.current_checkpoint = output_path

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        targets: dict,
        do_pruning: bool = True,
        do_distillation: bool = False,
        do_quantization: bool = False,
        teacher_checkpoint: str | None = None,
        dataset_path: str | None = None,
        distillation_steps: int = 1000,
        quantization_algorithm: str = "fp8",
        seq_length: int = 2048,
    ) -> str:
        """
        Orchestrates the full compression pipeline in order:
            1. Pruning   (optional but recommended first)
            2. Distillation (optional, recovers accuracy after pruning)
            3. Quantization (optional, always last)

        At least one step must be enabled.
        Distillation requires pruning to have run first (needs a student model)
        and a dataset (falls back to default wikitext if not provided).

        Returns:
            Path to the final compressed checkpoint as a string.
        """
        # --- Validate user choices before touching any files ---
        if not any([do_pruning, do_distillation, do_quantization]):
            raise ValueError(
                "Nothing to do: at least one of do_pruning, do_distillation, "
                "or do_quantization must be True."
            )

        if do_distillation and not do_pruning:
            raise ValueError(
                "Distillation requires a pruned student model. "
                "Enable do_pruning=True alongside do_distillation=True."
            )

        if do_distillation and not teacher_checkpoint:
            raise ValueError(
                "Distillation requires the original model as teacher. "
                "Provide teacher_checkpoint (path to original .nemo checkpoint)."
            )

        # --- Run pipeline in order ---
        if do_pruning:
            self.run_pruning(
                targets=targets,
                dataset_path=dataset_path,
                seq_length=seq_length,
            )

        if do_distillation:
            self.run_distillation(
                teacher_checkpoint=teacher_checkpoint,
                dataset_path=dataset_path,
                max_steps=distillation_steps,
                seq_length=seq_length,
            )

        if do_quantization:
            self.run_quantization(
                algorithm=quantization_algorithm,
            )

        logger.info(
            f"Compression pipeline complete. "
            f"Final checkpoint: {self.current_checkpoint}"
        )
        return str(self.current_checkpoint)