import logging
import shutil
from pathlib import Path

from nemo.collections import llm

from backend.engine.detector import detect_and_validate, SUPPORTED_ARCHITECTURES
from backend.engine.nemo_engine import NemoCompressionEngine
from backend.calculator import calculate_compression_targets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture → NeMo model class mapping
# When detector.py returns e.g. "LlamaForCausalLM" we look up the right
# NeMo model class to pass into import_ckpt / export_ckpt.
# ---------------------------------------------------------------------------
ARCHITECTURE_TO_NEMO_MODEL = {
    "LlamaForCausalLM":   llm.LlamaModel,
    "MistralForCausalLM": llm.MistralModel,
    "MixtralForCausalLM": llm.MixtralModel,
    "Qwen2ForCausalLM":   llm.Qwen2Model,
}


class CompressionEngine:
    """
    Top-level orchestrator for the full compression pipeline.

    Handles:
      1. Architecture detection and validation (detector.py)
      2. Compression target calculation (calculator.py)
      3. HuggingFace → NeMo format conversion
      4. Compression via NemoCompressionEngine (nemo_engine.py)
      5. NeMo → HuggingFace format conversion
      6. Cascade cleanup of intermediate checkpoints

    The user provides a HuggingFace model ID and gets a HuggingFace
    model back. All NeMo conversion is an internal implementation detail.

    Directory layout under work_dir for a given job:
        work_dir/
            hf_input/       ← downloaded HF weights (deleted after → .nemo)
            nemo_input/     ← converted .nemo (deleted after compression)
            nemo_output/    ← compressed .nemo steps (deleted after → HF)
            hf_output/      ← final HF checkpoint (returned to caller)
    """

    def __init__(
        self,
        work_dir: str,
        nemo_script_dir: str = "/opt/NeMo/scripts/llm",
        device_count: int = 1,
        alignment: int = 16,
    ):
        """
        Args:
            work_dir:        Root directory for all intermediate and final files.
                             In production this is the SageMaker job's local
                             scratch space, which maps to an S3 prefix.
            nemo_script_dir: Path to NeMo's scripts/llm/ inside the container.
            device_count:    Number of GPUs available.
            alignment:       Dimension alignment for calculator. Default 16.
        """
        self.work_dir        = Path(work_dir)
        self.nemo_script_dir = nemo_script_dir
        self.device_count    = device_count
        self.alignment       = alignment

        # subdirectories — created lazily in run()
        self.hf_input_dir   = self.work_dir / "hf_input"
        self.nemo_input_dir = self.work_dir / "nemo_input"
        self.nemo_output_dir = self.work_dir / "nemo_output"
        self.hf_output_dir  = self.work_dir / "hf_output"

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _setup_dirs(self) -> None:
        """Create all working directories."""
        for d in [
            self.hf_input_dir,
            self.nemo_input_dir,
            self.nemo_output_dir,
            self.hf_output_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _cleanup(self, path: Path, label: str) -> None:
        """
        Delete a directory and everything under it.
        Called after each cascade step to keep peak storage at ~2x model size.
        Logs but never raises — a cleanup failure should not abort the pipeline.
        """
        try:
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"[cleanup] Deleted {label}: {path}")
        except Exception as exc:
            logger.warning(f"[cleanup] Failed to delete {label} at {path}: {exc}")

    def _hf_to_nemo(
        self,
        hf_model_path: str,
        architecture: str,
        output_path: str,
    ) -> None:
        """
        Convert a HuggingFace checkpoint to .nemo format using llm.import_ckpt.
        The model class is looked up dynamically from the architecture string
        so we never hardcode a specific model config (unlike the course example
        which used llm.LlamaModel(llm.Llama32Config3B()) for a specific size).
        Passing no config lets NeMo infer it from the checkpoint itself.
        """
        model_class = ARCHITECTURE_TO_NEMO_MODEL[architecture]

        logger.info(
            f"[hf→nemo] Converting {architecture} from {hf_model_path} "
            f"to {output_path}"
        )

        llm.import_ckpt(
            model=model_class(),       # no hardcoded config — NeMo auto-detects
            source=f"hf:///{hf_model_path}",
            output_path=output_path,
            overwrite=True,
        )

        logger.info("[hf→nemo] Conversion complete.")

    def _nemo_to_hf(
        self,
        nemo_checkpoint: str,
        architecture: str,
        output_path: str,
    ) -> None:
        """
        Convert a .nemo checkpoint back to HuggingFace format using llm.export_ckpt.
        This is always the last step — the user receives a HF model they can
        use directly with transformers, upload to Hub, etc.
        """
        model_class = ARCHITECTURE_TO_NEMO_MODEL[architecture]

        logger.info(
            f"[nemo→hf] Converting {architecture} from {nemo_checkpoint} "
            f"to {output_path}"
        )

        llm.export_ckpt(
            model=model_class(),
            source=nemo_checkpoint,
            output_path=output_path,
            target="hf",
            overwrite=True,
        )

        logger.info("[nemo→hf] Conversion complete.")

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(
        self,
        model_id: str,
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
        Full end-to-end compression pipeline.

        Args:
            model_id:              HuggingFace model ID e.g. "meta-llama/Llama-3.1-8B"
            width_pruning_pct:     Fraction of layer params to remove via width pruning.
            depth_pruning_pct:     Fraction of layers to remove via depth pruning.
            do_pruning:            Whether to run pruning.
            do_distillation:       Whether to run distillation after pruning.
            do_quantization:       Whether to run quantization last.
            dataset_path:          Optional dataset for pruning/distillation.
                                   Falls back to default wikitext in S3 if None.
            distillation_steps:    Training steps for distillation.
            quantization_algorithm: Quantization algorithm (fp8, int8_sq, etc.)
            seq_length:            Sequence length for pruning/distillation.

        Returns:
            {
                "hf_output_path":   str,   # path to final HF checkpoint
                "compression_info": dict,  # original vs targets from calculator
            }
        """
        self._setup_dirs()

        # ------------------------------------------------------------------
        # Step 1: Detect architecture and validate
        # ------------------------------------------------------------------
        logger.info(f"[pipeline] Detecting architecture for {model_id}")
        config = detect_and_validate(model_id)
        architecture = config["architectures"][0]
        logger.info(f"[pipeline] Architecture: {architecture}")

        # ------------------------------------------------------------------
        # Step 2: Calculate compression targets
        # ------------------------------------------------------------------
        logger.info("[pipeline] Calculating compression targets")
        compression_info = calculate_compression_targets(
            config=config,
            width_pruning_pct=width_pruning_pct,
            depth_pruning_pct=depth_pruning_pct,
            alignment=self.alignment,
        )
        logger.info(
            f"[pipeline] Targets: "
            f"{compression_info['original']['total_params_B']}B → "
            f"{compression_info['expected_params_B']}B "
            f"({compression_info['compression_ratio']}x)"
        )

        # ------------------------------------------------------------------
        # Step 3: Download HF weights and convert to .nemo
        # Cascade step 1: HF weights → .nemo → delete HF weights
        # ------------------------------------------------------------------
        logger.info("[pipeline] Converting HF → NeMo")
        self._hf_to_nemo(
            hf_model_path=str(self.hf_input_dir),
            architecture=architecture,
            output_path=str(self.nemo_input_dir),
        )
        # HF weights no longer needed — delete to keep storage at ~2x model size
        self._cleanup(self.hf_input_dir, "hf_input")

        # ------------------------------------------------------------------
        # Step 4: Keep a reference to the teacher checkpoint BEFORE compression
        # This is the distillation exception — teacher must stay alive
        # throughout gpt_train.py and is only deleted after distillation ends.
        # ------------------------------------------------------------------
        teacher_checkpoint = str(self.nemo_input_dir) if do_distillation else None

        # ------------------------------------------------------------------
        # Step 5: Run compression via NemoCompressionEngine
        # ------------------------------------------------------------------
        logger.info("[pipeline] Starting NeMo compression")
        engine = NemoCompressionEngine(
            nemo_script_dir=self.nemo_script_dir,
            input_checkpoint=str(self.nemo_input_dir),
            output_dir=str(self.nemo_output_dir),
            device_count=self.device_count,
        )

        engine.run(
            targets=compression_info["targets"],
            do_pruning=do_pruning,
            do_distillation=do_distillation,
            do_quantization=do_quantization,
            teacher_checkpoint=teacher_checkpoint,
            dataset_path=dataset_path,
            distillation_steps=distillation_steps,
            quantization_algorithm=quantization_algorithm,
            seq_length=seq_length,
        )

        # ------------------------------------------------------------------
        # Step 6: Cascade cleanup of input .nemo
        # Now that compression is done, the original .nemo is no longer needed
        # (distillation has finished so teacher is safe to delete too).
        # ------------------------------------------------------------------
        self._cleanup(self.nemo_input_dir, "nemo_input")

        # ------------------------------------------------------------------
        # Step 7: Convert compressed .nemo → HuggingFace
        # Cascade step 3: compressed .nemo → HF → delete compressed .nemo
        # ------------------------------------------------------------------
        logger.info("[pipeline] Converting NeMo → HF")
        self._nemo_to_hf(
            nemo_checkpoint=engine.current_checkpoint,
            architecture=architecture,
            output_path=str(self.hf_output_dir),
        )
        # Compressed .nemo no longer needed once HF export is done
        self._cleanup(self.nemo_output_dir, "nemo_output")

        logger.info(
            f"[pipeline] Complete. "
            f"Final HF checkpoint at: {self.hf_output_dir}"
        )

        return {
            "hf_output_path":   str(self.hf_output_dir),
            "compression_info": compression_info,
        }