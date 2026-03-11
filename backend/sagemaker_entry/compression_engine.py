import json
import logging
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch
from calculator import calculate_compression_targets
from engine.detector import detect_and_validate
from engine.nemo_engine import NemoCompressionEngine
from huggingface_hub import snapshot_download
from nemo.collections import llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture → NeMo model class mapping
# When detector.py returns e.g. "LlamaForCausalLM" we look up the right
# NeMo model class to pass into import_ckpt / export_ckpt.
# ---------------------------------------------------------------------------
ARCHITECTURE_TO_NEMO_MODEL = {
    "LlamaForCausalLM": llm.LlamaModel,
    "MistralForCausalLM": llm.MistralModel,
    "MixtralForCausalLM": llm.MixtralModel,
    "Qwen2ForCausalLM": llm.Qwen2Model,
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
            hf_output/      ← final HF checkpoint (returned to caller, uploaded by SageMaker)
        /tmp/
            hf_input/       ← downloaded HF weights (deleted after → .nemo)
            nemo_input/     ← converted .nemo (deleted after compression)
            nemo_output/    ← compressed .nemo steps (deleted after → HF)
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
            work_dir:        Root directory for final output files.
                             In production this is the SageMaker job's output
                             directory, which maps to an S3 prefix.
            nemo_script_dir: Path to NeMo's scripts/llm/ inside the container.
            device_count:    Number of GPUs available.
            alignment:       Dimension alignment for calculator. Default 16.
        """
        self.work_dir = Path(work_dir)
        self.nemo_script_dir = nemo_script_dir
        self.device_count = device_count
        self.alignment = alignment

        # hf_output stays under work_dir — SageMaker uploads this to S3
        self.hf_output_dir = self.work_dir / "hf_output"

        # Intermediate dirs go to /tmp/ to avoid filling the SageMaker output volume
        self.hf_input_dir    = Path("/tmp/hf_input")
        self.nemo_input_dir  = Path("/tmp/nemo_input")
        self.nemo_output_dir = Path("/tmp/nemo_output")

        # Cached HF config dict — populated during _hf_to_nemo and reused by
        # _nemo_to_hf so we don't depend on hf_input_dir still existing at
        # export time (it's deleted by the cascade cleanup in between).
        self._cached_hf_config: dict | None = None

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

    def _load_hf_config(self, hf_model_path: str) -> dict:
        """
        Read and cache the HuggingFace config.json from a local model directory.
        Cached so _nemo_to_hf can reuse it after hf_input_dir has been deleted.
        """
        if self._cached_hf_config is None:
            config_path = Path(hf_model_path) / "config.json"
            with open(config_path) as f:
                self._cached_hf_config = json.load(f)
        return self._cached_hf_config

    @contextmanager
    def _dtype_context(self, hf_config: dict):
        """
        Context manager that temporarily sets torch.set_default_dtype to match
        the dtype declared in the HF config.json.

        This is the correct way to control NeMo model tensor allocation dtype —
        none of the NeMo model constructors (LlamaModel, MistralModel,
        MixtralModel, Qwen2Model) accept a dtype argument. Without this,
        NeMo defaults to float32, which causes the dtype mismatch assertion
        in nemo/lightning/io/state.py to fire when loading bfloat16 weights.
        """
        hf_dtype_str = hf_config.get("torch_dtype", "bfloat16")
        # HF sometimes stores values like "torch.bfloat16" – normalize to "bfloat16"
        if isinstance(hf_dtype_str, str) and hf_dtype_str.startswith("torch."):
            hf_dtype_str = hf_dtype_str.split(".", 1)[1]
        dtype = getattr(torch, hf_dtype_str)  # "bfloat16" → torch.bfloat16

        prev_dtype = torch.get_default_dtype()
        logger.info(f"[dtype] Setting default dtype: {prev_dtype} → {dtype}")
        torch.set_default_dtype(dtype)
        try:
            yield dtype
        finally:
            torch.set_default_dtype(prev_dtype)
            logger.info(f"[dtype] Restored default dtype: {prev_dtype}")

    def _build_nemo_model(self, architecture: str, hf_config: dict):
        """
        Build a NeMo model instance with config read directly from the
        HuggingFace config dict. This ensures:
          - No ZeroDivisionError from empty default NeMo configs
          - Works for any model size without hardcoding

        dtype is NOT passed here — none of the NeMo model constructors accept
        it as an argument. Callers must wrap this (and the import_ckpt /
        export_ckpt calls) inside _dtype_context() so torch.set_default_dtype
        controls tensor allocation with the correct dtype.

        Args:
            architecture: HF architecture string e.g. "Qwen2ForCausalLM"
            hf_config:    Parsed config.json dict from the HF model directory
        """
        hidden_size     = hf_config["hidden_size"]
        num_layers      = hf_config["num_hidden_layers"]
        num_heads       = hf_config["num_attention_heads"]
        num_kv_heads    = hf_config.get("num_key_value_heads", num_heads)
        ffn_hidden_size = hf_config["intermediate_size"]
        vocab_size      = hf_config["vocab_size"]

        if architecture == "LlamaForCausalLM":
            from nemo.collections.llm import LlamaConfig
            config = LlamaConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                num_query_groups=num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                vocab_size=vocab_size,
            )
            return llm.LlamaModel(config)

        elif architecture == "MistralForCausalLM":
            from nemo.collections.llm import MistralConfig
            config = MistralConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                num_query_groups=num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                vocab_size=vocab_size,
            )
            return llm.MistralModel(config)

        elif architecture == "Qwen2ForCausalLM":
            from nemo.collections.llm import Qwen2Config
            config = Qwen2Config(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                num_query_groups=num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                vocab_size=vocab_size,
            )
            return llm.Qwen2Model(config)

        elif architecture == "MixtralForCausalLM":
            from nemo.collections.llm import MixtralConfig
            # Mixtral-specific fields — read from config rather than assuming defaults
            num_experts = hf_config.get("num_local_experts", 8)
            top_k       = hf_config.get("num_experts_per_tok", 2)
            config = MixtralConfig(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_heads,
                num_query_groups=num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                vocab_size=vocab_size,
                num_moe_experts=num_experts,
                moe_router_topk=top_k,
            )
            return llm.MixtralModel(config)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def _hf_to_nemo(
        self,
        model_id: str,
        hf_model_path: str,
        architecture: str,
        output_path: str,
    ) -> None:
        """
        Convert a HuggingFace checkpoint to .nemo format using llm.import_ckpt.
        Wraps the conversion in _dtype_context so NeMo allocates tensors with
        the correct dtype, avoiding the dtype mismatch assertion in state.py.
        """
        logger.info(
            f"[hf→nemo] Converting {architecture} from {hf_model_path} to {output_path}"
        )

        hf_config = self._load_hf_config(hf_model_path)

        with self._dtype_context(hf_config):
            nemo_model = self._build_nemo_model(architecture, hf_config)
            llm.import_ckpt(
                model=nemo_model,
                source=f"hf://{model_id}",
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

        Uses the cached HF config (populated during _hf_to_nemo) so this works
        even after hf_input_dir has been deleted by the cascade cleanup.
        """
        logger.info(
            f"[nemo→hf] Converting {architecture} from {nemo_checkpoint} "
            f"to {output_path}"
        )

        if self._cached_hf_config is None:
            raise RuntimeError(
                "_nemo_to_hf called before _hf_to_nemo — HF config cache is empty. "
                "This is a bug; run() always calls _hf_to_nemo first."
            )

        with self._dtype_context(self._cached_hf_config):
            nemo_model = self._build_nemo_model(architecture, self._cached_hf_config)
            llm.export_ckpt(
                model=nemo_model,
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
            model_id:               HuggingFace model ID e.g. "meta-llama/Llama-3.1-8B"
            width_pruning_pct:      Fraction of layer params to remove via width pruning.
            depth_pruning_pct:      Fraction of layers to remove via depth pruning.
            do_pruning:             Whether to run pruning.
            do_distillation:        Whether to run distillation after pruning.
            do_quantization:        Whether to run quantization last.
            dataset_path:           Optional dataset for pruning/distillation.
                                    Falls back to default wikitext in S3 if None.
            distillation_steps:     Training steps for distillation.
            quantization_algorithm: Quantization algorithm (fp8, int8_sq, etc.)
            seq_length:             Sequence length for pruning/distillation.

        Returns:
            {
                "hf_output_path":   str,   # path to final HF checkpoint
                "compression_info": dict,  # original vs targets from calculator
            }
        """
        self._setup_dirs()

        # ------------------------------------------------------------------
        # Step 0: Download model from HuggingFace Hub to local /tmp/hf_input
        # ------------------------------------------------------------------
        logger.info(f"[pipeline] Downloading {model_id} from HuggingFace Hub")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(self.hf_input_dir),
            ignore_patterns=["*.md", "*.txt"],
        )
        logger.info("[pipeline] Download complete")

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
        # Step 3: Convert HF weights → .nemo
        # Also populates self._cached_hf_config for use in _nemo_to_hf later.
        # Cascade step 1: HF weights → .nemo → delete HF weights
        # ------------------------------------------------------------------
        logger.info("[pipeline] Converting HF → NeMo")
        self._hf_to_nemo(
            model_id=model_id,
            hf_model_path=str(self.hf_input_dir),
            architecture=architecture,
            output_path=str(self.nemo_input_dir),
        )
        # HF weights no longer needed — delete to keep storage at ~2x model size
        self._cleanup(self.hf_input_dir, "hf_input")

        # ------------------------------------------------------------------
        # Step 4: Keep a reference to the teacher checkpoint BEFORE compression
        # Distillation exception: teacher must stay alive throughout gpt_train.py
        # and is only deleted after distillation completes.
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
        # Distillation is done so teacher is safe to delete too.
        # ------------------------------------------------------------------
        self._cleanup(self.nemo_input_dir, "nemo_input")

        # ------------------------------------------------------------------
        # Step 7: Convert compressed .nemo → HuggingFace
        # Uses cached HF config — hf_input_dir is already gone at this point.
        # Cascade step 3: compressed .nemo → HF → delete compressed .nemo
        # ------------------------------------------------------------------
        logger.info("[pipeline] Converting NeMo → HF")
        self._nemo_to_hf(
            nemo_checkpoint=engine.current_checkpoint,
            architecture=architecture,
            output_path=str(self.hf_output_dir),
        )
        self._cleanup(self.nemo_output_dir, "nemo_output")

        logger.info(
            f"[pipeline] Complete. Final HF checkpoint at: {self.hf_output_dir}"
        )

        return {
            "hf_output_path": str(self.hf_output_dir),
            "compression_info": compression_info,
        }