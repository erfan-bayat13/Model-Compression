from typing import Iterable, Optional

from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoConfig, PretrainedConfig

SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",
    "MistralForCausalLM", 
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
}


def _load_config(model_id: str) -> PretrainedConfig:
    try:
        return AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    except HfHubHTTPError as exc:
        raise ValueError(f"Model '{model_id}' not found on Hugging Face Hub.") from exc
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed to load config for '{model_id}': {exc}") from exc


def fetch_model_architecture(model_id: str) -> str:
    config = _load_config(model_id)
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(
            f"Config for '{model_id}' does not declare an `architectures` field."
        )
    return architectures[0]


def validate_model_architecture(
    model_id: str, allowed_architectures: Optional[Iterable[str]] = None) -> str:
    architecture = fetch_model_architecture(model_id)
    if allowed_architectures and architecture not in set(allowed_architectures):
        raise ValueError(
            f"Model '{model_id}' has architecture '{architecture}', "
            f"expected one of {sorted(set(allowed_architectures))}."
        )
    return architecture

def detect_and_validate(model_id: str) -> dict:
    """
    Validates the model is supported and returns its config as a plain dict
    ready to pass into calculator.calculate_compression_targets().
    """
    config = _load_config(model_id)
    
    architectures = getattr(config, "architectures", None)
    if not architectures:
        raise ValueError(
            f"Config for '{model_id}' does not declare an `architectures` field."
        )
    
    architecture = architectures[0]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Architecture '{architecture}' is not supported. "
            f"Supported: {sorted(SUPPORTED_ARCHITECTURES)}"
        )
    
    return config.to_dict()