from unittest.mock import MagicMock, patch

import pytest

from services.detector import detect_and_validate


def _mock_config(architectures):
    """Build a minimal AutoConfig mock with the fields detect_and_validate reads."""
    config = MagicMock()
    config.architectures = architectures
    config.to_dict.return_value = {
        "architectures": architectures,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 128256,
    }
    return config


@patch("services.detector.AutoConfig.from_pretrained")
def test_llama_passes_validation(mock_from_pretrained):
    mock_from_pretrained.return_value = _mock_config(["LlamaForCausalLM"])

    result = detect_and_validate("meta-llama/Llama-3.1-8B")

    assert result["architectures"] == ["LlamaForCausalLM"]
    assert result["num_hidden_layers"] == 32
    mock_from_pretrained.assert_called_once_with(
        "meta-llama/Llama-3.1-8B", trust_remote_code=False
    )


@patch("services.detector.AutoConfig.from_pretrained")
def test_other_supported_architectures_pass(mock_from_pretrained):
    for arch in ["MistralForCausalLM", "MixtralForCausalLM", "Qwen2ForCausalLM"]:
        mock_from_pretrained.return_value = _mock_config([arch])
        result = detect_and_validate(f"some/{arch}")
        assert result["architectures"] == [arch]


@patch("services.detector.AutoConfig.from_pretrained")
def test_unsupported_architecture_raises(mock_from_pretrained):
    mock_from_pretrained.return_value = _mock_config(["GPT2LMHeadModel"])

    with pytest.raises(ValueError, match="not supported"):
        detect_and_validate("gpt2")


@patch("services.detector.AutoConfig.from_pretrained")
def test_missing_architectures_field_raises(mock_from_pretrained):
    config = MagicMock()
    config.architectures = None
    mock_from_pretrained.return_value = config

    with pytest.raises(ValueError, match="architectures"):
        detect_and_validate("some/model")


@patch("services.detector._load_config")
def test_model_not_found_raises(mock_load_config):
    # _load_config already converts HfHubHTTPError → ValueError, so mock at that level
    mock_load_config.side_effect = ValueError("Model 'bad/model' not found on Hugging Face Hub.")

    with pytest.raises(ValueError, match="not found"):
        detect_and_validate("bad/model")
