import pytest

from services.calculator import (
    _count_params,
    calculate_compression_targets,
)

# Real Llama 3.1 8B config fields used by the calculator
LLAMA_8B = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,   # GQA: 8 KV groups for 32 query heads
    "vocab_size": 128256,
}


def test_count_params_llama():
    result = _count_params(
        num_layers=32,
        hidden_size=4096,
        ffn_size=14336,
        num_heads=32,
        num_kv_groups=8,
        vocab_size=128256,
    )
    # The formula uses 2 FFN matrices (not 3 for SwiGLU) and omits the O projection,
    # so it gives ~5.6B rather than the "real" 8B — that's expected and intentional.
    total_B = result["total_params"] / 1e9
    assert abs(total_B - 5.614) < 0.01

    # Sanity checks on structure
    assert result["layer_params"] + result["embedding_params"] == result["total_params"]
    assert result["layer_params"] > result["embedding_params"]  # most params are in layers
    assert result["embedding_params"] > 0


def test_no_pruning_targets_match_originals():
    result = calculate_compression_targets(LLAMA_8B, width_pruning_pct=0.0, depth_pruning_pct=0.0)

    original = result["original"]
    targets  = result["targets"]

    assert targets["target_num_layers"]        == original["num_layers"]
    assert targets["target_hidden_size"]       == original["hidden_size"]
    assert targets["target_ffn_hidden_size"]   == original["ffn_hidden_size"]
    assert targets["layers_removed"]           == 0
    assert result["compression_ratio"]         == 1.0
    assert result["expected_params_B"]         == original["total_params_B"]


def test_width_pruning_25pct():
    result = calculate_compression_targets(LLAMA_8B, width_pruning_pct=0.25)
    targets = result["targets"]

    # Exact values derived from the formula: r_w = sqrt(0.75), rounded down to
    # nearest multiple of lcm(num_heads=32, alignment=16) = 32 for hidden,
    # and nearest 16 for FFN.
    assert targets["target_hidden_size"]     == 3520
    assert targets["target_ffn_hidden_size"] == 12400

    # Alignment constraints
    assert targets["target_hidden_size"]     % 32 == 0  # divisible by lcm(heads, kv_groups, align)
    assert targets["target_ffn_hidden_size"] % 16 == 0

    # Strictly smaller
    assert targets["target_hidden_size"]     < LLAMA_8B["hidden_size"]
    assert targets["target_ffn_hidden_size"] < LLAMA_8B["intermediate_size"]

    # Depth should be untouched
    assert targets["target_num_layers"] == LLAMA_8B["num_hidden_layers"]
    assert result["compression_ratio"]  > 1.0


def test_depth_pruning_25pct():
    result = calculate_compression_targets(LLAMA_8B, depth_pruning_pct=0.25)
    targets = result["targets"]

    assert targets["target_num_layers"] == 24
    assert targets["layers_removed"]    == 8

    # Width should be untouched
    assert targets["target_hidden_size"]     == LLAMA_8B["hidden_size"]
    assert targets["target_ffn_hidden_size"] == LLAMA_8B["intermediate_size"]
    assert result["compression_ratio"]       > 1.0


def test_depth_pruning_50pct():
    result = calculate_compression_targets(LLAMA_8B, depth_pruning_pct=0.5)
    targets = result["targets"]

    assert targets["target_num_layers"] == 16
    assert targets["layers_removed"]    == 16
    assert result["compression_ratio"]  > 1.0


def test_small_width_pruning_still_aligns():
    # Even tiny pruning should produce an aligned result smaller than original
    result = calculate_compression_targets(LLAMA_8B, width_pruning_pct=0.01)
    targets = result["targets"]

    assert targets["target_hidden_size"] == 4064  # floor(4075.47 / 32) * 32
    assert targets["target_hidden_size"] % 32 == 0
    assert targets["target_hidden_size"] < LLAMA_8B["hidden_size"]


def test_combined_pruning():
    result = calculate_compression_targets(
        LLAMA_8B, width_pruning_pct=0.25, depth_pruning_pct=0.25
    )
    targets = result["targets"]

    assert targets["target_num_layers"]      == 24
    assert targets["target_hidden_size"]     == 3520
    assert targets["target_ffn_hidden_size"] == 12400
    assert result["compression_ratio"]       > 1.0
    assert result["expected_params_B"]       < result["original"]["total_params_B"]
