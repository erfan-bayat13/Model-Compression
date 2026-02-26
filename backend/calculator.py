import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ALIGNMENT = 16
# GPUs execute matmuls in tiles sized to powers of 2. Misaligned dimensions
# cause padding, wasted compute, and NeMo's pruning API will reject them outright.


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _round_down_to_multiple(value: float, multiple: int) -> int:
    """
    Round DOWN to the nearest multiple.
    We always round down (not nearest) because rounding up would give
    us MORE params than requested — we'd overshoot the pruning target.
    """
    return (int(value) // multiple) * multiple


def _count_params(
    num_layers: int,
    hidden_size: int,
    ffn_size: int,
    num_heads: int,
    num_kv_groups: int,
    vocab_size: int,
) -> dict:
    """
    Count parameters using the GQA-aware formula from the NVIDIA course.

    Non-embedding params per layer:
      Q projection:   hidden_size^2
      K+V projection: 2 * hidden_size^2 * (num_kv_groups / num_heads)
        — when num_kv_groups == num_heads (standard MHA), this is just 2 * hidden_size^2
        — GQA reduces this proportionally as num_kv_groups < num_heads
      FFN (up + down): 2 * hidden_size * ffn_size

    Embedding params (fixed — cannot be pruned by width or depth pruning):
      2 * vocab_size * hidden_size  (input embedding + output head)
      Assumes untied embeddings (conservative estimate).
    """
    q_proj  = hidden_size ** 2
    kv_proj = 2 * (hidden_size ** 2) * (num_kv_groups / num_heads)
    ffn     = 2 * hidden_size * ffn_size

    layer_params     = num_layers * (q_proj + kv_proj + ffn)
    embedding_params = 2 * vocab_size * hidden_size
    total_params     = layer_params + embedding_params

    return {
        "layer_params":     layer_params,
        "embedding_params": embedding_params,
        "total_params":     total_params,
    }


def _calculate_width_targets(
    hidden_size: int,
    ffn_size: int,
    num_heads: int,
    num_kv_groups: int,
    width_pruning_pct: float,
    alignment: int = DEFAULT_ALIGNMENT,
) -> dict:
    """
    Compute target_hidden_size and target_ffn_hidden_size for gpt_prune.py.

    Uses R_W = sqrt(R_P) where R_P = 1 - width_pruning_pct.
    Operates on layer-dependent params only (embeddings are fixed).
    We treat width_pruning_pct as a fraction of layer params — this is the
    honest route: the user controls exactly how much of the compressible
    part is removed, without embedding size distorting their intent.

    Rounds DOWN to nearest combined multiple of `alignment` AND the head
    divisor — required by NeMo's pruning API. GPUs execute matmuls in tiles
    sized to powers of 2; misaligned dimensions cause padding, wasted compute,
    and NeMo will reject them outright.

    We use lcm(alignment, lcm(num_heads, num_kv_groups)) as the rounding
    base so both constraints are satisfied in one shot, avoiding the
    overshoot that comes from subtracting `alignment` in a loop.
    """
    print(f"DEBUG: num_heads={num_heads}, num_kv_groups={num_kv_groups}")
    print(f"DEBUG: head_divisor={math.lcm(num_heads, num_kv_groups)}, combined={math.lcm(math.lcm(num_heads, num_kv_groups), alignment)}")
    print(f"DEBUG: raw_hidden={hidden_size * math.sqrt(1.0 - width_pruning_pct):.2f}")
    # Step 1: reduction factors
    r_p = 1.0 - width_pruning_pct      # fraction of layer params to keep
    r_w = math.sqrt(r_p)               # width reduction factor (sqrt because
                                       # layer params scale with hidden_size^2)

    # Step 2: raw targets before rounding
    raw_hidden = hidden_size * r_w
    raw_ffn    = ffn_size * r_w

    # Step 3: compute combined rounding base
    # hidden_size must satisfy TWO constraints simultaneously:
    #   - divisible by alignment (hardware tile requirement, NeMo API)
    #   - divisible by lcm(num_heads, num_kv_groups) (attention head math)
    # lcm of both gives us the smallest number that satisfies both at once.
    head_divisor = math.lcm(num_heads, num_kv_groups)
    combined     = math.lcm(head_divisor, alignment)

    # Step 4: round down to combined multiple
    # ffn_size only needs alignment, not head divisibility
    target_hidden = _round_down_to_multiple(raw_hidden, combined)
    target_ffn    = _round_down_to_multiple(raw_ffn,    alignment)

    return {
        "target_hidden_size":     target_hidden,
        "target_ffn_hidden_size": target_ffn,
        "r_w":                    round(r_w, 4),
    }


def _calculate_depth_targets(
    num_layers: int,
    depth_pruning_pct: float,
) -> dict:
    """
    Compute target_num_layers for gpt_prune.py.

    Depth pruning simply removes entire transformer layers. The target
    layer count is floored (not rounded) — we never add layers.
    NeMo drops layers from the end of the stack by default.
    """
    target_num_layers = math.floor(num_layers * (1.0 - depth_pruning_pct))

    # Must keep at least 1 layer — a degenerate model is better than a crash
    target_num_layers = max(1, target_num_layers)

    layers_removed = num_layers - target_num_layers

    return {
        "target_num_layers": target_num_layers,
        "layers_removed":    layers_removed,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_compression_targets(
    config: dict,
    width_pruning_pct: float = 0.0,
    depth_pruning_pct: float = 0.0,
    alignment: int = DEFAULT_ALIGNMENT,
) -> dict:
    """
    Main entry point. Takes a raw HuggingFace config.json dict and pruning
    percentages, returns a structured breakdown ready for display in the UI
    and for passing as flags to gpt_prune.py.

    Args:
        config:             Raw config.json parsed as a Python dict.
        width_pruning_pct:  Fraction of layer-dependent params to remove via
                            width pruning. e.g. 0.25 = remove 25%.
        depth_pruning_pct:  Fraction of layers to remove. e.g. 0.25 = remove 25%.
        alignment:          Dimension alignment multiple. Default 64 (NeMo requirement).

    Returns:
        {
            "original": { num_layers, hidden_size, ... , total_params_B },
            "targets":  { target_num_layers, target_hidden_size, ... },
            "expected_params_B":  float,
            "compression_ratio":  float,
        }
    """
    # --- Parse config.json fields ---
    num_layers   = config["num_hidden_layers"]
    hidden_size  = config["hidden_size"]
    ffn_size     = config["intermediate_size"]
    num_heads    = config["num_attention_heads"]
    num_kv_groups = config.get("num_key_value_heads", num_heads)  # fallback = MHA
    vocab_size   = config["vocab_size"]

    # --- Count original params ---
    original_counts = _count_params(
        num_layers, hidden_size, ffn_size,
        num_heads, num_kv_groups, vocab_size
    )

    # --- Width targets ---
    if width_pruning_pct > 0:
        width_targets = _calculate_width_targets(
            hidden_size, ffn_size, num_heads, num_kv_groups,
            width_pruning_pct, alignment
        )
    else:
        width_targets = {
            "target_hidden_size":     hidden_size,
            "target_ffn_hidden_size": ffn_size,
            "r_w":                    1.0,
        }

    # --- Depth targets ---
    if depth_pruning_pct > 0:
        depth_targets = _calculate_depth_targets(num_layers, depth_pruning_pct)
    else:
        depth_targets = {
            "target_num_layers": num_layers,
            "layers_removed":    0,
        }

    # --- Expected param count after pruning ---
    expected_counts = _count_params(
        depth_targets["target_num_layers"],
        width_targets["target_hidden_size"],
        width_targets["target_ffn_hidden_size"],
        num_heads,
        num_kv_groups,
        vocab_size,
    )

    # --- Compression ratio ---
    compression_ratio = (
        original_counts["total_params"] / expected_counts["total_params"]
    )

    return {
        "original": {
            "num_layers":          num_layers,
            "hidden_size":         hidden_size,
            "ffn_hidden_size":     ffn_size,
            "num_attention_heads": num_heads,
            "num_kv_groups":       num_kv_groups,
            "vocab_size":          vocab_size,
            "layer_params_B":      round(original_counts["layer_params"]     / 1e9, 3),
            "embedding_params_B":  round(original_counts["embedding_params"] / 1e9, 3),
            "total_params_B":      round(original_counts["total_params"]     / 1e9, 3),
        },
        "targets": {
            "target_num_layers":        depth_targets["target_num_layers"],
            "layers_removed":           depth_targets["layers_removed"],
            "target_hidden_size":       width_targets["target_hidden_size"],
            "target_ffn_hidden_size":   width_targets["target_ffn_hidden_size"],
            "target_num_attention_heads": num_heads,      # unchanged for now
            "target_num_query_groups":    num_kv_groups,  # unchanged for now
            "r_w":                      width_targets["r_w"],
        },
        "expected_params_B": round(expected_counts["total_params"] / 1e9, 3),
        "compression_ratio": round(compression_ratio, 3),
    }