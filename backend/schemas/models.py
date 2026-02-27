from pydantic import BaseModel


class ModelDetectRequest(BaseModel):
    model_id: str


class ModelInfo(BaseModel):
    architecture: str
    total_params_B: float
    supported: bool


# Nested shapes for CalculatorResult — mirror calculate_compression_targets() output exactly

class OriginalParams(BaseModel):
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_kv_groups: int
    vocab_size: int
    layer_params_B: float
    embedding_params_B: float
    total_params_B: float


class PruningTargets(BaseModel):
    target_num_layers: int
    layers_removed: int
    target_hidden_size: int
    target_ffn_hidden_size: int
    target_num_attention_heads: int
    target_num_query_groups: int
    r_w: float


class CalculatorRequest(BaseModel):
    config: dict
    width_pruning_pct: float = 0.0
    depth_pruning_pct: float = 0.0


class CalculatorResult(BaseModel):
    original: OriginalParams
    targets: PruningTargets
    expected_params_B: float
    compression_ratio: float
