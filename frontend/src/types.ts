// --- models ---

export interface OriginalParams {
  num_layers: number;
  hidden_size: number;
  ffn_hidden_size: number;
  num_attention_heads: number;
  num_kv_groups: number;
  vocab_size: number;
  layer_params_B: number;
  embedding_params_B: number;
  total_params_B: number;
}

export interface PruningTargets {
  target_num_layers: number;
  layers_removed: number;
  target_hidden_size: number;
  target_ffn_hidden_size: number;
  target_num_attention_heads: number;
  target_num_query_groups: number;
  r_w: number;
}

export interface CalculatorResult {
  original: OriginalParams;
  targets: PruningTargets;
  expected_params_B: number;
  compression_ratio: number;
}

export interface ModelInfo {
  architecture: string;
  total_params_B: number;
  supported: boolean;
}

export interface DetectResponse {
  model_info: ModelInfo;
  calculator_result: CalculatorResult;
}

// --- compression ---

export interface CompressionRequest {
  model_id: string;
  width_pruning_pct?: number;
  depth_pruning_pct?: number;
  do_pruning?: boolean;
  do_distillation?: boolean;
  do_quantization?: boolean;
  dataset_path?: string | null;
  enable_mmlu?: boolean;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  created_at: string; // ISO datetime string from Python
}

export interface CompressionResult {
  job_id: string;
  download_url: string;
  compression_info: CalculatorResult;
}
