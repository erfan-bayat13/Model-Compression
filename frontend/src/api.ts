const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }

  return res.json();
}

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

export function detectModel(modelId: string) {
  return apiFetch("/models/detect", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

export function calculateTargets(config: object, widthPct: number, depthPct: number) {
  return apiFetch("/models/calculate", {
    method: "POST",
    body: JSON.stringify({
      config,
      width_pruning_pct: widthPct,
      depth_pruning_pct: depthPct,
    }),
  });
}

export function submitCompression(request: CompressionRequest) {
  return apiFetch("/compress", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function getJobStatus(jobId: string) {
  return apiFetch(`/jobs/${jobId}/status`);
}

export function getJobResult(jobId: string) {
  return apiFetch(`/jobs/${jobId}/result`);
}

export function getDownloadUrl(jobId: string) {
  return apiFetch(`/jobs/${jobId}/download`);
}
