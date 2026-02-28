import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { calculateTargets, submitCompression } from "../api";
import { CalculatorBreakdown } from "../components/CalculatorBreakdown";
import { CompressionOptions } from "../components/CompressionOptions";
import type { Options } from "../components/CompressionOptions";
import type { CalculatorResult, DetectResponse } from "../types";

const DEFAULT_OPTIONS: Options = {
  doDistillation: false,
  doQuantization: false,
  datasetPath: "",
  enableMmlu: false,
};

export function Configure() {
  const location = useLocation();
  const navigate = useNavigate();

  const detectResult = location.state?.detectResult as DetectResponse | undefined;
  const modelId = location.state?.modelId as string | undefined;

  const [widthPct, setWidthPct] = useState(0);
  const [depthPct, setDepthPct] = useState(0);
  const [calcResult, setCalcResult] = useState<CalculatorResult | null>(
    detectResult?.calculator_result ?? null,
  );
  const [options, setOptions] = useState<Options>(DEFAULT_OPTIONS);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Redirect if someone lands here directly without model data
  useEffect(() => {
    if (!detectResult) navigate("/", { replace: true });
  }, []);

  // Recalculate whenever sliders change — debounced to avoid hammering the API
  useEffect(() => {
    if (!detectResult) return;

    // The backend calculator reads raw HF config field names, not our renamed ones
    const { original } = detectResult.calculator_result;
    const hfConfig = {
      num_hidden_layers: original.num_layers,
      hidden_size: original.hidden_size,
      intermediate_size: original.ffn_hidden_size,
      num_attention_heads: original.num_attention_heads,
      num_key_value_heads: original.num_kv_groups,
      vocab_size: original.vocab_size,
    };

    const timer = setTimeout(async () => {
      try {
        const result = (await calculateTargets(
          hfConfig,
          widthPct / 100,
          depthPct / 100,
        )) as CalculatorResult;
        setCalcResult(result);
      } catch {
        // non-critical — keep showing the last valid result
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [widthPct, depthPct]);

  async function handleSubmit() {
    if (!detectResult) return;

    setSubmitting(true);
    setError(null);

    try {
      const response = (await submitCompression({
        model_id: modelId ?? detectResult.model_info.architecture,
        width_pruning_pct: widthPct / 100,
        depth_pruning_pct: depthPct / 100,
        do_pruning: widthPct > 0 || depthPct > 0,
        do_distillation: options.doDistillation,
        do_quantization: options.doQuantization,
        dataset_path: options.datasetPath || null,
        enable_mmlu: options.enableMmlu,
      })) as { job_id: string };

      navigate(`/progress/${response.job_id}`, {
        state: { calcResult, modelId },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit job.");
      setSubmitting(false);
    }
  }

  if (!detectResult) return null;

  return (
    <div className="max-w-2xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-primary)]">
          Configure compression
        </h1>
        <p className="mt-1 text-sm text-[var(--text-muted)] font-mono">
          {detectResult.model_info.architecture}
        </p>
      </div>

      <section className="space-y-5">
        <h2 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest">
          Pruning
        </h2>

        <div className="space-y-5">
          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm text-[var(--text-secondary)]">
                Width pruning
              </label>
              <span className="text-sm font-medium font-mono text-[var(--text-primary)]">
                {widthPct}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={50}
              value={widthPct}
              onChange={(e) => setWidthPct(Number(e.target.value))}
            />
            <p className="text-xs text-[var(--text-muted)] mt-1.5">
              Reduces hidden and FFN dimensions.
            </p>
          </div>

          <div>
            <div className="flex justify-between mb-2">
              <label className="text-sm text-[var(--text-secondary)]">
                Depth pruning
              </label>
              <span className="text-sm font-medium font-mono text-[var(--text-primary)]">
                {depthPct}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={50}
              value={depthPct}
              onChange={(e) => setDepthPct(Number(e.target.value))}
            />
            <p className="text-xs text-[var(--text-muted)] mt-1.5">
              Removes transformer layers.
            </p>
          </div>
        </div>
      </section>

      {calcResult && (
        <section className="space-y-3">
          <h2 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest">
            Estimated output
          </h2>
          <CalculatorBreakdown result={calcResult} />
        </section>
      )}

      <section className="space-y-3">
        <h2 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-widest">
          Options
        </h2>
        <CompressionOptions options={options} onChange={setOptions} />
      </section>

      {error && <p className="text-sm text-[var(--danger)]">{error}</p>}

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="w-full px-4 py-2.5 bg-[var(--accent)] text-black text-sm font-medium rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
      >
        {submitting ? "Submitting…" : "Compress model"}
      </button>
    </div>
  );
}
