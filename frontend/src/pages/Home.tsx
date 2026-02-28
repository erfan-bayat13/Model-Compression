import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { detectModel } from "../api";
import { ModelInput } from "../components/ModelInput";
import type { DetectResponse } from "../types";

export function Home() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DetectResponse | null>(null);
  const [detectedModelId, setDetectedModelId] = useState<string>("");

  async function handleDetect(modelId: string) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = (await detectModel(modelId)) as DetectResponse;
      setResult(data);
      setDetectedModelId(modelId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-[calc(100vh-53px)] flex flex-col items-center justify-center px-4 py-16">
      <div className="w-full max-w-lg space-y-8">
        <div>
          <h1 className="text-3xl font-semibold text-[var(--text-primary)] tracking-tight">
            Compress a model
          </h1>
          <p className="mt-2 text-sm text-[var(--text-secondary)]">
            Enter a HuggingFace model ID to detect architecture and configure
            compression.
          </p>
        </div>

        <ModelInput onDetect={handleDetect} loading={loading} error={error} />

        {result && (
          <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-5 space-y-4 fade-in">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-widest">
                Architecture
              </span>
              <span className="text-sm text-[var(--text-primary)] font-mono">
                {result.model_info.architecture}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-widest">
                Parameters
              </span>
              <span className="text-sm text-[var(--text-primary)] font-mono">
                {result.model_info.total_params_B}B
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-widest">
                Status
              </span>
              {result.model_info.supported ? (
                <span className="text-xs font-medium text-[var(--accent)] border border-[var(--accent-muted)] bg-[var(--accent-muted-bg)] px-2 py-0.5 rounded-full">
                  Supported
                </span>
              ) : (
                <span className="text-xs font-medium text-[var(--danger)] border border-red-800 bg-red-950/50 px-2 py-0.5 rounded-full">
                  Unsupported
                </span>
              )}
            </div>

            {result.model_info.supported && (
              <button
                onClick={() =>
                  navigate("/configure", {
                    state: { modelId: detectedModelId, detectResult: result },
                  })
                }
                className="w-full mt-2 px-4 py-2.5 bg-[var(--accent)] text-black text-sm font-medium rounded-lg hover:brightness-110 transition-all"
              >
                Continue →
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
