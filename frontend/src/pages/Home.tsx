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
      const data = await detectModel(modelId) as DetectResponse;
      setResult(data);
      setDetectedModelId(modelId);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Detection failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center px-4">
      <div className="w-full max-w-lg space-y-8">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Model Compression</h1>
          <p className="mt-1 text-sm text-gray-500">
            Enter a HuggingFace model ID to get started.
          </p>
        </div>

        <ModelInput onDetect={handleDetect} loading={loading} error={error} />

        {result && (
          <div className="bg-white border border-gray-200 rounded-lg p-5 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Architecture</span>
              <span className="text-sm text-gray-900 font-mono">{result.model_info.architecture}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Parameters</span>
              <span className="text-sm text-gray-900">{result.model_info.total_params_B}B</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Status</span>
              {result.model_info.supported ? (
                <span className="text-xs font-medium text-green-700 bg-green-100 px-2 py-0.5 rounded-full">
                  Supported
                </span>
              ) : (
                <span className="text-xs font-medium text-red-700 bg-red-100 px-2 py-0.5 rounded-full">
                  Unsupported
                </span>
              )}
            </div>

            {result.model_info.supported && (
              <button
                onClick={() => navigate("/configure", { state: { modelId: detectedModelId, detectResult: result } })}
                className="w-full mt-2 px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-700 transition-colors"
              >
                Continue
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
