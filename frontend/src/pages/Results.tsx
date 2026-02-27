import { useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";

import { getJobResult } from "../api";
import { MetricsComparison } from "../components/MetricsComparison";
import type { CalculatorResult } from "../types";

interface JobResult {
  job_id: string;
  download_url: string;
}

export function Results() {
  const { jobId } = useParams<{ jobId: string }>();
  const location = useLocation();
  const calcResult = location.state?.calcResult as CalculatorResult | undefined;

  const [jobResult, setJobResult] = useState<JobResult | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    getJobResult(jobId!)
      .then((data) => setJobResult(data as JobResult))
      .catch((err) => setFetchError(err instanceof Error ? err.message : "Failed to load result."));
  }, [jobId]);

  async function handleDownload() {
    if (!jobResult) return;
    setDownloading(true);
    window.open(jobResult.download_url, "_blank");
    setDownloading(false);
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Compression complete</h1>
        <p className="mt-1 text-sm text-gray-500 font-mono">{jobId}</p>
      </div>

      {calcResult ? (
        <section className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Results</h2>
          <MetricsComparison result={calcResult} />
        </section>
      ) : (
        <p className="text-sm text-gray-400">
          Metrics unavailable — navigate here from the app to see before/after stats.
        </p>
      )}

      <section className="space-y-3">
        <h2 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Download</h2>

        {fetchError && (
          <p className="text-sm text-red-600">{fetchError}</p>
        )}

        <p className="text-sm text-gray-500">
          The compressed model is packaged as <span className="font-mono">output.tar.gz</span> —
          a HuggingFace-compatible checkpoint. The download link expires in 1 hour.
        </p>

        <button
          onClick={handleDownload}
          disabled={!jobResult || downloading}
          className="w-full px-4 py-2.5 bg-gray-900 text-white text-sm font-medium rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {jobResult ? "Download model" : "Loading…"}
        </button>
      </section>
    </div>
  );
}
