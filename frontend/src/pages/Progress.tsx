import { useLocation, useNavigate, useParams } from "react-router-dom";

import { JobProgress } from "../components/JobProgress";
import { useJobPolling } from "../hooks/useJobPolling";

export function Progress() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const location = useLocation();
  const { status, error, isPolling } = useJobPolling(jobId!);

  return (
    <div className="max-w-lg mx-auto px-4 py-10 space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900">Compression in progress</h1>
        <p className="mt-1 text-sm text-gray-500 font-mono">{jobId}</p>
      </div>

      <JobProgress status={status} />

      {isPolling && (
        <p className="text-xs text-gray-400">Checking every 10 seconds…</p>
      )}

      {status === "Completed" && (
        <div className="space-y-3">
          <p className="text-sm text-green-700">Your compressed model is ready.</p>
          <button
            onClick={() => navigate(`/results/${jobId}`, { state: location.state })}
            className="w-full px-4 py-2.5 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 transition-colors"
          >
            View results
          </button>
        </div>
      )}

      {(status === "Failed" || status === "Stopped") && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-sm font-medium text-red-800">Job {status?.toLowerCase()}</p>
          {error && <p className="mt-1 text-sm text-red-600">{error}</p>}
          <p className="mt-2 text-xs text-red-500">
            Check CloudWatch logs for the job <span className="font-mono">{jobId}</span> for details.
          </p>
        </div>
      )}

      {!status && !error && (
        <p className="text-sm text-gray-400">Fetching job status…</p>
      )}
    </div>
  );
}
