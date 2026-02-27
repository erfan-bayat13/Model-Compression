import { useEffect, useRef, useState } from "react";

import { getJobStatus } from "../api";
import type { JobStatusResponse } from "../types";

const TERMINAL_STATUSES = new Set(["Completed", "Failed", "Stopped"]);
const POLL_INTERVAL_MS = 10_000;

interface PollingState {
  status: string | null;
  createdAt: string | null;
  error: string | null;
  isPolling: boolean;
}

export function useJobPolling(jobId: string): PollingState {
  const [state, setState] = useState<PollingState>({
    status: null,
    createdAt: null,
    error: null,
    isPolling: true,
  });

  // Ref so the interval callback can clear itself without stale closure issues
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function fetchStatus() {
      try {
        const data = await getJobStatus(jobId) as JobStatusResponse;
        const terminal = TERMINAL_STATUSES.has(data.status);

        setState({
          status: data.status,
          createdAt: data.created_at,
          error: null,
          isPolling: !terminal,
        });

        if (terminal && intervalRef.current !== null) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : "Failed to fetch job status.",
          isPolling: false,
        }));
        if (intervalRef.current !== null) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      }
    }

    fetchStatus();
    intervalRef.current = setInterval(fetchStatus, POLL_INTERVAL_MS);

    return () => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId]);

  return state;
}
