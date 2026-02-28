interface Props {
  status: string | null;
}

const STATUS_STYLES: Record<string, { dot: string; text: string; label: string }> = {
  InProgress: {
    dot: "bg-yellow-400 animate-pulse",
    text: "text-yellow-400",
    label: "Running",
  },
  Completed: {
    dot: "bg-[var(--accent)]",
    text: "text-[var(--accent)]",
    label: "Completed",
  },
  Failed: {
    dot: "bg-[var(--danger)]",
    text: "text-[var(--danger)]",
    label: "Failed",
  },
  Stopped: {
    dot: "bg-[var(--text-muted)]",
    text: "text-[var(--text-secondary)]",
    label: "Stopped",
  },
};

const DEFAULT_STYLE = {
  dot: "bg-[var(--border-bright)] animate-pulse",
  text: "text-[var(--text-muted)]",
  label: "Waiting…",
};

export function JobProgress({ status }: Props) {
  const style = (status && STATUS_STYLES[status]) ?? DEFAULT_STYLE;

  return (
    <div className="flex items-center gap-2.5">
      <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${style.dot}`} />
      <span className={`text-sm font-medium ${style.text}`}>{style.label}</span>
      {status &&
        status !== "Completed" &&
        status !== "Failed" &&
        status !== "Stopped" && (
          <span className="text-xs text-[var(--text-muted)]">({status})</span>
        )}
    </div>
  );
}
