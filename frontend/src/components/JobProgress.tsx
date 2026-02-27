interface Props {
  status: string | null;
}

const STATUS_STYLES: Record<string, { dot: string; text: string; label: string }> = {
  InProgress: {
    dot: "bg-yellow-400 animate-pulse",
    text: "text-yellow-700",
    label: "Running",
  },
  Completed: {
    dot: "bg-green-500",
    text: "text-green-700",
    label: "Completed",
  },
  Failed: {
    dot: "bg-red-500",
    text: "text-red-700",
    label: "Failed",
  },
  Stopped: {
    dot: "bg-gray-400",
    text: "text-gray-600",
    label: "Stopped",
  },
};

const DEFAULT_STYLE = {
  dot: "bg-gray-300 animate-pulse",
  text: "text-gray-500",
  label: "Waiting…",
};

export function JobProgress({ status }: Props) {
  const style = (status && STATUS_STYLES[status]) ?? DEFAULT_STYLE;

  return (
    <div className="flex items-center gap-2.5">
      <span className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${style.dot}`} />
      <span className={`text-sm font-medium ${style.text}`}>{style.label}</span>
      {status && status !== "Completed" && status !== "Failed" && status !== "Stopped" && (
        <span className="text-xs text-gray-400">({status})</span>
      )}
    </div>
  );
}
