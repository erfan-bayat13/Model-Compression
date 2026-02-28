import { useState } from "react";

interface Props {
  onDetect: (modelId: string) => void;
  loading: boolean;
  error: string | null;
}

export function ModelInput({ onDetect, loading, error }: Props) {
  const [value, setValue] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (value.trim()) onDetect(value.trim());
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="meta-llama/Llama-3.1-8B"
          disabled={loading}
          className="flex-1 px-4 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-sm text-[var(--text-primary)] placeholder:text-[var(--text-muted)] font-mono focus:outline-none focus:border-[var(--border-bright)] focus:ring-1 focus:ring-[var(--accent)]/30 disabled:opacity-50 transition-colors"
        />
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="px-5 py-2.5 bg-[var(--accent)] text-black text-sm font-medium rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        >
          {loading ? "Detecting…" : "Detect"}
        </button>
      </div>

      {error && <p className="mt-2 text-sm text-[var(--danger)]">{error}</p>}
    </form>
  );
}
