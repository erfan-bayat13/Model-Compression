export interface Options {
  doDistillation: boolean;
  doQuantization: boolean;
  datasetPath: string;
  enableMmlu: boolean;
}

interface Props {
  options: Options;
  onChange: (updated: Options) => void;
}

export function CompressionOptions({ options, onChange }: Props) {
  function patch(partial: Partial<Options>) {
    onChange({ ...options, ...partial });
  }

  return (
    <div className="space-y-4">
      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.doDistillation}
          onChange={(e) =>
            patch({ doDistillation: e.target.checked, datasetPath: "" })
          }
          className="mt-0.5 accent-[#22c55e]"
        />
        <div>
          <span className="text-sm font-medium text-[var(--text-primary)]">
            Knowledge distillation
          </span>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">
            Fine-tunes the pruned model using the original as a teacher.
            Requires a dataset.
          </p>
        </div>
      </label>

      {options.doDistillation && (
        <div className="ml-6">
          <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1.5">
            Dataset S3 path{" "}
            <span className="text-[var(--text-muted)] font-normal">
              (leave blank to use default wikitext)
            </span>
          </label>
          <input
            type="text"
            value={options.datasetPath}
            onChange={(e) => patch({ datasetPath: e.target.value })}
            placeholder="s3://your-bucket/dataset/"
            className="w-full px-3 py-1.5 text-sm bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-[var(--text-primary)] placeholder:text-[var(--text-muted)] font-mono focus:outline-none focus:border-[var(--border-bright)] focus:ring-1 focus:ring-[var(--accent)]/30 transition-colors"
          />
        </div>
      )}

      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.doQuantization}
          onChange={(e) => patch({ doQuantization: e.target.checked })}
          className="mt-0.5 accent-[#22c55e]"
        />
        <div>
          <span className="text-sm font-medium text-[var(--text-primary)]">
            Quantization (FP8)
          </span>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">
            Reduces weight precision after compression. Cuts memory roughly in
            half.
          </p>
        </div>
      </label>

      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.enableMmlu}
          onChange={(e) => patch({ enableMmlu: e.target.checked })}
          className="mt-0.5 accent-[#22c55e]"
        />
        <div>
          <span className="text-sm font-medium text-[var(--text-primary)]">
            Run MMLU evaluation
          </span>
          <p className="text-xs text-[var(--text-secondary)] mt-0.5">
            Benchmarks the compressed model on MMLU after the job completes.
          </p>
        </div>
      </label>
    </div>
  );
}
