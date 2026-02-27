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
    <div className="space-y-3">
      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.doDistillation}
          onChange={(e) => patch({ doDistillation: e.target.checked, datasetPath: "" })}
          className="mt-0.5"
        />
        <div>
          <span className="text-sm font-medium text-gray-800">Knowledge distillation</span>
          <p className="text-xs text-gray-500 mt-0.5">
            Fine-tunes the pruned model using the original as a teacher. Requires a dataset.
          </p>
        </div>
      </label>

      {options.doDistillation && (
        <div className="ml-6">
          <label className="block text-xs font-medium text-gray-600 mb-1">
            Dataset S3 path <span className="text-gray-400 font-normal">(leave blank to use default wikitext)</span>
          </label>
          <input
            type="text"
            value={options.datasetPath}
            onChange={(e) => patch({ datasetPath: e.target.value })}
            placeholder="s3://your-bucket/dataset/"
            className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
      )}

      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.doQuantization}
          onChange={(e) => patch({ doQuantization: e.target.checked })}
          className="mt-0.5"
        />
        <div>
          <span className="text-sm font-medium text-gray-800">Quantization (FP8)</span>
          <p className="text-xs text-gray-500 mt-0.5">
            Reduces weight precision after compression. Cuts memory roughly in half.
          </p>
        </div>
      </label>

      <label className="flex items-start gap-3 cursor-pointer">
        <input
          type="checkbox"
          checked={options.enableMmlu}
          onChange={(e) => patch({ enableMmlu: e.target.checked })}
          className="mt-0.5"
        />
        <div>
          <span className="text-sm font-medium text-gray-800">Run MMLU evaluation</span>
          <p className="text-xs text-gray-500 mt-0.5">
            Benchmarks the compressed model on MMLU after the job completes.
          </p>
        </div>
      </label>
    </div>
  );
}
