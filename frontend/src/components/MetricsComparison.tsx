import type { CalculatorResult } from "../types";

interface Props {
  result: CalculatorResult;
}

// Rough fp16 model size estimate: 2 bytes per parameter
function estimateSizeGb(paramsB: number): string {
  return `${(paramsB * 2).toFixed(1)} GB`;
}

function Row({
  label,
  original,
  compressed,
  highlight,
}: {
  label: string;
  original: string;
  compressed: string;
  highlight?: boolean;
}) {
  return (
    <tr className={highlight ? "bg-blue-50" : "border-t border-gray-100"}>
      <td className="py-3 pl-4 pr-6 text-sm text-gray-500 whitespace-nowrap">{label}</td>
      <td className="py-3 px-4 text-sm text-gray-900 text-right">{original}</td>
      <td className="py-3 pl-4 pr-4 text-sm text-right font-medium text-blue-700">{compressed}</td>
    </tr>
  );
}

export function MetricsComparison({ result }: Props) {
  const { original, targets, expected_params_B, compression_ratio } = result;

  // Theoretical speedup ≈ compression ratio (linear approximation — real speedup
  // depends on hardware and memory bandwidth, but this is a reasonable estimate)
  const speedup = `~${compression_ratio}x`;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-gray-50">
            <th className="py-2.5 pl-4 pr-6 text-xs font-medium text-gray-400 text-left">Metric</th>
            <th className="py-2.5 px-4 text-xs font-medium text-gray-400 text-right">Original</th>
            <th className="py-2.5 pl-4 pr-4 text-xs font-medium text-gray-400 text-right">Compressed</th>
          </tr>
        </thead>
        <tbody>
          <Row
            label="Parameters"
            original={`${original.total_params_B}B`}
            compressed={`${expected_params_B}B`}
          />
          <Row
            label="Model size (fp16 est.)"
            original={estimateSizeGb(original.total_params_B)}
            compressed={estimateSizeGb(expected_params_B)}
          />
          <Row
            label="Layers"
            original={String(original.num_layers)}
            compressed={
              targets.layers_removed > 0
                ? `${targets.target_num_layers} (−${targets.layers_removed})`
                : String(targets.target_num_layers)
            }
          />
          <Row
            label="Hidden size"
            original={String(original.hidden_size)}
            compressed={String(targets.target_hidden_size)}
          />
          <Row
            label="FFN size"
            original={String(original.ffn_hidden_size)}
            compressed={String(targets.target_ffn_hidden_size)}
          />
          <Row
            label="Compression ratio"
            original="—"
            compressed={`${compression_ratio}x`}
            highlight
          />
          <Row
            label="Theoretical speedup"
            original="—"
            compressed={speedup}
            highlight
          />
        </tbody>
      </table>
    </div>
  );
}
