import type { CalculatorResult } from "../types";

interface Props {
  result: CalculatorResult;
}

function Row({ label, original, after }: { label: string; original: string; after: string }) {
  return (
    <tr className="border-t border-gray-100">
      <td className="py-2.5 pr-4 text-sm text-gray-500 whitespace-nowrap">{label}</td>
      <td className="py-2.5 px-4 text-sm text-gray-900 text-right">{original}</td>
      <td className="py-2.5 pl-4 text-sm text-right font-medium text-blue-700">{after}</td>
    </tr>
  );
}

export function CalculatorBreakdown({ result }: Props) {
  const { original, targets, expected_params_B, compression_ratio } = result;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-gray-50">
            <th className="py-2.5 pr-4 text-xs font-medium text-gray-400 text-left pl-4">Metric</th>
            <th className="py-2.5 px-4 text-xs font-medium text-gray-400 text-right">Original</th>
            <th className="py-2.5 pl-4 pr-4 text-xs font-medium text-gray-400 text-right">After Compression</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-50 px-4">
          <tr className="border-t border-gray-100">
            <td className="py-2.5 pr-4 pl-4 text-sm text-gray-500">Parameters</td>
            <td className="py-2.5 px-4 text-sm text-gray-900 text-right">{original.total_params_B}B</td>
            <td className="py-2.5 pl-4 pr-4 text-sm text-right font-medium text-blue-700">{expected_params_B}B</td>
          </tr>
          <tr className="border-t border-gray-100">
            <td className="py-2.5 pr-4 pl-4 text-sm text-gray-500">Layers</td>
            <td className="py-2.5 px-4 text-sm text-gray-900 text-right">{original.num_layers}</td>
            <td className="py-2.5 pl-4 pr-4 text-sm text-right font-medium text-blue-700">
              {targets.target_num_layers}
              {targets.layers_removed > 0 && (
                <span className="ml-1 text-xs text-gray-400">(-{targets.layers_removed})</span>
              )}
            </td>
          </tr>
          <tr className="border-t border-gray-100">
            <td className="py-2.5 pr-4 pl-4 text-sm text-gray-500">Hidden size</td>
            <td className="py-2.5 px-4 text-sm text-gray-900 text-right">{original.hidden_size}</td>
            <td className="py-2.5 pl-4 pr-4 text-sm text-right font-medium text-blue-700">{targets.target_hidden_size}</td>
          </tr>
          <tr className="border-t border-gray-100">
            <td className="py-2.5 pr-4 pl-4 text-sm text-gray-500">FFN size</td>
            <td className="py-2.5 px-4 text-sm text-gray-900 text-right">{original.ffn_hidden_size}</td>
            <td className="py-2.5 pl-4 pr-4 text-sm text-right font-medium text-blue-700">{targets.target_ffn_hidden_size}</td>
          </tr>
          <tr className="border-t border-gray-100 bg-gray-50">
            <td className="py-2.5 pr-4 pl-4 text-sm font-medium text-gray-700">Compression ratio</td>
            <td className="py-2.5 px-4 text-sm text-gray-400 text-right">—</td>
            <td className="py-2.5 pl-4 pr-4 text-sm text-right font-semibold text-blue-700">{compression_ratio}x</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
