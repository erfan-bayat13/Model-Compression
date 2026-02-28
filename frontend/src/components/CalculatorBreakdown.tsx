import type { CalculatorResult } from "../types";

interface Props {
  result: CalculatorResult;
}

function Row({
  label,
  original,
  after,
  highlight,
}: {
  label: string;
  original: string;
  after: string;
  highlight?: boolean;
}) {
  return (
    <tr
      className={
        highlight
          ? "bg-[var(--accent-muted-bg)] border-t border-[var(--border)]"
          : "border-t border-[var(--border)]"
      }
    >
      <td className="py-2.5 pl-4 pr-4 text-sm text-[var(--text-secondary)] whitespace-nowrap">
        {label}
      </td>
      <td className="py-2.5 px-4 text-sm text-[var(--text-primary)] font-mono text-right">
        {original}
      </td>
      <td
        className={`py-2.5 pl-4 pr-4 text-sm text-right font-medium font-mono ${highlight ? "text-[var(--accent)]" : "text-[var(--accent)]"}`}
      >
        {after}
      </td>
    </tr>
  );
}

export function CalculatorBreakdown({ result }: Props) {
  const { original, targets, expected_params_B, compression_ratio } = result;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg overflow-hidden">
      <table className="w-full">
        <thead>
          <tr className="bg-[var(--bg-tertiary)]">
            <th className="py-2.5 pl-4 pr-4 text-xs font-medium text-[var(--text-muted)] text-left">
              Metric
            </th>
            <th className="py-2.5 px-4 text-xs font-medium text-[var(--text-muted)] text-right">
              Original
            </th>
            <th className="py-2.5 pl-4 pr-4 text-xs font-medium text-[var(--text-muted)] text-right">
              After compression
            </th>
          </tr>
        </thead>
        <tbody>
          <Row
            label="Parameters"
            original={`${original.total_params_B}B`}
            after={`${expected_params_B}B`}
          />
          <Row
            label="Layers"
            original={String(original.num_layers)}
            after={
              targets.layers_removed > 0
                ? `${targets.target_num_layers} (−${targets.layers_removed})`
                : String(targets.target_num_layers)
            }
          />
          <Row
            label="Hidden size"
            original={String(original.hidden_size)}
            after={String(targets.target_hidden_size)}
          />
          <Row
            label="FFN size"
            original={String(original.ffn_hidden_size)}
            after={String(targets.target_ffn_hidden_size)}
          />
          <Row
            label="Compression ratio"
            original="—"
            after={`${compression_ratio}x`}
            highlight
          />
        </tbody>
      </table>
    </div>
  );
}
