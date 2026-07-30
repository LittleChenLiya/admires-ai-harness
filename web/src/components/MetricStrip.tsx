import { CircleDollarSign, Clock3, Cpu, Wrench } from "lucide-react";

import type { Metrics } from "../types";

export function MetricStrip({ metrics }: { metrics: Metrics }) {
  const values = [
    {
      icon: Cpu,
      label: "Tokens",
      value: metrics.inputTokens + metrics.outputTokens,
    },
    { icon: Wrench, label: "Tools", value: metrics.toolCalls },
    {
      icon: CircleDollarSign,
      label: "Cost",
      value: `$${metrics.costUsd.toFixed(4)}`,
    },
    {
      icon: Clock3,
      label: "Duration",
      value:
        metrics.durationMs === null
          ? "—"
          : `${(metrics.durationMs / 1000).toFixed(2)}s`,
    },
  ];
  return (
    <div className="metric-strip">
      {values.map(({ icon: Icon, label, value }) => (
        <div className="metric" key={label}>
          <Icon size={15} />
          <div>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        </div>
      ))}
    </div>
  );
}
