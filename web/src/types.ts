export type RunStatus =
  | "created"
  | "running"
  | "waiting_approval"
  | "paused"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "timed_out"
  | "budget_exceeded";

export interface Demo {
  id: string;
  title: string;
  description: string;
  agents: string[];
  defaultInput: string;
  capabilities: string[];
}

export interface Metrics {
  steps: number;
  modelCalls: number;
  toolCalls: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  durationMs: number | null;
}

export interface Run {
  id: string;
  demo: string;
  input: string;
  status: RunStatus;
  createdAt: string;
  updatedAt: string;
  output: string | null;
  error: string | null;
  metrics: Metrics;
}

export interface RunEvent {
  schema_version: "1";
  event_id: string;
  run_id: string;
  sequence: number;
  type: string;
  timestamp: string;
  parent_run_id: string | null;
  node_id: string | null;
  trace_id: string;
  span_id: string;
  payload: Record<string, unknown>;
}

export interface TraceExport {
  schema_version: "1";
  run_id: string;
  trace_id: string;
  status: RunStatus;
  exported_at: string;
  total_events: number;
  truncated: boolean;
  events: RunEvent[];
}

export interface Approval {
  requestId: string;
  toolName: string;
  reason: string;
  risk: "low" | "medium" | "high" | "critical";
  expiresAt: string;
  arguments: Record<string, unknown>;
  decided: boolean;
  approved: boolean | null;
}
