import type { Approval, Demo, Run, RunEvent, TraceExport } from "./types";

const API_ROOT = import.meta.env.VITE_API_ROOT ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  demos: () => request<Demo[]>("/demos"),
  runs: () => request<Run[]>("/runs"),
  run: (id: string) => request<Run>(`/runs/${id}`),
  events: (id: string) => request<RunEvent[]>(`/runs/${id}/events`),
  approvals: (id: string) => request<Approval[]>(`/runs/${id}/approvals`),
  start: (demo: string, input: string) =>
    request<Run>("/runs", {
      method: "POST",
      body: JSON.stringify({ demo, input }),
    }),
  decide: (runId: string, requestId: string, approved: boolean) =>
    request<Run>(`/runs/${runId}/approvals/${requestId}`, {
      method: "POST",
      body: JSON.stringify({ approved, decidedBy: "console-user" }),
    }),
  resume: (id: string) =>
    request<Run>(`/runs/${id}/resume`, { method: "POST" }),
  cancel: (id: string) =>
    request<Run>(`/runs/${id}/cancel`, { method: "POST" }),
  exportTrace: (id: string) => request<TraceExport>(`/runs/${id}/export`),
};

export function streamUrl(runId: string, after: number): string {
  return `${API_ROOT}/runs/${runId}/stream?after=${after}`;
}
