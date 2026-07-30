import type { RunStatus } from "../types";

export function StatusPill({ status }: { status: RunStatus }) {
  return (
    <span className={`status status-${status}`}>
      <span className="status-dot" />
      {status.replace("_", " ")}
    </span>
  );
}
