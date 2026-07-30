import { Clock3 } from "lucide-react";

import type { Run } from "../types";
import { StatusPill } from "./StatusPill";

interface Props {
  runs: Run[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function RunList({ runs, selectedId, onSelect }: Props) {
  return (
    <section className="run-list panel" aria-labelledby="runs-title">
      <div className="section-heading compact">
        <h2 id="runs-title">Recent runs</h2>
        <span>{runs.length}</span>
      </div>
      <div className="run-list-items">
        {runs.length === 0 ? (
          <div className="empty">
            Launch a workflow to create the first trace.
          </div>
        ) : (
          runs.map((run) => (
            <button
              type="button"
              className={run.id === selectedId ? "run-row selected" : "run-row"}
              key={run.id}
              onClick={() => onSelect(run.id)}
            >
              <div className="run-row-top">
                <strong>{run.demo}</strong>
                <StatusPill status={run.status} />
              </div>
              <p>{run.input}</p>
              <span className="run-time">
                <Clock3 size={12} /> {new Date(run.createdAt).toLocaleString()}
              </span>
            </button>
          ))
        )}
      </div>
    </section>
  );
}
