import { Ban, Download, RotateCcw } from "lucide-react";
import { useState } from "react";

import type { Approval, Run, RunEvent } from "../types";
import { ApprovalCard } from "./ApprovalCard";
import { EventTimeline } from "./EventTimeline";
import { MetricStrip } from "./MetricStrip";
import { StatusPill } from "./StatusPill";

interface Props {
  run: Run | null;
  events: RunEvent[];
  approvals: Approval[];
  onDecide: (requestId: string, approved: boolean) => Promise<void>;
  onResume: () => Promise<void>;
  onCancel: () => Promise<void>;
  onExport: () => Promise<void>;
}

export function RunInspector({
  run,
  events,
  approvals,
  onDecide,
  onResume,
  onCancel,
  onExport,
}: Props) {
  const [exportError, setExportError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);

  if (!run) {
    return (
      <section className="inspector panel empty-inspector">
        <p className="eyebrow">TRACE EXPLORER</p>
        <h2>Select or launch a run</h2>
        <p>
          Every model call, tool boundary, checkpoint, delegation, and policy
          decision is observable.
        </p>
      </section>
    );
  }

  const canResume = run.status === "paused";
  const canCancel = [
    "created",
    "running",
    "paused",
    "waiting_approval",
  ].includes(run.status);

  async function handleExport() {
    setExportError(null);
    setExporting(true);
    try {
      await onExport();
    } catch (reason) {
      setExportError(
        reason instanceof Error ? reason.message : "Trace export failed",
      );
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className="inspector panel" aria-labelledby="inspector-title">
      <div className="inspector-header">
        <div>
          <div className="title-with-status">
            <h2 id="inspector-title">{run.demo} trace</h2>
            <StatusPill status={run.status} />
          </div>
          <code>{run.id}</code>
        </div>
        <div className="run-actions">
          <button
            type="button"
            onClick={() => void handleExport()}
            disabled={exporting}
          >
            <Download size={14} /> {exporting ? "Exporting…" : "Export trace"}
          </button>
          {canResume && (
            <button type="button" onClick={() => void onResume()}>
              <RotateCcw size={14} /> Resume
            </button>
          )}
          {canCancel && (
            <button type="button" onClick={() => void onCancel()}>
              <Ban size={14} /> Cancel
            </button>
          )}
        </div>
      </div>
      {exportError && (
        <div className="error-card" role="alert">
          <span>Export failed</span>
          <p>{exportError}</p>
        </div>
      )}
      <MetricStrip metrics={run.metrics} />
      {approvals.map((approval) => (
        <ApprovalCard
          key={approval.requestId}
          approval={approval}
          onDecide={onDecide}
        />
      ))}
      {run.output && (
        <div className="output-card">
          <span>Final output</span>
          <p>{run.output}</p>
        </div>
      )}
      {run.error && (
        <div className="error-card">
          <span>Run failed</span>
          <p>{run.error}</p>
        </div>
      )}
      <div className="timeline-heading">
        <h3>Event timeline</h3>
        <span>{events.length} events · live</span>
      </div>
      <EventTimeline events={events} />
    </section>
  );
}
