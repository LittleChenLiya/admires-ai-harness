import { AlertTriangle } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "./api";
import { Brand } from "./components/Brand";
import { DemoLauncher } from "./components/DemoLauncher";
import { RunInspector } from "./components/RunInspector";
import { RunList } from "./components/RunList";
import { useRunStream } from "./hooks/useRunStream";
import type { Approval, Demo, Run, RunEvent } from "./types";

export function App() {
  const [demos, setDemos] = useState<Demo[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [seedEvents, setSeedEvents] = useState<RunEvent[]>([]);
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const events = useRunStream(selectedId, seedEvents);
  const selected = useMemo(
    () => runs.find((run) => run.id === selectedId) ?? null,
    [runs, selectedId],
  );

  const refreshRuns = useCallback(async () => {
    const nextRuns = await api.runs();
    setRuns(nextRuns);
    return nextRuns;
  }, []);

  const selectRun = useCallback(async (id: string) => {
    setSelectedId(id);
    const [run, nextEvents, nextApprovals] = await Promise.all([
      api.run(id),
      api.events(id),
      api.approvals(id),
    ]);
    setRuns((current) => [run, ...current.filter((item) => item.id !== id)]);
    setSeedEvents(nextEvents);
    setApprovals(nextApprovals);
  }, []);

  useEffect(() => {
    void Promise.all([api.demos(), refreshRuns()])
      .then(([nextDemos, nextRuns]) => {
        setDemos(nextDemos);
        if (nextRuns[0]) void selectRun(nextRuns[0].id);
      })
      .catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : String(reason)),
      );
  }, [refreshRuns, selectRun]);

  useEffect(() => {
    const event = events.at(-1);
    if (!selectedId || !event) return;
    if (event.type.startsWith("run.") || event.type.startsWith("approval.")) {
      void Promise.all([api.run(selectedId), api.approvals(selectedId)]).then(
        ([nextRun, nextApprovals]) => {
          setRuns((current) => [
            nextRun,
            ...current.filter((item) => item.id !== selectedId),
          ]);
          setApprovals(nextApprovals);
        },
      );
    }
  }, [events, selectedId]);

  async function launch(demo: Demo, input: string) {
    setBusy(true);
    setError(null);
    try {
      const run = await api.start(demo.id, input);
      setRuns((current) => [run, ...current]);
      await selectRun(run.id);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  async function decide(requestId: string, approved: boolean) {
    if (!selectedId) return;
    await api.decide(selectedId, requestId, approved);
    await selectRun(selectedId);
  }

  async function act(action: "resume" | "cancel") {
    if (!selectedId) return;
    await api[action](selectedId);
    await selectRun(selectedId);
  }

  async function exportTrace() {
    if (!selectedId) return;
    const trace = await api.exportTrace(selectedId);
    const blob = new Blob([JSON.stringify(trace, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `trace-${selectedId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-shell">
      <Brand />
      {error && (
        <div className="global-error" role="alert">
          <AlertTriangle size={16} />
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
      <main>
        <aside>
          <DemoLauncher demos={demos} busy={busy} onLaunch={launch} />
          <RunList
            runs={runs}
            selectedId={selectedId}
            onSelect={(id) => void selectRun(id)}
          />
        </aside>
        <RunInspector
          run={selected}
          events={events}
          approvals={approvals}
          onDecide={decide}
          onResume={() => act("resume")}
          onCancel={() => act("cancel")}
          onExport={exportTrace}
        />
      </main>
      <footer>
        <span>Event-sourced · Safe by default · Fully replayable</span>
        <span>v1.0.0</span>
      </footer>
    </div>
  );
}
