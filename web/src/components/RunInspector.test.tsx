import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Run } from "../types";
import { RunInspector } from "./RunInspector";

const run: Run = {
  id: "run-1",
  demo: "research",
  input: "probe",
  status: "succeeded",
  createdAt: "2026-07-29T00:00:00Z",
  updatedAt: "2026-07-29T00:00:01Z",
  output: "done",
  error: null,
  metrics: {
    steps: 2,
    modelCalls: 1,
    toolCalls: 1,
    inputTokens: 10,
    outputTokens: 4,
    costUsd: 0,
    durationMs: 1000,
  },
};

function renderInspector(onExport: () => Promise<void>) {
  return render(
    <RunInspector
      run={run}
      events={[]}
      approvals={[]}
      onDecide={async () => {}}
      onResume={async () => {}}
      onCancel={async () => {}}
      onExport={onExport}
    />,
  );
}

describe("RunInspector export", () => {
  it("triggers the export action", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn().mockResolvedValue(undefined);
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export trace/i }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it("surfaces export failures inline", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn().mockRejectedValue(new Error("run not found"));
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export trace/i }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Export failed");
    expect(screen.getByRole("alert")).toHaveTextContent("run not found");
  });
});
