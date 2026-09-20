import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Run } from "../types";
import { RunInspector } from "./RunInspector";

const run: Run = {
  id: "run-1",
  demo: "research",
  input: "Why use event sourcing?",
  status: "succeeded",
  createdAt: "2026-07-29T00:00:00Z",
  updatedAt: "2026-07-29T00:00:05Z",
  output: "answer",
  error: null,
  metrics: {
    steps: 2,
    modelCalls: 1,
    toolCalls: 1,
    inputTokens: 10,
    outputTokens: 4,
    costUsd: 0,
    durationMs: 5000,
  },
};

function renderInspector(onExport: () => Promise<void>) {
  return render(
    <RunInspector
      run={run}
      events={[]}
      approvals={[]}
      onDecide={() => Promise.resolve()}
      onResume={() => Promise.resolve()}
      onCancel={() => Promise.resolve()}
      onExport={onExport}
    />,
  );
}

describe("RunInspector export", () => {
  it("invokes the export action from the inspector header", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn().mockResolvedValue(undefined);
    renderInspector(onExport);

    await user.click(screen.getByRole("button", { name: /export/i }));

    expect(onExport).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("surfaces an inline error when the export fails", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn().mockRejectedValue(new Error("run not found"));
    renderInspector(onExport);

    await user.click(screen.getByRole("button", { name: /export/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("run not found");
  });
});
