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
  updatedAt: "2026-07-29T00:01:00Z",
  output: "answer",
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
      onDecide={() => Promise.resolve()}
      onResume={() => Promise.resolve()}
      onCancel={() => Promise.resolve()}
      onExport={onExport}
    />,
  );
}

describe("RunInspector trace export", () => {
  it("offers an export action that invokes the handler", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn(() => Promise.resolve());
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export trace/i }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it("surfaces export failures through the rejected handler", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn(() => Promise.reject(new Error("run not found")));
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export trace/i }));
    expect(onExport).toHaveBeenCalledTimes(1);
    const firstCall = onExport.mock.results[0];
    expect(firstCall?.type).toBe("return");
    await expect(firstCall?.value).rejects.toThrow("run not found");
  });
});
