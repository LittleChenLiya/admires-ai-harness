import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { Run } from "../types";
import { RunInspector } from "./RunInspector";

const run: Run = {
  id: "run-1",
  demo: "research",
  input: "Why use event sourcing?",
  status: "succeeded",
  createdAt: "2026-07-29T00:00:00Z",
  updatedAt: "2026-07-29T00:00:01Z",
  output: "done",
  error: null,
  metrics: {
    steps: 2,
    modelCalls: 1,
    toolCalls: 0,
    inputTokens: 10,
    outputTokens: 5,
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

describe("RunInspector export", () => {
  it("triggers the export action from the toolbar", async () => {
    const user = userEvent.setup();
    const onExport = vi.fn<() => Promise<void>>().mockResolvedValue();
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export/i }));
    expect(onExport).toHaveBeenCalledTimes(1);
  });

  it("surfaces an error when the export request fails", async () => {
    const user = userEvent.setup();
    const onExport = vi
      .fn<() => Promise<void>>()
      .mockRejectedValue(new Error("run not found"));
    renderInspector(onExport);
    await user.click(screen.getByRole("button", { name: /export/i }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Export failed");
    expect(alert).toHaveTextContent("run not found");
  });

  it("disables the button while an export is in flight", async () => {
    const user = userEvent.setup();
    let release: () => void = () => undefined;
    const onExport = vi.fn<() => Promise<void>>(
      () =>
        new Promise<void>((resolve) => {
          release = resolve;
        }),
    );
    renderInspector(onExport);
    const button = screen.getByRole("button", { name: /export/i });
    await user.click(button);
    await waitFor(() => expect(button).toBeDisabled());
    release();
    await waitFor(() => expect(button).toBeEnabled());
  });
});
