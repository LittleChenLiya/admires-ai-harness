import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { RunEvent } from "../types";
import { EventTimeline } from "./EventTimeline";

const event: RunEvent = {
  schema_version: "1",
  event_id: "event-1",
  run_id: "run-1",
  sequence: 3,
  type: "tool.completed",
  timestamp: "2026-07-29T00:00:00Z",
  parent_run_id: null,
  node_id: "search",
  trace_id: "trace-1",
  span_id: "span-1",
  payload: { tool: "workspace.search", matches: 2 },
};

describe("EventTimeline", () => {
  it("reveals structured payload on demand", async () => {
    const user = userEvent.setup();
    render(<EventTimeline events={[event]} />);
    await user.click(screen.getByRole("button", { expanded: false }));
    expect(screen.getByText(/"matches": 2/)).toBeInTheDocument();
  });
});
