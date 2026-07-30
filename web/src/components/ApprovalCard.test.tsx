import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import type { Approval } from "../types";
import { ApprovalCard } from "./ApprovalCard";

const approval: Approval = {
  requestId: "request-1",
  toolName: "workspace.write",
  reason: "Writing a file changes the workspace.",
  risk: "medium",
  expiresAt: "2030-01-01T00:00:00Z",
  arguments: { path: "slugify.py" },
  decided: true,
  approved: true,
};

describe("ApprovalCard", () => {
  it("only allows retrying the persisted decision during recovery", async () => {
    const user = userEvent.setup();
    const onDecide = vi.fn().mockResolvedValue(undefined);
    render(<ApprovalCard approval={approval} onDecide={onDecide} />);

    expect(screen.getByRole("button", { name: "Deny" })).toBeDisabled();
    await user.click(
      screen.getByRole("button", { name: "Resume approved action" }),
    );

    expect(onDecide).toHaveBeenCalledOnce();
    expect(onDecide).toHaveBeenCalledWith("request-1", true);
  });

  it("only allows retrying a persisted denial", async () => {
    const user = userEvent.setup();
    const onDecide = vi.fn().mockResolvedValue(undefined);
    render(
      <ApprovalCard
        approval={{ ...approval, approved: false }}
        onDecide={onDecide}
      />,
    );

    expect(screen.getByRole("button", { name: "Approve once" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Retry denial" }));

    expect(onDecide).toHaveBeenCalledOnce();
    expect(onDecide).toHaveBeenCalledWith("request-1", false);
  });
});
