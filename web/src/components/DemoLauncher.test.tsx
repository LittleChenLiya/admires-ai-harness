import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import type { Demo } from "../types";
import { DemoLauncher } from "./DemoLauncher";

const demos: Demo[] = [
  {
    id: "research",
    title: "Research",
    description: "Synthesize evidence.",
    agents: ["researcher"],
    defaultInput: "Explain event sourcing.",
    capabilities: ["RAG", "citations"],
  },
];

describe("DemoLauncher", () => {
  it("launches the selected demo with editable input", async () => {
    const user = userEvent.setup();
    const onLaunch = vi.fn().mockResolvedValue(undefined);
    render(<DemoLauncher demos={demos} busy={false} onLaunch={onLaunch} />);
    await user.clear(screen.getByLabelText("Task input"));
    await user.type(screen.getByLabelText("Task input"), "New question");
    await user.click(screen.getByRole("button", { name: /run workflow/i }));
    expect(onLaunch).toHaveBeenCalledWith(demos[0], "New question");
  });
});
