import { render, screen } from "@testing-library/react";

import { StatusPill } from "./StatusPill";

describe("StatusPill", () => {
  it("formats compound status labels", () => {
    render(<StatusPill status="waiting_approval" />);
    expect(screen.getByText("waiting approval")).toBeInTheDocument();
  });
});
