import { api } from "./api";
import type { TraceExport } from "./types";

const trace: TraceExport = {
  schema_version: "1",
  run_id: "run-1",
  trace_id: "trace-1",
  status: "succeeded",
  exported_at: "2026-07-29T00:01:00Z",
  total_events: 2,
  truncated: false,
  events: [],
};

describe("api.exportTrace", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("requests the export endpoint and parses the envelope", async () => {
    const fetchMock = vi.fn(() =>
      Promise.resolve(
        new Response(JSON.stringify(trace), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const result = await api.exportTrace("run-1");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/runs/run-1/export",
      expect.objectContaining({
        headers: expect.objectContaining({
          "Content-Type": "application/json",
        }),
      }),
    );
    expect(result.schema_version).toBe("1");
    expect(result.total_events).toBe(2);
  });

  it("rejects with the server detail when the run is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response(JSON.stringify({ detail: "run not found" }), {
            status: 404,
          }),
        ),
      ),
    );
    await expect(api.exportTrace("missing")).rejects.toThrow(/run not found/);
  });
});
