import { defineConfig, devices } from "@playwright/test";

const runSuffix = `${Date.now()}-${process.pid}`;

export default defineConfig({
  testDir: "./tests/e2e",
  // The reference backend deliberately uses one local SQLite database. Serial
  // browser workflows make the CI path deterministic and keep traces readable.
  fullyParallel: false,
  workers: 1,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      command: "uv run admire serve --host 127.0.0.1 --port 8000",
      cwd: "..",
      url: "http://127.0.0.1:8000/api/health",
      reuseExistingServer: false,
      timeout: 60_000,
      env: {
        ADMIRE_DATABASE_PATH: `.admire/e2e-${runSuffix}/harness.db`,
        ADMIRE_WORKSPACE_ROOT: `.admire/e2e-${runSuffix}/workspaces`,
        ADMIRE_MODEL_PROVIDER: "mock",
        ADMIRE_ALLOW_NETWORK: "false",
      },
    },
    {
      command: "pnpm dev --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
