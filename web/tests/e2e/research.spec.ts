import { expect, test } from "@playwright/test";

test("a research run completes and exposes cited output and replayable events", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("tab", { name: "Research" }).click();
  await page
    .getByLabel("Task input")
    .fill("Why are event-sourced agent runtimes easier to debug?");
  const startResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/runs") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run workflow" }).click();
  const started = (await (await startResponse).json()) as { id: string };

  const inspector = page.getByRole("region", { name: "research trace" });
  await expect(inspector.getByText(started.id, { exact: true })).toBeVisible();
  await expect(
    inspector.locator(".title-with-status .status-succeeded"),
  ).toHaveText("succeeded", { timeout: 15_000 });
  await expect(inspector.getByText(/Sources/)).toBeVisible();
  await expect(inspector.getByText(/architecture.md/)).toBeVisible();
  await expect(
    inspector.getByText("run.completed", { exact: true }),
  ).toBeVisible();
});
