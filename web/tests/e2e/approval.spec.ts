import { expect, test } from "@playwright/test";

test("a coding write waits for one-shot approval and exposes its trace", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("tab", { name: "Coding" }).click();
  await page
    .getByLabel("Task input")
    .fill("Create a browser-tested slugify utility");
  const startResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/runs") &&
      response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run workflow" }).click();
  const started = (await (await startResponse).json()) as { id: string };

  const inspector = page.getByRole("region", { name: "coding trace" });
  await expect(inspector.getByText(started.id, { exact: true })).toBeVisible();
  await expect(
    inspector.getByText("workspace.write", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    inspector.getByRole("button", { name: "Approve once" }),
  ).toBeVisible();
  await inspector.getByRole("button", { name: "Approve once" }).click();

  await expect(
    inspector.locator(".title-with-status .status-succeeded"),
  ).toHaveText("succeeded", { timeout: 15_000 });
  await expect(inspector.getByText(/Created `slugify.py`/)).toBeVisible();
  await expect(
    inspector.getByText("approval.consumed", { exact: true }),
  ).toBeVisible();
  await expect(
    inspector.getByText("run.completed", { exact: true }),
  ).toBeVisible();
});
