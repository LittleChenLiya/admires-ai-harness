import { expect, test } from "@playwright/test";

test("a research run completes and exposes cited output and replayable events", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("tab", { name: "Research" }).click();
  await page
    .getByLabel("Task input")
    .fill("Why are event-sourced agent runtimes easier to debug?");
  await page.getByRole("button", { name: "Run workflow" }).click();

  await expect(
    page.getByText("succeeded", { exact: true }).first(),
  ).toBeVisible();
  await expect(page.getByText(/Sources/)).toBeVisible();
  await expect(page.getByText(/architecture.md/)).toBeVisible();
  await expect(page.getByText("run.completed", { exact: true })).toBeVisible();
});
