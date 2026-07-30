import { expect, test } from "@playwright/test";

test("a coding write waits for one-shot approval and exposes its trace", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("tab", { name: "Coding" }).click();
  await page
    .getByLabel("Task input")
    .fill("Create a browser-tested slugify utility");
  await page.getByRole("button", { name: "Run workflow" }).click();

  await expect(
    page.getByText("workspace.write", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Approve once" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Approve once" }).click();

  await expect(
    page.getByText("succeeded", { exact: true }).first(),
  ).toBeVisible();
  await expect(page.getByText(/Created `slugify.py`/)).toBeVisible();
  await expect(
    page.getByText("approval.consumed", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText("run.completed", { exact: true })).toBeVisible();
});
