import { expect, test } from "@playwright/test";

test("dashboard renders the web upload flow and platform picker", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "VidMeta AI" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Generate platform metadata" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Resumable browser upload" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Local path or folder" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Upload and analyze" })).toBeVisible();
  await expect(page.getByText("Choose social platforms")).toBeVisible();
  await expect(page.getByRole("tab", { name: /Core video networks/ })).toBeVisible();

  await page.getByRole("button", { name: "Clear" }).click();
  await expect(page.getByText("0 selected for generation")).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
