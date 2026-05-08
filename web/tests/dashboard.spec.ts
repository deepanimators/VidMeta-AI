import { expect, test } from "@playwright/test";

test("dashboard renders and validates local path input", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "VidMeta AI" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Generate platform metadata" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Analyze local path" })).toBeVisible();

  await page.getByRole("button", { name: "Analyze local path" }).click();
  await expect(page.getByText("Enter a local file or folder path")).toBeVisible();
  expect(consoleErrors).toEqual([]);
});
