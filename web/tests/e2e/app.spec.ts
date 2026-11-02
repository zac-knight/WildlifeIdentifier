import { expect, test } from "@playwright/test";

// A generated 256 x 256 PNG, built in the browser so that no image is stored in the repository.
async function samplePhoto(page: import("@playwright/test").Page): Promise<Buffer> {
  await page.goto("/");
  const base64 = await page.evaluate(async () => {
    const canvas = new OffscreenCanvas(256, 256);
    const context = canvas.getContext("2d")!;
    const gradient = context.createLinearGradient(0, 0, 256, 256);
    gradient.addColorStop(0, "#3a7");
    gradient.addColorStop(1, "#c84");
    context.fillStyle = gradient;
    context.fillRect(0, 0, 256, 256);
    const blob = await canvas.convertToBlob({ type: "image/png" });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return btoa(String.fromCharCode(...bytes));
  });
  return Buffer.from(base64, "base64");
}

test("navigation reaches every main page", async ({ page }) => {
  for (const [path, heading] of [
    ["/identify", "Identify a bird"],
    ["/species", "Species"],
    ["/camera-trap", "Camera trap batches"],
    ["/evaluation", "Evaluation"],
    ["/about", "About WildLens"],
  ]) {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading, level: 1 })).toBeVisible();
  }
});

test("the identify button waits for input", async ({ page }) => {
  await page.goto("/identify");
  await expect(page.getByRole("button", { name: "Identify", exact: true })).toBeDisabled();
});

test("a photo runs through the real browser pipeline", async ({ page }) => {
  test.setTimeout(120_000);
  const photo = await samplePhoto(page);
  const uploads: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST") uploads.push(request.url());
  });
  await page.goto("/identify");
  await page
    .getByLabel("Photo")
    .setInputFiles({ name: "bird.png", mimeType: "image/png", buffer: photo });
  await page.getByRole("button", { name: "Identify", exact: true }).click();
  await expect(page.getByTestId("result-summary")).toBeVisible({ timeout: 90_000 });
  await expect(page.getByTestId("candidates").locator("li")).toHaveCount(5);
  await expect(page.getByTestId("cam-canvas")).toBeAttached();
  // Tiny random models are never confident, so this must not be presented as an identification.
  await expect(page.getByTestId("result-summary")).not.toContainText("Very likely");
  expect(uploads).toEqual([]);
});

test("feedback explains when it is switched off", async ({ page }) => {
  test.setTimeout(120_000);
  const photo = await samplePhoto(page);
  await page.goto("/identify");
  await page
    .getByLabel("Photo")
    .setInputFiles({ name: "bird.png", mimeType: "image/png", buffer: photo });
  await page.getByRole("button", { name: "Identify", exact: true }).click();
  await page.getByRole("button", { name: "Yes, that's right" }).click({ timeout: 90_000 });
  await expect(page.getByText("Feedback is switched off")).toBeVisible();
});

test("a corrupt image gives a clear message", async ({ page }) => {
  await page.goto("/identify");
  await page.getByLabel("Photo").setInputFiles({
    name: "bad.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("not an image"),
  });
  await expect(page.getByText("could not be read as an image")).toBeVisible();
});

test("species search filters the list", async ({ page }) => {
  await page.goto("/species");
  await page.getByLabel("Search by name or family").fill("zzzzzz");
  await expect(page.getByText("No species match")).toBeVisible();
});

test("a species page loads", async ({ page }) => {
  await page.goto("/species/gymnorhina-tibicen");
  await expect(page.getByRole("heading", { name: "Australian Magpie", level: 1 })).toBeVisible();
});

test("an unknown species gives a not found page", async ({ page }) => {
  const response = await page.goto("/species/not-a-real-bird");
  expect(response?.status()).toBe(404);
});

test("keyboard users can skip to the content", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to content" })).toBeFocused();
});

test("the theme can be switched", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Switch to dark theme/ }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
});

test("the page fits a phone screen", async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 700 });
  for (const path of ["/", "/identify", "/evaluation"]) {
    await page.goto(path);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow, path).toBeLessThanOrEqual(0);
  }
});
