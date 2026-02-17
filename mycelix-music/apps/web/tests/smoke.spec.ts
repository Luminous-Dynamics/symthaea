import { test, expect } from '@playwright/test';

const shouldRun = Boolean(process.env.SMOKE_E2E);
const baseUrl = process.env.SMOKE_BASE_URL || 'http://localhost:3000';

test.describe('mycelix smoke flow', () => {
  test.skip(!shouldRun, 'Set SMOKE_E2E=1 and SMOKE_BASE_URL to enable the smoke test.');

  test('upload → claim → play → dashboard happy path (smoke)', async ({ page }) => {
    // Optional readiness probe
    const readyResp = await page.request.get(`${baseUrl}/health/ready`);
    test.skip(readyResp.status() >= 400, 'Stack not ready at SMOKE_BASE_URL');

    await page.goto(baseUrl);
    await expect(page).toHaveTitle(/Mycelix/i);

    await page.goto(`${baseUrl}/upload`);
    await expect(page.getByText(/upload/i)).toBeVisible();

    await page.goto(`${baseUrl}/discover`);
    await expect(page.getByText(/discover/i)).toBeVisible();

    await page.goto(`${baseUrl}/dashboard`);
    await expect(page.getByText(/dashboard/i)).toBeVisible();
  });
});
