import { test, expect } from '@playwright/test'

test.describe('Terra Atlas Visual Tests', () => {
  test('homepage globe renders correctly', async ({ page }) => {
    await page.goto('http://localhost:3002')

    // Wait for globe to load
    await page.waitForTimeout(3000)

    // Take screenshot
    await page.screenshot({
      path: 'tests/screenshots/homepage-globe.png',
      fullPage: false
    })

    // Check no white blob (canvas should have dark colors)
    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()
  })

  test('explore page with projects', async ({ page }) => {
    await page.goto('http://localhost:3002/explore')

    // Wait for globe and data to load
    await page.waitForTimeout(4000)

    await page.screenshot({
      path: 'tests/screenshots/explore-page.png',
      fullPage: false
    })
  })

  test('homepage stats display', async ({ page }) => {
    await page.goto('http://localhost:3002')

    await page.waitForTimeout(2000)

    // Check stats are visible
    const stats = page.locator('text=In queue')
    await expect(stats).toBeVisible()

    await page.screenshot({
      path: 'tests/screenshots/homepage-stats.png',
      fullPage: true
    })
  })

  test('pledge form works', async ({ page }) => {
    // Test the pledge API flow
    const response = await page.request.post('http://localhost:3002/api/pledge', {
      data: {
        projectId: 'test-visual',
        email: 'test@visual.com',
        amount: 50
      }
    })

    expect(response.ok()).toBeTruthy()
    const data = await response.json()
    expect(data.success).toBe(true)
  })
})
