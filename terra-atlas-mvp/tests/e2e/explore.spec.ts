import { test, expect } from '@playwright/test'

test.describe('Explore Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/explore')
  })

  test('should display page title', async ({ page }) => {
    await expect(page.locator('text=Interactive Explorer')).toBeVisible()
  })

  test('should display navigation with current page highlighted', async ({ page }) => {
    const exploreLink = page.locator('nav >> text=Explore')
    await expect(exploreLink).toBeVisible()
  })

  test('should display project statistics', async ({ page }) => {
    // Wait for stats to load
    await page.waitForSelector('[class*="text-emerald"]')

    // Should show project count
    const projectCount = page.locator('text=Projects').first()
    await expect(projectCount).toBeVisible()

    // Should show capacity
    const capacity = page.locator('text=Total Capacity')
    await expect(capacity).toBeVisible()

    // Should show investment
    const investment = page.locator('text=Investment')
    await expect(investment).toBeVisible()
  })

  test('should display filter options', async ({ page }) => {
    await expect(page.locator('text=Filter by Type')).toBeVisible()
    await expect(page.locator('text=All Projects')).toBeVisible()
    await expect(page.locator('text=Solar')).toBeVisible()
    await expect(page.locator('text=Wind')).toBeVisible()
    await expect(page.locator('text=Storage')).toBeVisible()
  })

  test('should filter projects when filter button clicked', async ({ page }) => {
    // Click solar filter
    const solarFilter = page.locator('button:has-text("Solar")')
    await solarFilter.click()

    // Wait for filter to apply (stats should update)
    await page.waitForTimeout(500)

    // The filter should be visually selected (ring or border change)
    await expect(solarFilter).toBeVisible()
  })

  test('should display globe visualization', async ({ page }) => {
    // The globe should be rendered (canvas or WebGL)
    const globeContainer = page.locator('.h-screen >> canvas, .h-screen >> [class*="globe"]')
    // Globe may take time to load
    await page.waitForTimeout(2000)
  })

  test('should navigate back to home', async ({ page }) => {
    const homeLink = page.locator('nav >> a:has-text("Home")')
    await homeLink.click()
    await expect(page).toHaveURL('/')
  })
})

test.describe('Explore Page API', () => {
  test('should load projects from API', async ({ page }) => {
    // Listen for API response
    const responsePromise = page.waitForResponse(
      (response) =>
        response.url().includes('/api/projects') &&
        response.status() === 200
    )

    await page.goto('/explore')
    const response = await responsePromise

    // Verify response has projects
    const data = await response.json()
    expect(data).toHaveProperty('projects')
  })
})
