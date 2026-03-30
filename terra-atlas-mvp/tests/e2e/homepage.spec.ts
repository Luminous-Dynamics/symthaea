// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { test, expect } from '@playwright/test'

test.describe('Homepage', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should display Terra Atlas branding', async ({ page }) => {
    await expect(page.locator('text=Terra')).toBeVisible()
    await expect(page.locator('text=Atlas')).toBeVisible()
  })

  test('should display hero headline', async ({ page }) => {
    await expect(page.locator('text=What if energy was')).toBeVisible()
    await expect(page.locator('text=essentially free?')).toBeVisible()
  })

  test('should display stats cards', async ({ page }) => {
    // Wait for stats to load (either fallback or real data)
    await expect(page.locator('text=Projects')).toBeVisible()
    await expect(page.locator('text=Total capacity')).toBeVisible()
    await expect(page.locator('text=Jobs enabled')).toBeVisible()
    await expect(page.locator('text=Homes powered')).toBeVisible()
  })

  test('should have working navigation links', async ({ page }) => {
    // Desktop navigation
    const exploreLink = page.locator('nav >> text=Explore').first()
    await expect(exploreLink).toBeVisible()

    const horizonLink = page.locator('nav >> text=Horizon').first()
    await expect(horizonLink).toBeVisible()

    const apiLink = page.locator('nav >> text=API').first()
    await expect(apiLink).toBeVisible()
  })

  test('should display primary CTA button', async ({ page }) => {
    const ctaButton = page.locator('text=Explore Energy Projects')
    await expect(ctaButton).toBeVisible()
  })

  test('should display trust badges', async ({ page }) => {
    await expect(page.locator('text=Live data streams')).toBeVisible()
    await expect(page.locator('text=Regulated & compliant')).toBeVisible()
  })

  test('should navigate to explore page when CTA clicked', async ({ page }) => {
    const ctaButton = page.locator('a:has-text("Explore Energy Projects")')
    await ctaButton.click()
    await expect(page).toHaveURL(/\/explore/)
  })

  test('should display scroll-to-proof button', async ({ page }) => {
    const scrollButton = page.locator('[aria-label="Scroll to proof section"]')
    await expect(scrollButton).toBeVisible()
  })
})

test.describe('Homepage Mobile', () => {
  test.use({ viewport: { width: 375, height: 667 } })

  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('should show hamburger menu on mobile', async ({ page }) => {
    const menuButton = page.locator('[aria-label="Open menu"]')
    await expect(menuButton).toBeVisible()
  })

  test('should open mobile menu when hamburger clicked', async ({ page }) => {
    const menuButton = page.locator('[aria-label="Open menu"]')
    await menuButton.click()

    // Menu should show navigation links
    await expect(page.locator('text=Explore').last()).toBeVisible()
    await expect(page.locator('text=Horizon').last()).toBeVisible()
    await expect(page.locator('text=API').last()).toBeVisible()
    await expect(page.locator('text=Start Investing').last()).toBeVisible()
  })

  test('should close mobile menu when overlay clicked', async ({ page }) => {
    const menuButton = page.locator('[aria-label="Open menu"]')
    await menuButton.click()

    // Wait for menu to appear
    await page.waitForTimeout(300)

    // Click overlay to close
    await page.locator('.bg-black\\/80').click()

    // Menu should close (hamburger visible again)
    await expect(page.locator('[aria-label="Open menu"]')).toBeVisible()
  })

  test('should close mobile menu after navigation', async ({ page }) => {
    const menuButton = page.locator('[aria-label="Open menu"]')
    await menuButton.click()

    // Click a navigation link
    await page.locator('a:has-text("Explore")').last().click()

    // Should navigate to explore
    await expect(page).toHaveURL(/\/explore/)
  })
})
