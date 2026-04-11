// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * E2E Tests - Trust System (MATL Algorithm)
 *
 * Tests trust attestations, network visualization, and MATL scoring
 */

import { test, expect, Page } from '@playwright/test';

test.describe('Trust System', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    const context = await browser.newContext();
    page = await context.newPage();
    await page.goto('/');
    await expect(page.getByTestId('inbox-view')).toBeVisible({ timeout: 10000 });
  });

  test.afterEach(async () => {
    await page.close();
  });

  test('should display trust network visualization', async () => {
    // Navigate to trust section
    await page.getByRole('link', { name: /trust/i }).click();

    // Wait for trust network graph to load
    await expect(page.getByTestId('trust-network-graph')).toBeVisible({ timeout: 5000 });

    // Verify SVG graph is rendered
    const svg = page.locator('svg.trust-graph');
    await expect(svg).toBeVisible();

    // Verify nodes are present
    const nodes = page.locator('.trust-graph .node');
    await expect(nodes).toHaveCount(1, { timeout: 5000 }); // At minimum, user's own node

    // Verify "me" node is highlighted
    await expect(page.locator('.node.me')).toBeVisible();
  });

  test('should create a trust attestation', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click "Add Trust" button
    await page.getByRole('button', { name: /add trust/i }).click();

    // Wait for attestation modal
    await expect(page.getByTestId('attestation-modal')).toBeVisible();

    // Fill in agent details
    await page.getByLabel('Agent ID or Email').fill('alice@example.com');

    // Set trust level using slider
    const slider = page.getByRole('slider', { name: /trust level/i });
    await slider.fill('80'); // 80% trust

    // Add context
    await page.getByLabel('Context').fill('colleague');

    // Submit attestation
    await page.getByRole('button', { name: /create attestation/i }).click();

    // Verify success
    await expect(page.getByText(/attestation created/i)).toBeVisible();

    // Verify new node appears in graph
    await expect(page.getByText('alice@example.com')).toBeVisible();
  });

  test('should view trust score details', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click on a node in the graph
    const node = page.locator('.node').first();
    await node.click();

    // Wait for details panel
    await expect(page.getByTestId('trust-details-panel')).toBeVisible();

    // Verify trust score is displayed
    await expect(page.getByText(/trust score/i)).toBeVisible();
    await expect(page.getByText(/%/)).toBeVisible();

    // Verify MATL breakdown
    await expect(page.getByText(/matl/i)).toBeVisible();
    await expect(page.getByText(/direct trust/i)).toBeVisible();
    await expect(page.getByText(/network trust/i)).toBeVisible();
    await expect(page.getByText(/temporal/i)).toBeVisible();
  });

  test('should update an existing attestation', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click on an existing connection
    const connection = page.locator('.trust-edge').first();
    if (await connection.isVisible()) {
      await connection.click();

      // Wait for edit panel
      await expect(page.getByTestId('attestation-edit-panel')).toBeVisible();

      // Update trust level
      const slider = page.getByRole('slider', { name: /trust level/i });
      const currentValue = await slider.inputValue();
      const newValue = parseInt(currentValue) + 10;
      await slider.fill(String(Math.min(newValue, 100)));

      // Save changes
      await page.getByRole('button', { name: /save/i }).click();

      // Verify success
      await expect(page.getByText(/attestation updated/i)).toBeVisible();
    }
  });

  test('should revoke a trust attestation', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click on an existing node
    const node = page.locator('.node:not(.me)').first();
    if (await node.isVisible()) {
      await node.click();

      // Click revoke button
      await page.getByRole('button', { name: /revoke/i }).click();

      // Confirm revocation
      await expect(page.getByTestId('confirm-dialog')).toBeVisible();
      await page.getByRole('button', { name: /confirm/i }).click();

      // Verify success
      await expect(page.getByText(/attestation revoked/i)).toBeVisible();
    }
  });

  test('should filter trust network by group', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click filter dropdown
    await page.getByRole('combobox', { name: /filter/i }).click();

    // Select a group
    await page.getByRole('option', { name: /work/i }).click();

    // Verify nodes are filtered
    await page.waitForTimeout(500);
    const nodes = page.locator('.node:not(.me)');
    const filteredNodes = await nodes.all();

    for (const node of filteredNodes) {
      const group = await node.getAttribute('data-group');
      expect(group).toBe('work');
    }
  });

  test('should change color scheme', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click color scheme selector
    await page.getByRole('button', { name: /color scheme/i }).click();

    // Select "group" scheme
    await page.getByRole('option', { name: /group/i }).click();

    // Verify nodes are recolored by group
    await expect(page.locator('.color-scheme-group')).toBeVisible();
  });

  test('should zoom and pan the graph', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    const graph = page.getByTestId('trust-network-graph');

    // Get initial transform
    const initialTransform = await graph.locator('g.container').getAttribute('transform');

    // Zoom in using button
    await page.getByRole('button', { name: /zoom in/i }).click();
    await page.waitForTimeout(300);

    // Verify zoom changed
    const zoomedTransform = await graph.locator('g.container').getAttribute('transform');
    expect(zoomedTransform).not.toBe(initialTransform);

    // Reset zoom
    await page.getByRole('button', { name: /reset/i }).click();
    await page.waitForTimeout(300);

    // Verify zoom reset
    const resetTransform = await graph.locator('g.container').getAttribute('transform');
    expect(resetTransform).toContain('scale(1)');
  });

  test('should show trust statistics', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Verify stats panel is visible
    const statsPanel = page.getByTestId('trust-stats');
    await expect(statsPanel).toBeVisible();

    // Verify stats are displayed
    await expect(statsPanel.getByText(/nodes/i)).toBeVisible();
    await expect(statsPanel.getByText(/connections/i)).toBeVisible();
    await expect(statsPanel.getByText(/avg trust/i)).toBeVisible();
  });

  test('should export trust network', async () => {
    await page.getByRole('link', { name: /trust/i }).click();

    // Click export button
    await page.getByRole('button', { name: /export/i }).click();

    // Select export format
    await page.getByRole('option', { name: /json/i }).click();

    // Wait for download
    const download = await page.waitForEvent('download');

    // Verify download started
    expect(download.suggestedFilename()).toContain('trust-network');
    expect(download.suggestedFilename()).toContain('.json');
  });

  test('should display trust indicators in inbox', async () => {
    // Navigate to inbox
    await page.getByRole('link', { name: /inbox/i }).click();

    // Check that trust badges are visible on emails
    const emails = page.getByTestId('email-list-item');
    const count = await emails.count();

    if (count > 0) {
      // Verify first email has trust indicator
      const firstEmail = emails.first();
      const trustBadge = firstEmail.getByTestId('trust-badge');

      // Trust badge should be visible
      await expect(trustBadge).toBeVisible();

      // Verify trust level is displayed
      const trustLevel = await trustBadge.textContent();
      expect(trustLevel).toMatch(/%|trusted|unknown/i);
    }
  });
});
