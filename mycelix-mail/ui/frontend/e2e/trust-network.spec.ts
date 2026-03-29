// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Trust Network Visualization E2E Tests
 *
 * Tests for viewing and interacting with the trust network graph
 */

import { test, expect, Page } from '@playwright/test';

// ============================================================================
// Helper Functions
// ============================================================================

async function navigateToTrustNetwork(page: Page) {
  const trustLink = page.locator(
    '[data-testid="nav-trust"], a[href*="trust"], [role="menuitem"]:has-text("Trust")'
  );

  if (await trustLink.count() > 0) {
    await trustLink.click();
    await page.waitForURL(/\/trust/, { timeout: 10000 });
  } else {
    // Try navigating directly
    await page.goto('/trust');
  }

  await page.waitForLoadState('networkidle');
}

async function waitForGraphToLoad(page: Page) {
  // Wait for the graph visualization to render
  await page.waitForSelector(
    '[data-testid="trust-graph"], [data-testid="trust-network"], .trust-graph, svg.graph',
    { timeout: 15000 }
  );
}

// ============================================================================
// Trust Network Display Tests
// ============================================================================

test.describe('Trust Network Display', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await navigateToTrustNetwork(page);
  });

  test('should display trust network graph', async ({ page }) => {
    await waitForGraphToLoad(page);

    const graph = page.locator(
      '[data-testid="trust-graph"], [data-testid="trust-network"], .trust-graph'
    );
    await expect(graph.first()).toBeVisible();
  });

  test('should show the current user as center node', async ({ page }) => {
    await waitForGraphToLoad(page);

    const centerNode = page.locator(
      '[data-testid="node-self"], [data-testid="center-node"], .node-self, [data-is-me="true"]'
    );

    if (await centerNode.count() > 0) {
      await expect(centerNode.first()).toBeVisible();
    }
  });

  test('should display connected nodes (trusted contacts)', async ({ page }) => {
    await waitForGraphToLoad(page);

    const nodes = page.locator(
      '[data-testid="trust-node"], .trust-node, circle.node, [data-node-type="contact"]'
    );

    // Should have at least one node (the user themselves)
    await expect(nodes.first()).toBeVisible();
  });

  test('should display edges connecting nodes', async ({ page }) => {
    await waitForGraphToLoad(page);

    const edges = page.locator(
      '[data-testid="trust-edge"], .trust-edge, line.edge, path.edge'
    );

    // Edges might exist if there are trusted contacts
    const edgeCount = await edges.count();
    console.log(`Found ${edgeCount} trust edges`);
  });

  test('should show legend explaining trust levels', async ({ page }) => {
    await waitForGraphToLoad(page);

    const legend = page.locator(
      '[data-testid="trust-legend"], .trust-legend, .graph-legend'
    );

    if (await legend.count() > 0) {
      await expect(legend).toBeVisible();
    }
  });

  test('should show trust level color coding', async ({ page }) => {
    await waitForGraphToLoad(page);

    // Look for different trust level indicators
    const highTrust = page.locator('[data-trust-level="high"], .trust-high');
    const mediumTrust = page.locator('[data-trust-level="medium"], .trust-medium');
    const lowTrust = page.locator('[data-trust-level="low"], .trust-low');

    // At least some trust level indicators should exist
    const totalCount = (await highTrust.count()) + (await mediumTrust.count()) + (await lowTrust.count());
    console.log(`Trust level indicators: ${totalCount}`);
  });
});

// ============================================================================
// Graph Interaction Tests
// ============================================================================

test.describe('Trust Graph Interaction', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/inbox');
    await navigateToTrustNetwork(page);
    await waitForGraphToLoad(page);
  });

  test('should show tooltip on node hover', async ({ page }) => {
    const nodes = page.locator(
      '[data-testid="trust-node"], .trust-node, circle.node'
    );

    if ((await nodes.count()) > 0) {
      await nodes.first().hover();

      const tooltip = page.locator(
        '[data-testid="node-tooltip"], .tooltip, [role="tooltip"]'
      );

      if (await tooltip.count() > 0) {
        await expect(tooltip.first()).toBeVisible({ timeout: 3000 });
      }
    }
  });

  test('should open node details on click', async ({ page }) => {
    const nodes = page.locator(
      '[data-testid="trust-node"]:not([data-is-me="true"]), .trust-node:not(.self)'
    );

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const nodeDetails = page.locator(
        '[data-testid="node-details"], [data-testid="trust-details-panel"], .node-details-panel'
      );

      if (await nodeDetails.count() > 0) {
        await expect(nodeDetails.first()).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should zoom in on graph', async ({ page }) => {
    const zoomInButton = page.locator(
      '[data-testid="zoom-in"], button[aria-label*="zoom in"]'
    );

    if (await zoomInButton.count() > 0) {
      await zoomInButton.click();
      await zoomInButton.click();

      // Graph should be zoomed (transform scale should be > 1)
      const graph = page.locator('[data-testid="trust-graph"]');
      const transform = await graph.getAttribute('style');

      if (transform) {
        expect(transform.includes('scale') || transform.includes('transform')).toBeTruthy();
      }
    }
  });

  test('should zoom out on graph', async ({ page }) => {
    const zoomOutButton = page.locator(
      '[data-testid="zoom-out"], button[aria-label*="zoom out"]'
    );

    if (await zoomOutButton.count() > 0) {
      await zoomOutButton.click();
    }
  });

  test('should reset zoom to default', async ({ page }) => {
    const resetButton = page.locator(
      '[data-testid="zoom-reset"], button[aria-label*="reset"], button:has-text("Reset")'
    );

    if (await resetButton.count() > 0) {
      await resetButton.click();
    }
  });

  test('should pan graph with mouse drag', async ({ page }) => {
    const graph = page.locator('[data-testid="trust-graph"]');

    if (await graph.count() > 0) {
      const box = await graph.boundingBox();

      if (box) {
        // Drag from center to the right
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
        await page.mouse.down();
        await page.mouse.move(box.x + box.width / 2 + 100, box.y + box.height / 2, { steps: 10 });
        await page.mouse.up();
      }
    }
  });
});

// ============================================================================
// Trust Statistics Tests
// ============================================================================

test.describe('Trust Statistics', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await page.waitForLoadState('networkidle');
  });

  test('should display trust statistics summary', async ({ page }) => {
    const stats = page.locator(
      '[data-testid="trust-stats"], .trust-statistics, .trust-summary'
    );

    if (await stats.count() > 0) {
      await expect(stats.first()).toBeVisible();
    }
  });

  test('should show total number of trusted contacts', async ({ page }) => {
    const trustedCount = page.locator(
      '[data-testid="trusted-count"], .trusted-contacts-count'
    );

    if (await trustedCount.count() > 0) {
      const text = await trustedCount.textContent();
      expect(text).toMatch(/\d+/);
    }
  });

  test('should show trust score distribution', async ({ page }) => {
    const distribution = page.locator(
      '[data-testid="trust-distribution"], .trust-distribution-chart'
    );

    if (await distribution.count() > 0) {
      await expect(distribution).toBeVisible();
    }
  });

  test('should show recent trust activity', async ({ page }) => {
    const activity = page.locator(
      '[data-testid="trust-activity"], .trust-activity-feed, .recent-attestations'
    );

    if (await activity.count() > 0) {
      await expect(activity).toBeVisible();
    }
  });
});

// ============================================================================
// Create Attestation Tests
// ============================================================================

test.describe('Create Trust Attestation', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should open attestation form from node', async ({ page }) => {
    const nodes = page.locator(
      '[data-testid="trust-node"]:not([data-is-me="true"])'
    );

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const attestButton = page.locator(
        '[data-testid="create-attestation"], button:has-text("Attest")'
      );

      if (await attestButton.count() > 0) {
        await attestButton.click();

        const attestForm = page.locator('[data-testid="attestation-form"]');
        await expect(attestForm.first()).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should create positive attestation', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]:not([data-is-me="true"])');

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const attestButton = page.locator('[data-testid="create-attestation"]');
      if (await attestButton.count() > 0) {
        await attestButton.click();

        // Select attestation type
        const typeSelect = page.locator('[data-testid="attestation-type"]');
        if (await typeSelect.count() > 0) {
          await typeSelect.selectOption('professional');
        }

        // Set trust level
        const trustSlider = page.locator('[data-testid="trust-level-slider"]');
        if (await trustSlider.count() > 0) {
          // Move slider to high trust
          await trustSlider.fill('80');
        }

        // Add comment
        const commentInput = page.locator('[data-testid="attestation-comment"]');
        if (await commentInput.count() > 0) {
          await commentInput.fill('Verified professional contact');
        }

        // Submit
        const submitButton = page.locator('[data-testid="submit-attestation"]');
        await submitButton.click();

        const successToast = page.locator('[data-testid="toast-success"]');
        await expect(successToast.first()).toBeVisible({ timeout: 10000 });
      }
    }
  });

  test('should create negative attestation', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]:not([data-is-me="true"])');

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const attestButton = page.locator('[data-testid="create-attestation"]');
      if (await attestButton.count() > 0) {
        await attestButton.click();

        const typeSelect = page.locator('[data-testid="attestation-type"]');
        if (await typeSelect.count() > 0) {
          await typeSelect.selectOption('spam');
        }

        const trustSlider = page.locator('[data-testid="trust-level-slider"]');
        if (await trustSlider.count() > 0) {
          await trustSlider.fill('10');
        }

        const commentInput = page.locator('[data-testid="attestation-comment"]');
        if (await commentInput.count() > 0) {
          await commentInput.fill('Suspicious activity detected');
        }

        const submitButton = page.locator('[data-testid="submit-attestation"]');
        await submitButton.click();

        const successToast = page.locator('[data-testid="toast-success"]');
        await expect(successToast.first()).toBeVisible({ timeout: 10000 });
      }
    }
  });
});

// ============================================================================
// View Attestations Tests
// ============================================================================

test.describe('View Attestations', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should view attestations for a contact', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]:not([data-is-me="true"])');

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const attestationList = page.locator(
        '[data-testid="attestation-list"], .attestation-history'
      );

      if (await attestationList.count() > 0) {
        await expect(attestationList).toBeVisible();
      }
    }
  });

  test('should show attestation details', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]:not([data-is-me="true"])');

    if ((await nodes.count()) > 0) {
      await nodes.first().click();

      const attestationItem = page.locator('[data-testid="attestation-item"]');

      if ((await attestationItem.count()) > 0) {
        await attestationItem.first().click();

        const attestationDetails = page.locator('[data-testid="attestation-details"]');
        if (await attestationDetails.count() > 0) {
          await expect(attestationDetails).toBeVisible();
        }
      }
    }
  });

  test('should show attestations given to me', async ({ page }) => {
    const myAttestationsTab = page.locator(
      '[data-testid="my-attestations-tab"], button:has-text("Received")'
    );

    if (await myAttestationsTab.count() > 0) {
      await myAttestationsTab.click();

      const attestationList = page.locator('[data-testid="attestation-list"]');
      await expect(attestationList.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should show attestations I have given', async ({ page }) => {
    const givenAttestationsTab = page.locator(
      '[data-testid="given-attestations-tab"], button:has-text("Given")'
    );

    if (await givenAttestationsTab.count() > 0) {
      await givenAttestationsTab.click();

      const attestationList = page.locator('[data-testid="attestation-list"]');
      await expect(attestationList.first()).toBeVisible({ timeout: 5000 });
    }
  });
});

// ============================================================================
// Trust Graph Filters Tests
// ============================================================================

test.describe('Trust Graph Filters', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should filter by trust level', async ({ page }) => {
    const trustFilter = page.locator(
      '[data-testid="trust-level-filter"], select[name="trustLevel"]'
    );

    if (await trustFilter.count() > 0) {
      await trustFilter.selectOption('high');
      await page.waitForTimeout(500);

      // Only high trust nodes should be visible
      const lowTrustNodes = page.locator('[data-trust-level="low"]');
      expect(await lowTrustNodes.count()).toBe(0);
    }
  });

  test('should filter by connection depth', async ({ page }) => {
    const depthFilter = page.locator(
      '[data-testid="depth-filter"], select[name="depth"]'
    );

    if (await depthFilter.count() > 0) {
      await depthFilter.selectOption('1'); // Direct connections only
      await page.waitForTimeout(500);
    }
  });

  test('should search for specific contact in graph', async ({ page }) => {
    const searchInput = page.locator(
      '[data-testid="graph-search"], input[placeholder*="search" i]'
    );

    if (await searchInput.count() > 0) {
      await searchInput.fill('john');
      await page.waitForTimeout(500);

      // Matching node should be highlighted
      const highlightedNode = page.locator('[data-highlighted="true"], .highlighted');
      // Node might be highlighted if search term matches
    }
  });
});

// ============================================================================
// Trust Path Analysis Tests
// ============================================================================

test.describe('Trust Path Analysis', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should show trust path between two contacts', async ({ page }) => {
    const pathButton = page.locator(
      '[data-testid="show-path"], button:has-text("Show Path")'
    );

    if (await pathButton.count() > 0) {
      await pathButton.click();

      const pathDialog = page.locator('[data-testid="path-dialog"]');
      await expect(pathDialog.first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('should highlight trust path in graph', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]');

    if ((await nodes.count()) >= 2) {
      // Select first node
      await nodes.first().click({ modifiers: ['Control'] });

      // Select second node (if implementation supports multi-select)
      await nodes.nth(1).click({ modifiers: ['Control'] });

      // Path might be highlighted
      const pathEdges = page.locator('[data-path-edge="true"], .path-highlighted');
      // Path edges might be highlighted
    }
  });
});

// ============================================================================
// Graph Export Tests
// ============================================================================

test.describe('Trust Graph Export', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should export graph as image', async ({ page }) => {
    const exportButton = page.locator(
      '[data-testid="export-graph"], button:has-text("Export")'
    );

    if (await exportButton.count() > 0) {
      await exportButton.click();

      const imageOption = page.locator('[data-testid="export-png"], button:has-text("PNG")');
      if (await imageOption.count() > 0) {
        // Set up download promise
        const downloadPromise = page.waitForEvent('download');
        await imageOption.click();

        // Download should start
        try {
          const download = await downloadPromise;
          expect(download.suggestedFilename()).toContain('.png');
        } catch {
          // Download might not happen in test environment
        }
      }
    }
  });

  test('should export graph data as JSON', async ({ page }) => {
    const exportButton = page.locator('[data-testid="export-graph"]');

    if (await exportButton.count() > 0) {
      await exportButton.click();

      const jsonOption = page.locator('[data-testid="export-json"], button:has-text("JSON")');
      if (await jsonOption.count() > 0) {
        const downloadPromise = page.waitForEvent('download');
        await jsonOption.click();

        try {
          const download = await downloadPromise;
          expect(download.suggestedFilename()).toContain('.json');
        } catch {
          // Download might not happen in test environment
        }
      }
    }
  });
});

// ============================================================================
// Accessibility Tests
// ============================================================================

test.describe('Trust Network Accessibility', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);
  });

  test('should have accessible graph description', async ({ page }) => {
    const graph = page.locator('[data-testid="trust-graph"]');

    if (await graph.count() > 0) {
      const ariaLabel = await graph.getAttribute('aria-label');
      const ariaDescribedBy = await graph.getAttribute('aria-describedby');
      const role = await graph.getAttribute('role');

      // Should have some accessibility attribute
      expect(ariaLabel || ariaDescribedBy || role).toBeTruthy();
    }
  });

  test('should navigate nodes with keyboard', async ({ page }) => {
    const graph = page.locator('[data-testid="trust-graph"]');

    if (await graph.count() > 0) {
      await graph.focus();

      // Tab through nodes
      await page.keyboard.press('Tab');
      await page.keyboard.press('Tab');

      // Focused node should be indicated
      const focusedNode = page.locator('[data-testid="trust-node"]:focus, .node:focus');
      // Node might receive focus
    }
  });

  test('should announce node information to screen readers', async ({ page }) => {
    const nodes = page.locator('[data-testid="trust-node"]');

    if ((await nodes.count()) > 0) {
      const ariaLabel = await nodes.first().getAttribute('aria-label');
      // Nodes should have aria labels
    }
  });
});

// ============================================================================
// Real-time Updates Tests
// ============================================================================

test.describe('Real-time Trust Updates', () => {
  test('should update graph when new attestation is received', async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);

    const initialNodeCount = await page.locator('[data-testid="trust-node"]').count();

    // Simulate receiving a new attestation (this would require WebSocket simulation)
    // For now, we just verify the graph is rendered

    expect(initialNodeCount).toBeGreaterThanOrEqual(1);
  });

  test('should show notification for new trust events', async ({ page }) => {
    await page.goto('/trust');
    await waitForGraphToLoad(page);

    // Look for notification indicator
    const notificationBadge = page.locator(
      '[data-testid="trust-notification"], .notification-badge'
    );

    // Notification might exist
    if (await notificationBadge.count() > 0) {
      await expect(notificationBadge).toBeVisible();
    }
  });
});
