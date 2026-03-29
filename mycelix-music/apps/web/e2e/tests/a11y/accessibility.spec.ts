// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Tests
 *
 * WCAG 2.1 AA compliance testing with axe-core.
 */

import { test, expect, Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// ==================== Types ====================

interface A11yResult {
  violations: A11yViolation[];
  passes: number;
  incomplete: number;
}

interface A11yViolation {
  id: string;
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
  nodes: number;
}

// ==================== Helpers ====================

async function runAxe(page: Page, options?: { include?: string[]; exclude?: string[] }): Promise<A11yResult> {
  let builder = new AxeBuilder({ page });

  if (options?.include) {
    builder = builder.include(options.include);
  }
  if (options?.exclude) {
    builder = builder.exclude(options.exclude);
  }

  const results = await builder.analyze();

  return {
    violations: results.violations.map(v => ({
      id: v.id,
      impact: v.impact as A11yViolation['impact'],
      description: v.description,
      nodes: v.nodes.length,
    })),
    passes: results.passes.length,
    incomplete: results.incomplete.length,
  };
}

function logViolations(violations: A11yViolation[]): void {
  if (violations.length > 0) {
    console.log('\nAccessibility Violations:');
    violations.forEach(v => {
      console.log(`  [${v.impact.toUpperCase()}] ${v.id}: ${v.description} (${v.nodes} nodes)`);
    });
  }
}

// ==================== Tests ====================

test.describe('Accessibility - Pages', () => {
  const pages = [
    { name: 'Home', path: '/' },
    { name: 'Library', path: '/library' },
    { name: 'Search', path: '/search' },
    { name: 'Playlist', path: '/playlist/featured' },
    { name: 'Settings', path: '/settings' },
    { name: 'Login', path: '/login' },
  ];

  for (const { name, path } of pages) {
    test(`${name} page has no critical violations`, async ({ page }) => {
      await page.goto(path);
      await page.waitForLoadState('networkidle');

      const results = await runAxe(page);
      logViolations(results.violations);

      const criticalViolations = results.violations.filter(
        v => v.impact === 'critical' || v.impact === 'serious'
      );

      expect(criticalViolations).toHaveLength(0);
    });
  }
});

test.describe('Accessibility - Keyboard Navigation', () => {
  test('can navigate all interactive elements with Tab', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Start from beginning
    await page.keyboard.press('Tab');

    // Track focused elements
    const focusedElements: string[] = [];
    let loopCount = 0;
    const maxLoops = 50;

    while (loopCount < maxLoops) {
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return null;
        return {
          tag: el.tagName,
          role: el.getAttribute('role'),
          text: el.textContent?.slice(0, 30),
        };
      });

      if (!focused) break;

      const key = `${focused.tag}:${focused.role || 'none'}:${focused.text || ''}`;
      if (focusedElements.includes(key)) break; // Looped back

      focusedElements.push(key);
      await page.keyboard.press('Tab');
      loopCount++;
    }

    console.log(`Found ${focusedElements.length} focusable elements`);
    expect(focusedElements.length).toBeGreaterThan(5);
  });

  test('focus is visible on all interactive elements', async ({ page }) => {
    await page.goto('/');

    // Tab through elements and verify focus visibility
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');

      const hasFocusStyle = await page.evaluate(() => {
        const el = document.activeElement;
        if (!el || el === document.body) return true;

        const styles = window.getComputedStyle(el);
        const focusStyles = window.getComputedStyle(el, ':focus');

        // Check for visible focus indicators
        const hasOutline = styles.outlineWidth !== '0px' && styles.outlineStyle !== 'none';
        const hasBoxShadow = styles.boxShadow !== 'none';
        const hasBorder = styles.borderColor !== 'transparent';

        return hasOutline || hasBoxShadow || hasBorder;
      });

      expect(hasFocusStyle).toBe(true);
    }
  });

  test('skip link works correctly', async ({ page }) => {
    await page.goto('/');

    // Focus skip link
    await page.keyboard.press('Tab');

    const skipLink = await page.locator('[data-testid="skip-link"]');
    if (await skipLink.isVisible()) {
      await page.keyboard.press('Enter');

      // Verify focus moved to main content
      const focusedId = await page.evaluate(() => document.activeElement?.id);
      expect(focusedId).toBe('main-content');
    }
  });

  test('modal traps focus correctly', async ({ page }) => {
    await page.goto('/');
    await page.click('[data-testid="settings-button"]');
    await page.waitForSelector('[data-testid="settings-modal"]');

    // Tab through modal
    const focusedElements: string[] = [];
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('Tab');
      const focused = await page.evaluate(() => {
        const el = document.activeElement;
        return el?.closest('[data-testid="settings-modal"]') ? 'inside' : 'outside';
      });
      focusedElements.push(focused);
    }

    // All focus should be inside modal
    expect(focusedElements.every(f => f === 'inside')).toBe(true);
  });

  test('Escape closes modal', async ({ page }) => {
    await page.goto('/');
    await page.click('[data-testid="settings-button"]');
    await page.waitForSelector('[data-testid="settings-modal"]');

    await page.keyboard.press('Escape');

    await expect(page.locator('[data-testid="settings-modal"]')).not.toBeVisible();
  });
});

test.describe('Accessibility - Screen Reader', () => {
  test('all images have alt text', async ({ page }) => {
    await page.goto('/library');
    await page.waitForLoadState('networkidle');

    const imagesWithoutAlt = await page.evaluate(() => {
      const images = document.querySelectorAll('img');
      return Array.from(images).filter(img => !img.alt && !img.getAttribute('aria-hidden')).length;
    });

    expect(imagesWithoutAlt).toBe(0);
  });

  test('form inputs have labels', async ({ page }) => {
    await page.goto('/settings');
    await page.waitForLoadState('networkidle');

    const results = await runAxe(page, {
      include: ['form', 'input', 'select', 'textarea'],
    });

    const labelViolations = results.violations.filter(
      v => v.id === 'label' || v.id === 'label-title-only'
    );

    logViolations(labelViolations);
    expect(labelViolations).toHaveLength(0);
  });

  test('headings are properly structured', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const headingStructure = await page.evaluate(() => {
      const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      const levels: number[] = [];

      headings.forEach(h => {
        const level = parseInt(h.tagName[1], 10);
        levels.push(level);
      });

      // Check for proper nesting (no skipping levels)
      let previousLevel = 0;
      for (const level of levels) {
        if (level > previousLevel + 1 && previousLevel !== 0) {
          return { valid: false, levels, skippedAt: level };
        }
        previousLevel = level;
      }

      return { valid: true, levels };
    });

    console.log('Heading structure:', headingStructure.levels.join(' -> '));
    expect(headingStructure.valid).toBe(true);
  });

  test('interactive elements have accessible names', async ({ page }) => {
    await page.goto('/');

    const elementsWithoutNames = await page.evaluate(() => {
      const interactiveSelectors = 'button, a, input, select, textarea, [role="button"], [role="link"]';
      const elements = document.querySelectorAll(interactiveSelectors);

      let count = 0;
      elements.forEach(el => {
        const hasName = !!(
          el.getAttribute('aria-label') ||
          el.getAttribute('aria-labelledby') ||
          el.textContent?.trim() ||
          (el as HTMLInputElement).placeholder ||
          el.getAttribute('title')
        );

        if (!hasName && !el.getAttribute('aria-hidden')) {
          count++;
          console.log('Missing accessible name:', el.outerHTML.slice(0, 100));
        }
      });

      return count;
    });

    expect(elementsWithoutNames).toBe(0);
  });

  test('ARIA attributes are valid', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const results = await runAxe(page);
    const ariaViolations = results.violations.filter(v => v.id.startsWith('aria'));

    logViolations(ariaViolations);
    expect(ariaViolations).toHaveLength(0);
  });

  test('live regions announce updates', async ({ page }) => {
    await page.goto('/library');

    // Click like button to trigger toast
    await page.click('[data-testid^="track-"] [data-testid="like-button"]:first-of-type');

    // Verify toast has proper live region attributes
    const toast = page.locator('[data-testid="toast"]');
    await expect(toast).toBeVisible();

    const ariaLive = await toast.getAttribute('aria-live');
    expect(ariaLive).toBe('polite');
  });
});

test.describe('Accessibility - Color and Contrast', () => {
  test('text has sufficient color contrast', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    const results = await runAxe(page);
    const contrastViolations = results.violations.filter(
      v => v.id === 'color-contrast' || v.id === 'color-contrast-enhanced'
    );

    logViolations(contrastViolations);

    // Allow minor contrast issues but no serious/critical
    const seriousContrastIssues = contrastViolations.filter(
      v => v.impact === 'serious' || v.impact === 'critical'
    );
    expect(seriousContrastIssues).toHaveLength(0);
  });

  test('focus indicators have sufficient contrast', async ({ page }) => {
    await page.goto('/');

    await page.keyboard.press('Tab');

    const focusContrast = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el) return true;

      const styles = window.getComputedStyle(el);
      const outlineColor = styles.outlineColor;

      // Simple check: outline should not be transparent
      return outlineColor !== 'transparent' && outlineColor !== 'rgba(0, 0, 0, 0)';
    });

    expect(focusContrast).toBe(true);
  });

  test('information is not conveyed by color alone', async ({ page }) => {
    await page.goto('/library');

    // Check that status indicators have text or icons, not just color
    const statusElements = await page.locator('[data-status]').all();

    for (const el of statusElements) {
      const hasNonColorIndicator = await el.evaluate(node => {
        // Should have either text content or an icon
        const hasText = node.textContent && node.textContent.trim().length > 0;
        const hasIcon = node.querySelector('svg, img, [aria-label]') !== null;
        const hasAriaLabel = node.getAttribute('aria-label') !== null;

        return hasText || hasIcon || hasAriaLabel;
      });

      expect(hasNonColorIndicator).toBe(true);
    }
  });
});

test.describe('Accessibility - Motion and Animation', () => {
  test('respects prefers-reduced-motion', async ({ page }) => {
    // Emulate reduced motion preference
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Check that animations are disabled
    const hasReducedMotion = await page.evaluate(() => {
      const el = document.body;
      const styles = window.getComputedStyle(el);
      const hasTransitions = styles.transitionDuration !== '0s';
      const hasAnimations = styles.animationDuration !== '0s';

      // Also check a specific animated element
      const waveform = document.querySelector('[data-testid="waveform"]');
      if (waveform) {
        const waveformStyles = window.getComputedStyle(waveform);
        return waveformStyles.animationDuration === '0s';
      }

      return !hasTransitions && !hasAnimations;
    });

    expect(hasReducedMotion).toBe(true);
  });

  test('no content flashes rapidly', async ({ page }) => {
    await page.goto('/');

    // Monitor for rapid visual changes
    const flashDetected = await page.evaluate(() => {
      return new Promise<boolean>((resolve) => {
        let flashCount = 0;
        let lastState = '';

        const observer = new MutationObserver(() => {
          const currentState = document.body.innerHTML.length.toString();
          if (currentState !== lastState) {
            flashCount++;
            lastState = currentState;
          }
        });

        observer.observe(document.body, {
          childList: true,
          subtree: true,
          attributes: true,
        });

        setTimeout(() => {
          observer.disconnect();
          // More than 3 flashes per second is dangerous
          resolve(flashCount > 9); // 3 flashes * 3 seconds
        }, 3000);
      });
    });

    expect(flashDetected).toBe(false);
  });
});

test.describe('Accessibility - Audio Player', () => {
  test('player controls are accessible', async ({ page }) => {
    await page.goto('/');

    const results = await runAxe(page, {
      include: ['[data-testid="player"]'],
    });

    logViolations(results.violations);
    expect(results.violations.filter(v => v.impact === 'critical')).toHaveLength(0);
  });

  test('progress bar is keyboard accessible', async ({ page }) => {
    await page.goto('/library');
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
    await page.waitForSelector('[data-testid="player-playing"]');

    const progressBar = page.locator('[data-testid="progress-bar"]');

    // Check it has proper ARIA attributes
    await expect(progressBar).toHaveAttribute('role', 'slider');
    await expect(progressBar).toHaveAttribute('aria-valuemin');
    await expect(progressBar).toHaveAttribute('aria-valuemax');
    await expect(progressBar).toHaveAttribute('aria-valuenow');
    await expect(progressBar).toHaveAttribute('aria-label');
  });

  test('volume control announces changes', async ({ page }) => {
    await page.goto('/');

    const volumeSlider = page.locator('[data-testid="volume-slider"]');

    // Check ARIA attributes
    await expect(volumeSlider).toHaveAttribute('role', 'slider');
    await expect(volumeSlider).toHaveAttribute('aria-label');
    await expect(volumeSlider).toHaveAttribute('aria-valuetext');
  });

  test('now playing information is announced', async ({ page }) => {
    await page.goto('/library');
    await page.click('[data-testid^="track-"] [data-testid="play-button"]:first-of-type');
    await page.waitForSelector('[data-testid="player-playing"]');

    // Check for live region
    const nowPlaying = page.locator('[data-testid="now-playing"]');
    const ariaLive = await nowPlaying.getAttribute('aria-live');

    expect(ariaLive).toBe('polite');
  });
});
