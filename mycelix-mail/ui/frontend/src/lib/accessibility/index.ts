// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Utilities
 *
 * WCAG 2.1 AA compliance helpers and screen reader support
 */

import { useEffect, useRef, useCallback, useState } from 'react';

// ============================================================================
// Focus Management
// ============================================================================

/**
 * Trap focus within a container (for modals, dialogs)
 */
export function useFocusTrap(isActive: boolean = true) {
  const containerRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    const focusableSelectors = [
      'button:not([disabled])',
      'a[href]',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      '[tabindex]:not([tabindex="-1"])',
    ].join(',');

    const focusableElements = container.querySelectorAll<HTMLElement>(focusableSelectors);
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Focus first element
    firstElement?.focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [isActive]);

  return containerRef;
}

/**
 * Return focus to previous element when component unmounts
 */
export function useReturnFocus() {
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    previousFocusRef.current = document.activeElement as HTMLElement;

    return () => {
      previousFocusRef.current?.focus();
    };
  }, []);
}

/**
 * Focus management for roving tabindex pattern
 */
export function useRovingTabIndex<T extends HTMLElement>(
  items: T[],
  options: { orientation?: 'horizontal' | 'vertical' | 'both'; loop?: boolean } = {}
) {
  const { orientation = 'vertical', loop = true } = options;
  const [focusedIndex, setFocusedIndex] = useState(0);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const isHorizontal = orientation === 'horizontal' || orientation === 'both';
      const isVertical = orientation === 'vertical' || orientation === 'both';

      let newIndex = focusedIndex;

      switch (e.key) {
        case 'ArrowUp':
          if (isVertical) {
            e.preventDefault();
            newIndex = focusedIndex - 1;
          }
          break;
        case 'ArrowDown':
          if (isVertical) {
            e.preventDefault();
            newIndex = focusedIndex + 1;
          }
          break;
        case 'ArrowLeft':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = focusedIndex - 1;
          }
          break;
        case 'ArrowRight':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = focusedIndex + 1;
          }
          break;
        case 'Home':
          e.preventDefault();
          newIndex = 0;
          break;
        case 'End':
          e.preventDefault();
          newIndex = items.length - 1;
          break;
      }

      if (loop) {
        if (newIndex < 0) newIndex = items.length - 1;
        if (newIndex >= items.length) newIndex = 0;
      } else {
        newIndex = Math.max(0, Math.min(items.length - 1, newIndex));
      }

      if (newIndex !== focusedIndex) {
        setFocusedIndex(newIndex);
        items[newIndex]?.focus();
      }
    },
    [focusedIndex, items, orientation, loop]
  );

  return { focusedIndex, setFocusedIndex, handleKeyDown };
}

// ============================================================================
// Screen Reader Announcements
// ============================================================================

/**
 * Live region for screen reader announcements
 */
export function useAnnounce() {
  const regionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    // Create live region if it doesn't exist
    let region = document.getElementById('sr-announcer') as HTMLDivElement;

    if (!region) {
      region = document.createElement('div');
      region.id = 'sr-announcer';
      region.setAttribute('aria-live', 'polite');
      region.setAttribute('aria-atomic', 'true');
      region.className = 'sr-only';
      region.style.cssText = `
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
      `;
      document.body.appendChild(region);
    }

    regionRef.current = region;
  }, []);

  const announce = useCallback(
    (message: string, priority: 'polite' | 'assertive' = 'polite') => {
      if (!regionRef.current) return;

      regionRef.current.setAttribute('aria-live', priority);
      regionRef.current.textContent = '';

      // Force reflow for screen readers
      void regionRef.current.offsetWidth;

      regionRef.current.textContent = message;
    },
    []
  );

  return announce;
}

/**
 * Announce loading states
 */
export function useLoadingAnnouncement(isLoading: boolean, loadingMessage = 'Loading') {
  const announce = useAnnounce();
  const prevLoading = useRef(isLoading);

  useEffect(() => {
    if (isLoading && !prevLoading.current) {
      announce(loadingMessage);
    }
    prevLoading.current = isLoading;
  }, [isLoading, loadingMessage, announce]);
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

/**
 * Skip link hook for main content navigation
 */
export function useSkipLink(targetId: string) {
  const handleClick = useCallback(
    (e: React.MouseEvent | React.KeyboardEvent) => {
      e.preventDefault();
      const target = document.getElementById(targetId);
      if (target) {
        target.setAttribute('tabindex', '-1');
        target.focus();
        target.scrollIntoView();
      }
    },
    [targetId]
  );

  return handleClick;
}

/**
 * Keyboard shortcut handler
 */
export function useKeyboardShortcuts(
  shortcuts: Record<string, () => void>,
  options: { enabled?: boolean } = {}
) {
  const { enabled = true } = options;

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        (e.target as HTMLElement).isContentEditable
      ) {
        return;
      }

      // Build key combination string
      const parts: string[] = [];
      if (e.metaKey || e.ctrlKey) parts.push('Ctrl');
      if (e.altKey) parts.push('Alt');
      if (e.shiftKey) parts.push('Shift');
      parts.push(e.key.toUpperCase());

      const combo = parts.join('+');

      if (shortcuts[combo]) {
        e.preventDefault();
        shortcuts[combo]();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);
}

// ============================================================================
// Color Contrast & Visual Accessibility
// ============================================================================

/**
 * Check if color contrast meets WCAG requirements
 */
export function checkContrast(
  foreground: string,
  background: string,
  level: 'AA' | 'AAA' = 'AA'
): { passes: boolean; ratio: number } {
  const getLuminance = (r: number, g: number, b: number): number => {
    const [rs, gs, bs] = [r, g, b].map((c) => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
  };

  const parseColor = (color: string): [number, number, number] => {
    // Handle hex colors
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      const r = parseInt(hex.slice(0, 2), 16);
      const g = parseInt(hex.slice(2, 4), 16);
      const b = parseInt(hex.slice(4, 6), 16);
      return [r, g, b];
    }
    // Handle rgb colors
    const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match) {
      return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
    }
    return [0, 0, 0];
  };

  const l1 = getLuminance(...parseColor(foreground));
  const l2 = getLuminance(...parseColor(background));

  const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

  const requiredRatio = level === 'AAA' ? 7 : 4.5;

  return { passes: ratio >= requiredRatio, ratio };
}

/**
 * Detect reduced motion preference
 */
export function useReducedMotion(): boolean {
  const [prefersReduced, setPrefersReduced] = useState(false);

  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReduced(query.matches);

    const handler = (e: MediaQueryListEvent) => setPrefersReduced(e.matches);
    query.addEventListener('change', handler);
    return () => query.removeEventListener('change', handler);
  }, []);

  return prefersReduced;
}

/**
 * Detect high contrast mode
 */
export function useHighContrast(): boolean {
  const [highContrast, setHighContrast] = useState(false);

  useEffect(() => {
    const query = window.matchMedia('(prefers-contrast: more)');
    setHighContrast(query.matches);

    const handler = (e: MediaQueryListEvent) => setHighContrast(e.matches);
    query.addEventListener('change', handler);
    return () => query.removeEventListener('change', handler);
  }, []);

  return highContrast;
}

// ============================================================================
// Form Accessibility
// ============================================================================

/**
 * Generate accessible form field props
 */
export function useFormField(options: {
  id: string;
  label: string;
  error?: string;
  hint?: string;
  required?: boolean;
}) {
  const { id, label, error, hint, required } = options;

  const labelId = `${id}-label`;
  const errorId = `${id}-error`;
  const hintId = `${id}-hint`;

  const describedBy = [error && errorId, hint && hintId].filter(Boolean).join(' ') || undefined;

  return {
    labelProps: {
      id: labelId,
      htmlFor: id,
    },
    inputProps: {
      id,
      'aria-labelledby': labelId,
      'aria-describedby': describedBy,
      'aria-invalid': !!error,
      'aria-required': required,
    },
    errorProps: {
      id: errorId,
      role: 'alert' as const,
    },
    hintProps: {
      id: hintId,
    },
  };
}

// ============================================================================
// Landmark Utilities
// ============================================================================

/**
 * Standard ARIA landmark roles
 */
export const LANDMARKS = {
  banner: 'banner',           // Header
  navigation: 'navigation',   // Nav menus
  main: 'main',              // Main content
  complementary: 'complementary', // Sidebar
  contentinfo: 'contentinfo', // Footer
  search: 'search',          // Search
  form: 'form',              // Forms with labels
  region: 'region',          // Named sections
} as const;

// ============================================================================
// Component Helpers
// ============================================================================

/**
 * Generate unique ID for accessibility attributes
 */
let idCounter = 0;
export function generateId(prefix = 'a11y'): string {
  return `${prefix}-${++idCounter}`;
}

/**
 * Visually hidden styles for screen reader only content
 */
export const visuallyHidden: React.CSSProperties = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: 0,
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  border: 0,
};

/**
 * CSS class for visually hidden content
 */
export const SR_ONLY_CLASS = 'sr-only';

// ============================================================================
// Accessibility Audit
// ============================================================================

/**
 * Basic accessibility checker for development
 */
export function runAccessibilityAudit(): string[] {
  const issues: string[] = [];

  // Check for images without alt text
  document.querySelectorAll('img:not([alt])').forEach((img) => {
    issues.push(`Image missing alt text: ${(img as HTMLImageElement).src}`);
  });

  // Check for buttons without accessible names
  document.querySelectorAll('button').forEach((btn) => {
    if (!btn.textContent?.trim() && !btn.getAttribute('aria-label')) {
      issues.push(`Button missing accessible name`);
    }
  });

  // Check for form inputs without labels
  document.querySelectorAll('input, select, textarea').forEach((input) => {
    const id = input.id;
    if (!id) {
      issues.push(`Form input missing id attribute`);
    } else if (!document.querySelector(`label[for="${id}"]`) && !input.getAttribute('aria-label')) {
      issues.push(`Form input "${id}" missing label`);
    }
  });

  // Check for links without text
  document.querySelectorAll('a[href]').forEach((link) => {
    if (!link.textContent?.trim() && !link.getAttribute('aria-label')) {
      issues.push(`Link missing accessible text: ${(link as HTMLAnchorElement).href}`);
    }
  });

  // Check for missing main landmark
  if (!document.querySelector('main, [role="main"]')) {
    issues.push('Page missing main landmark');
  }

  // Check for missing page title
  if (!document.title) {
    issues.push('Page missing title');
  }

  // Check for missing lang attribute
  if (!document.documentElement.lang) {
    issues.push('HTML element missing lang attribute');
  }

  return issues;
}
