// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Utilities
 *
 * Helpers for ensuring ARIA compliance and screen reader support:
 * - Focus management
 * - Live region announcements
 * - Keyboard navigation helpers
 * - Screen reader utilities
 */

import { useEffect, useRef, useCallback, useState } from 'react';

// ============================================
// Live Region Announcements
// ============================================

/**
 * Creates a live region for screen reader announcements
 */
export function createLiveRegion(politeness: 'polite' | 'assertive' = 'polite'): HTMLDivElement {
  const region = document.createElement('div');
  region.setAttribute('role', 'status');
  region.setAttribute('aria-live', politeness);
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
  return region;
}

let liveRegion: HTMLDivElement | null = null;

/**
 * Announces a message to screen readers
 */
export function announce(message: string, politeness: 'polite' | 'assertive' = 'polite'): void {
  if (!liveRegion) {
    liveRegion = createLiveRegion(politeness);
  }

  // Clear and set message (forces re-announcement)
  liveRegion.textContent = '';
  requestAnimationFrame(() => {
    if (liveRegion) {
      liveRegion.setAttribute('aria-live', politeness);
      liveRegion.textContent = message;
    }
  });
}

/**
 * Hook for announcing messages to screen readers
 */
export function useAnnounce() {
  return useCallback((message: string, politeness: 'polite' | 'assertive' = 'polite') => {
    announce(message, politeness);
  }, []);
}

// ============================================
// Focus Management
// ============================================

/**
 * Focuses the first focusable element within a container
 */
export function focusFirstElement(container: HTMLElement): boolean {
  const focusable = getFocusableElements(container);
  if (focusable.length > 0) {
    focusable[0].focus();
    return true;
  }
  return false;
}

/**
 * Gets all focusable elements within a container
 */
export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const selector = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
  ].join(', ');

  return Array.from(container.querySelectorAll<HTMLElement>(selector));
}

/**
 * Hook for trapping focus within a container (for modals)
 */
export function useFocusTrap(isActive: boolean = true) {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    // Store previously focused element
    previousFocusRef.current = document.activeElement as HTMLElement;

    // Focus first element in container
    focusFirstElement(containerRef.current);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab' || !containerRef.current) return;

      const focusable = getFocusableElements(containerRef.current);
      if (focusable.length === 0) return;

      const first = focusable[0];
      const last = focusable[focusable.length - 1];

      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);

      // Restore focus to previously focused element
      if (previousFocusRef.current && previousFocusRef.current.focus) {
        previousFocusRef.current.focus();
      }
    };
  }, [isActive]);

  return containerRef;
}

/**
 * Hook for managing focus when a list changes
 */
export function useFocusOnChange<T>(
  items: T[],
  getItemId: (item: T) => string,
  onFocus?: (item: T) => void
) {
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const itemRefs = useRef<Map<string, HTMLElement>>(new Map());

  const registerRef = useCallback((id: string, element: HTMLElement | null) => {
    if (element) {
      itemRefs.current.set(id, element);
    } else {
      itemRefs.current.delete(id);
    }
  }, []);

  const focusItem = useCallback((id: string) => {
    const element = itemRefs.current.get(id);
    if (element) {
      element.focus();
      setFocusedId(id);
      const item = items.find((i) => getItemId(i) === id);
      if (item && onFocus) {
        onFocus(item);
      }
    }
  }, [items, getItemId, onFocus]);

  const focusNext = useCallback(() => {
    const currentIndex = focusedId
      ? items.findIndex((i) => getItemId(i) === focusedId)
      : -1;
    const nextIndex = Math.min(currentIndex + 1, items.length - 1);
    if (nextIndex >= 0 && items[nextIndex]) {
      focusItem(getItemId(items[nextIndex]));
    }
  }, [items, focusedId, getItemId, focusItem]);

  const focusPrevious = useCallback(() => {
    const currentIndex = focusedId
      ? items.findIndex((i) => getItemId(i) === focusedId)
      : items.length;
    const prevIndex = Math.max(currentIndex - 1, 0);
    if (items[prevIndex]) {
      focusItem(getItemId(items[prevIndex]));
    }
  }, [items, focusedId, getItemId, focusItem]);

  return {
    focusedId,
    registerRef,
    focusItem,
    focusNext,
    focusPrevious,
  };
}

// ============================================
// Keyboard Navigation
// ============================================

/**
 * Hook for arrow key navigation in a list
 */
export function useArrowNavigation<T>(
  items: T[],
  options: {
    onSelect?: (item: T, index: number) => void;
    onEscape?: () => void;
    orientation?: 'vertical' | 'horizontal';
    wrap?: boolean;
    enabled?: boolean;
  } = {}
) {
  const {
    onSelect,
    onEscape,
    orientation = 'vertical',
    wrap = true,
    enabled = true,
  } = options;

  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't handle if typing in input
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const isVertical = orientation === 'vertical';
      const nextKey = isVertical ? 'ArrowDown' : 'ArrowRight';
      const prevKey = isVertical ? 'ArrowUp' : 'ArrowLeft';

      switch (e.key) {
        case nextKey:
          e.preventDefault();
          setSelectedIndex((i) => {
            const next = i + 1;
            if (next >= items.length) {
              return wrap ? 0 : i;
            }
            return next;
          });
          break;

        case prevKey:
          e.preventDefault();
          setSelectedIndex((i) => {
            const prev = i - 1;
            if (prev < 0) {
              return wrap ? items.length - 1 : 0;
            }
            return prev;
          });
          break;

        case 'Enter':
        case ' ':
          e.preventDefault();
          if (items[selectedIndex] && onSelect) {
            onSelect(items[selectedIndex], selectedIndex);
          }
          break;

        case 'Escape':
          if (onEscape) {
            e.preventDefault();
            onEscape();
          }
          break;

        case 'Home':
          e.preventDefault();
          setSelectedIndex(0);
          break;

        case 'End':
          e.preventDefault();
          setSelectedIndex(items.length - 1);
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [items, selectedIndex, orientation, wrap, enabled, onSelect, onEscape]);

  return {
    selectedIndex,
    setSelectedIndex,
    selectedItem: items[selectedIndex],
  };
}

// ============================================
// ARIA Helpers
// ============================================

/**
 * Generates ARIA props for a listbox
 */
export function getListboxProps(options: {
  label: string;
  multiselect?: boolean;
  orientation?: 'vertical' | 'horizontal';
}) {
  return {
    role: 'listbox',
    'aria-label': options.label,
    'aria-multiselectable': options.multiselect || undefined,
    'aria-orientation': options.orientation || 'vertical',
  };
}

/**
 * Generates ARIA props for a listbox option
 */
export function getOptionProps(options: {
  selected: boolean;
  disabled?: boolean;
  index: number;
}) {
  return {
    role: 'option',
    'aria-selected': options.selected,
    'aria-disabled': options.disabled || undefined,
    'aria-posinset': options.index + 1,
    tabIndex: options.selected ? 0 : -1,
  };
}

/**
 * Generates ARIA props for a dialog/modal
 */
export function getDialogProps(options: {
  title: string;
  describedBy?: string;
}) {
  return {
    role: 'dialog',
    'aria-modal': true,
    'aria-labelledby': `dialog-title-${options.title.replace(/\s+/g, '-').toLowerCase()}`,
    'aria-describedby': options.describedBy,
  };
}

/**
 * Generates ARIA props for a tab panel
 */
export function getTabPanelProps(options: {
  id: string;
  selected: boolean;
}) {
  return {
    role: 'tabpanel',
    id: `tabpanel-${options.id}`,
    'aria-labelledby': `tab-${options.id}`,
    hidden: !options.selected,
    tabIndex: options.selected ? 0 : -1,
  };
}

/**
 * Generates ARIA props for a tab
 */
export function getTabProps(options: {
  id: string;
  selected: boolean;
  disabled?: boolean;
}) {
  return {
    role: 'tab',
    id: `tab-${options.id}`,
    'aria-selected': options.selected,
    'aria-disabled': options.disabled || undefined,
    'aria-controls': `tabpanel-${options.id}`,
    tabIndex: options.selected ? 0 : -1,
  };
}

// ============================================
// Screen Reader Text
// ============================================

/**
 * Generates visually hidden CSS class
 */
export const srOnlyStyles = `
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

/**
 * Component for visually hidden text (screen reader only)
 */
export function ScreenReaderOnly({ children }: { children: React.ReactNode }) {
  return <span className="sr-only">{children}</span>;
}

/**
 * Formats a trust score for screen readers
 */
export function formatTrustScoreForSR(score: number): string {
  const percentage = Math.round(score * 100);
  if (percentage >= 70) return `High trust, ${percentage} percent`;
  if (percentage >= 40) return `Medium trust, ${percentage} percent`;
  return `Low trust, ${percentage} percent`;
}

/**
 * Formats an epistemic tier for screen readers
 */
export function formatTierForSR(tier: number): string {
  const descriptions: Record<number, string> = {
    0: 'Tier 0, unverifiable sender',
    1: 'Tier 1, sender email verified',
    2: 'Tier 2, sender identity verified',
    3: 'Tier 3, sender in trust network',
    4: 'Tier 4, fully attested sender',
  };
  return descriptions[tier] || `Tier ${tier}`;
}

/**
 * Formats an assurance level for screen readers
 */
export function formatAssuranceLevelForSR(level: string): string {
  const descriptions: Record<string, string> = {
    e0_anonymous: 'E0, anonymous, no identity verification',
    e1_verified_email: 'E1, email verified, low assurance',
    e2_gitcoin_passport: 'E2, Gitcoin Passport verified, medium assurance',
    e3_multi_factor: 'E3, multi-factor verified, high assurance',
    e4_constitutional: 'E4, constitutional identity, highest assurance',
  };
  return descriptions[level] || level;
}

export default {
  announce,
  useAnnounce,
  focusFirstElement,
  getFocusableElements,
  useFocusTrap,
  useFocusOnChange,
  useArrowNavigation,
  getListboxProps,
  getOptionProps,
  getDialogProps,
  getTabPanelProps,
  getTabProps,
  ScreenReaderOnly,
  formatTrustScoreForSR,
  formatTierForSR,
  formatAssuranceLevelForSR,
};
