// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility (a11y) Utilities for Mycelix Mail
 *
 * WCAG 2.1 AA compliant utilities for screen readers,
 * keyboard navigation, focus management, and live regions.
 */

// ============================================================================
// ARIA Live Announcements
// ============================================================================

type AriaLive = 'polite' | 'assertive' | 'off';

let announcer: HTMLElement | null = null;

/**
 * Initialize the ARIA live region announcer
 */
export function initAnnouncer(): void {
  if (typeof document === 'undefined') return;

  if (!announcer) {
    announcer = document.createElement('div');
    announcer.setAttribute('role', 'status');
    announcer.setAttribute('aria-live', 'polite');
    announcer.setAttribute('aria-atomic', 'true');
    announcer.className = 'sr-only';
    announcer.style.cssText = `
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
    document.body.appendChild(announcer);
  }
}

/**
 * Announce a message to screen readers
 */
export function announce(message: string, priority: AriaLive = 'polite'): void {
  if (!announcer) {
    initAnnouncer();
  }

  if (announcer) {
    announcer.setAttribute('aria-live', priority);
    // Clear and re-set to trigger announcement
    announcer.textContent = '';
    setTimeout(() => {
      announcer!.textContent = message;
    }, 50);
  }
}

/**
 * Announce urgently (interrupts current announcements)
 */
export function announceUrgent(message: string): void {
  announce(message, 'assertive');
}

// ============================================================================
// Focus Management
// ============================================================================

const focusStack: HTMLElement[] = [];

/**
 * Save current focus for later restoration
 */
export function saveFocus(): void {
  const activeElement = document.activeElement as HTMLElement;
  if (activeElement) {
    focusStack.push(activeElement);
  }
}

/**
 * Restore previously saved focus
 */
export function restoreFocus(): void {
  const element = focusStack.pop();
  if (element && typeof element.focus === 'function') {
    setTimeout(() => element.focus(), 0);
  }
}

/**
 * Focus the first focusable element within a container
 */
export function focusFirst(container: HTMLElement): void {
  const focusable = getFocusableElements(container);
  if (focusable.length > 0) {
    focusable[0].focus();
  }
}

/**
 * Get all focusable elements within a container
 */
export function getFocusableElements(container: HTMLElement): HTMLElement[] {
  const focusableSelectors = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[tabindex]:not([tabindex="-1"])',
    '[contenteditable="true"]',
  ].join(', ');

  return Array.from(container.querySelectorAll<HTMLElement>(focusableSelectors)).filter(
    (el) => !el.hasAttribute('disabled') && el.offsetParent !== null
  );
}

/**
 * Trap focus within a container (for modals)
 */
export function createFocusTrap(container: HTMLElement): () => void {
  const handleKeyDown = (event: KeyboardEvent) => {
    if (event.key !== 'Tab') return;

    const focusable = getFocusableElements(container);
    if (focusable.length === 0) return;

    const first = focusable[0];
    const last = focusable[focusable.length - 1];

    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  container.addEventListener('keydown', handleKeyDown);
  focusFirst(container);

  return () => {
    container.removeEventListener('keydown', handleKeyDown);
  };
}

// ============================================================================
// Keyboard Navigation
// ============================================================================

export interface KeyboardHandlers {
  [key: string]: (event: KeyboardEvent) => void;
}

/**
 * Create a keyboard navigation handler
 */
export function createKeyboardHandler(handlers: KeyboardHandlers): (event: KeyboardEvent) => void {
  return (event: KeyboardEvent) => {
    const key = getKeyName(event);
    const handler = handlers[key];

    if (handler) {
      handler(event);
    }
  };
}

/**
 * Get normalized key name including modifiers
 */
export function getKeyName(event: KeyboardEvent): string {
  const parts: string[] = [];

  if (event.ctrlKey || event.metaKey) parts.push('Ctrl');
  if (event.altKey) parts.push('Alt');
  if (event.shiftKey) parts.push('Shift');

  const key = event.key === ' ' ? 'Space' : event.key;
  parts.push(key);

  return parts.join('+');
}

/**
 * Arrow key navigation for lists
 */
export function handleListNavigation(
  event: KeyboardEvent,
  items: HTMLElement[],
  currentIndex: number,
  options: {
    loop?: boolean;
    horizontal?: boolean;
    onSelect?: (index: number) => void;
  } = {}
): number {
  const { loop = true, horizontal = false, onSelect } = options;

  const prevKey = horizontal ? 'ArrowLeft' : 'ArrowUp';
  const nextKey = horizontal ? 'ArrowRight' : 'ArrowDown';

  let newIndex = currentIndex;

  switch (event.key) {
    case prevKey:
      event.preventDefault();
      newIndex = loop
        ? (currentIndex - 1 + items.length) % items.length
        : Math.max(0, currentIndex - 1);
      break;

    case nextKey:
      event.preventDefault();
      newIndex = loop
        ? (currentIndex + 1) % items.length
        : Math.min(items.length - 1, currentIndex + 1);
      break;

    case 'Home':
      event.preventDefault();
      newIndex = 0;
      break;

    case 'End':
      event.preventDefault();
      newIndex = items.length - 1;
      break;

    case 'Enter':
    case ' ':
      event.preventDefault();
      onSelect?.(currentIndex);
      return currentIndex;
  }

  if (newIndex !== currentIndex && items[newIndex]) {
    items[newIndex].focus();
  }

  return newIndex;
}

// ============================================================================
// Color & Contrast
// ============================================================================

/**
 * Calculate relative luminance of a color
 */
export function getLuminance(r: number, g: number, b: number): number {
  const [rs, gs, bs] = [r, g, b].map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });

  return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
}

/**
 * Calculate contrast ratio between two colors
 */
export function getContrastRatio(color1: [number, number, number], color2: [number, number, number]): number {
  const l1 = getLuminance(...color1);
  const l2 = getLuminance(...color2);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast meets WCAG AA requirements
 * Normal text: 4.5:1, Large text: 3:1
 */
export function meetsContrastAA(
  foreground: [number, number, number],
  background: [number, number, number],
  isLargeText = false
): boolean {
  const ratio = getContrastRatio(foreground, background);
  return ratio >= (isLargeText ? 3 : 4.5);
}

/**
 * Parse CSS color to RGB
 */
export function parseColor(color: string): [number, number, number] | null {
  // Handle hex colors
  const hexMatch = color.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hexMatch) {
    let hex = hexMatch[1];
    if (hex.length === 3) {
      hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    }
    return [
      parseInt(hex.slice(0, 2), 16),
      parseInt(hex.slice(2, 4), 16),
      parseInt(hex.slice(4, 6), 16),
    ];
  }

  // Handle rgb/rgba
  const rgbMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
  if (rgbMatch) {
    return [parseInt(rgbMatch[1]), parseInt(rgbMatch[2]), parseInt(rgbMatch[3])];
  }

  return null;
}

// ============================================================================
// Reduced Motion
// ============================================================================

/**
 * Check if user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Get animation duration based on motion preference
 */
export function getAnimationDuration(normalDuration: number): number {
  return prefersReducedMotion() ? 0 : normalDuration;
}

// ============================================================================
// Skip Links
// ============================================================================

/**
 * Create skip link targets
 */
export const skipTargets = {
  mainContent: 'main-content',
  navigation: 'main-nav',
  search: 'search-input',
  emailList: 'email-list',
  composeButton: 'compose-btn',
};

/**
 * Skip to a target element
 */
export function skipTo(targetId: string): void {
  const target = document.getElementById(targetId);
  if (target) {
    target.focus();
    target.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth' });
  }
}

// ============================================================================
// Form Accessibility
// ============================================================================

export interface FormFieldA11y {
  id: string;
  labelId: string;
  descriptionId?: string;
  errorId?: string;
  ariaDescribedBy: string;
}

/**
 * Generate accessibility IDs for a form field
 */
export function getFormFieldA11y(fieldName: string, hasError: boolean, hasDescription: boolean): FormFieldA11y {
  const id = `field-${fieldName}`;
  const labelId = `${id}-label`;
  const descriptionId = hasDescription ? `${id}-description` : undefined;
  const errorId = hasError ? `${id}-error` : undefined;

  const describedBy = [descriptionId, errorId].filter(Boolean).join(' ');

  return {
    id,
    labelId,
    descriptionId,
    errorId,
    ariaDescribedBy: describedBy,
  };
}

// ============================================================================
// Screen Reader Utilities
// ============================================================================

/**
 * Generate a screen-reader-only class styles
 */
export const srOnlyStyles: React.CSSProperties = {
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
 * Format a date for screen readers
 */
export function formatDateForSR(date: Date): string {
  return date.toLocaleDateString(undefined, {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
  });
}

/**
 * Format email count for screen readers
 */
export function formatEmailCountForSR(count: number, unread: number): string {
  const total = count === 1 ? '1 email' : `${count} emails`;
  const unreadText = unread === 0 ? 'none unread' : unread === 1 ? '1 unread' : `${unread} unread`;

  return `${total}, ${unreadText}`;
}

// ============================================================================
// ARIA Utilities
// ============================================================================

/**
 * Generate props for an expandable element
 */
export function getExpandableProps(id: string, isExpanded: boolean) {
  return {
    'aria-expanded': isExpanded,
    'aria-controls': `${id}-content`,
  };
}

/**
 * Generate props for a selectable item
 */
export function getSelectableProps(isSelected: boolean) {
  return {
    'aria-selected': isSelected,
    role: 'option',
  };
}

/**
 * Generate props for a checkable item
 */
export function getCheckableProps(isChecked: boolean | 'mixed') {
  return {
    'aria-checked': isChecked,
    role: 'checkbox',
  };
}
