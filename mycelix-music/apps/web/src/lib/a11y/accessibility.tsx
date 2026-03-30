// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Utilities
 * WCAG 2.1 AA compliance helpers, screen reader support, and keyboard navigation
 */

'use client';

import React, {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useRef,
  useState,
  ReactNode,
  KeyboardEvent,
  FocusEvent,
} from 'react';

// Accessibility context
interface A11yContextValue {
  announcements: string[];
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
  reducedMotion: boolean;
  highContrast: boolean;
  screenReaderActive: boolean;
  focusVisible: boolean;
}

const A11yContext = createContext<A11yContextValue | null>(null);

/**
 * Accessibility Provider
 */
export function A11yProvider({ children }: { children: ReactNode }) {
  const [announcements, setAnnouncements] = useState<string[]>([]);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [screenReaderActive, setScreenReaderActive] = useState(false);
  const [focusVisible, setFocusVisible] = useState(false);

  // Detect user preferences
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Reduced motion preference
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(motionQuery.matches);
    motionQuery.addEventListener('change', (e) => setReducedMotion(e.matches));

    // High contrast preference
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');
    setHighContrast(contrastQuery.matches);
    contrastQuery.addEventListener('change', (e) => setHighContrast(e.matches));

    // Detect keyboard navigation (focus-visible)
    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Tab') {
        setFocusVisible(true);
      }
    };

    const handleMouseDown = () => {
      setFocusVisible(false);
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('mousedown', handleMouseDown);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  // Screen reader announcement
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    setAnnouncements((prev) => [...prev, message]);

    // Clear after announcement
    setTimeout(() => {
      setAnnouncements((prev) => prev.filter((m) => m !== message));
    }, 1000);
  }, []);

  return (
    <A11yContext.Provider
      value={{
        announcements,
        announce,
        reducedMotion,
        highContrast,
        screenReaderActive,
        focusVisible,
      }}
    >
      {children}
      {/* Live region for announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcements[announcements.length - 1]}
      </div>
    </A11yContext.Provider>
  );
}

export function useA11y() {
  const context = useContext(A11yContext);
  if (!context) {
    throw new Error('useA11y must be used within A11yProvider');
  }
  return context;
}

/**
 * Skip Link Component
 */
export function SkipLink({ href = '#main-content', children = 'Skip to main content' }: {
  href?: string;
  children?: ReactNode;
}) {
  return (
    <a
      href={href}
      className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-primary-600 focus:text-white focus:rounded-lg focus:outline-none focus:ring-2 focus:ring-white"
    >
      {children}
    </a>
  );
}

/**
 * Visually Hidden Component (for screen readers)
 */
export function VisuallyHidden({ children, as: Component = 'span' }: {
  children: ReactNode;
  as?: keyof JSX.IntrinsicElements;
}) {
  return (
    <Component className="sr-only">
      {children}
    </Component>
  );
}

/**
 * Focus Trap Hook
 */
export function useFocusTrap(isActive: boolean) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Focus first element
    firstElement.focus();

    const handleKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    container.addEventListener('keydown', handleKeyDown);
    return () => container.removeEventListener('keydown', handleKeyDown);
  }, [isActive]);

  return containerRef;
}

/**
 * Roving Tabindex Hook (for toolbar/menu navigation)
 */
export function useRovingTabindex<T extends HTMLElement>(
  items: T[],
  options: {
    orientation?: 'horizontal' | 'vertical' | 'both';
    loop?: boolean;
  } = {}
) {
  const { orientation = 'horizontal', loop = true } = options;
  const [focusedIndex, setFocusedIndex] = useState(0);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent, index: number) => {
      const isHorizontal = orientation === 'horizontal' || orientation === 'both';
      const isVertical = orientation === 'vertical' || orientation === 'both';

      let newIndex = index;

      switch (e.key) {
        case 'ArrowRight':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = loop
              ? (index + 1) % items.length
              : Math.min(index + 1, items.length - 1);
          }
          break;
        case 'ArrowLeft':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = loop
              ? (index - 1 + items.length) % items.length
              : Math.max(index - 1, 0);
          }
          break;
        case 'ArrowDown':
          if (isVertical) {
            e.preventDefault();
            newIndex = loop
              ? (index + 1) % items.length
              : Math.min(index + 1, items.length - 1);
          }
          break;
        case 'ArrowUp':
          if (isVertical) {
            e.preventDefault();
            newIndex = loop
              ? (index - 1 + items.length) % items.length
              : Math.max(index - 1, 0);
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

      if (newIndex !== index) {
        setFocusedIndex(newIndex);
        items[newIndex]?.focus();
      }
    },
    [items, orientation, loop]
  );

  return { focusedIndex, handleKeyDown };
}

/**
 * Focus Restore Hook
 */
export function useFocusRestore(isOpen: boolean) {
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement as HTMLElement;
    } else if (previousFocusRef.current) {
      previousFocusRef.current.focus();
    }
  }, [isOpen]);
}

/**
 * Accessible Icon Button
 */
export function IconButton({
  icon,
  label,
  onClick,
  disabled = false,
  className = '',
}: {
  icon: ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={`p-2 rounded-lg hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
    >
      {icon}
    </button>
  );
}

/**
 * Live Region Component
 */
export function LiveRegion({
  message,
  priority = 'polite',
  atomic = true,
}: {
  message: string;
  priority?: 'polite' | 'assertive';
  atomic?: boolean;
}) {
  return (
    <div
      role="status"
      aria-live={priority}
      aria-atomic={atomic}
      className="sr-only"
    >
      {message}
    </div>
  );
}

/**
 * Progress Indicator (accessible)
 */
export function ProgressIndicator({
  value,
  max = 100,
  label,
  showValue = true,
}: {
  value: number;
  max?: number;
  label: string;
  showValue?: boolean;
}) {
  const percentage = Math.round((value / max) * 100);

  return (
    <div className="w-full">
      <div className="flex justify-between mb-1">
        <span className="text-sm font-medium text-white">{label}</span>
        {showValue && (
          <span className="text-sm text-gray-400">{percentage}%</span>
        )}
      </div>
      <div
        role="progressbar"
        aria-valuenow={value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label}
        className="w-full h-2 bg-gray-700 rounded-full overflow-hidden"
      >
        <div
          className="h-full bg-primary-500 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Accessible Loading State
 */
export function LoadingState({
  label = 'Loading...',
  size = 'md',
}: {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
}) {
  const sizes = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12',
  };

  return (
    <div
      role="status"
      aria-label={label}
      className="flex items-center gap-3"
    >
      <div
        className={`${sizes[size]} animate-spin rounded-full border-2 border-gray-600 border-t-primary-500`}
      />
      <span className="sr-only">{label}</span>
    </div>
  );
}

/**
 * Keyboard Shortcut Display
 */
export function KeyboardShortcut({ keys }: { keys: string[] }) {
  return (
    <kbd className="inline-flex items-center gap-0.5 text-xs text-gray-400">
      {keys.map((key, i) => (
        <React.Fragment key={key}>
          {i > 0 && <span className="mx-0.5">+</span>}
          <span className="px-1.5 py-0.5 bg-gray-800 border border-gray-700 rounded text-gray-300 font-mono">
            {key}
          </span>
        </React.Fragment>
      ))}
    </kbd>
  );
}

/**
 * Color Contrast Checker
 */
export function getContrastRatio(color1: string, color2: string): number {
  const getLuminance = (hex: string): number => {
    const rgb = hex
      .replace('#', '')
      .match(/.{2}/g)!
      .map((x) => parseInt(x, 16) / 255);

    const [r, g, b] = rgb.map((c) =>
      c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
    );

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const l1 = getLuminance(color1);
  const l2 = getLuminance(color2);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast meets WCAG requirements
 */
export function meetsContrastRequirements(
  color1: string,
  color2: string,
  level: 'AA' | 'AAA' = 'AA',
  isLargeText = false
): boolean {
  const ratio = getContrastRatio(color1, color2);

  if (level === 'AAA') {
    return isLargeText ? ratio >= 4.5 : ratio >= 7;
  }

  return isLargeText ? ratio >= 3 : ratio >= 4.5;
}

/**
 * Accessible Tabs Component
 */
export function Tabs({
  tabs,
  activeTab,
  onTabChange,
  label,
}: {
  tabs: { id: string; label: string; content: ReactNode }[];
  activeTab: string;
  onTabChange: (id: string) => void;
  label: string;
}) {
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const activeIndex = tabs.findIndex((t) => t.id === activeTab);

  const handleKeyDown = (e: KeyboardEvent, index: number) => {
    let newIndex = index;

    switch (e.key) {
      case 'ArrowRight':
        e.preventDefault();
        newIndex = (index + 1) % tabs.length;
        break;
      case 'ArrowLeft':
        e.preventDefault();
        newIndex = (index - 1 + tabs.length) % tabs.length;
        break;
      case 'Home':
        e.preventDefault();
        newIndex = 0;
        break;
      case 'End':
        e.preventDefault();
        newIndex = tabs.length - 1;
        break;
    }

    if (newIndex !== index) {
      tabRefs.current[newIndex]?.focus();
      onTabChange(tabs[newIndex].id);
    }
  };

  return (
    <div>
      <div
        role="tablist"
        aria-label={label}
        className="flex border-b border-gray-700"
      >
        {tabs.map((tab, index) => (
          <button
            key={tab.id}
            ref={(el) => { tabRefs.current[index] = el; }}
            role="tab"
            id={`tab-${tab.id}`}
            aria-selected={tab.id === activeTab}
            aria-controls={`panel-${tab.id}`}
            tabIndex={tab.id === activeTab ? 0 : -1}
            onClick={() => onTabChange(tab.id)}
            onKeyDown={(e) => handleKeyDown(e, index)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              tab.id === activeTab
                ? 'text-primary-500 border-primary-500'
                : 'text-gray-400 border-transparent hover:text-white hover:border-gray-600'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {tabs.map((tab) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={`panel-${tab.id}`}
          aria-labelledby={`tab-${tab.id}`}
          hidden={tab.id !== activeTab}
          tabIndex={0}
          className="p-4 focus:outline-none"
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}

/**
 * Focus indicator styles (add to global CSS)
 */
export const focusStyles = `
  /* Focus-visible polyfill */
  .focus-visible:focus {
    outline: 2px solid var(--primary-500);
    outline-offset: 2px;
  }

  /* Reduce motion for users who prefer it */
  @media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      scroll-behavior: auto !important;
    }
  }

  /* High contrast mode */
  @media (prefers-contrast: more) {
    :root {
      --border-color: #fff;
      --text-color: #fff;
      --bg-color: #000;
    }
  }

  /* Screen reader only */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }

  .sr-only:focus,
  .sr-only:active {
    position: static;
    width: auto;
    height: auto;
    padding: inherit;
    margin: inherit;
    overflow: visible;
    clip: auto;
    white-space: normal;
  }
`;

export default {
  A11yProvider,
  useA11y,
  SkipLink,
  VisuallyHidden,
  useFocusTrap,
  useRovingTabindex,
  useFocusRestore,
  IconButton,
  LiveRegion,
  ProgressIndicator,
  LoadingState,
  KeyboardShortcut,
  Tabs,
  getContrastRatio,
  meetsContrastRequirements,
};
