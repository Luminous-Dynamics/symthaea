// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Provider for Mycelix Mail
 *
 * WCAG 2.1 AA compliant accessibility features
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

// Types
export interface AccessibilitySettings {
  reduceMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large';
  focusVisible: boolean;
  announcePageChanges: boolean;
  keyboardShortcutsEnabled: boolean;
}

export interface AccessibilityContextValue extends AccessibilitySettings {
  updateSetting: <K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => void;
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
  focusTrap: (container: HTMLElement | null) => (() => void) | undefined;
}

const defaultSettings: AccessibilitySettings = {
  reduceMotion: false,
  highContrast: false,
  fontSize: 'medium',
  focusVisible: true,
  announcePageChanges: true,
  keyboardShortcutsEnabled: true,
};

const STORAGE_KEY = 'mycelix_a11y';

// Context
const AccessibilityContext = createContext<AccessibilityContextValue | null>(null);

// Provider component
export function AccessibilityProvider({ children }: { children: React.ReactNode }) {
  const [settings, setSettings] = useState<AccessibilitySettings>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try {
          return { ...defaultSettings, ...JSON.parse(stored) };
        } catch {
          // Ignore parse errors
        }
      }

      // Detect system preferences
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const prefersHighContrast = window.matchMedia('(prefers-contrast: more)').matches;

      return {
        ...defaultSettings,
        reduceMotion: prefersReducedMotion,
        highContrast: prefersHighContrast,
      };
    }
    return defaultSettings;
  });

  // Persist settings
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  // Apply CSS custom properties
  useEffect(() => {
    const root = document.documentElement;

    // Font size
    const fontSizes = { small: '14px', medium: '16px', large: '18px' };
    root.style.setProperty('--base-font-size', fontSizes[settings.fontSize]);

    // Reduce motion
    if (settings.reduceMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }

    // High contrast
    if (settings.highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Focus visible
    if (settings.focusVisible) {
      root.classList.add('focus-visible-enabled');
    } else {
      root.classList.remove('focus-visible-enabled');
    }
  }, [settings]);

  // Listen for system preference changes
  useEffect(() => {
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setSettings((s) => ({ ...s, reduceMotion: e.matches }));
    };

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setSettings((s) => ({ ...s, highContrast: e.matches }));
    };

    motionQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  const updateSetting = useCallback(
    <K extends keyof AccessibilitySettings>(key: K, value: AccessibilitySettings[K]) => {
      setSettings((s) => ({ ...s, [key]: value }));
    },
    []
  );

  // Live region announcer
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    const announcer = document.getElementById('a11y-announcer');
    if (announcer) {
      announcer.setAttribute('aria-live', priority);
      announcer.textContent = '';
      // Small delay to ensure announcement
      setTimeout(() => {
        announcer.textContent = message;
      }, 100);
    }
  }, []);

  // Focus trap for modals
  const focusTrap = useCallback((container: HTMLElement | null) => {
    if (!container) return undefined;

    const focusableElements = container.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return undefined;

    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    const handleKeyDown = (e: KeyboardEvent) => {
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
    firstElement.focus();

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const value: AccessibilityContextValue = {
    ...settings,
    updateSetting,
    announce,
    focusTrap,
  };

  return React.createElement(
    AccessibilityContext.Provider,
    { value },
    children,
    // Live region for announcements
    React.createElement('div', {
      id: 'a11y-announcer',
      role: 'status',
      'aria-live': 'polite',
      'aria-atomic': 'true',
      className: 'sr-only',
      style: {
        position: 'absolute',
        width: '1px',
        height: '1px',
        padding: 0,
        margin: '-1px',
        overflow: 'hidden',
        clip: 'rect(0, 0, 0, 0)',
        whiteSpace: 'nowrap',
        border: 0,
      },
    })
  );
}

// Hook
export function useAccessibility(): AccessibilityContextValue {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
}

// Skip link component
export function SkipLink({ targetId, children }: { targetId: string; children: React.ReactNode }) {
  return React.createElement(
    'a',
    {
      href: `#${targetId}`,
      className: 'skip-link',
      style: {
        position: 'absolute',
        top: '-40px',
        left: 0,
        background: 'var(--color-primary)',
        color: 'white',
        padding: '8px 16px',
        zIndex: 100,
        textDecoration: 'none',
        transition: 'top 0.2s',
      },
      onFocus: (e: React.FocusEvent<HTMLAnchorElement>) => {
        e.currentTarget.style.top = '0';
      },
      onBlur: (e: React.FocusEvent<HTMLAnchorElement>) => {
        e.currentTarget.style.top = '-40px';
      },
    },
    children
  );
}

// Focus visible hook
export function useFocusVisible() {
  const [isFocusVisible, setIsFocusVisible] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setIsFocusVisible(true);
      }
    };

    const handleMouseDown = () => {
      setIsFocusVisible(false);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  return isFocusVisible;
}

// Keyboard shortcut hook
export function useKeyboardShortcut(
  key: string,
  callback: () => void,
  options: { ctrl?: boolean; alt?: boolean; shift?: boolean; enabled?: boolean } = {}
) {
  const { keyboardShortcutsEnabled } = useAccessibility();
  const { ctrl = false, alt = false, shift = false, enabled = true } = options;

  useEffect(() => {
    if (!enabled || !keyboardShortcutsEnabled) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (
        e.key.toLowerCase() === key.toLowerCase() &&
        e.ctrlKey === ctrl &&
        e.altKey === alt &&
        e.shiftKey === shift
      ) {
        // Don't trigger in input fields
        const target = e.target as HTMLElement;
        if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

        e.preventDefault();
        callback();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [key, ctrl, alt, shift, callback, enabled, keyboardShortcutsEnabled]);
}

// CSS for accessibility features
export const accessibilityStyles = `
  /* Reduce motion */
  .reduce-motion * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }

  /* High contrast */
  .high-contrast {
    --color-text: #000000;
    --color-background: #ffffff;
    --color-primary: #0000ff;
    --color-error: #ff0000;
    --color-border: #000000;
  }

  .high-contrast a {
    text-decoration: underline;
  }

  .high-contrast button,
  .high-contrast input,
  .high-contrast select,
  .high-contrast textarea {
    border: 2px solid #000000;
  }

  /* Focus visible */
  .focus-visible-enabled *:focus {
    outline: 3px solid var(--color-focus, #4d90fe);
    outline-offset: 2px;
  }

  .focus-visible-enabled *:focus:not(:focus-visible) {
    outline: none;
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
    border: 0;
  }

  /* Font sizes */
  html {
    font-size: var(--base-font-size, 16px);
  }
`;
