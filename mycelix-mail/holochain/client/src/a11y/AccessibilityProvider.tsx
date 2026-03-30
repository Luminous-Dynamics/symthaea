// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Provider - Comprehensive A11y Support
 *
 * Features:
 * - Focus management
 * - Screen reader announcements
 * - Keyboard navigation
 * - Reduced motion support
 * - High contrast mode
 * - Font size scaling
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useRef,
  ReactNode,
} from 'react';

export interface A11ySettings {
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'x-large';
  keyboardNavigation: boolean;
  screenReaderMode: boolean;
  focusIndicator: 'default' | 'enhanced' | 'custom';
  announceNotifications: boolean;
  autoFocusNewContent: boolean;
}

export interface A11yContextValue {
  settings: A11ySettings;
  updateSettings: (settings: Partial<A11ySettings>) => void;
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
  setFocus: (elementId: string) => void;
  trapFocus: (containerId: string) => () => void;
  skipToMain: () => void;
  registerLandmark: (id: string, role: string, label: string) => void;
  unregisterLandmark: (id: string) => void;
  getLandmarks: () => Landmark[];
}

interface Landmark {
  id: string;
  role: string;
  label: string;
}

const defaultSettings: A11ySettings = {
  reducedMotion: false,
  highContrast: false,
  fontSize: 'medium',
  keyboardNavigation: true,
  screenReaderMode: false,
  focusIndicator: 'default',
  announceNotifications: true,
  autoFocusNewContent: false,
};

const A11yContext = createContext<A11yContextValue | null>(null);

export function useA11y(): A11yContextValue {
  const context = useContext(A11yContext);
  if (!context) {
    throw new Error('useA11y must be used within A11yProvider');
  }
  return context;
}

interface A11yProviderProps {
  children: ReactNode;
  initialSettings?: Partial<A11ySettings>;
}

export const A11yProvider: React.FC<A11yProviderProps> = ({
  children,
  initialSettings = {},
}) => {
  const [settings, setSettings] = useState<A11ySettings>(() => {
    // Load from localStorage
    const saved = localStorage.getItem('mycelix-a11y');
    if (saved) {
      return { ...defaultSettings, ...JSON.parse(saved), ...initialSettings };
    }
    return { ...defaultSettings, ...initialSettings };
  });

  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const announcerRef = useRef<HTMLDivElement>(null);
  const politeAnnouncerRef = useRef<HTMLDivElement>(null);

  // Detect system preferences
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setSettings((prev) => ({ ...prev, reducedMotion: e.matches }));
    };

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setSettings((prev) => ({ ...prev, highContrast: e.matches }));
    };

    mediaQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    // Set initial values from system
    if (mediaQuery.matches && !initialSettings.reducedMotion) {
      setSettings((prev) => ({ ...prev, reducedMotion: true }));
    }
    if (contrastQuery.matches && !initialSettings.highContrast) {
      setSettings((prev) => ({ ...prev, highContrast: true }));
    }

    return () => {
      mediaQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, [initialSettings]);

  // Save settings to localStorage
  useEffect(() => {
    localStorage.setItem('mycelix-a11y', JSON.stringify(settings));

    // Apply settings to document
    document.documentElement.dataset.reducedMotion = String(settings.reducedMotion);
    document.documentElement.dataset.highContrast = String(settings.highContrast);
    document.documentElement.dataset.fontSize = settings.fontSize;
    document.documentElement.dataset.focusIndicator = settings.focusIndicator;

    // Apply font size
    const fontSizes = {
      small: '14px',
      medium: '16px',
      large: '18px',
      'x-large': '20px',
    };
    document.documentElement.style.fontSize = fontSizes[settings.fontSize];
  }, [settings]);

  const updateSettings = useCallback((newSettings: Partial<A11ySettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  }, []);

  const announce = useCallback(
    (message: string, priority: 'polite' | 'assertive' = 'polite') => {
      const announcer =
        priority === 'assertive' ? announcerRef.current : politeAnnouncerRef.current;

      if (announcer) {
        // Clear and re-announce for screen readers
        announcer.textContent = '';
        requestAnimationFrame(() => {
          announcer.textContent = message;
        });
      }
    },
    []
  );

  const setFocus = useCallback((elementId: string) => {
    const element = document.getElementById(elementId);
    if (element) {
      element.focus();
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, []);

  const trapFocus = useCallback((containerId: string) => {
    const container = document.getElementById(containerId);
    if (!container) return () => {};

    const focusableSelector =
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    const focusableElements = container.querySelectorAll(focusableSelector);
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

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
    firstElement?.focus();

    return () => {
      container.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const skipToMain = useCallback(() => {
    const main = document.querySelector('main, [role="main"]');
    if (main) {
      (main as HTMLElement).tabIndex = -1;
      (main as HTMLElement).focus();
    }
  }, []);

  const registerLandmark = useCallback((id: string, role: string, label: string) => {
    setLandmarks((prev) => {
      if (prev.some((l) => l.id === id)) return prev;
      return [...prev, { id, role, label }];
    });
  }, []);

  const unregisterLandmark = useCallback((id: string) => {
    setLandmarks((prev) => prev.filter((l) => l.id !== id));
  }, []);

  const getLandmarks = useCallback(() => landmarks, [landmarks]);

  const value: A11yContextValue = {
    settings,
    updateSettings,
    announce,
    setFocus,
    trapFocus,
    skipToMain,
    registerLandmark,
    unregisterLandmark,
    getLandmarks,
  };

  return (
    <A11yContext.Provider value={value}>
      {/* Skip to main link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-0 focus:left-0 focus:z-50 focus:p-4 focus:bg-blue-600 focus:text-white"
        onClick={(e) => {
          e.preventDefault();
          skipToMain();
        }}
      >
        Skip to main content
      </a>

      {/* Screen reader announcer (assertive) */}
      <div
        ref={announcerRef}
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      />

      {/* Screen reader announcer (polite) */}
      <div
        ref={politeAnnouncerRef}
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      />

      {children}
    </A11yContext.Provider>
  );
};

// Accessibility-enhanced components
export const VisuallyHidden: React.FC<{ children: ReactNode }> = ({ children }) => (
  <span className="sr-only">{children}</span>
);

export interface FocusTrapProps {
  children: ReactNode;
  active?: boolean;
  returnFocus?: boolean;
}

export const FocusTrap: React.FC<FocusTrapProps> = ({
  children,
  active = true,
  returnFocus = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (active) {
      previousFocusRef.current = document.activeElement as HTMLElement;

      const container = containerRef.current;
      if (!container) return;

      const focusableSelector =
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
      const focusableElements = container.querySelectorAll(focusableSelector);
      const firstElement = focusableElements[0] as HTMLElement;

      firstElement?.focus();

      return () => {
        if (returnFocus && previousFocusRef.current) {
          previousFocusRef.current.focus();
        }
      };
    }
  }, [active, returnFocus]);

  return <div ref={containerRef}>{children}</div>;
};

export interface LiveRegionProps {
  children: ReactNode;
  mode?: 'polite' | 'assertive' | 'off';
  atomic?: boolean;
  relevant?: 'additions' | 'removals' | 'text' | 'all';
}

export const LiveRegion: React.FC<LiveRegionProps> = ({
  children,
  mode = 'polite',
  atomic = true,
  relevant = 'additions text',
}) => (
  <div
    role={mode === 'assertive' ? 'alert' : 'status'}
    aria-live={mode}
    aria-atomic={atomic}
    aria-relevant={relevant}
  >
    {children}
  </div>
);

// Keyboard navigation hook
export function useKeyboardNavigation(
  items: string[],
  options: {
    orientation?: 'horizontal' | 'vertical' | 'both';
    loop?: boolean;
    onSelect?: (item: string) => void;
  } = {}
) {
  const { orientation = 'vertical', loop = true, onSelect } = options;
  const [activeIndex, setActiveIndex] = useState(0);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      const isVertical = orientation === 'vertical' || orientation === 'both';
      const isHorizontal = orientation === 'horizontal' || orientation === 'both';

      let newIndex = activeIndex;

      switch (e.key) {
        case 'ArrowUp':
          if (isVertical) {
            e.preventDefault();
            newIndex = activeIndex - 1;
          }
          break;
        case 'ArrowDown':
          if (isVertical) {
            e.preventDefault();
            newIndex = activeIndex + 1;
          }
          break;
        case 'ArrowLeft':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = activeIndex - 1;
          }
          break;
        case 'ArrowRight':
          if (isHorizontal) {
            e.preventDefault();
            newIndex = activeIndex + 1;
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
        case 'Enter':
        case ' ':
          e.preventDefault();
          onSelect?.(items[activeIndex]);
          return;
      }

      // Handle bounds
      if (loop) {
        if (newIndex < 0) newIndex = items.length - 1;
        if (newIndex >= items.length) newIndex = 0;
      } else {
        newIndex = Math.max(0, Math.min(items.length - 1, newIndex));
      }

      setActiveIndex(newIndex);
    },
    [activeIndex, items, orientation, loop, onSelect]
  );

  return {
    activeIndex,
    setActiveIndex,
    handleKeyDown,
    getItemProps: (index: number) => ({
      tabIndex: index === activeIndex ? 0 : -1,
      'aria-selected': index === activeIndex,
    }),
  };
}

// Focus visible hook (for custom focus indicators)
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

export default A11yProvider;
