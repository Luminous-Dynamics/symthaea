// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility Hook
 *
 * Comprehensive accessibility features:
 * - Screen reader announcements
 * - Focus management
 * - Reduced motion support
 * - High contrast mode
 * - Keyboard navigation
 */

import { useState, useCallback, useEffect, useRef, createContext, useContext } from 'react';

// Accessibility preferences
export interface AccessibilityPreferences {
  // Visual
  reducedMotion: boolean;
  highContrast: boolean;
  largeText: boolean;
  dyslexiaFriendly: boolean;
  colorBlindMode: 'none' | 'protanopia' | 'deuteranopia' | 'tritanopia';

  // Audio
  visualizeAudio: boolean;  // Show captions/visualizations for audio cues
  monoAudio: boolean;       // Combine stereo to mono

  // Interaction
  stickyKeys: boolean;      // Hold modifier keys
  slowKeys: boolean;        // Delay before key repeat
  keyRepeatDelay: number;   // ms before repeat

  // Screen reader
  announceTrackChanges: boolean;
  announceProgress: boolean;
  verbosityLevel: 'minimal' | 'normal' | 'verbose';
}

// Focus trap for modals
export interface FocusTrapOptions {
  initialFocus?: HTMLElement | string;
  returnFocus?: boolean;
  escapeDeactivates?: boolean;
}

// Live region for announcements
export type AnnouncementPriority = 'polite' | 'assertive';

export interface Announcement {
  message: string;
  priority: AnnouncementPriority;
  timestamp: number;
}

// Default preferences
const DEFAULT_PREFERENCES: AccessibilityPreferences = {
  reducedMotion: false,
  highContrast: false,
  largeText: false,
  dyslexiaFriendly: false,
  colorBlindMode: 'none',
  visualizeAudio: false,
  monoAudio: false,
  stickyKeys: false,
  slowKeys: false,
  keyRepeatDelay: 500,
  announceTrackChanges: true,
  announceProgress: true,
  verbosityLevel: 'normal',
};

// Storage key
const STORAGE_KEY = 'mycelix-a11y-preferences';

export function useAccessibility() {
  const [preferences, setPreferences] = useState<AccessibilityPreferences>(() => {
    if (typeof window === 'undefined') return DEFAULT_PREFERENCES;

    // Load from storage
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        return { ...DEFAULT_PREFERENCES, ...JSON.parse(stored) };
      } catch {
        return DEFAULT_PREFERENCES;
      }
    }

    // Detect system preferences
    return {
      ...DEFAULT_PREFERENCES,
      reducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
      highContrast: window.matchMedia('(prefers-contrast: more)').matches,
    };
  });

  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const announcementTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Save preferences to storage
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    }
  }, [preferences]);

  // Apply CSS classes to document
  useEffect(() => {
    if (typeof document === 'undefined') return;

    const classes = document.documentElement.classList;

    classes.toggle('reduce-motion', preferences.reducedMotion);
    classes.toggle('high-contrast', preferences.highContrast);
    classes.toggle('large-text', preferences.largeText);
    classes.toggle('dyslexia-friendly', preferences.dyslexiaFriendly);
    classes.toggle(`colorblind-${preferences.colorBlindMode}`, preferences.colorBlindMode !== 'none');
  }, [preferences]);

  // Listen for system preference changes
  useEffect(() => {
    if (typeof window === 'undefined') return;

    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');

    const handleMotionChange = (e: MediaQueryListEvent) => {
      setPreferences(prev => ({ ...prev, reducedMotion: e.matches }));
    };

    const handleContrastChange = (e: MediaQueryListEvent) => {
      setPreferences(prev => ({ ...prev, highContrast: e.matches }));
    };

    motionQuery.addEventListener('change', handleMotionChange);
    contrastQuery.addEventListener('change', handleContrastChange);

    return () => {
      motionQuery.removeEventListener('change', handleMotionChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  /**
   * Update preferences
   */
  const updatePreference = useCallback(<K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  }, []);

  /**
   * Reset to defaults
   */
  const resetPreferences = useCallback(() => {
    setPreferences(DEFAULT_PREFERENCES);
  }, []);

  /**
   * Announce to screen readers
   */
  const announce = useCallback((
    message: string,
    priority: AnnouncementPriority = 'polite'
  ) => {
    const announcement: Announcement = {
      message,
      priority,
      timestamp: Date.now(),
    };

    setAnnouncements(prev => [...prev, announcement]);

    // Clear old announcements
    if (announcementTimeoutRef.current) {
      clearTimeout(announcementTimeoutRef.current);
    }

    announcementTimeoutRef.current = setTimeout(() => {
      setAnnouncements([]);
    }, 5000);
  }, []);

  /**
   * Announce track change
   */
  const announceTrack = useCallback((
    title: string,
    artist: string,
    isPlaying: boolean
  ) => {
    if (!preferences.announceTrackChanges) return;

    const action = isPlaying ? 'Now playing' : 'Paused';
    const message = `${action}: ${title} by ${artist}`;
    announce(message, 'polite');
  }, [preferences.announceTrackChanges, announce]);

  /**
   * Announce playback progress
   */
  const announceProgress = useCallback((
    currentTime: number,
    duration: number
  ) => {
    if (!preferences.announceProgress) return;

    const percent = Math.round((currentTime / duration) * 100);
    const currentFormatted = formatTime(currentTime);
    const durationFormatted = formatTime(duration);

    announce(`${currentFormatted} of ${durationFormatted}, ${percent}% complete`, 'polite');
  }, [preferences.announceProgress, announce]);

  return {
    preferences,
    updatePreference,
    resetPreferences,
    announcements,
    announce,
    announceTrack,
    announceProgress,
  };
}

/**
 * Focus trap hook for modals
 */
export function useFocusTrap(
  containerRef: React.RefObject<HTMLElement>,
  isActive: boolean,
  options: FocusTrapOptions = {}
) {
  const { initialFocus, returnFocus = true, escapeDeactivates = true } = options;
  const previousActiveElement = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    // Store current focus
    previousActiveElement.current = document.activeElement as HTMLElement;

    // Get focusable elements
    const focusableSelector = [
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'a[href]',
      '[tabindex]:not([tabindex="-1"])',
    ].join(', ');

    const container = containerRef.current;
    const focusableElements = container.querySelectorAll<HTMLElement>(focusableSelector);
    const firstElement = focusableElements[0];
    const lastElement = focusableElements[focusableElements.length - 1];

    // Set initial focus
    if (initialFocus) {
      const target = typeof initialFocus === 'string'
        ? container.querySelector<HTMLElement>(initialFocus)
        : initialFocus;
      target?.focus();
    } else {
      firstElement?.focus();
    }

    // Handle tab key
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement?.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement?.focus();
        }
      }

      if (escapeDeactivates && e.key === 'Escape') {
        // Let parent handle escape
      }
    };

    container.addEventListener('keydown', handleKeyDown);

    return () => {
      container.removeEventListener('keydown', handleKeyDown);

      // Return focus
      if (returnFocus && previousActiveElement.current) {
        previousActiveElement.current.focus();
      }
    };
  }, [isActive, containerRef, initialFocus, returnFocus, escapeDeactivates]);
}

/**
 * Skip link hook
 */
export function useSkipLink(targetId: string) {
  const handleClick = useCallback((e: React.MouseEvent | React.KeyboardEvent) => {
    e.preventDefault();
    const target = document.getElementById(targetId);
    if (target) {
      target.focus();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  }, [targetId]);

  return { onClick: handleClick, onKeyDown: handleClick };
}

/**
 * Roving tabindex for lists/toolbars
 */
export function useRovingTabIndex<T extends HTMLElement>(
  items: React.RefObject<T>[],
  options: {
    orientation?: 'horizontal' | 'vertical' | 'both';
    loop?: boolean;
  } = {}
) {
  const { orientation = 'horizontal', loop = true } = options;
  const [activeIndex, setActiveIndex] = useState(0);

  const handleKeyDown = useCallback((e: React.KeyboardEvent, index: number) => {
    const isHorizontal = orientation === 'horizontal' || orientation === 'both';
    const isVertical = orientation === 'vertical' || orientation === 'both';

    let newIndex = index;

    if ((e.key === 'ArrowRight' && isHorizontal) || (e.key === 'ArrowDown' && isVertical)) {
      e.preventDefault();
      newIndex = index + 1;
      if (newIndex >= items.length) {
        newIndex = loop ? 0 : items.length - 1;
      }
    }

    if ((e.key === 'ArrowLeft' && isHorizontal) || (e.key === 'ArrowUp' && isVertical)) {
      e.preventDefault();
      newIndex = index - 1;
      if (newIndex < 0) {
        newIndex = loop ? items.length - 1 : 0;
      }
    }

    if (e.key === 'Home') {
      e.preventDefault();
      newIndex = 0;
    }

    if (e.key === 'End') {
      e.preventDefault();
      newIndex = items.length - 1;
    }

    if (newIndex !== index) {
      setActiveIndex(newIndex);
      items[newIndex]?.current?.focus();
    }
  }, [items, orientation, loop]);

  return {
    activeIndex,
    setActiveIndex,
    getTabIndex: (index: number) => index === activeIndex ? 0 : -1,
    handleKeyDown,
  };
}

/**
 * Live region component for announcements
 */
export function LiveRegion({ announcements }: { announcements: Announcement[] }) {
  return (
    <>
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcements
          .filter(a => a.priority === 'polite')
          .map(a => a.message)
          .join('. ')}
      </div>
      <div
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      >
        {announcements
          .filter(a => a.priority === 'assertive')
          .map(a => a.message)
          .join('. ')}
      </div>
    </>
  );
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ============================================================================
// Context for app-wide accessibility
// ============================================================================

interface AccessibilityContextValue {
  preferences: AccessibilityPreferences;
  updatePreference: <K extends keyof AccessibilityPreferences>(
    key: K,
    value: AccessibilityPreferences[K]
  ) => void;
  announce: (message: string, priority?: AnnouncementPriority) => void;
}

const AccessibilityContext = createContext<AccessibilityContextValue | null>(null);

export function useAccessibilityContext() {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibilityContext must be used within AccessibilityProvider');
  }
  return context;
}

export { AccessibilityContext };

export default useAccessibility;
