// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Accessibility React Hooks for Mycelix Mail
 *
 * Custom hooks for managing accessibility features in React components
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  announce,
  announceUrgent,
  saveFocus,
  restoreFocus,
  createFocusTrap,
  getFocusableElements,
  handleListNavigation,
  prefersReducedMotion,
  skipTo,
} from './a11y';

// ============================================================================
// useAnnounce - Screen Reader Announcements
// ============================================================================

export function useAnnounce() {
  return useCallback(
    (message: string, urgent = false) => {
      if (urgent) {
        announceUrgent(message);
      } else {
        announce(message);
      }
    },
    []
  );
}

// ============================================================================
// useFocusTrap - Modal Focus Management
// ============================================================================

export function useFocusTrap(isActive: boolean) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isActive || !containerRef.current) return;

    saveFocus();
    const cleanup = createFocusTrap(containerRef.current);

    return () => {
      cleanup();
      restoreFocus();
    };
  }, [isActive]);

  return containerRef;
}

// ============================================================================
// useArrowNavigation - List Navigation
// ============================================================================

interface UseArrowNavigationOptions {
  loop?: boolean;
  horizontal?: boolean;
  onSelect?: (index: number) => void;
}

export function useArrowNavigation(
  itemCount: number,
  options: UseArrowNavigationOptions = {}
) {
  const [activeIndex, setActiveIndex] = useState(0);
  const itemsRef = useRef<HTMLElement[]>([]);

  const registerItem = useCallback((index: number, element: HTMLElement | null) => {
    if (element) {
      itemsRef.current[index] = element;
    }
  }, []);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      const newIndex = handleListNavigation(
        event.nativeEvent,
        itemsRef.current,
        activeIndex,
        {
          ...options,
          onSelect: (index) => {
            options.onSelect?.(index);
          },
        }
      );

      setActiveIndex(newIndex);
    },
    [activeIndex, options]
  );

  const getItemProps = useCallback(
    (index: number) => ({
      ref: (el: HTMLElement | null) => registerItem(index, el),
      tabIndex: index === activeIndex ? 0 : -1,
      'aria-selected': index === activeIndex,
    }),
    [activeIndex, registerItem]
  );

  return {
    activeIndex,
    setActiveIndex,
    handleKeyDown,
    getItemProps,
  };
}

// ============================================================================
// useReducedMotion - Animation Preference
// ============================================================================

export function useReducedMotion(): boolean {
  const [reducedMotion, setReducedMotion] = useState(prefersReducedMotion);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');

    const handleChange = (event: MediaQueryListEvent) => {
      setReducedMotion(event.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return reducedMotion;
}

// ============================================================================
// useHighContrast - Color Scheme Preference
// ============================================================================

export function useHighContrast(): boolean {
  const [highContrast, setHighContrast] = useState(false);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-contrast: more)');
    setHighContrast(mediaQuery.matches);

    const handleChange = (event: MediaQueryListEvent) => {
      setHighContrast(event.matches);
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  return highContrast;
}

// ============================================================================
// useFocusVisible - Focus Visibility Detection
// ============================================================================

export function useFocusVisible() {
  const [isFocusVisible, setIsFocusVisible] = useState(false);
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    let hadKeyboardEvent = false;

    const onKeyDown = () => {
      hadKeyboardEvent = true;
    };

    const onPointerDown = () => {
      hadKeyboardEvent = false;
    };

    const onFocus = () => {
      if (hadKeyboardEvent) {
        setIsFocusVisible(true);
      }
    };

    const onBlur = () => {
      setIsFocusVisible(false);
    };

    document.addEventListener('keydown', onKeyDown, true);
    document.addEventListener('pointerdown', onPointerDown, true);
    element.addEventListener('focus', onFocus);
    element.addEventListener('blur', onBlur);

    return () => {
      document.removeEventListener('keydown', onKeyDown, true);
      document.removeEventListener('pointerdown', onPointerDown, true);
      element.removeEventListener('focus', onFocus);
      element.removeEventListener('blur', onBlur);
    };
  }, []);

  return { ref, isFocusVisible };
}

// ============================================================================
// useSkipLink - Skip Navigation
// ============================================================================

export function useSkipLink(targetId: string) {
  const handleClick = useCallback(
    (event: React.MouseEvent | React.KeyboardEvent) => {
      event.preventDefault();
      skipTo(targetId);
    },
    [targetId]
  );

  return {
    href: `#${targetId}`,
    onClick: handleClick,
    onKeyDown: (event: React.KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        handleClick(event);
      }
    },
  };
}

// ============================================================================
// useRoving TabIndex - Composite Widgets
// ============================================================================

interface UseRovingTabIndexOptions {
  initialIndex?: number;
  vertical?: boolean;
}

export function useRovingTabIndex<T extends HTMLElement>(
  items: T[],
  options: UseRovingTabIndexOptions = {}
) {
  const { initialIndex = 0, vertical = true } = options;
  const [focusedIndex, setFocusedIndex] = useState(initialIndex);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      const nextKey = vertical ? 'ArrowDown' : 'ArrowRight';
      const prevKey = vertical ? 'ArrowUp' : 'ArrowLeft';

      switch (event.key) {
        case nextKey:
          event.preventDefault();
          setFocusedIndex((prev) => (prev + 1) % items.length);
          break;
        case prevKey:
          event.preventDefault();
          setFocusedIndex((prev) => (prev - 1 + items.length) % items.length);
          break;
        case 'Home':
          event.preventDefault();
          setFocusedIndex(0);
          break;
        case 'End':
          event.preventDefault();
          setFocusedIndex(items.length - 1);
          break;
      }
    },
    [items.length, vertical]
  );

  useEffect(() => {
    items[focusedIndex]?.focus();
  }, [focusedIndex, items]);

  return {
    focusedIndex,
    setFocusedIndex,
    getTabIndex: (index: number) => (index === focusedIndex ? 0 : -1),
    handleKeyDown,
  };
}

// ============================================================================
// useId - Unique ID Generation
// ============================================================================

let idCounter = 0;

export function useId(prefix = 'mycelix'): string {
  const [id] = useState(() => `${prefix}-${++idCounter}`);
  return id;
}

// ============================================================================
// useLiveRegion - Dynamic Live Region
// ============================================================================

interface UseLiveRegionOptions {
  politeness?: 'polite' | 'assertive';
  atomic?: boolean;
  relevant?: 'additions' | 'removals' | 'text' | 'all';
}

export function useLiveRegion(options: UseLiveRegionOptions = {}) {
  const { politeness = 'polite', atomic = true, relevant = 'additions' } = options;
  const [message, setMessage] = useState('');

  const regionProps = {
    role: 'status',
    'aria-live': politeness,
    'aria-atomic': atomic,
    'aria-relevant': relevant,
  };

  const announce = useCallback((newMessage: string) => {
    // Clear and reset to ensure announcement
    setMessage('');
    setTimeout(() => setMessage(newMessage), 50);
  }, []);

  return {
    message,
    announce,
    regionProps,
  };
}

// ============================================================================
// useDescription - Accessible Descriptions
// ============================================================================

export function useDescription(description: string) {
  const id = useId('description');

  return {
    descriptionId: id,
    descriptionProps: {
      id,
      children: description,
    },
    inputProps: {
      'aria-describedby': id,
    },
  };
}

// ============================================================================
// useErrorAnnouncement - Form Error Handling
// ============================================================================

export function useErrorAnnouncement() {
  const announceMessage = useAnnounce();
  const errorsRef = useRef<Set<string>>(new Set());

  const announceError = useCallback(
    (fieldName: string, error: string) => {
      const errorKey = `${fieldName}: ${error}`;
      if (!errorsRef.current.has(errorKey)) {
        errorsRef.current.add(errorKey);
        announceMessage(`Error in ${fieldName}: ${error}`, true);
      }
    },
    [announceMessage]
  );

  const clearErrors = useCallback(() => {
    errorsRef.current.clear();
  }, []);

  return { announceError, clearErrors };
}

// ============================================================================
// useDialogA11y - Dialog/Modal Accessibility
// ============================================================================

interface UseDialogA11yOptions {
  title: string;
  description?: string;
}

export function useDialogA11y(isOpen: boolean, options: UseDialogA11yOptions) {
  const { title, description } = options;
  const titleId = useId('dialog-title');
  const descId = useId('dialog-desc');
  const containerRef = useFocusTrap(isOpen);

  useEffect(() => {
    if (isOpen) {
      // Hide other content from screen readers
      const main = document.querySelector('main');
      if (main) {
        main.setAttribute('aria-hidden', 'true');
      }
    }

    return () => {
      const main = document.querySelector('main');
      if (main) {
        main.removeAttribute('aria-hidden');
      }
    };
  }, [isOpen]);

  return {
    containerRef,
    dialogProps: {
      role: 'dialog',
      'aria-modal': true,
      'aria-labelledby': titleId,
      'aria-describedby': description ? descId : undefined,
    },
    titleProps: {
      id: titleId,
      children: title,
    },
    descriptionProps: description
      ? {
          id: descId,
          children: description,
        }
      : undefined,
  };
}
