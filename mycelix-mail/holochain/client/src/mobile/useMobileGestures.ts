// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mobile Gestures
 *
 * Provides easy-to-use hooks for gesture handling in React components
 */

import { useRef, useEffect, useCallback } from 'react';
import {
  GestureManager,
  GestureConfig,
  GestureHandlers,
  PullToRefreshManager,
  PullToRefreshConfig,
  SwipeableItem,
  SwipeableItemConfig,
  SwipeEvent,
  TouchPoint,
  PinchEvent,
} from './GestureManager';

/**
 * Hook for general gesture detection
 */
export function useGestures<T extends HTMLElement = HTMLElement>(
  handlers: GestureHandlers,
  config?: GestureConfig
): React.RefObject<T> {
  const ref = useRef<T>(null);
  const managerRef = useRef<GestureManager | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    managerRef.current = new GestureManager(ref.current, handlers, config);

    return () => {
      managerRef.current?.destroy();
    };
  }, [handlers, config]);

  return ref;
}

/**
 * Hook for swipe gestures with actions
 */
export interface UseSwipeOptions {
  threshold?: number;
  velocityThreshold?: number;
}

export function useSwipe<T extends HTMLElement = HTMLElement>(
  onSwipe: (direction: 'left' | 'right' | 'up' | 'down', event: SwipeEvent) => void,
  options?: UseSwipeOptions
): React.RefObject<T> {
  const handlers: GestureHandlers = {
    onSwipeLeft: (event) => onSwipe('left', event),
    onSwipeRight: (event) => onSwipe('right', event),
    onSwipeUp: (event) => onSwipe('up', event),
    onSwipeDown: (event) => onSwipe('down', event),
  };

  return useGestures<T>(handlers, {
    swipeThreshold: options?.threshold,
    swipeVelocityThreshold: options?.velocityThreshold,
  });
}

/**
 * Hook for swipe to reveal actions (email list items)
 */
export interface SwipeAction {
  id: string;
  label: string;
  icon?: string;
  color: string;
  backgroundColor: string;
}

export interface UseSwipeActionsOptions {
  leftActions?: SwipeAction[];
  rightActions?: SwipeAction[];
  actionThreshold?: number;
}

export function useSwipeActions<T extends HTMLElement = HTMLElement>(
  onAction: (actionId: string) => void,
  options?: UseSwipeActionsOptions
): React.RefObject<T> {
  const ref = useRef<T>(null);
  const managerRef = useRef<SwipeableItem | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    managerRef.current = new SwipeableItem(ref.current, onAction, {
      leftActions: options?.leftActions,
      rightActions: options?.rightActions,
      actionThreshold: options?.actionThreshold,
    });

    return () => {
      managerRef.current?.destroy();
    };
  }, [onAction, options]);

  return ref;
}

/**
 * Hook for pull-to-refresh
 */
export function usePullToRefresh<T extends HTMLElement = HTMLElement>(
  onRefresh: () => Promise<void>,
  config?: PullToRefreshConfig
): React.RefObject<T> {
  const ref = useRef<T>(null);
  const managerRef = useRef<PullToRefreshManager | null>(null);

  useEffect(() => {
    if (!ref.current) return;

    managerRef.current = new PullToRefreshManager(ref.current, onRefresh, config);

    return () => {
      managerRef.current?.destroy();
    };
  }, [onRefresh, config]);

  return ref;
}

/**
 * Hook for long press detection
 */
export function useLongPress<T extends HTMLElement = HTMLElement>(
  onLongPress: (point: TouchPoint) => void,
  delay?: number
): React.RefObject<T> {
  return useGestures<T>(
    { onLongPress },
    { longPressDelay: delay }
  );
}

/**
 * Hook for double tap detection
 */
export function useDoubleTap<T extends HTMLElement = HTMLElement>(
  onDoubleTap: (point: TouchPoint) => void,
  delay?: number
): React.RefObject<T> {
  return useGestures<T>(
    { onDoubleTap },
    { doubleTapDelay: delay }
  );
}

/**
 * Hook for pinch-to-zoom
 */
export interface UsePinchZoomOptions {
  minScale?: number;
  maxScale?: number;
  threshold?: number;
}

export function usePinchZoom<T extends HTMLElement = HTMLElement>(
  onZoom: (scale: number) => void,
  options?: UsePinchZoomOptions
): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const scaleRef = useRef(1);
  const baseScaleRef = useRef(1);

  const minScale = options?.minScale ?? 0.5;
  const maxScale = options?.maxScale ?? 3;

  const handlePinch = useCallback(
    (event: PinchEvent) => {
      const newScale = Math.max(minScale, Math.min(maxScale, baseScaleRef.current * event.scale));
      scaleRef.current = newScale;
      onZoom(newScale);
    },
    [onZoom, minScale, maxScale]
  );

  const handlePinchEnd = useCallback(() => {
    baseScaleRef.current = scaleRef.current;
  }, []);

  const handlers: GestureHandlers = {
    onPinch: handlePinch,
    onPinchEnd: handlePinchEnd,
  };

  useEffect(() => {
    if (!ref.current) return;

    const manager = new GestureManager(ref.current, handlers, {
      pinchThreshold: options?.threshold,
    });

    return () => {
      manager.destroy();
    };
  }, [handlers, options?.threshold]);

  return [ref, scaleRef.current];
}

/**
 * Hook for detecting mobile/touch device
 */
export function useIsTouchDevice(): boolean {
  if (typeof window === 'undefined') return false;

  return (
    'ontouchstart' in window ||
    navigator.maxTouchPoints > 0 ||
    // @ts-ignore - msMaxTouchPoints is IE-specific
    navigator.msMaxTouchPoints > 0
  );
}

/**
 * Hook for responsive breakpoints
 */
export interface Breakpoints {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  width: number;
}

export function useBreakpoints(): Breakpoints {
  const getBreakpoints = useCallback((): Breakpoints => {
    if (typeof window === 'undefined') {
      return { isMobile: false, isTablet: false, isDesktop: true, width: 1024 };
    }

    const width = window.innerWidth;
    return {
      isMobile: width < 640,
      isTablet: width >= 640 && width < 1024,
      isDesktop: width >= 1024,
      width,
    };
  }, []);

  const breakpointsRef = useRef(getBreakpoints());

  useEffect(() => {
    const handleResize = () => {
      breakpointsRef.current = getBreakpoints();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [getBreakpoints]);

  return breakpointsRef.current;
}

/**
 * Hook for safe area insets (notch, home indicator)
 */
export interface SafeAreaInsets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export function useSafeAreaInsets(): SafeAreaInsets {
  const getInsets = useCallback((): SafeAreaInsets => {
    if (typeof window === 'undefined') {
      return { top: 0, right: 0, bottom: 0, left: 0 };
    }

    const style = getComputedStyle(document.documentElement);

    return {
      top: parseInt(style.getPropertyValue('--safe-area-inset-top') || '0', 10),
      right: parseInt(style.getPropertyValue('--safe-area-inset-right') || '0', 10),
      bottom: parseInt(style.getPropertyValue('--safe-area-inset-bottom') || '0', 10),
      left: parseInt(style.getPropertyValue('--safe-area-inset-left') || '0', 10),
    };
  }, []);

  return getInsets();
}

/**
 * Hook for viewport height (fixes 100vh issues on mobile)
 */
export function useViewportHeight(): number {
  const heightRef = useRef(typeof window !== 'undefined' ? window.innerHeight : 0);

  useEffect(() => {
    const updateHeight = () => {
      heightRef.current = window.innerHeight;
      document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
    };

    updateHeight();
    window.addEventListener('resize', updateHeight);
    window.addEventListener('orientationchange', updateHeight);

    return () => {
      window.removeEventListener('resize', updateHeight);
      window.removeEventListener('orientationchange', updateHeight);
    };
  }, []);

  return heightRef.current;
}

/**
 * Hook for keyboard visibility (mobile keyboard)
 */
export function useKeyboardVisibility(): boolean {
  const isVisibleRef = useRef(false);

  useEffect(() => {
    if (typeof window === 'undefined' || !('visualViewport' in window)) {
      return;
    }

    const viewport = window.visualViewport!;
    const initialHeight = viewport.height;

    const handleResize = () => {
      // Keyboard is visible if viewport height decreased significantly
      isVisibleRef.current = viewport.height < initialHeight * 0.75;
    };

    viewport.addEventListener('resize', handleResize);
    return () => viewport.removeEventListener('resize', handleResize);
  }, []);

  return isVisibleRef.current;
}

export default {
  useGestures,
  useSwipe,
  useSwipeActions,
  usePullToRefresh,
  useLongPress,
  useDoubleTap,
  usePinchZoom,
  useIsTouchDevice,
  useBreakpoints,
  useSafeAreaInsets,
  useViewportHeight,
  useKeyboardVisibility,
};
