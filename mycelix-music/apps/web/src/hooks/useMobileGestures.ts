// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mobile Gestures Hook
 *
 * Touch gesture handling for mobile:
 * - Swipe gestures for navigation
 * - Pinch-to-zoom for waveforms
 * - Long press for context menus
 * - Drag and drop reordering
 */

import { useState, useCallback, useEffect, useRef } from 'react';

// Gesture types
export type GestureType =
  | 'tap'
  | 'doubletap'
  | 'longpress'
  | 'swipeleft'
  | 'swiperight'
  | 'swipeup'
  | 'swipedown'
  | 'pinch'
  | 'rotate'
  | 'pan';

export interface Point {
  x: number;
  y: number;
}

export interface GestureEvent {
  type: GestureType;
  target: HTMLElement;
  startPoint: Point;
  endPoint: Point;
  deltaX: number;
  deltaY: number;
  velocity: Point;
  scale?: number;
  rotation?: number;
  duration: number;
  timestamp: number;
}

export interface GestureConfig {
  // Thresholds
  swipeThreshold: number;           // Min distance for swipe (px)
  swipeVelocityThreshold: number;   // Min velocity for swipe (px/ms)
  longPressDelay: number;           // Delay for long press (ms)
  doubleTapDelay: number;           // Max delay between taps (ms)
  pinchThreshold: number;           // Min scale change for pinch

  // Enabled gestures
  enableSwipe: boolean;
  enablePinch: boolean;
  enableRotate: boolean;
  enableLongPress: boolean;
  enableDoubleTap: boolean;
  enablePan: boolean;
}

export interface GestureHandlers {
  onTap?: (e: GestureEvent) => void;
  onDoubleTap?: (e: GestureEvent) => void;
  onLongPress?: (e: GestureEvent) => void;
  onSwipeLeft?: (e: GestureEvent) => void;
  onSwipeRight?: (e: GestureEvent) => void;
  onSwipeUp?: (e: GestureEvent) => void;
  onSwipeDown?: (e: GestureEvent) => void;
  onPinch?: (e: GestureEvent) => void;
  onRotate?: (e: GestureEvent) => void;
  onPan?: (e: GestureEvent) => void;
  onPanStart?: (e: GestureEvent) => void;
  onPanEnd?: (e: GestureEvent) => void;
}

const DEFAULT_CONFIG: GestureConfig = {
  swipeThreshold: 50,
  swipeVelocityThreshold: 0.3,
  longPressDelay: 500,
  doubleTapDelay: 300,
  pinchThreshold: 0.1,
  enableSwipe: true,
  enablePinch: true,
  enableRotate: false,
  enableLongPress: true,
  enableDoubleTap: true,
  enablePan: true,
};

export function useMobileGestures(
  ref: React.RefObject<HTMLElement>,
  handlers: GestureHandlers,
  config: Partial<GestureConfig> = {}
) {
  const fullConfig = { ...DEFAULT_CONFIG, ...config };

  const [isPanning, setIsPanning] = useState(false);
  const [isPinching, setIsPinching] = useState(false);

  // Touch tracking
  const touchStartRef = useRef<Touch[]>([]);
  const touchStartTimeRef = useRef(0);
  const lastTapTimeRef = useRef(0);
  const longPressTimerRef = useRef<NodeJS.Timeout | null>(null);
  const initialDistanceRef = useRef(0);
  const initialAngleRef = useRef(0);

  // Calculate distance between two points
  const getDistance = useCallback((p1: Point, p2: Point): number => {
    return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
  }, []);

  // Calculate angle between two points
  const getAngle = useCallback((p1: Point, p2: Point): number => {
    return Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
  }, []);

  // Create gesture event
  const createGestureEvent = useCallback((
    type: GestureType,
    target: HTMLElement,
    startTouches: Touch[],
    endTouches: Touch[],
    extraData?: Partial<GestureEvent>
  ): GestureEvent => {
    const startPoint = { x: startTouches[0].clientX, y: startTouches[0].clientY };
    const endPoint = { x: endTouches[0].clientX, y: endTouches[0].clientY };
    const duration = Date.now() - touchStartTimeRef.current;

    return {
      type,
      target,
      startPoint,
      endPoint,
      deltaX: endPoint.x - startPoint.x,
      deltaY: endPoint.y - startPoint.y,
      velocity: {
        x: (endPoint.x - startPoint.x) / duration,
        y: (endPoint.y - startPoint.y) / duration,
      },
      duration,
      timestamp: Date.now(),
      ...extraData,
    };
  }, []);

  // Handle touch start
  const handleTouchStart = useCallback((e: TouchEvent) => {
    const touches = Array.from(e.touches);
    touchStartRef.current = touches;
    touchStartTimeRef.current = Date.now();

    // Clear any existing long press timer
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
    }

    // Two-finger gesture start
    if (touches.length === 2 && (fullConfig.enablePinch || fullConfig.enableRotate)) {
      setIsPinching(true);
      const p1 = { x: touches[0].clientX, y: touches[0].clientY };
      const p2 = { x: touches[1].clientX, y: touches[1].clientY };
      initialDistanceRef.current = getDistance(p1, p2);
      initialAngleRef.current = getAngle(p1, p2);
    }

    // Long press detection
    if (touches.length === 1 && fullConfig.enableLongPress) {
      longPressTimerRef.current = setTimeout(() => {
        if (touchStartRef.current.length === 1) {
          const event = createGestureEvent(
            'longpress',
            e.target as HTMLElement,
            touchStartRef.current,
            Array.from(e.touches)
          );
          handlers.onLongPress?.(event);
        }
      }, fullConfig.longPressDelay);
    }
  }, [fullConfig, getDistance, getAngle, createGestureEvent, handlers]);

  // Handle touch move
  const handleTouchMove = useCallback((e: TouchEvent) => {
    const touches = Array.from(e.touches);

    // Cancel long press on move
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }

    // Two-finger gestures
    if (touches.length === 2 && isPinching) {
      const p1 = { x: touches[0].clientX, y: touches[0].clientY };
      const p2 = { x: touches[1].clientX, y: touches[1].clientY };

      // Pinch
      if (fullConfig.enablePinch) {
        const currentDistance = getDistance(p1, p2);
        const scale = currentDistance / initialDistanceRef.current;

        if (Math.abs(scale - 1) > fullConfig.pinchThreshold) {
          const event = createGestureEvent(
            'pinch',
            e.target as HTMLElement,
            touchStartRef.current,
            touches,
            { scale }
          );
          handlers.onPinch?.(event);
        }
      }

      // Rotate
      if (fullConfig.enableRotate) {
        const currentAngle = getAngle(p1, p2);
        const rotation = currentAngle - initialAngleRef.current;

        if (Math.abs(rotation) > 5) {
          const event = createGestureEvent(
            'rotate',
            e.target as HTMLElement,
            touchStartRef.current,
            touches,
            { rotation }
          );
          handlers.onRotate?.(event);
        }
      }
    }

    // Pan gesture
    if (touches.length === 1 && fullConfig.enablePan) {
      if (!isPanning) {
        setIsPanning(true);
        const event = createGestureEvent(
          'pan',
          e.target as HTMLElement,
          touchStartRef.current,
          touches
        );
        handlers.onPanStart?.(event);
      }

      const event = createGestureEvent(
        'pan',
        e.target as HTMLElement,
        touchStartRef.current,
        touches
      );
      handlers.onPan?.(event);
    }
  }, [fullConfig, isPanning, isPinching, getDistance, getAngle, createGestureEvent, handlers]);

  // Handle touch end
  const handleTouchEnd = useCallback((e: TouchEvent) => {
    const touches = Array.from(e.changedTouches);
    const startTouches = touchStartRef.current;

    // Clear long press timer
    if (longPressTimerRef.current) {
      clearTimeout(longPressTimerRef.current);
      longPressTimerRef.current = null;
    }

    // End pinching
    if (isPinching) {
      setIsPinching(false);
    }

    // End panning
    if (isPanning) {
      setIsPanning(false);
      const event = createGestureEvent(
        'pan',
        e.target as HTMLElement,
        startTouches,
        touches
      );
      handlers.onPanEnd?.(event);
    }

    if (startTouches.length !== 1 || touches.length !== 1) return;

    const event = createGestureEvent(
      'tap',
      e.target as HTMLElement,
      startTouches,
      touches
    );

    // Check for swipe
    if (fullConfig.enableSwipe) {
      const { deltaX, deltaY, velocity } = event;
      const absX = Math.abs(deltaX);
      const absY = Math.abs(deltaY);
      const absVelocityX = Math.abs(velocity.x);
      const absVelocityY = Math.abs(velocity.y);

      if (absX > fullConfig.swipeThreshold || absY > fullConfig.swipeThreshold) {
        if (absX > absY && absVelocityX > fullConfig.swipeVelocityThreshold) {
          if (deltaX > 0) {
            handlers.onSwipeRight?.({ ...event, type: 'swiperight' });
          } else {
            handlers.onSwipeLeft?.({ ...event, type: 'swipeleft' });
          }
          return;
        } else if (absY > absX && absVelocityY > fullConfig.swipeVelocityThreshold) {
          if (deltaY > 0) {
            handlers.onSwipeDown?.({ ...event, type: 'swipedown' });
          } else {
            handlers.onSwipeUp?.({ ...event, type: 'swipeup' });
          }
          return;
        }
      }
    }

    // Check for double tap
    if (fullConfig.enableDoubleTap) {
      const now = Date.now();
      if (now - lastTapTimeRef.current < fullConfig.doubleTapDelay) {
        handlers.onDoubleTap?.({ ...event, type: 'doubletap' });
        lastTapTimeRef.current = 0;
        return;
      }
      lastTapTimeRef.current = now;
    }

    // Single tap
    handlers.onTap?.(event);
  }, [fullConfig, isPanning, isPinching, createGestureEvent, handlers]);

  // Set up event listeners
  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    element.addEventListener('touchstart', handleTouchStart, { passive: true });
    element.addEventListener('touchmove', handleTouchMove, { passive: true });
    element.addEventListener('touchend', handleTouchEnd, { passive: true });
    element.addEventListener('touchcancel', handleTouchEnd, { passive: true });

    return () => {
      element.removeEventListener('touchstart', handleTouchStart);
      element.removeEventListener('touchmove', handleTouchMove);
      element.removeEventListener('touchend', handleTouchEnd);
      element.removeEventListener('touchcancel', handleTouchEnd);

      if (longPressTimerRef.current) {
        clearTimeout(longPressTimerRef.current);
      }
    };
  }, [ref, handleTouchStart, handleTouchMove, handleTouchEnd]);

  return {
    isPanning,
    isPinching,
  };
}

/**
 * Hook for swipe-to-dismiss behavior
 */
export function useSwipeToDismiss(
  ref: React.RefObject<HTMLElement>,
  onDismiss: () => void,
  options: {
    direction?: 'left' | 'right' | 'up' | 'down';
    threshold?: number;
  } = {}
) {
  const { direction = 'down', threshold = 100 } = options;
  const [offset, setOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  useMobileGestures(ref, {
    onPanStart: () => setIsDragging(true),
    onPan: (e) => {
      let delta = 0;
      switch (direction) {
        case 'left':
          delta = Math.min(0, e.deltaX);
          break;
        case 'right':
          delta = Math.max(0, e.deltaX);
          break;
        case 'up':
          delta = Math.min(0, e.deltaY);
          break;
        case 'down':
          delta = Math.max(0, e.deltaY);
          break;
      }
      setOffset(delta);
    },
    onPanEnd: (e) => {
      setIsDragging(false);
      const absOffset = Math.abs(offset);
      if (absOffset > threshold) {
        onDismiss();
      }
      setOffset(0);
    },
  });

  return {
    offset,
    isDragging,
    style: {
      transform: direction === 'left' || direction === 'right'
        ? `translateX(${offset}px)`
        : `translateY(${offset}px)`,
      transition: isDragging ? 'none' : 'transform 0.3s ease-out',
    },
  };
}

/**
 * Hook for pull-to-refresh behavior
 */
export function usePullToRefresh(
  ref: React.RefObject<HTMLElement>,
  onRefresh: () => Promise<void>,
  options: {
    threshold?: number;
    maxPull?: number;
  } = {}
) {
  const { threshold = 80, maxPull = 120 } = options;
  const [pullDistance, setPullDistance] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isPulling, setIsPulling] = useState(false);

  useMobileGestures(ref, {
    onPanStart: () => {
      if (ref.current?.scrollTop === 0) {
        setIsPulling(true);
      }
    },
    onPan: (e) => {
      if (!isPulling) return;
      const distance = Math.min(Math.max(0, e.deltaY), maxPull);
      setPullDistance(distance);
    },
    onPanEnd: async () => {
      if (!isPulling) return;
      setIsPulling(false);

      if (pullDistance >= threshold && !isRefreshing) {
        setIsRefreshing(true);
        await onRefresh();
        setIsRefreshing(false);
      }

      setPullDistance(0);
    },
  });

  const progress = Math.min(pullDistance / threshold, 1);

  return {
    pullDistance,
    isRefreshing,
    isPulling,
    progress,
    shouldRefresh: pullDistance >= threshold,
  };
}

export default useMobileGestures;
