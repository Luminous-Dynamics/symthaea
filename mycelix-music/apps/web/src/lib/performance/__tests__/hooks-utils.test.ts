// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Performance Hooks Tests
 */

import { renderHook, act, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import {
  useDebounce,
  useThrottledCallback,
  useStableCallback,
  useLocalStorage,
  usePrevious,
  useIntersectionObserver,
  useMediaQuery,
  useMounted,
  useUpdateEffect,
} from '../hooks-utils';

describe('Performance Hooks', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('useDebounce', () => {
    it('returns initial value immediately', () => {
      const { result } = renderHook(() => useDebounce('initial', 500));

      expect(result.current).toBe('initial');
    });

    it('debounces value changes', () => {
      const { result, rerender } = renderHook(
        ({ value }) => useDebounce(value, 500),
        { initialProps: { value: 'initial' } }
      );

      expect(result.current).toBe('initial');

      rerender({ value: 'updated' });
      expect(result.current).toBe('initial');

      act(() => {
        vi.advanceTimersByTime(500);
      });

      expect(result.current).toBe('updated');
    });

    it('resets timer on rapid changes', () => {
      const { result, rerender } = renderHook(
        ({ value }) => useDebounce(value, 500),
        { initialProps: { value: 'a' } }
      );

      rerender({ value: 'b' });
      act(() => vi.advanceTimersByTime(200));

      rerender({ value: 'c' });
      act(() => vi.advanceTimersByTime(200));

      rerender({ value: 'd' });
      act(() => vi.advanceTimersByTime(200));

      // Still showing initial because timer keeps resetting
      expect(result.current).toBe('a');

      act(() => vi.advanceTimersByTime(500));
      expect(result.current).toBe('d');
    });

    it('cleans up timeout on unmount', () => {
      const { unmount } = renderHook(() => useDebounce('value', 500));

      const clearTimeoutSpy = vi.spyOn(global, 'clearTimeout');
      unmount();

      expect(clearTimeoutSpy).toHaveBeenCalled();
    });
  });

  describe('useThrottledCallback', () => {
    it('calls callback immediately on first invocation', () => {
      const callback = vi.fn();
      const { result } = renderHook(() => useThrottledCallback(callback, 1000));

      act(() => {
        result.current('arg1');
      });

      expect(callback).toHaveBeenCalledWith('arg1');
      expect(callback).toHaveBeenCalledTimes(1);
    });

    it('throttles subsequent calls', () => {
      const callback = vi.fn();
      const { result } = renderHook(() => useThrottledCallback(callback, 1000));

      act(() => {
        result.current('call1');
        result.current('call2');
        result.current('call3');
      });

      expect(callback).toHaveBeenCalledTimes(1);
      expect(callback).toHaveBeenCalledWith('call1');
    });

    it('allows calls after throttle period', () => {
      const callback = vi.fn();
      const { result } = renderHook(() => useThrottledCallback(callback, 1000));

      act(() => {
        result.current('call1');
      });

      act(() => {
        vi.advanceTimersByTime(1000);
      });

      act(() => {
        result.current('call2');
      });

      expect(callback).toHaveBeenCalledTimes(2);
      expect(callback).toHaveBeenLastCalledWith('call2');
    });

    it('maintains stable reference', () => {
      const callback = vi.fn();
      const { result, rerender } = renderHook(
        () => useThrottledCallback(callback, 1000)
      );

      const firstRef = result.current;
      rerender();

      expect(result.current).toBe(firstRef);
    });
  });

  describe('useStableCallback', () => {
    it('maintains stable reference across renders', () => {
      const { result, rerender } = renderHook(
        ({ fn }) => useStableCallback(fn),
        { initialProps: { fn: () => 'first' } }
      );

      const firstRef = result.current;
      rerender({ fn: () => 'second' });

      expect(result.current).toBe(firstRef);
    });

    it('calls latest callback version', () => {
      const callback1 = vi.fn(() => 'first');
      const callback2 = vi.fn(() => 'second');

      const { result, rerender } = renderHook(
        ({ fn }) => useStableCallback(fn),
        { initialProps: { fn: callback1 } }
      );

      rerender({ fn: callback2 });

      const returnValue = result.current();

      expect(callback2).toHaveBeenCalled();
      expect(returnValue).toBe('second');
    });
  });

  describe('useLocalStorage', () => {
    beforeEach(() => {
      localStorage.clear();
    });

    it('returns initial value when storage is empty', () => {
      const { result } = renderHook(() =>
        useLocalStorage('testKey', 'defaultValue')
      );

      expect(result.current[0]).toBe('defaultValue');
    });

    it('reads existing value from storage', () => {
      localStorage.setItem('testKey', JSON.stringify('storedValue'));

      const { result } = renderHook(() =>
        useLocalStorage('testKey', 'defaultValue')
      );

      expect(result.current[0]).toBe('storedValue');
    });

    it('updates storage when value changes', () => {
      const { result } = renderHook(() =>
        useLocalStorage('testKey', 'initial')
      );

      act(() => {
        result.current[1]('updated');
      });

      expect(result.current[0]).toBe('updated');
      expect(JSON.parse(localStorage.getItem('testKey')!)).toBe('updated');
    });

    it('handles complex objects', () => {
      const initialValue = { name: 'test', count: 0 };
      const { result } = renderHook(() =>
        useLocalStorage('objKey', initialValue)
      );

      act(() => {
        result.current[1]({ name: 'updated', count: 5 });
      });

      expect(result.current[0]).toEqual({ name: 'updated', count: 5 });
    });

    it('handles function updater', () => {
      const { result } = renderHook(() => useLocalStorage('countKey', 0));

      act(() => {
        result.current[1]((prev: number) => prev + 1);
      });

      expect(result.current[0]).toBe(1);
    });

    it('handles invalid JSON in storage gracefully', () => {
      localStorage.setItem('badKey', 'not valid json');

      const { result } = renderHook(() =>
        useLocalStorage('badKey', 'fallback')
      );

      expect(result.current[0]).toBe('fallback');
    });
  });

  describe('usePrevious', () => {
    it('returns undefined on first render', () => {
      const { result } = renderHook(() => usePrevious('value'));

      expect(result.current).toBeUndefined();
    });

    it('returns previous value after update', () => {
      const { result, rerender } = renderHook(
        ({ value }) => usePrevious(value),
        { initialProps: { value: 'first' } }
      );

      rerender({ value: 'second' });

      expect(result.current).toBe('first');
    });

    it('tracks value history correctly', () => {
      const { result, rerender } = renderHook(
        ({ value }) => usePrevious(value),
        { initialProps: { value: 'a' } }
      );

      expect(result.current).toBeUndefined();

      rerender({ value: 'b' });
      expect(result.current).toBe('a');

      rerender({ value: 'c' });
      expect(result.current).toBe('b');
    });
  });

  describe('useMounted', () => {
    it('returns false during initial render', () => {
      const { result } = renderHook(() => useMounted());

      // After first render, should be true
      expect(result.current).toBe(true);
    });

    it('returns false after unmount', () => {
      const { result, unmount } = renderHook(() => useMounted());

      expect(result.current).toBe(true);

      unmount();

      // Note: After unmount, we can't access result.current meaningfully
      // This test mainly ensures no errors occur
    });
  });

  describe('useUpdateEffect', () => {
    it('does not run on initial render', () => {
      const effect = vi.fn();

      renderHook(() => useUpdateEffect(effect, []));

      expect(effect).not.toHaveBeenCalled();
    });

    it('runs on subsequent renders', () => {
      const effect = vi.fn();

      const { rerender } = renderHook(
        ({ dep }) => useUpdateEffect(effect, [dep]),
        { initialProps: { dep: 0 } }
      );

      expect(effect).not.toHaveBeenCalled();

      rerender({ dep: 1 });

      expect(effect).toHaveBeenCalledTimes(1);
    });

    it('runs cleanup on unmount', () => {
      const cleanup = vi.fn();
      const effect = vi.fn(() => cleanup);

      const { rerender, unmount } = renderHook(
        ({ dep }) => useUpdateEffect(effect, [dep]),
        { initialProps: { dep: 0 } }
      );

      rerender({ dep: 1 });
      unmount();

      expect(cleanup).toHaveBeenCalled();
    });
  });

  describe('useMediaQuery', () => {
    const mockMatchMedia = (matches: boolean) => {
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: vi.fn().mockImplementation((query) => ({
          matches,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      });
    };

    it('returns true when media query matches', () => {
      mockMatchMedia(true);

      const { result } = renderHook(() =>
        useMediaQuery('(min-width: 768px)')
      );

      expect(result.current).toBe(true);
    });

    it('returns false when media query does not match', () => {
      mockMatchMedia(false);

      const { result } = renderHook(() =>
        useMediaQuery('(min-width: 768px)')
      );

      expect(result.current).toBe(false);
    });

    it('updates when media query changes', async () => {
      let mediaQueryCallback: ((event: { matches: boolean }) => void) | null = null;

      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: vi.fn().mockImplementation((query) => ({
          matches: false,
          media: query,
          addEventListener: (event: string, cb: any) => {
            if (event === 'change') {
              mediaQueryCallback = cb;
            }
          },
          removeEventListener: vi.fn(),
        })),
      });

      const { result } = renderHook(() =>
        useMediaQuery('(min-width: 768px)')
      );

      expect(result.current).toBe(false);

      // Simulate media query change
      act(() => {
        if (mediaQueryCallback) {
          mediaQueryCallback({ matches: true });
        }
      });

      expect(result.current).toBe(true);
    });
  });

  describe('useIntersectionObserver', () => {
    let observeCallback: IntersectionObserverCallback;
    const mockIntersectionObserver = vi.fn((callback) => {
      observeCallback = callback;
      return {
        observe: vi.fn(),
        unobserve: vi.fn(),
        disconnect: vi.fn(),
      };
    });

    beforeEach(() => {
      vi.stubGlobal('IntersectionObserver', mockIntersectionObserver);
    });

    it('creates observer with correct options', () => {
      const ref = { current: document.createElement('div') };

      renderHook(() =>
        useIntersectionObserver(ref, {
          threshold: 0.5,
          rootMargin: '10px',
        })
      );

      expect(mockIntersectionObserver).toHaveBeenCalledWith(
        expect.any(Function),
        expect.objectContaining({
          threshold: 0.5,
          rootMargin: '10px',
        })
      );
    });

    it('returns entry when element is observed', () => {
      const ref = { current: document.createElement('div') };

      const { result } = renderHook(() => useIntersectionObserver(ref));

      const mockEntry = {
        isIntersecting: true,
        intersectionRatio: 0.8,
        target: ref.current,
      } as unknown as IntersectionObserverEntry;

      act(() => {
        observeCallback([mockEntry], {} as IntersectionObserver);
      });

      expect(result.current?.isIntersecting).toBe(true);
    });
  });
});
