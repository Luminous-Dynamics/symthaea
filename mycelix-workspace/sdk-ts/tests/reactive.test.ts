// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Reactive Module Tests
 *
 * Tests for Observable patterns and reactive state management.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  BehaviorSubject,
  Observable,
  Subject,
  ReplaySubject,
  Store,
  map,
  filter,
  distinctUntilChanged,
  debounceTime,
  take,
  skip,
  combineLatest,
  merge,
  fromPromise,
  interval,
  timer,
  type Action,
} from '../src/reactive/index.js';

// =============================================================================
// BehaviorSubject Tests
// =============================================================================

describe('BehaviorSubject', () => {
  it('should emit current value to new subscribers', () => {
    const subject = new BehaviorSubject(42);
    const values: number[] = [];

    subject.subscribe((v) => values.push(v));

    expect(values).toEqual([42]);
  });

  it('should emit new values to all subscribers', () => {
    const subject = new BehaviorSubject(0);
    const values1: number[] = [];
    const values2: number[] = [];

    subject.subscribe((v) => values1.push(v));
    subject.subscribe((v) => values2.push(v));

    subject.next(1);
    subject.next(2);

    expect(values1).toEqual([0, 1, 2]);
    expect(values2).toEqual([0, 1, 2]);
  });

  it('should allow synchronous value access', () => {
    const subject = new BehaviorSubject('initial');

    expect(subject.value).toBe('initial');

    subject.next('updated');
    expect(subject.value).toBe('updated');
  });

  it('should support unsubscribe', () => {
    const subject = new BehaviorSubject(0);
    const values: number[] = [];

    const sub = subject.subscribe((v) => values.push(v));
    subject.next(1);

    sub.unsubscribe();
    subject.next(2);

    expect(values).toEqual([0, 1]);
  });

  it('should handle complete', () => {
    const subject = new BehaviorSubject(0);
    let completed = false;

    subject.subscribe({
      next: () => {},
      complete: () => {
        completed = true;
      },
    });

    subject.complete();

    expect(completed).toBe(true);
    expect(subject.closed).toBe(true);
  });

  it('should not emit after complete', () => {
    const subject = new BehaviorSubject(0);
    const values: number[] = [];

    subject.subscribe((v) => values.push(v));
    subject.complete();
    subject.next(1);

    expect(values).toEqual([0]);
  });

  it('should handle error callback', () => {
    const subject = new BehaviorSubject(0);
    const errors: Error[] = [];

    subject.subscribe({
      next: () => {
        throw new Error('Test error');
      },
      error: (err) => errors.push(err),
    });

    expect(errors.length).toBe(1);
    expect(errors[0].message).toBe('Test error');
  });

  it('should report closed state correctly', () => {
    const subject = new BehaviorSubject(0);

    expect(subject.closed).toBe(false);

    subject.complete();

    expect(subject.closed).toBe(true);
  });
});

// =============================================================================
// Subject Tests
// =============================================================================

describe('Subject', () => {
  it('should not emit to new subscribers until next value', () => {
    const subject = new Subject<number>();
    const values: number[] = [];

    subject.subscribe((v) => values.push(v));

    expect(values).toEqual([]);

    subject.next(1);
    expect(values).toEqual([1]);
  });

  it('should emit to multiple subscribers', () => {
    const subject = new Subject<string>();
    const values1: string[] = [];
    const values2: string[] = [];

    subject.subscribe((v) => values1.push(v));
    subject.subscribe((v) => values2.push(v));

    subject.next('hello');

    expect(values1).toEqual(['hello']);
    expect(values2).toEqual(['hello']);
  });

  it('should handle complete', () => {
    const subject = new Subject<number>();
    let completed = false;

    subject.subscribe({
      next: () => {},
      complete: () => {
        completed = true;
      },
    });

    subject.complete();

    expect(completed).toBe(true);
  });

  it('should not emit after complete', () => {
    const subject = new Subject<number>();
    const values: number[] = [];

    subject.subscribe((v) => values.push(v));
    subject.next(1);
    subject.complete();
    subject.next(2);

    expect(values).toEqual([1]);
  });
});

// =============================================================================
// ReplaySubject Tests
// =============================================================================

describe('ReplaySubject', () => {
  it('should replay last N values to new subscribers', () => {
    const subject = new ReplaySubject<number>(3);

    subject.next(1);
    subject.next(2);
    subject.next(3);
    subject.next(4);

    const values: number[] = [];
    subject.subscribe((v) => values.push(v));

    // Should replay last 3 values: 2, 3, 4
    expect(values).toEqual([2, 3, 4]);
  });

  it('should replay all values when fewer than buffer size', () => {
    const subject = new ReplaySubject<string>(5);

    subject.next('a');
    subject.next('b');

    const values: string[] = [];
    subject.subscribe((v) => values.push(v));

    expect(values).toEqual(['a', 'b']);
  });

  it('should emit new values after replay', () => {
    const subject = new ReplaySubject<number>(2);

    subject.next(1);
    subject.next(2);

    const values: number[] = [];
    subject.subscribe((v) => values.push(v));

    subject.next(3);

    expect(values).toEqual([1, 2, 3]);
  });

  it('should handle complete', () => {
    const subject = new ReplaySubject<number>(2);
    let completed = false;

    subject.next(1);

    subject.subscribe({
      next: () => {},
      complete: () => {
        completed = true;
      },
    });

    subject.complete();

    expect(completed).toBe(true);
  });
});

// =============================================================================
// Observable Tests
// =============================================================================

describe('Observable', () => {
  it('should create from subscribe function', () => {
    const observable = new Observable<number>((observer) => {
      observer.next?.(1);
      observer.next?.(2);
      observer.complete?.();
    });

    const values: number[] = [];
    let completed = false;

    observable.subscribe({
      next: (v) => values.push(v),
      complete: () => {
        completed = true;
      },
    });

    expect(values).toEqual([1, 2]);
    expect(completed).toBe(true);
  });

  it('should support function-style subscribe', () => {
    const observable = new Observable<string>((observer) => {
      observer.next?.('hello');
    });

    const values: string[] = [];
    observable.subscribe((v) => values.push(v));

    expect(values).toEqual(['hello']);
  });

  it('should handle unsubscribe', () => {
    let teardownCalled = false;

    const observable = new Observable<number>((observer) => {
      const id = setInterval(() => observer.next?.(1), 10);
      return () => {
        clearInterval(id);
        teardownCalled = true;
      };
    });

    const sub = observable.subscribe(() => {});
    sub.unsubscribe();

    expect(teardownCalled).toBe(true);
    expect(sub.closed).toBe(true);
  });
});

// =============================================================================
// Operator Tests
// =============================================================================

describe('Operators', () => {
  describe('map', () => {
    it('should transform values', () => {
      const subject = new Subject<number>();
      const mapped = map<number, string>((n) => `value: ${n}`)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: string[] = [];
      mapped.subscribe((v) => values.push(v));

      subject.next(1);
      subject.next(2);

      expect(values).toEqual(['value: 1', 'value: 2']);
    });
  });

  describe('filter', () => {
    it('should only pass matching values', () => {
      const subject = new Subject<number>();
      const filtered = filter<number>((n) => n > 5)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: number[] = [];
      filtered.subscribe((v) => values.push(v));

      subject.next(3);
      subject.next(7);
      subject.next(4);
      subject.next(10);

      expect(values).toEqual([7, 10]);
    });
  });

  describe('distinctUntilChanged', () => {
    it('should dedupe consecutive values', () => {
      const subject = new Subject<number>();
      const distinct = distinctUntilChanged<number>()(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: number[] = [];
      distinct.subscribe((v) => values.push(v));

      subject.next(1);
      subject.next(1);
      subject.next(2);
      subject.next(2);
      subject.next(1);

      expect(values).toEqual([1, 2, 1]);
    });

    it('should use custom comparator', () => {
      const subject = new Subject<{ id: number }>();
      const distinct = distinctUntilChanged<{ id: number }>((a, b) => a.id === b.id)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: Array<{ id: number }> = [];
      distinct.subscribe((v) => values.push(v));

      subject.next({ id: 1 });
      subject.next({ id: 1 });
      subject.next({ id: 2 });

      expect(values.length).toBe(2);
      expect(values[0].id).toBe(1);
      expect(values[1].id).toBe(2);
    });
  });

  describe('debounceTime', () => {
    it('should debounce rapid emissions', async () => {
      const subject = new Subject<number>();
      const debounced = debounceTime<number>(50)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: number[] = [];
      debounced.subscribe((v) => values.push(v));

      subject.next(1);
      subject.next(2);
      subject.next(3);

      await new Promise((r) => setTimeout(r, 100));

      expect(values).toEqual([3]);
    });
  });

  describe('take', () => {
    it('should take only N values', () => {
      const subject = new Subject<number>();
      const taken = take<number>(2)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: number[] = [];
      let completed = false;

      taken.subscribe({
        next: (v) => values.push(v),
        complete: () => {
          completed = true;
        },
      });

      subject.next(1);
      subject.next(2);
      subject.next(3);

      expect(values).toEqual([1, 2]);
      expect(completed).toBe(true);
    });
  });

  describe('skip', () => {
    it('should skip first N values', () => {
      const subject = new Subject<number>();
      const skipped = skip<number>(2)(
        new Observable((obs) => {
          subject.subscribe(obs);
        })
      );

      const values: number[] = [];
      skipped.subscribe((v) => values.push(v));

      subject.next(1);
      subject.next(2);
      subject.next(3);
      subject.next(4);

      expect(values).toEqual([3, 4]);
    });
  });
});

// =============================================================================
// Combiner Tests
// =============================================================================

describe('Combiners', () => {
  describe('combineLatest', () => {
    it('should combine latest values from multiple sources', () => {
      const a$ = new BehaviorSubject(1);
      const b$ = new BehaviorSubject('a');

      // combineLatest takes variadic args, not array
      const combined = combineLatest(a$, b$);

      const values: Array<[number, string]> = [];
      combined.subscribe((v) => values.push(v as [number, string]));

      expect(values).toEqual([[1, 'a']]);

      a$.next(2);
      expect(values).toEqual([
        [1, 'a'],
        [2, 'a'],
      ]);

      b$.next('b');
      expect(values).toEqual([
        [1, 'a'],
        [2, 'a'],
        [2, 'b'],
      ]);
    });
  });

  describe('merge', () => {
    it('should merge emissions from multiple sources', () => {
      const a$ = new Subject<number>();
      const b$ = new Subject<number>();

      const merged = merge(a$, b$);

      const values: number[] = [];
      merged.subscribe((v) => values.push(v));

      a$.next(1);
      b$.next(2);
      a$.next(3);

      expect(values).toEqual([1, 2, 3]);
    });
  });
});

// =============================================================================
// Creator Tests
// =============================================================================

describe('Creators', () => {
  describe('fromPromise', () => {
    it('should emit resolved value and complete', async () => {
      const promise = Promise.resolve(42);
      const observable = fromPromise(promise);

      const values: number[] = [];
      let completed = false;

      observable.subscribe({
        next: (v) => values.push(v),
        complete: () => {
          completed = true;
        },
      });

      await promise;
      await new Promise((r) => setTimeout(r, 0));

      expect(values).toEqual([42]);
      expect(completed).toBe(true);
    });

    it('should emit error on rejection', async () => {
      const promise = Promise.reject(new Error('Failed'));
      const observable = fromPromise(promise);

      let error: Error | null = null;

      observable.subscribe({
        next: () => {},
        error: (err) => {
          error = err;
        },
      });

      await new Promise((r) => setTimeout(r, 10));

      expect(error).not.toBeNull();
      expect(error?.message).toBe('Failed');
    });
  });

  describe('interval', () => {
    it('should emit incrementing numbers', async () => {
      const int$ = interval(50);

      const values: number[] = [];
      const sub = int$.subscribe((v) => values.push(v));

      await new Promise((r) => setTimeout(r, 175));

      sub.unsubscribe();

      expect(values.length).toBeGreaterThanOrEqual(2);
      expect(values[0]).toBe(0);
      expect(values[1]).toBe(1);
    });
  });

  describe('timer', () => {
    it('should emit after delay and complete', async () => {
      const timer$ = timer(50);

      const values: number[] = [];
      let completed = false;

      timer$.subscribe({
        next: (v) => values.push(v),
        complete: () => {
          completed = true;
        },
      });

      await new Promise((r) => setTimeout(r, 100));

      expect(values).toEqual([0]);
      expect(completed).toBe(true);
    });
  });
});

// =============================================================================
// Store Tests
// =============================================================================

describe('Store', () => {
  interface CounterState {
    count: number;
  }

  interface IncrementAction extends Action<'INCREMENT'> {
    type: 'INCREMENT';
  }

  interface DecrementAction extends Action<'DECREMENT'> {
    type: 'DECREMENT';
  }

  interface AddAction extends Action<'ADD'> {
    type: 'ADD';
    payload: number;
  }

  type CounterAction = IncrementAction | DecrementAction | AddAction;

  const counterReducer = (state: CounterState, action: CounterAction): CounterState => {
    switch (action.type) {
      case 'INCREMENT':
        return { count: state.count + 1 };
      case 'DECREMENT':
        return { count: state.count - 1 };
      case 'ADD':
        return { count: state.count + action.payload };
      default:
        return state;
    }
  };

  it('should initialize with initial state', () => {
    const store = new Store<CounterState, CounterAction>(counterReducer, { count: 0 });

    expect(store.state).toEqual({ count: 0 });
  });

  it('should dispatch actions and update state', () => {
    const store = new Store<CounterState, CounterAction>(counterReducer, { count: 0 });

    store.dispatch({ type: 'INCREMENT' });
    expect(store.state).toEqual({ count: 1 });

    store.dispatch({ type: 'INCREMENT' });
    expect(store.state).toEqual({ count: 2 });

    store.dispatch({ type: 'DECREMENT' });
    expect(store.state).toEqual({ count: 1 });
  });

  it('should handle actions with payload', () => {
    const store = new Store<CounterState, CounterAction>(counterReducer, { count: 0 });

    store.dispatch({ type: 'ADD', payload: 10 });
    expect(store.state).toEqual({ count: 10 });
  });

  it('should notify subscribers on state change', () => {
    const store = new Store<CounterState, CounterAction>(counterReducer, { count: 0 });

    const states: CounterState[] = [];
    store.subscribe((state) => states.push(state));

    store.dispatch({ type: 'INCREMENT' });
    store.dispatch({ type: 'INCREMENT' });

    expect(states).toEqual([{ count: 0 }, { count: 1 }, { count: 2 }]);
  });

  it('should allow selecting state slices', () => {
    const store = new Store<CounterState, CounterAction>(counterReducer, { count: 5 });

    const counts: number[] = [];
    store.select((s) => s.count).subscribe((count) => counts.push(count));

    store.dispatch({ type: 'INCREMENT' });

    expect(counts).toEqual([5, 6]);
  });
});
