// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Reactive State Utilities
 *
 * Provides Observable patterns for "alive" UIs that automatically
 * update when underlying state changes. Essential for the Glass-Top
 * architecture where users see instant, reactive updates.
 *
 * @example
 * ```typescript
 * // Create observable state
 * const balance$ = new BehaviorSubject<number>(0);
 *
 * // Subscribe to changes (UI updates automatically)
 * balance$.subscribe(bal => console.log(`Balance: ${bal}`));
 *
 * // Update triggers all subscribers
 * balance$.next(100); // Logs: "Balance: 100"
 * ```
 */

// =============================================================================
// Core Observable Types
// =============================================================================

/**
 * Observer interface - receives values from an Observable
 */
export interface Observer<T> {
  next: (value: T) => void;
  error?: (err: Error) => void;
  complete?: () => void;
}

/**
 * Subscription - allows unsubscribing from an Observable
 */
export interface Subscription {
  unsubscribe: () => void;
  readonly closed: boolean;
}

/**
 * Subscribable interface - can be subscribed to
 */
export interface Subscribable<T> {
  subscribe(observer: Partial<Observer<T>>): Subscription;
  subscribe(next: (value: T) => void): Subscription;
}

// =============================================================================
// BehaviorSubject - Observable with current value
// =============================================================================

/**
 * BehaviorSubject holds a current value and emits it to new subscribers.
 * Perfect for state management where you always need the latest value.
 *
 * @example
 * ```typescript
 * const state$ = new BehaviorSubject({ balance: 100 });
 *
 * // Get current value synchronously
 * console.log(state$.value); // { balance: 100 }
 *
 * // Subscribe to future changes
 * state$.subscribe(s => updateUI(s));
 *
 * // Update state
 * state$.next({ balance: 150 });
 * ```
 */
export class BehaviorSubject<T> implements Subscribable<T> {
  private _value: T;
  private observers: Set<Partial<Observer<T>>> = new Set();
  private _closed = false;

  constructor(initialValue: T) {
    this._value = initialValue;
  }

  /** Current value (synchronous access) */
  get value(): T {
    return this._value;
  }

  /** Whether this subject is closed */
  get closed(): boolean {
    return this._closed;
  }

  /** Emit a new value to all subscribers */
  next(value: T): void {
    if (this._closed) return;
    this._value = value;
    this.observers.forEach((observer) => {
      try {
        observer.next?.(value);
      } catch (err) {
        observer.error?.(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  /** Signal an error to all subscribers */
  error(err: Error): void {
    if (this._closed) return;
    this.observers.forEach((observer) => {
      observer.error?.(err);
    });
  }

  /** Complete this subject (no more values) */
  complete(): void {
    if (this._closed) return;
    this._closed = true;
    this.observers.forEach((observer) => {
      observer.complete?.();
    });
    this.observers.clear();
  }

  /** Subscribe to value changes */
  subscribe(observerOrNext: Partial<Observer<T>> | ((value: T) => void)): Subscription {
    const observer: Partial<Observer<T>> =
      typeof observerOrNext === 'function' ? { next: observerOrNext } : observerOrNext;

    this.observers.add(observer);

    // Immediately emit current value to new subscriber
    if (observer.next && !this._closed) {
      try {
        observer.next(this._value);
      } catch (err) {
        observer.error?.(err instanceof Error ? err : new Error(String(err)));
      }
    }

    let closed = false;
    return {
      unsubscribe: () => {
        if (!closed) {
          closed = true;
          this.observers.delete(observer);
        }
      },
      get closed() {
        return closed;
      },
    };
  }

  /** Get an Observable that doesn't allow .next() */
  asObservable(): Observable<T> {
    return new Observable((observer) => {
      return this.subscribe(observer);
    });
  }
}

// =============================================================================
// Observable - Lazy push-based collection
// =============================================================================

/**
 * Observable represents a lazy push-based collection.
 * Values are only computed when someone subscribes.
 */
export class Observable<T> implements Subscribable<T> {
  constructor(
    private _subscribe: (observer: Observer<T>) => Subscription | (() => void) | void
  ) {}

  subscribe(observerOrNext: Partial<Observer<T>> | ((value: T) => void)): Subscription {
    const observer: Observer<T> =
      typeof observerOrNext === 'function'
        ? { next: observerOrNext, error: () => {}, complete: () => {} }
        : {
            next: observerOrNext.next ?? (() => {}),
            error: observerOrNext.error ?? (() => {}),
            complete: observerOrNext.complete ?? (() => {}),
          };

    const result = this._subscribe(observer);

    if (!result) {
      return { unsubscribe: () => {}, closed: false };
    }

    if (typeof result === 'function') {
      let closed = false;
      return {
        unsubscribe: () => {
          if (!closed) {
            closed = true;
            result();
          }
        },
        get closed() {
          return closed;
        },
      };
    }

    return result;
  }

  /** Transform values with operators */
  // Overloads for better type inference with common patterns
  pipe<A>(op1: (source: Observable<T>) => Observable<A>): Observable<A>;
  pipe<A, B>(
    op1: (source: Observable<T>) => Observable<A>,
    op2: (source: Observable<A>) => Observable<B>
  ): Observable<B>;
  pipe<A, B, C>(
    op1: (source: Observable<T>) => Observable<A>,
    op2: (source: Observable<A>) => Observable<B>,
    op3: (source: Observable<B>) => Observable<C>
  ): Observable<C>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pipe(...operators: Array<(source: Observable<any>) => Observable<any>>): Observable<any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pipe(...operators: Array<(source: Observable<any>) => Observable<any>>): Observable<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return operators.reduce((prev, op) => op(prev), this as any);
  }
}

// =============================================================================
// Subject - Observable that allows external .next() calls
// =============================================================================

/**
 * Subject is both an Observable and an Observer.
 * Unlike BehaviorSubject, it doesn't hold a current value.
 */
export class Subject<T> implements Subscribable<T> {
  private observers: Set<Partial<Observer<T>>> = new Set();
  private _closed = false;

  get closed(): boolean {
    return this._closed;
  }

  next(value: T): void {
    if (this._closed) return;
    this.observers.forEach((observer) => {
      try {
        observer.next?.(value);
      } catch (err) {
        observer.error?.(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  error(err: Error): void {
    if (this._closed) return;
    this.observers.forEach((observer) => {
      observer.error?.(err);
    });
  }

  complete(): void {
    if (this._closed) return;
    this._closed = true;
    this.observers.forEach((observer) => {
      observer.complete?.();
    });
    this.observers.clear();
  }

  subscribe(observerOrNext: Partial<Observer<T>> | ((value: T) => void)): Subscription {
    const observer: Partial<Observer<T>> =
      typeof observerOrNext === 'function' ? { next: observerOrNext } : observerOrNext;

    this.observers.add(observer);

    let closed = false;
    return {
      unsubscribe: () => {
        if (!closed) {
          closed = true;
          this.observers.delete(observer);
        }
      },
      get closed() {
        return closed;
      },
    };
  }

  asObservable(): Observable<T> {
    return new Observable((observer) => {
      return this.subscribe(observer);
    });
  }
}

// =============================================================================
// ReplaySubject - Replays N values to new subscribers
// =============================================================================

/**
 * ReplaySubject replays the last N values to new subscribers.
 * Useful for events where new subscribers need recent history.
 */
export class ReplaySubject<T> implements Subscribable<T> {
  private observers: Set<Partial<Observer<T>>> = new Set();
  private buffer: T[] = [];
  private _closed = false;

  constructor(private bufferSize: number = Infinity) {}

  get closed(): boolean {
    return this._closed;
  }

  next(value: T): void {
    if (this._closed) return;

    this.buffer.push(value);
    if (this.buffer.length > this.bufferSize) {
      this.buffer.shift();
    }

    this.observers.forEach((observer) => {
      try {
        observer.next?.(value);
      } catch (err) {
        observer.error?.(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  error(err: Error): void {
    if (this._closed) return;
    this.observers.forEach((observer) => {
      observer.error?.(err);
    });
  }

  complete(): void {
    if (this._closed) return;
    this._closed = true;
    this.observers.forEach((observer) => {
      observer.complete?.();
    });
    this.observers.clear();
  }

  subscribe(observerOrNext: Partial<Observer<T>> | ((value: T) => void)): Subscription {
    const observer: Partial<Observer<T>> =
      typeof observerOrNext === 'function' ? { next: observerOrNext } : observerOrNext;

    // Replay buffered values
    if (observer.next && !this._closed) {
      for (const value of this.buffer) {
        try {
          observer.next(value);
        } catch (err) {
          observer.error?.(err instanceof Error ? err : new Error(String(err)));
        }
      }
    }

    this.observers.add(observer);

    let closed = false;
    return {
      unsubscribe: () => {
        if (!closed) {
          closed = true;
          this.observers.delete(observer);
        }
      },
      get closed() {
        return closed;
      },
    };
  }
}

// =============================================================================
// Operators
// =============================================================================

/**
 * Map operator - transform each emitted value
 */
export function map<T, R>(project: (value: T) => R): (source: Observable<T>) => Observable<R> {
  return (source) =>
    new Observable((observer) => {
      return source.subscribe({
        next: (value) => observer.next(project(value)),
        error: (err) => observer.error?.(err),
        complete: () => observer.complete?.(),
      });
    });
}

/**
 * Filter operator - only emit values that pass the predicate
 */
export function filter<T>(predicate: (value: T) => boolean): (source: Observable<T>) => Observable<T> {
  return (source) =>
    new Observable((observer) => {
      return source.subscribe({
        next: (value) => {
          if (predicate(value)) observer.next(value);
        },
        error: (err) => observer.error?.(err),
        complete: () => observer.complete?.(),
      });
    });
}

/**
 * Distinct operator - only emit when value changes
 */
export function distinctUntilChanged<T>(
  compare?: (prev: T, curr: T) => boolean
): (source: Observable<T>) => Observable<T> {
  const compareFn = compare ?? ((a, b) => a === b);
  return (source) =>
    new Observable((observer) => {
      let hasPrev = false;
      let prev: T;

      return source.subscribe({
        next: (value) => {
          if (!hasPrev || !compareFn(prev, value)) {
            hasPrev = true;
            prev = value;
            observer.next(value);
          }
        },
        error: (err) => observer.error?.(err),
        complete: () => observer.complete?.(),
      });
    });
}

/**
 * Debounce operator - only emit after silence period
 */
export function debounceTime<T>(ms: number): (source: Observable<T>) => Observable<T> {
  return (source) =>
    new Observable((observer) => {
      let timeoutId: ReturnType<typeof setTimeout> | null = null;

      const subscription = source.subscribe({
        next: (value) => {
          if (timeoutId) clearTimeout(timeoutId);
          timeoutId = setTimeout(() => observer.next(value), ms);
        },
        error: (err) => observer.error?.(err),
        complete: () => {
          if (timeoutId) clearTimeout(timeoutId);
          observer.complete?.();
        },
      });

      return {
        unsubscribe: () => {
          if (timeoutId) clearTimeout(timeoutId);
          subscription.unsubscribe();
        },
        closed: subscription.closed,
      };
    });
}

/**
 * Take operator - only emit first N values
 */
export function take<T>(count: number): (source: Observable<T>) => Observable<T> {
  return (source) =>
    new Observable((observer) => {
      let taken = 0;

      const subscription = source.subscribe({
        next: (value) => {
          if (taken < count) {
            taken++;
            observer.next(value);
            if (taken >= count) {
              observer.complete?.();
              subscription?.unsubscribe();
            }
          }
        },
        error: (err) => observer.error?.(err),
        complete: () => observer.complete?.(),
      });

      return subscription;
    });
}

/**
 * Skip operator - skip first N values
 */
export function skip<T>(count: number): (source: Observable<T>) => Observable<T> {
  return (source) =>
    new Observable((observer) => {
      let skipped = 0;

      return source.subscribe({
        next: (value) => {
          if (skipped >= count) {
            observer.next(value);
          } else {
            skipped++;
          }
        },
        error: (err) => observer.error?.(err),
        complete: () => observer.complete?.(),
      });
    });
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Combine multiple observables into one that emits arrays
 */
export function combineLatest<T extends unknown[]>(
  ...sources: { [K in keyof T]: Subscribable<T[K]> }
): Observable<T> {
  return new Observable((observer) => {
    const values: T = new Array(sources.length) as T;
    const hasValue: boolean[] = new Array(sources.length).fill(false);
    let completed = 0;

    const subscriptions = sources.map((source, index) =>
      source.subscribe({
        next: (value) => {
          values[index] = value;
          hasValue[index] = true;
          if (hasValue.every(Boolean)) {
            observer.next([...values] as T);
          }
        },
        error: (err) => observer.error?.(err),
        complete: () => {
          completed++;
          if (completed === sources.length) {
            observer.complete?.();
          }
        },
      })
    );

    return {
      unsubscribe: () => subscriptions.forEach((s) => s.unsubscribe()),
      closed: false,
    };
  });
}

/**
 * Merge multiple observables into one
 */
export function merge<T>(...sources: Array<Subscribable<T>>): Observable<T> {
  return new Observable((observer) => {
    let completed = 0;

    const subscriptions = sources.map((source) =>
      source.subscribe({
        next: (value) => observer.next(value),
        error: (err) => observer.error?.(err),
        complete: () => {
          completed++;
          if (completed === sources.length) {
            observer.complete?.();
          }
        },
      })
    );

    return {
      unsubscribe: () => subscriptions.forEach((s) => s.unsubscribe()),
      closed: false,
    };
  });
}

/**
 * Create an Observable from a Promise
 */
export function fromPromise<T>(promise: Promise<T>): Observable<T> {
  return new Observable((observer) => {
    let cancelled = false;

    promise
      .then((value) => {
        if (!cancelled) {
          observer.next(value);
          observer.complete?.();
        }
      })
      .catch((err) => {
        if (!cancelled) {
          observer.error?.(err instanceof Error ? err : new Error(String(err)));
        }
      });

    return () => {
      cancelled = true;
    };
  });
}

/**
 * Create an Observable that emits at intervals
 */
export function interval(ms: number): Observable<number> {
  return new Observable((observer) => {
    let count = 0;
    const id = setInterval(() => observer.next(count++), ms);
    return () => clearInterval(id);
  });
}

/**
 * Create an Observable that emits after a delay
 */
export function timer(ms: number): Observable<0> {
  return new Observable((observer) => {
    const id = setTimeout(() => {
      observer.next(0);
      observer.complete?.();
    }, ms);
    return () => clearTimeout(id);
  });
}

// =============================================================================
// Reactive Store (Simplified Redux-like pattern)
// =============================================================================

/**
 * Action type for store dispatches
 */
export interface Action<T = unknown> {
  type: string;
  payload?: T;
}

/**
 * Reducer function type
 */
export type Reducer<S, A extends Action = Action> = (state: S, action: A) => S;

/**
 * Simple reactive store with time-travel debugging support
 */
export class Store<S, A extends Action = Action> {
  private state$: BehaviorSubject<S>;
  private actions$: Subject<A> = new Subject();
  private history: Array<{ action: A; state: S }> = [];
  private maxHistory: number;

  constructor(
    private reducer: Reducer<S, A>,
    initialState: S,
    options?: { maxHistory?: number }
  ) {
    this.state$ = new BehaviorSubject(initialState);
    this.maxHistory = options?.maxHistory ?? 100;
  }

  /** Current state */
  get state(): S {
    return this.state$.value;
  }

  /** Observable of state changes */
  get state$Observable(): Observable<S> {
    return this.state$.asObservable();
  }

  /** Observable of dispatched actions */
  get actions$Observable(): Observable<A> {
    return this.actions$.asObservable();
  }

  /** Dispatch an action */
  dispatch(action: A): void {
    const prevState = this.state$.value;
    const nextState = this.reducer(prevState, action);

    // Store history for debugging
    this.history.push({ action, state: nextState });
    if (this.history.length > this.maxHistory) {
      this.history.shift();
    }

    this.state$.next(nextState);
    this.actions$.next(action);
  }

  /** Subscribe to state changes */
  subscribe(observer: (state: S) => void): Subscription {
    return this.state$.subscribe(observer);
  }

  /** Select a slice of state */
  select<R>(selector: (state: S) => R): Observable<R> {
    return this.state$.asObservable().pipe(
      map(selector),
      distinctUntilChanged<R>()
    );
  }

  /** Get action history for debugging */
  getHistory(): Array<{ action: A; state: S }> {
    return [...this.history];
  }

  /** Time travel to a previous state (for debugging) */
  timeTravelTo(index: number): void {
    if (index >= 0 && index < this.history.length) {
      this.state$.next(this.history[index].state);
    }
  }
}

// Note: Observer, Subscription, Subscribable are exported at their definitions above
