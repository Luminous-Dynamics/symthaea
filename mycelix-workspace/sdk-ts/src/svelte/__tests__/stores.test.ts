/**
 * Svelte Integration Tests
 *
 * Tests for Svelte stores, reactive updates, and component lifecycle.
 *
 * @module @mycelix/sdk/svelte/__tests__
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  writable,
  derived,
  setMycelixContext,
  getContextStore,
  createQueryStore,
  createMutationStore,
  type Readable,
  type Writable,
  type QueryState,
  type MutationState,
  type MycelixContext,
} from '../index';

// Mock context
const mockContext: MycelixContext = {
  identityService: {},
  financeService: {},
  propertyService: {},
  energyService: {},
  mediaService: {},
  governanceService: {},
  justiceService: {},
  knowledgeService: {},
};

describe('Writable Store', () => {
  it('should create a writable store with initial value', () => {
    const store = writable(42);
    let value: number | undefined;

    const unsubscribe = store.subscribe((v) => {
      value = v;
    });

    expect(value).toBe(42);
    unsubscribe();
  });

  it('should update value with set()', () => {
    const store = writable('initial');
    let value: string | undefined;

    store.subscribe((v) => {
      value = v;
    });

    store.set('updated');
    expect(value).toBe('updated');
  });

  it('should update value with update()', () => {
    const store = writable(10);
    let value: number | undefined;

    store.subscribe((v) => {
      value = v;
    });

    store.update((v) => v * 2);
    expect(value).toBe(20);
  });

  it('should notify all subscribers', () => {
    const store = writable(0);
    const values1: number[] = [];
    const values2: number[] = [];

    store.subscribe((v) => values1.push(v));
    store.subscribe((v) => values2.push(v));

    store.set(1);
    store.set(2);

    expect(values1).toEqual([0, 1, 2]);
    expect(values2).toEqual([0, 1, 2]);
  });

  it('should stop notifying after unsubscribe', () => {
    const store = writable(0);
    const values: number[] = [];

    const unsubscribe = store.subscribe((v) => values.push(v));

    store.set(1);
    unsubscribe();
    store.set(2);

    expect(values).toEqual([0, 1]);
  });
});

describe('Derived Store', () => {
  it('should derive value from source store', () => {
    const source = writable(5);
    const doubled = derived(source, (v) => v * 2);
    let value: number | undefined;

    doubled.subscribe((v) => {
      value = v;
    });

    expect(value).toBe(10);
  });

  it('should update when source updates', () => {
    const source = writable('hello');
    const uppercased = derived(source, (v) => v.toUpperCase());
    let value: string | undefined;

    uppercased.subscribe((v) => {
      value = v;
    });

    source.set('world');
    expect(value).toBe('WORLD');
  });

  it('should work with complex transformations', () => {
    interface User {
      name: string;
      age: number;
    }

    const userStore = writable<User>({ name: 'Alice', age: 30 });
    const nameStore = derived(userStore, (user) => user.name);
    const ageStore = derived(userStore, (user) => user.age);
    const isAdultStore = derived(ageStore, (age) => age >= 18);

    let name: string | undefined;
    let isAdult: boolean | undefined;

    nameStore.subscribe((v) => { name = v; });
    isAdultStore.subscribe((v) => { isAdult = v; });

    expect(name).toBe('Alice');
    expect(isAdult).toBe(true);

    userStore.set({ name: 'Bob', age: 15 });
    expect(name).toBe('Bob');
    expect(isAdult).toBe(false);
  });
});

describe('Context Store', () => {
  beforeEach(() => {
    // Reset context before each test
    setMycelixContext(null as unknown as MycelixContext);
  });

  it('should start with null context', () => {
    const store = getContextStore();
    let value: MycelixContext | null | undefined;

    store.subscribe((v) => {
      value = v;
    });

    expect(value).toBeNull();
  });

  it('should update when context is set', () => {
    const store = getContextStore();
    let value: MycelixContext | null | undefined;

    store.subscribe((v) => {
      value = v;
    });

    setMycelixContext(mockContext);
    expect(value).toBe(mockContext);
  });
});

describe('QueryStore', () => {
  beforeEach(() => {
    setMycelixContext(mockContext);
  });

  it('should create a query store with loading state', () => {
    const fetcher = vi.fn().mockResolvedValue('data');
    const store = createQueryStore(fetcher, undefined);
    let state: QueryState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    expect(state?.loading).toBe(true);
    expect(state?.data).toBeUndefined();
    expect(state?.error).toBeUndefined();
  });

  it('should update state on successful fetch', async () => {
    const fetcher = vi.fn().mockResolvedValue('test-data');
    const store = createQueryStore(fetcher, undefined);
    let state: QueryState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    // Wait for fetch to complete
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(state?.loading).toBe(false);
    expect(state?.data).toBe('test-data');
    expect(state?.error).toBeUndefined();
  });

  it('should update state on fetch error', async () => {
    const error = new Error('Fetch failed');
    const fetcher = vi.fn().mockRejectedValue(error);
    const store = createQueryStore(fetcher, undefined);
    let state: QueryState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    // Wait for fetch to fail
    await new Promise((resolve) => setTimeout(resolve, 10));

    expect(state?.loading).toBe(false);
    expect(state?.data).toBeUndefined();
    expect(state?.error).toBe(error);
  });

  it('should support refetch', async () => {
    let callCount = 0;
    const fetcher = vi.fn().mockImplementation(() => {
      callCount++;
      return Promise.resolve(`data-${callCount}`);
    });
    const store = createQueryStore(fetcher, undefined);
    let state: QueryState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(state?.data).toBe('data-1');

    await store.refetch();
    expect(state?.data).toBe('data-2');
  });
});

describe('MutationStore', () => {
  beforeEach(() => {
    setMycelixContext(mockContext);
  });

  it('should create a mutation store with initial state', () => {
    const mutator = vi.fn().mockResolvedValue('result');
    const store = createMutationStore(mutator);
    let state: MutationState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    expect(state?.loading).toBe(false);
    expect(state?.data).toBeUndefined();
    expect(state?.error).toBeUndefined();
  });

  it('should update state during mutation', async () => {
    const mutator = vi.fn().mockImplementation(() =>
      new Promise((resolve) => setTimeout(() => resolve('result'), 50))
    );
    const store = createMutationStore(mutator);
    const states: MutationState<string>[] = [];

    store.subscribe((s) => {
      states.push({ ...s });
    });

    const mutatePromise = store.mutate({ input: 'test' });

    // Check loading state
    await new Promise((resolve) => setTimeout(resolve, 10));
    const loadingState = states.find((s) => s.loading);
    expect(loadingState?.loading).toBe(true);

    await mutatePromise;

    // Check final state
    const finalState = states[states.length - 1];
    expect(finalState?.loading).toBe(false);
    expect(finalState?.data).toBe('result');
  });

  it('should handle mutation error', async () => {
    const error = new Error('Mutation failed');
    const mutator = vi.fn().mockRejectedValue(error);
    const store = createMutationStore(mutator);
    let state: MutationState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    try {
      await store.mutate({ input: 'test' });
    } catch {
      // Expected error
    }

    expect(state?.loading).toBe(false);
    expect(state?.error).toBe(error);
  });

  it('should support reset', async () => {
    const mutator = vi.fn().mockResolvedValue('result');
    const store = createMutationStore(mutator);
    let state: MutationState<string> | undefined;

    store.subscribe((s) => {
      state = s;
    });

    await store.mutate({ input: 'test' });
    expect(state?.data).toBe('result');

    store.reset();
    expect(state?.data).toBeUndefined();
    expect(state?.error).toBeUndefined();
    expect(state?.loading).toBe(false);
  });
});

describe('Reactive Updates', () => {
  it('should propagate updates synchronously', () => {
    const store = writable(0);
    const updates: number[] = [];

    store.subscribe((v) => updates.push(v));

    store.set(1);
    store.set(2);
    store.set(3);

    // All updates should be synchronous
    expect(updates).toEqual([0, 1, 2, 3]);
  });

  it('should support chained derivations', () => {
    const a = writable(1);
    const b = derived(a, (v) => v + 1);
    const c = derived(b, (v) => v * 2);
    const d = derived(c, (v) => `result: ${v}`);

    let value: string | undefined;
    d.subscribe((v) => { value = v; });

    expect(value).toBe('result: 4'); // (1 + 1) * 2 = 4

    a.set(5);
    expect(value).toBe('result: 12'); // (5 + 1) * 2 = 12
  });
});

describe('Component Lifecycle', () => {
  it('should clean up subscriptions', () => {
    const store = writable(0);
    const subscribeFn = vi.fn();

    const unsubscribe = store.subscribe(subscribeFn);
    expect(subscribeFn).toHaveBeenCalledTimes(1);

    store.set(1);
    expect(subscribeFn).toHaveBeenCalledTimes(2);

    unsubscribe();

    store.set(2);
    // Should not be called after unsubscribe
    expect(subscribeFn).toHaveBeenCalledTimes(2);
  });

  it('should handle multiple subscribe/unsubscribe cycles', () => {
    const store = writable('value');
    const values: string[] = [];

    for (let i = 0; i < 3; i++) {
      const unsubscribe = store.subscribe((v) => values.push(v));
      unsubscribe();
    }

    // Each subscription should receive initial value
    expect(values).toEqual(['value', 'value', 'value']);

    // After unsubscribe, updates should not be received
    store.set('new-value');
    expect(values).toEqual(['value', 'value', 'value']);
  });
});

describe('Type Safety', () => {
  it('should preserve types through derivations', () => {
    interface Config {
      theme: 'light' | 'dark';
      fontSize: number;
    }

    const configStore = writable<Config>({
      theme: 'light',
      fontSize: 14,
    });

    const themeStore = derived(configStore, (c) => c.theme);
    const isDarkStore = derived(themeStore, (t) => t === 'dark');

    let isDark: boolean | undefined;
    isDarkStore.subscribe((v) => { isDark = v; });

    expect(isDark).toBe(false);

    configStore.update((c) => ({ ...c, theme: 'dark' }));
    expect(isDark).toBe(true);
  });
});
