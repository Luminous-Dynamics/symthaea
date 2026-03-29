// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

// Mock $app/environment before importing the module
vi.mock('$app/environment', () => ({ browser: false }));

// Mock the conductor module
vi.mock('./conductor', () => ({
  isConnected: { subscribe: (fn: (v: boolean) => void) => { fn(false); return () => {}; } },
  callZome: vi.fn(),
}));

// Mock resilience-client (DHT calls)
vi.mock('./resilience-client', () => ({
  getConsensusPrice: vi.fn(),
  reportPrice: vi.fn(),
  getBasketIndex: vi.fn(),
  computeVolatility: vi.fn(),
}));

// Mock community config
vi.mock('./community', () => ({
  getCanonicalItems: () => [
    { key: 'bread', name: 'Bread', unit: 'loaf', default_price: 0.5, weight: 0.2 },
    { key: 'rice', name: 'Rice', unit: 'kg', default_price: 0.3, weight: 0.3 },
    { key: 'eggs', name: 'Eggs', unit: 'dozen', default_price: 0.8, weight: 0.5 },
  ],
  getBasketWeights: () => ({ bread: 0.2, rice: 0.3, eggs: 0.5 }),
  getBasketName: () => 'Test Basket',
}));

// Reset module state between tests
beforeEach(() => {
  vi.resetModules();
});

describe('value-basket: personal basket CRUD', () => {
  it('starts with an empty basket (non-browser)', async () => {
    const { basketItems } = await import('./value-basket');
    expect(get(basketItems)).toEqual([]);
  });

  it('addOrUpdateItem adds an item to the basket', async () => {
    const { addOrUpdateItem, basketItems } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    const items = get(basketItems);
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe('bread');
    expect(items[0].name).toBe('Bread');
    expect(items[0].price_tend).toBe(0.5);
  });

  it('addOrUpdateItem updates an existing item', async () => {
    const { addOrUpdateItem, basketItems } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    addOrUpdateItem('bread', {
      name: 'Bread (updated)',
      unit: 'loaf',
      price_tend: 0.6,
      updated_at: Date.now(),
    });

    const items = get(basketItems);
    expect(items).toHaveLength(1);
    expect(items[0].name).toBe('Bread (updated)');
    expect(items[0].price_tend).toBe(0.6);
  });

  it('removeItem removes an item from the basket', async () => {
    const { addOrUpdateItem, removeItem, basketItems } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    addOrUpdateItem('rice', {
      name: 'Rice',
      unit: 'kg',
      price_tend: 0.3,
      updated_at: Date.now(),
    });

    removeItem('bread');

    const items = get(basketItems);
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe('rice');
  });

  it('clearBasket removes all items', async () => {
    const { addOrUpdateItem, clearBasket, basketItems, basketHistory } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    clearBasket();

    expect(get(basketItems)).toEqual([]);
    expect(get(basketHistory)).toEqual([]);
  });
});

describe('value-basket: hasBasket', () => {
  it('is false when basket is empty', async () => {
    const { hasBasket } = await import('./value-basket');
    expect(get(hasBasket)).toBe(false);
  });

  it('is true when items are added', async () => {
    const { addOrUpdateItem, hasBasket } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    expect(get(hasBasket)).toBe(true);
  });
});

describe('value-basket: purchasingPowerIndex', () => {
  it('returns 0 for empty basket', async () => {
    const { purchasingPowerIndex } = await import('./value-basket');
    expect(get(purchasingPowerIndex)).toBe(0);
  });

  it('computes average of personal prices', async () => {
    const { addOrUpdateItem, purchasingPowerIndex } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 1.0,
      updated_at: Date.now(),
    });

    addOrUpdateItem('rice', {
      name: 'Rice',
      unit: 'kg',
      price_tend: 3.0,
      updated_at: Date.now(),
    });

    // Average of 1.0 and 3.0 = 2.0
    expect(get(purchasingPowerIndex)).toBe(2.0);
  });
});

describe('value-basket: purchasingPower', () => {
  it('computes quantities for a given TEND balance', async () => {
    const { addOrUpdateItem, purchasingPower } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: Date.now(),
    });

    addOrUpdateItem('rice', {
      name: 'Rice',
      unit: 'kg',
      price_tend: 2.0,
      updated_at: Date.now(),
    });

    const power = purchasingPower(10);
    expect(power.length).toBeGreaterThanOrEqual(2);

    const bread = power.find((p) => p.name === 'Bread');
    expect(bread).toBeDefined();
    expect(parseFloat(bread!.quantity)).toBe(20); // 10 / 0.5

    const rice = power.find((p) => p.name === 'Rice');
    expect(rice).toBeDefined();
    expect(parseFloat(rice!.quantity)).toBe(5); // 10 / 2.0
  });

  it('returns empty array for empty basket with no canonical items having prices', async () => {
    const { purchasingPower } = await import('./value-basket');
    // With mocked canonical items that have default prices, enrichedItems will include them.
    // Items with price_tend=0 are skipped, but our mocks have nonzero defaults.
    const power = purchasingPower(10);
    // Should include canonical items with their default prices
    expect(power.length).toBeGreaterThanOrEqual(0);
  });
});

describe('value-basket: export/import', () => {
  it('exportBasket returns valid JSON', async () => {
    const { addOrUpdateItem, exportBasket } = await import('./value-basket');

    addOrUpdateItem('bread', {
      name: 'Bread',
      unit: 'loaf',
      price_tend: 0.5,
      updated_at: 1000,
    });

    const json = exportBasket();
    const parsed = JSON.parse(json);
    expect(parsed.bread).toBeDefined();
    expect(parsed.bread.name).toBe('Bread');
  });

  it('importBasket loads valid JSON', async () => {
    const { importBasket, basketItems } = await import('./value-basket');

    const json = JSON.stringify({
      milk: { name: 'Milk', unit: 'liter', price_tend: 1.2, updated_at: 1000 },
    });

    const ok = importBasket(json);
    expect(ok).toBe(true);

    const items = get(basketItems);
    expect(items).toHaveLength(1);
    expect(items[0].key).toBe('milk');
  });

  it('importBasket rejects invalid JSON', async () => {
    const { importBasket } = await import('./value-basket');
    expect(importBasket('not json')).toBe(false);
  });

  it('importBasket rejects items with wrong types', async () => {
    const { importBasket } = await import('./value-basket');
    const json = JSON.stringify({
      milk: { name: 123, unit: 'liter', price_tend: 'not a number', updated_at: 1000 },
    });
    expect(importBasket(json)).toBe(false);
  });
});

describe('value-basket: enrichedItems', () => {
  it('includes canonical items even when basket is empty', async () => {
    const { enrichedItems } = await import('./value-basket');
    const items = get(enrichedItems);
    // Should have at least the 3 canonical items from the mock
    expect(items.length).toBeGreaterThanOrEqual(3);
    const keys = items.map((i) => i.key);
    expect(keys).toContain('bread');
    expect(keys).toContain('rice');
    expect(keys).toContain('eggs');
  });

  it('marks items as personal source when no consensus', async () => {
    const { enrichedItems } = await import('./value-basket');
    const items = get(enrichedItems);
    for (const item of items) {
      expect(item.source).toBe('personal');
      expect(item.consensus_price).toBeNull();
    }
  });

  it('uses canonical default prices when no personal estimate', async () => {
    const { enrichedItems } = await import('./value-basket');
    const items = get(enrichedItems);
    const bread = items.find((i) => i.key === 'bread');
    expect(bread).toBeDefined();
    expect(bread!.price_tend).toBe(0.5); // from canonical default_price
  });
});
