import { describe, it, expect } from 'vitest';
import config from './community-config.json';

// ============================================================================
// JSON Structure Validation
// ============================================================================

describe('community-config: JSON structure', () => {
  it('parses as a valid object', () => {
    expect(config).toBeDefined();
    expect(typeof config).toBe('object');
    expect(config).not.toBeNull();
  });

  it('has all required top-level fields', () => {
    const required = [
      'community_name',
      'basket_name',
      'dao_did',
      'currency_code',
      'currency_symbol',
      'labor_hour_value',
      'basket_items',
    ] as const;

    for (const field of required) {
      expect(config).toHaveProperty(field);
    }
  });

  it('has string fields with non-empty values', () => {
    expect(typeof config.community_name).toBe('string');
    expect(config.community_name.length).toBeGreaterThan(0);

    expect(typeof config.basket_name).toBe('string');
    expect(config.basket_name.length).toBeGreaterThan(0);

    expect(typeof config.dao_did).toBe('string');
    expect(config.dao_did.length).toBeGreaterThan(0);

    expect(typeof config.currency_symbol).toBe('string');
    expect(config.currency_symbol.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Currency Validation
// ============================================================================

describe('community-config: currency', () => {
  it('has a valid ISO 4217 currency code (3 uppercase letters)', () => {
    expect(config.currency_code).toMatch(/^[A-Z]{3}$/);
  });

  it('has a positive labor_hour_value', () => {
    expect(typeof config.labor_hour_value).toBe('number');
    expect(config.labor_hour_value).toBeGreaterThan(0);
    expect(Number.isFinite(config.labor_hour_value)).toBe(true);
  });
});

// ============================================================================
// Basket Items Validation
// ============================================================================

describe('community-config: basket_items', () => {
  it('is an array with at least 1 item', () => {
    expect(Array.isArray(config.basket_items)).toBe(true);
    expect(config.basket_items.length).toBeGreaterThanOrEqual(1);
  });

  it('each item has required fields: name, unit, default_price, weight', () => {
    for (const item of config.basket_items) {
      expect(typeof item.name).toBe('string');
      expect(item.name.length).toBeGreaterThan(0);

      expect(typeof item.unit).toBe('string');
      expect(item.unit.length).toBeGreaterThan(0);

      expect(typeof item.default_price).toBe('number');
      expect(typeof item.weight).toBe('number');
    }
  });

  it('each item has a positive default_price', () => {
    for (const item of config.basket_items) {
      expect(item.default_price).toBeGreaterThan(0);
    }
  });

  it('each item weight is between 0 and 1 (exclusive/inclusive)', () => {
    for (const item of config.basket_items) {
      expect(item.weight).toBeGreaterThan(0);
      expect(item.weight).toBeLessThanOrEqual(1);
    }
  });

  it('weights sum to approximately 1.0', () => {
    const totalWeight = config.basket_items.reduce((sum, item) => sum + item.weight, 0);
    expect(totalWeight).toBeCloseTo(1.0, 1); // within 0.05
  });

  it('has no duplicate item names', () => {
    const names = config.basket_items.map((item) => item.name);
    const unique = new Set(names);
    expect(unique.size).toBe(names.length);
  });

  it('has no duplicate item keys', () => {
    const keys = config.basket_items.map((item) => item.key);
    const unique = new Set(keys);
    expect(unique.size).toBe(keys.length);
  });
});
