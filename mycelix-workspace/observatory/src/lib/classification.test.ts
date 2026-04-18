// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
import { describe, it, expect } from 'vitest';
import {
  ALL_CLASSIFICATIONS,
  checkReadiness,
  classify,
  countByClassification,
  deserializeOverlay,
  emptyOverlay,
  isExpense,
  isIncome,
  isNeutral,
  serializeOverlay,
  setClassification,
  unclassify,
  unclassifiedFromList,
  type ClassificationOverlay,
} from './classification';

describe('classification overlay', () => {
  it('starts empty', () => {
    const o = emptyOverlay();
    expect(o.version).toBe(1);
    expect(Object.keys(o.entries)).toHaveLength(0);
  });

  it('classifies unknown hash as Unclassified', () => {
    const o = emptyOverlay();
    expect(classify(o, 'hash-never-seen')).toBe('Unclassified');
  });

  it('set then read round-trips', () => {
    let o: ClassificationOverlay = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'BarterLabor');
    expect(classify(o, 'tx-1')).toBe('BarterLabor');
  });

  it('set returns a new overlay (treat as immutable)', () => {
    const o = emptyOverlay();
    const updated = setClassification(o, 'tx-1', 'sap', 'Wage');
    expect(o.entries).not.toBe(updated.entries);
    expect(classify(o, 'tx-1')).toBe('Unclassified');
    expect(classify(updated, 'tx-1')).toBe('Wage');
  });

  it('unclassify removes the entry', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'marketplace', 'Sale');
    o = unclassify(o, 'tx-1');
    expect(classify(o, 'tx-1')).toBe('Unclassified');
  });

  it('later setClassification overrides earlier', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'Gift');
    o = setClassification(o, 'tx-1', 'tend', 'BarterLabor');
    expect(classify(o, 'tx-1')).toBe('BarterLabor');
  });

  it('updatedAt is set on every change', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'Gift');
    const first = o.entries['tx-1'].updatedAt;
    expect(first).toMatch(/^\d{4}-\d{2}-\d{2}T/);
  });
});

describe('classification category predicates', () => {
  it('INCOME set contains expected classifications', () => {
    expect(isIncome('Wage')).toBe(true);
    expect(isIncome('Sale')).toBe(true);
    expect(isIncome('Distribution')).toBe(true);
    expect(isIncome('BarterLabor')).toBe(true);
  });

  it('Purchase and CommonsContribution are expenses', () => {
    expect(isExpense('Purchase')).toBe(true);
    expect(isExpense('CommonsContribution')).toBe(true);
    expect(isExpense('Wage')).toBe(false);
  });

  it('Gift and InternalTransfer and Unclassified are neutral', () => {
    expect(isNeutral('Gift')).toBe(true);
    expect(isNeutral('InternalTransfer')).toBe(true);
    expect(isNeutral('Unclassified')).toBe(true);
    expect(isNeutral('Wage')).toBe(false);
  });

  it('income + expense + neutral partition the space (except BarterLabor straddles)', () => {
    for (const c of ALL_CLASSIFICATIONS) {
      const counted = Number(isIncome(c)) + Number(isExpense(c)) + Number(isNeutral(c));
      expect(counted).toBeGreaterThanOrEqual(1);
    }
  });
});

describe('classification queries', () => {
  it('unclassifiedFromList surfaces missing hashes', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'Gift');
    o = setClassification(o, 'tx-2', 'sap', 'Wage');
    const miss = unclassifiedFromList(o, ['tx-1', 'tx-2', 'tx-3', 'tx-4']);
    expect(miss).toEqual(['tx-3', 'tx-4']);
  });

  it('unclassifiedFromList treats Unclassified entries as missing', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'Unclassified');
    expect(unclassifiedFromList(o, ['tx-1'])).toEqual(['tx-1']);
  });

  it('countByClassification tallies correctly', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'Gift');
    o = setClassification(o, 'tx-2', 'tend', 'Gift');
    o = setClassification(o, 'tx-3', 'sap', 'Wage');
    const c = countByClassification(o);
    expect(c.Gift).toBe(2);
    expect(c.Wage).toBe(1);
    expect(c.Distribution).toBe(0);
  });
});

describe('export readiness', () => {
  it('ready when all hashes classified', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'a', 'tend', 'Gift');
    o = setClassification(o, 'b', 'sap', 'Wage');
    const r = checkReadiness(o, ['a', 'b']);
    expect(r.ready).toBe(true);
    expect(r.unclassifiedCount).toBe(0);
    expect(r.classified).toBe(2);
    expect(r.total).toBe(2);
  });

  it('not ready when some unclassified', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'a', 'tend', 'Gift');
    const r = checkReadiness(o, ['a', 'b', 'c']);
    expect(r.ready).toBe(false);
    expect(r.unclassifiedCount).toBe(2);
    expect(r.unclassifiedSample).toEqual(['b', 'c']);
  });

  it('unclassifiedSample capped at 10', () => {
    const o = emptyOverlay();
    const txs = Array.from({ length: 25 }, (_, i) => `tx-${i}`);
    const r = checkReadiness(o, txs);
    expect(r.unclassifiedCount).toBe(25);
    expect(r.unclassifiedSample).toHaveLength(10);
  });
});

describe('serialization round-trip', () => {
  it('preserves entries', () => {
    let o = emptyOverlay();
    o = setClassification(o, 'tx-1', 'tend', 'BarterLabor', 'cleaned Alice\'s garden');
    o = setClassification(o, 'tx-2', 'marketplace', 'Sale');
    const serialized = serializeOverlay(o);
    const restored = deserializeOverlay(serialized);
    expect(classify(restored, 'tx-1')).toBe('BarterLabor');
    expect(restored.entries['tx-1'].note).toBe('cleaned Alice\'s garden');
    expect(classify(restored, 'tx-2')).toBe('Sale');
  });

  it('rejects non-object JSON', () => {
    expect(() => deserializeOverlay('42')).toThrow(/not an object/);
    expect(() => deserializeOverlay('null')).toThrow(/not an object/);
  });

  it('rejects mismatched version', () => {
    expect(() => deserializeOverlay('{"version":2,"entries":{}}')).toThrow(/unsupported version/);
  });

  it('rejects missing entries', () => {
    expect(() => deserializeOverlay('{"version":1}')).toThrow(/entries missing/);
  });
});
