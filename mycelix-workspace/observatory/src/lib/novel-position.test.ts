// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
import { describe, it, expect } from 'vitest';
import {
  conservativeDefaults,
  footnotes,
  QUESTION_LABELS,
  renderFootnotesText,
  tendContributesToIncome,
  type NovelPositionSet,
} from './novel-position';

describe('novel-position stances', () => {
  it('conservative defaults are a complete position set', () => {
    const d = conservativeDefaults();
    expect(d.tendBarter).toBe('TreatAsBarterFMV');
    expect(d.demurrage).toBe('NotTaxable');
    expect(d.mycel).toBe('NonPropertyNotIncome');
    expect(d.compost).toBe('DisclaimedViaNonControl');
  });

  it('conservative TEND stance contributes to income', () => {
    expect(tendContributesToIncome(conservativeDefaults())).toBe(true);
  });

  it('zero-sum TEND stance does NOT contribute to income', () => {
    const set: NovelPositionSet = {
      ...conservativeDefaults(),
      tendBarter: 'TreatAsZeroSumNonIncome',
    };
    expect(tendContributesToIncome(set)).toBe(false);
  });

  it('gift TEND stance does NOT contribute to income', () => {
    const set: NovelPositionSet = {
      ...conservativeDefaults(),
      tendBarter: 'TreatAsGift',
    };
    expect(tendContributesToIncome(set)).toBe(false);
  });
});

describe('footnote generation', () => {
  it('produces four footnotes in fixed order', () => {
    const f = footnotes(conservativeDefaults());
    expect(f).toHaveLength(4);
    expect(f[0].question).toBe('tendBarter');
    expect(f[1].question).toBe('demurrage');
    expect(f[2].question).toBe('mycel');
    expect(f[3].question).toBe('compost');
  });

  it('each footnote carries a non-empty reasoning string', () => {
    const f = footnotes(conservativeDefaults());
    for (const footnote of f) {
      expect(footnote.reasoning.length).toBeGreaterThan(40);
      expect(footnote.chosenStance.length).toBeGreaterThan(0);
      expect(footnote.questionLabel.length).toBeGreaterThan(0);
    }
  });

  it('reasoning changes with stance', () => {
    const a = footnotes({
      ...conservativeDefaults(),
      tendBarter: 'TreatAsBarterFMV',
    });
    const b = footnotes({
      ...conservativeDefaults(),
      tendBarter: 'TreatAsZeroSumNonIncome',
    });
    expect(a[0].reasoning).not.toBe(b[0].reasoning);
  });

  it('cites expected legal authorities for each question', () => {
    const f = footnotes(conservativeDefaults());
    // Q1 TEND barter default (FMV) should cite IRS Topic 420 or SARS.
    expect(f[0].reasoning).toMatch(/(IRS Topic 420|SARS)/);
    // Q2 demurrage NotTaxable should cite IRC §1001 realization.
    expect(f[1].reasoning).toMatch(/§?1001|realization/);
    // Q3 MYCEL NonPropertyNotIncome should cite Glenshaw Glass.
    expect(f[2].reasoning).toMatch(/Glenshaw Glass/);
    // Q4 Compost DisclaimedViaNonControl should cite constructive-receipt.
    expect(f[3].reasoning).toMatch(/constructive[- ]receipt|Helvering/);
  });
});

describe('rendered text output', () => {
  it('renders a multi-line block', () => {
    const text = renderFootnotesText(conservativeDefaults());
    expect(text).toContain('NOVEL TAX POSITIONS');
    expect(text).toContain('Q1:');
    expect(text).toContain('Q2:');
    expect(text).toContain('Q3:');
    expect(text).toContain('Q4:');
  });

  it('lines are CSV-comment-safe (every non-empty line starts with #)', () => {
    const text = renderFootnotesText(conservativeDefaults());
    for (const line of text.split('\n')) {
      if (line.trim().length === 0) continue;
      expect(line.startsWith('#')).toBe(true);
    }
  });

  it('wraps long reasoning to readable width', () => {
    const text = renderFootnotesText(conservativeDefaults());
    for (const line of text.split('\n')) {
      // Allow some slack for the header decoration, but lines shouldn't run
      // wildly long — wrapping target is ~80 cols.
      expect(line.length).toBeLessThan(120);
    }
  });
});

describe('question labels', () => {
  it('exports human-readable labels for every question', () => {
    expect(QUESTION_LABELS.tendBarter).toContain('TEND');
    expect(QUESTION_LABELS.demurrage).toContain('demurrage');
    expect(QUESTION_LABELS.mycel).toContain('MYCEL');
    expect(QUESTION_LABELS.compost).toContain('commons');
  });
});
