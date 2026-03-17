import { describe, it, expect, vi, beforeEach } from 'vitest';

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

// Mock community config with a controlled test configuration
vi.mock('./community', () => ({
  getCommunityConfig: () => ({
    community_name: 'Test Cooperative',
    basket_name: 'Test Basket',
    dao_did: 'test-dao',
    currency_code: 'ZAR',
    currency_symbol: 'R',
    labor_hour_value: 27.58,
    labor_hour_source: 'Test source',
    tax_year_start_month: 3,
    tax_form_name: 'IT12',
    tax_authority: 'SARS',
    basket_items: [],
  }),
  getTaxYearBounds: (year: string) => {
    const y = parseInt(year);
    return {
      start: new Date(y, 2, 1),       // Mar 1
      end: new Date(y + 1, 1, 28, 23, 59, 59), // Feb 28 next year
    };
  },
  formatCurrency: (v: number) => `R${v.toFixed(2)}`,
}));

import type { ExchangeRecord } from './resilience-client';
import { generateTaxExport, exportToCsv, exportToJson } from './tax-export';
import type { TaxExportSummary } from './tax-export';

// ============================================================================
// Helpers
// ============================================================================

function makeExchange(overrides: Partial<ExchangeRecord> = {}): ExchangeRecord {
  return {
    id: 'ex-001',
    provider_did: 'did:test:provider',
    receiver_did: 'did:test:receiver',
    hours: 2,
    service_description: 'Plumbing repair',
    service_category: 'Maintenance',
    status: 'Confirmed',
    timestamp: new Date(2026, 5, 15).getTime(), // Jun 15, 2026 — inside Mar 2026-Feb 2027
    ...overrides,
  };
}

// ============================================================================
// generateTaxExport
// ============================================================================

describe('generateTaxExport', () => {
  it('produces a valid summary with required fields', () => {
    const exchanges = [makeExchange()];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');

    expect(summary.member_did).toBe('did:test:provider');
    expect(summary.dao_did).toBe('test-dao');
    expect(summary.tax_year).toBe('2026');
    expect(summary.currency_code).toBe('ZAR');
    expect(summary.currency_symbol).toBe('R');
    expect(summary.hourly_rate).toBe(27.58);
    expect(summary.tax_authority).toBe('SARS');
    expect(summary.tax_form).toBe('IT12');
    expect(summary.period_start).toBe('2026-03-01');
    expect(summary.period_end).toBe('2027-02-28');
  });

  it('returns empty rows for empty exchange list', () => {
    const summary = generateTaxExport([], 'did:test:me', 'test-dao', '2026');
    expect(summary.rows).toEqual([]);
    expect(summary.exchange_count).toBe(0);
    expect(summary.total_hours_provided).toBe(0);
    expect(summary.total_hours_received).toBe(0);
    expect(summary.net_hours).toBe(0);
    expect(summary.estimated_income).toBe(0);
    expect(summary.estimated_expense).toBe(0);
  });

  it('filters by Confirmed status only', () => {
    const exchanges = [
      makeExchange({ id: 'ex-1', status: 'Confirmed' }),
      makeExchange({ id: 'ex-2', status: 'Proposed' }),
      makeExchange({ id: 'ex-3', status: 'Disputed' }),
      makeExchange({ id: 'ex-4', status: 'Cancelled' }),
    ];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    expect(summary.exchange_count).toBe(1);
    expect(summary.rows).toHaveLength(1);
  });

  it('filters by tax year date range', () => {
    const insideRange = makeExchange({
      id: 'ex-in',
      timestamp: new Date(2026, 5, 15).getTime(), // Jun 2026
    });
    const beforeRange = makeExchange({
      id: 'ex-before',
      timestamp: new Date(2026, 0, 15).getTime(), // Jan 2026 — before Mar 1
    });
    const afterRange = makeExchange({
      id: 'ex-after',
      timestamp: new Date(2027, 5, 15).getTime(), // Jun 2027 — after Feb 28
    });

    const summary = generateTaxExport(
      [insideRange, beforeRange, afterRange],
      'did:test:provider',
      'test-dao',
      '2026',
    );
    expect(summary.exchange_count).toBe(1);
  });

  it('classifies provided vs received correctly', () => {
    const memberDid = 'did:test:member';
    const exchanges = [
      makeExchange({ provider_did: memberDid, receiver_did: 'did:test:other', hours: 3 }),
      makeExchange({ provider_did: 'did:test:other', receiver_did: memberDid, hours: 1 }),
    ];
    const summary = generateTaxExport(exchanges, memberDid, 'test-dao', '2026');

    expect(summary.total_hours_provided).toBe(3);
    expect(summary.total_hours_received).toBe(1);
    expect(summary.net_hours).toBe(2);

    const provided = summary.rows.find((r) => r.direction === 'provided');
    const received = summary.rows.find((r) => r.direction === 'received');
    expect(provided).toBeDefined();
    expect(received).toBeDefined();
  });

  it('computes estimated_local_value using hourly rate', () => {
    const exchanges = [makeExchange({ hours: 4 })];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');

    // 4 hours * 27.58 = 110.32
    expect(summary.estimated_income).toBe(110.32);
    expect(summary.rows[0].estimated_local_value).toBe(110.32);
  });

  it('sorts rows by date ascending', () => {
    const exchanges = [
      makeExchange({ id: 'ex-late', timestamp: new Date(2026, 8, 1).getTime() }),
      makeExchange({ id: 'ex-early', timestamp: new Date(2026, 3, 1).getTime() }),
      makeExchange({ id: 'ex-mid', timestamp: new Date(2026, 5, 1).getTime() }),
    ];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const dates = summary.rows.map((r) => r.date);
    expect(dates).toEqual([...dates].sort());
  });

  it('truncates counterparty DIDs for privacy', () => {
    const longDid = 'did:holo:abcdefghijklmnopqrstuvwxyz1234567890';
    const exchanges = [makeExchange({ receiver_did: longDid })];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');

    const cp = summary.rows[0].counterparty;
    expect(cp.length).toBeLessThan(longDid.length);
    expect(cp).toContain('...');
  });
});

// ============================================================================
// exportToCsv
// ============================================================================

describe('exportToCsv', () => {
  function makeSummary(rows: TaxExportSummary['rows'] = []): TaxExportSummary {
    return generateTaxExport(
      rows.length === 0 ? [] : [makeExchange()],
      'did:test:provider',
      'test-dao',
      '2026',
    );
  }

  it('produces valid CSV with header row', () => {
    const summary = makeSummary();
    const csv = exportToCsv(summary);
    const lines = csv.split('\n');

    // Find the CSV header line (first non-comment line)
    const headerLine = lines.find((l) => !l.startsWith('#'));
    expect(headerLine).toBeDefined();
    expect(headerLine).toContain('Date');
    expect(headerLine).toContain('Direction');
    expect(headerLine).toContain('Category');
    expect(headerLine).toContain('Description');
    expect(headerLine).toContain('Hours');
    expect(headerLine).toContain('Counterparty');
    expect(headerLine).toContain('Status');
  });

  it('includes metadata comment header', () => {
    const summary = makeSummary();
    const csv = exportToCsv(summary);
    expect(csv).toContain('# MYCELIX COOPERATIVE');
    expect(csv).toContain('# Tax Year: 2026');
    expect(csv).toContain('# Tax Authority: SARS');
  });

  it('produces valid empty output for zero exchanges', () => {
    const summary = generateTaxExport([], 'did:test:me', 'test-dao', '2026');
    const csv = exportToCsv(summary);

    // Should still have headers
    expect(csv).toContain('Date,Direction');
    // Should have no data rows after the CSV header
    const lines = csv.split('\n');
    const headerIdx = lines.findIndex((l) => l.startsWith('Date,'));
    const dataLines = lines.slice(headerIdx + 1).filter((l) => l.trim().length > 0);
    expect(dataLines).toHaveLength(0);
  });

  it('escapes commas in description fields', () => {
    const exchanges = [makeExchange({ service_description: 'Fix plumbing, electrical' })];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const csv = exportToCsv(summary);

    // The field with comma should be quoted
    expect(csv).toContain('"Fix plumbing, electrical"');
  });

  it('escapes double quotes in description fields', () => {
    const exchanges = [makeExchange({ service_description: 'Install "premium" fixtures' })];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const csv = exportToCsv(summary);

    // Double quotes should be doubled per CSV spec
    expect(csv).toContain('"Install ""premium"" fixtures"');
  });

  it('escapes newlines in description fields', () => {
    const exchanges = [makeExchange({ service_description: 'Line1\nLine2' })];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const csv = exportToCsv(summary);

    expect(csv).toContain('"Line1\nLine2"');
  });
});

// ============================================================================
// exportToJson
// ============================================================================

describe('exportToJson', () => {
  it('produces valid parseable JSON', () => {
    const exchanges = [makeExchange()];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const json = exportToJson(summary);

    const parsed = JSON.parse(json);
    expect(parsed).toBeDefined();
    expect(typeof parsed).toBe('object');
  });

  it('includes all required summary fields', () => {
    const exchanges = [makeExchange()];
    const summary = generateTaxExport(exchanges, 'did:test:provider', 'test-dao', '2026');
    const parsed = JSON.parse(exportToJson(summary));

    expect(parsed.member_did).toBe('did:test:provider');
    expect(parsed.dao_did).toBe('test-dao');
    expect(parsed.tax_year).toBe('2026');
    expect(parsed.currency_code).toBe('ZAR');
    expect(parsed.hourly_rate).toBe(27.58);
    expect(parsed.rows).toBeInstanceOf(Array);
    expect(parsed.rows.length).toBe(1);
    expect(parsed.exchange_count).toBe(1);
    expect(parsed.total_hours_provided).toBeGreaterThan(0);
    expect(parsed.estimated_income).toBeGreaterThan(0);
  });

  it('produces valid empty JSON for zero exchanges', () => {
    const summary = generateTaxExport([], 'did:test:me', 'test-dao', '2026');
    const parsed = JSON.parse(exportToJson(summary));

    expect(parsed.rows).toEqual([]);
    expect(parsed.exchange_count).toBe(0);
  });
});
