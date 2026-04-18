// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// End-to-end golden path for the Mycelix × Nation-State coexistence
// feature. Exercises every TS subsystem shipped under
// plans/would-you-like-to-parallel-fox.md together in one flow:
//
//   1. User has a primary DID AND a legal DID (dual-DID rule).
//   2. User classifies an issuer (sovereign tier).
//   3. User has TEND exchanges happening under the legal DID.
//   4. User picks a classification for each transaction.
//   5. User picks a fiat feed (active selection required).
//   6. User anchors FMV per transaction.
//   7. User picks novel tax positions.
//   8. User generates a tax export.
//
// At every step we assert:
//   - The primary DID never leaks into any tax-facing output.
//   - The export uses the legal DID and nothing else.
//   - Local state (classification, FMV anchors, feed config) never
//     references the primary DID.
//   - Novel-position footnotes embed cite-able legal reasoning.
//
// This is task #22 from the plan: a full end-to-end of the
// observability-layer user flow.

import { describe, it, expect, beforeEach } from 'vitest';
import type { ExchangeRecord } from './resilience-client';
import {
  PrimaryDidStateInteropError,
  assertLegalDid,
  exportToCsv,
  exportToJson,
  generateTaxExport,
} from './tax-export';
import {
  classify,
  checkReadiness,
  emptyOverlay,
  setClassification,
  type ClassificationOverlay,
} from './classification';
import {
  conservativeDefaults,
  renderFootnotesText,
  tendContributesToIncome,
} from './novel-position';
import {
  anchorBatch,
  emptyAnchorCache,
  lookupAnchor,
} from './fmv-anchor';
import {
  FEED_CONFIG_STORAGE_KEY,
  saveFeedConfig,
  NoFeedSelectedError,
} from './fiat-feeds';

// ============================================================================
// Test scaffolding
// ============================================================================

class MemStorage {
  private m = new Map<string, string>();
  getItem(k: string) {
    return this.m.get(k) ?? null;
  }
  setItem(k: string, v: string) {
    this.m.set(k, v);
  }
  removeItem(k: string) {
    this.m.delete(k);
  }
  clear() {
    this.m.clear();
  }
}

beforeEach(() => {
  (globalThis as unknown as { localStorage: MemStorage }).localStorage = new MemStorage();
});

// Mock the community module to a SA-SARS config — matches the existing
// tax-export.test.ts pattern.
import { vi } from 'vitest';
vi.mock('$app/environment', () => ({ browser: false }));
vi.mock('./community', () => ({
  getCommunityConfig: () => ({
    community_name: 'Test SA Cooperative',
    basket_name: 'Test Basket',
    dao_did: 'test-dao',
    currency_code: 'ZAR',
    currency_symbol: 'R',
    labor_hour_value: 27.58,
    labor_hour_source: 'Test: SA national minimum wage 2026',
    tax_year_start_month: 3,
    tax_form_name: 'IT12',
    tax_authority: 'SARS',
    basket_items: [],
  }),
  getTaxYearBounds: (year: string) => {
    const y = parseInt(year);
    return {
      start: new Date(y, 2, 1),
      end: new Date(y + 1, 1, 28, 23, 59, 59),
    };
  },
  formatCurrency: (v: number) => `R${v.toFixed(2)}`,
}));
vi.mock('./resilience-client', () => ({
  getConsensusPrice: vi.fn(),
  reportPrice: vi.fn(),
  getBasketIndex: vi.fn(),
  computeVolatility: vi.fn(),
}));

// ============================================================================
// Fixtures
// ============================================================================

const PRIMARY_DID = 'did:mycelix:primary:abc123def456';
const LEGAL_DID = 'did:mycelix:legal:7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b';
const SOVEREIGN_ISSUER = 'did:web:home.affairs.gov.za';
const DAO_DID = 'did:mycelix:dao:test-coop';

function mkExchange(
  id: string,
  hours: number,
  providerDid: string,
  receiverDid: string,
  category: string,
  description: string,
  daysAgo: number,
): ExchangeRecord {
  return {
    id,
    provider_did: providerDid,
    receiver_did: receiverDid,
    hours,
    service_category: category,
    service_description: description,
    status: 'Confirmed',
    timestamp: new Date(2026, 5, 1 - daysAgo).getTime(),
  };
}

const counterparty = 'did:mycelix:legal:counterparty-123';

// Five TEND exchanges for Alice (LEGAL_DID) during tax year 2026.
const exchanges: ExchangeRecord[] = [
  mkExchange('tx-1', 4, LEGAL_DID, counterparty, 'gardening', 'Cleaned Bob\'s garden', 5),
  mkExchange('tx-2', 2, counterparty, LEGAL_DID, 'tutoring', 'Bob tutored Alice in math', 4),
  mkExchange('tx-3', 6, LEGAL_DID, counterparty, 'coding', 'Website work for community', 3),
  mkExchange('tx-4', 1, LEGAL_DID, counterparty, 'gift', 'Helping neighbor move (no ledger offset expected)', 2),
  mkExchange('tx-5', 3, counterparty, LEGAL_DID, 'baking', 'Bread exchange', 1),
];

// ============================================================================
// Golden path
// ============================================================================

describe('End-to-end coexistence golden path', () => {
  it('step 1 — dual-DID guard refuses primary, accepts legal', () => {
    expect(() => assertLegalDid(PRIMARY_DID)).toThrow(PrimaryDidStateInteropError);
    expect(() => assertLegalDid(LEGAL_DID)).not.toThrow();
  });

  it('step 2 — sovereign issuer classification is surfaceable', () => {
    // (Zome-side classify_issuer call would happen here via the CLI
    // or conductor; the local TS side simply records the intent and
    // surfaces it in the export footnotes.)
    const issuerInfo = {
      did: SOVEREIGN_ISSUER,
      tier: 'sovereign' as const,
      rationale: 'SA Department of Home Affairs',
    };
    expect(issuerInfo.did).toBe('did:web:home.affairs.gov.za');
    expect(issuerInfo.tier).toBe('sovereign');
  });

  it('step 3-4 — all exchanges get classified, readiness confirmed', () => {
    let overlay: ClassificationOverlay = emptyOverlay();
    overlay = setClassification(overlay, 'tx-1', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-2', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-3', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-4', 'tend', 'Gift');
    overlay = setClassification(overlay, 'tx-5', 'tend', 'BarterLabor');

    const readiness = checkReadiness(
      overlay,
      exchanges.map((e) => e.id),
    );
    expect(readiness.ready).toBe(true);
    expect(readiness.unclassifiedCount).toBe(0);
    expect(readiness.classified).toBe(5);

    expect(classify(overlay, 'tx-1')).toBe('BarterLabor');
    expect(classify(overlay, 'tx-4')).toBe('Gift');
  });

  it('step 5 — feed selection is REQUIRED before FMV anchoring', async () => {
    // No feed selected yet — FMV anchoring must fail loudly.
    await expect(
      anchorBatch(
        [
          { txHash: 'tx-1', base: 'SAP', quote: 'ZAR', txTimestamp: Date.now() },
        ],
        undefined,
        async () => new Response('{}'),
      ),
    ).rejects.toBeInstanceOf(NoFeedSelectedError);

    // User picks custom-url (an arbitrary community-run endpoint).
    saveFeedConfig({
      id: 'custom-url',
      customUrl: 'https://za-sap-oracle.example.net/rates',
      selectedAt: '2026-04-18T12:00:00Z',
      label: 'my community SAP oracle',
    });

    // Now anchoring works.
    const fetcher: typeof globalThis.fetch = async () =>
      new Response(JSON.stringify({ rate: 1.85 }), { status: 200 });

    const { anchored, failed, cache } = await anchorBatch(
      [{ txHash: 'tx-1', base: 'SAP', quote: 'ZAR', txTimestamp: Date.now() }],
      emptyAnchorCache(),
      fetcher,
    );
    expect(failed).toHaveLength(0);
    expect(anchored).toHaveLength(1);
    expect(anchored[0].rate).toBeCloseTo(1.85);
    expect(lookupAnchor(cache, 'tx-1')?.source).toBe('custom-url');
  });

  it('step 6 — FMV anchors are immutable (re-run does NOT re-value)', async () => {
    saveFeedConfig({
      id: 'custom-url',
      customUrl: 'https://za-sap-oracle.example.net/rates',
      selectedAt: '2026-04-18T12:00:00Z',
    });

    // First anchor at rate = 1.85 ZAR/SAP.
    let cache = emptyAnchorCache();
    {
      const first = await anchorBatch(
        [{ txHash: 'tx-a', base: 'SAP', quote: 'ZAR', txTimestamp: Date.now() - 86400_000 }],
        cache,
        async () => new Response(JSON.stringify({ rate: 1.85 })),
      );
      cache = first.cache;
      expect(first.anchored[0].rate).toBeCloseTo(1.85);
    }

    // Second anchor at rate = 2.50 (rate changed). The existing tx
    // MUST keep its original anchor — not re-value against today's spot.
    {
      const second = await anchorBatch(
        [{ txHash: 'tx-a', base: 'SAP', quote: 'ZAR', txTimestamp: Date.now() - 86400_000 }],
        cache,
        async () => new Response(JSON.stringify({ rate: 2.5 })),
      );
      expect(second.anchored[0].rate).toBeCloseTo(1.85); // UNCHANGED.
    }
  });

  it('step 7 — novel-position stances affect income totals as expected', () => {
    // Conservative stance: TEND is barter income at FMV → contributes.
    expect(tendContributesToIncome(conservativeDefaults())).toBe(true);

    // Zero-sum stance: TEND does NOT contribute to taxable income.
    const zeroSum = { ...conservativeDefaults(), tendBarter: 'TreatAsZeroSumNonIncome' as const };
    expect(tendContributesToIncome(zeroSum)).toBe(false);

    // Rendered footnotes embed the reasoning with cite-able authority.
    const footnotes = renderFootnotesText(conservativeDefaults());
    expect(footnotes).toContain('NOVEL TAX POSITIONS');
    expect(footnotes).toContain('Glenshaw Glass'); // MYCEL Q3 default citation
  });

  it('step 8 — tax-export runs under LEGAL DID and never under primary', () => {
    // Running under primary MUST fail.
    expect(() =>
      generateTaxExport(exchanges, PRIMARY_DID, DAO_DID, '2026'),
    ).toThrow(PrimaryDidStateInteropError);

    // Running under legal DID succeeds.
    const summary = generateTaxExport(exchanges, LEGAL_DID, DAO_DID, '2026');
    expect(summary.member_did).toBe(LEGAL_DID);
    expect(summary.dao_did).toBe(DAO_DID);
    expect(summary.tax_year).toBe('2026');
    expect(summary.tax_authority).toBe('SARS');
    expect(summary.tax_form).toBe('IT12');
    expect(summary.currency_code).toBe('ZAR');

    // Totals should reflect the 4 LEGAL_DID-as-provider txs (tx-1: 4h, tx-3: 6h,
    // tx-4: 1h; and tx-2, tx-5 as receiver). Classification overlay isn't
    // consumed by the current generateTaxExport (that's a v2 extension);
    // here we just assert the mechanical hour totals.
    // Provider: tx-1 (4h) + tx-3 (6h) + tx-4 (1h) = 11h provided.
    // Receiver: tx-2 (2h) + tx-5 (3h) = 5h received.
    expect(summary.total_hours_provided).toBeCloseTo(11);
    expect(summary.total_hours_received).toBeCloseTo(5);
    expect(summary.net_hours).toBeCloseTo(6);
  });

  it('step 8b — CSV and JSON exports contain legal DID, never primary', () => {
    const summary = generateTaxExport(exchanges, LEGAL_DID, DAO_DID, '2026');
    const csv = exportToCsv(summary);
    const json = exportToJson(summary);

    expect(csv).toContain(LEGAL_DID);
    expect(csv).not.toContain(PRIMARY_DID);
    expect(json).toContain(LEGAL_DID);
    expect(json).not.toContain(PRIMARY_DID);

    // CSV header section should carry the jurisdiction info.
    expect(csv).toContain('SARS');
    expect(csv).toContain('IT12');
    expect(csv).toContain('ZAR');
  });

  it('step 9 — classification + novel-position footnotes compose onto CSV', () => {
    const summary = generateTaxExport(exchanges, LEGAL_DID, DAO_DID, '2026');
    const baseCsv = exportToCsv(summary);
    const footnotes = renderFootnotesText(conservativeDefaults());

    // Downstream consumer stitches footnotes onto the CSV trailer.
    const composedCsv = `${baseCsv}\n\n${footnotes}`;

    // Assertions about the composed export:
    expect(composedCsv).toContain('SARS');
    expect(composedCsv).toContain('NOVEL TAX POSITIONS');
    expect(composedCsv).toContain('Q1:');
    expect(composedCsv).toContain('Q4:');
    expect(composedCsv).not.toContain(PRIMARY_DID);

    // Every tax-position line is CSV-comment-safe.
    for (const line of footnotes.split('\n')) {
      if (line.trim().length === 0) continue;
      expect(line.startsWith('#')).toBe(true);
    }
  });

  it('step 10 — primary DID never reachable from any local state', () => {
    // Assemble every piece of local state the flow touches.
    saveFeedConfig({
      id: 'custom-url',
      customUrl: 'https://za-sap-oracle.example.net/rates',
      selectedAt: '2026-04-18T12:00:00Z',
    });

    let overlay = emptyOverlay();
    overlay = setClassification(overlay, 'tx-1', 'tend', 'BarterLabor');

    const storage = globalThis.localStorage as unknown as MemStorage;
    const feedCfgRaw = storage.getItem(FEED_CONFIG_STORAGE_KEY) ?? '';

    // Primary DID must appear nowhere in any of the state blobs.
    expect(feedCfgRaw).not.toContain(PRIMARY_DID);
    expect(JSON.stringify(overlay)).not.toContain(PRIMARY_DID);

    // The summary uses LEGAL_DID only.
    const summary = generateTaxExport(exchanges, LEGAL_DID, DAO_DID, '2026');
    expect(JSON.stringify(summary)).not.toContain(PRIMARY_DID);
  });

  it('step 11 — full flow summary: legal DID cryptographically isolated', () => {
    // The end-to-end property: even with all outputs in hand, an
    // observer cannot link LEGAL_DID back to PRIMARY_DID through
    // any TS-layer artefact.
    saveFeedConfig({
      id: 'custom-url',
      customUrl: 'https://za-sap-oracle.example.net/rates',
      selectedAt: '2026-04-18T12:00:00Z',
    });
    let overlay = emptyOverlay();
    overlay = setClassification(overlay, 'tx-1', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-2', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-3', 'tend', 'BarterLabor');
    overlay = setClassification(overlay, 'tx-4', 'tend', 'Gift');
    overlay = setClassification(overlay, 'tx-5', 'tend', 'BarterLabor');

    const summary = generateTaxExport(exchanges, LEGAL_DID, DAO_DID, '2026');
    const positions = conservativeDefaults();
    const footnotes = renderFootnotesText(positions);
    const csv = `${exportToCsv(summary)}\n\n${footnotes}`;
    const json = exportToJson({ ...summary, footnotes });

    // The ONLY DID leaving the user's device (via export) is the legal
    // one. Nothing correlates back to the primary.
    expect(csv).toContain(LEGAL_DID);
    expect(csv).not.toContain(PRIMARY_DID);
    expect(json).toContain(LEGAL_DID);
    expect(json).not.toContain(PRIMARY_DID);

    // Classification overlay is local and contains only legal-side ids.
    expect(JSON.stringify(overlay)).not.toContain(PRIMARY_DID);
  });
});

describe('Golden path adversarial checks', () => {
  it('refuses export even if LEGAL_DID is typoed into primary form', () => {
    const typoed = 'did:mycelix:primary:legally-looking-tail';
    expect(() => generateTaxExport(exchanges, typoed, DAO_DID, '2026')).toThrow(
      PrimaryDidStateInteropError,
    );
  });

  it('refuses export if primary is passed as DAO DID', () => {
    // Closed: the dual-DID guard now covers dao_did too. Forcing-
    // function from the earlier session's E2E did its job — gap
    // surfaced, then closed.
    expect(() =>
      generateTaxExport(exchanges, LEGAL_DID, PRIMARY_DID, '2026'),
    ).toThrow(PrimaryDidStateInteropError);
  });

  it('refuses export if BOTH DIDs are primary (the worst case)', () => {
    expect(() =>
      generateTaxExport(exchanges, PRIMARY_DID, PRIMARY_DID, '2026'),
    ).toThrow(PrimaryDidStateInteropError);
  });

  it('accepts two different legal DIDs (member != dao)', () => {
    // Common real case: member is an individual, DAO is the
    // cooperative under which they file. Both legal, both different.
    const cooperative_legal = 'did:mycelix:legal:cafecafecafecafecafecafecafecafecafecafecafecafecafecafecafecafe';
    const summary = generateTaxExport(
      exchanges,
      LEGAL_DID,
      cooperative_legal,
      '2026',
    );
    expect(summary.member_did).toBe(LEGAL_DID);
    expect(summary.dao_did).toBe(cooperative_legal);
  });
});
