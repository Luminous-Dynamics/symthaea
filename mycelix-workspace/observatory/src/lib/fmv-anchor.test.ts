// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
import { describe, it, expect, beforeEach } from 'vitest';
import {
  FMV_ANCHOR_STORAGE_KEY,
  anchorBatch,
  anchorFmv,
  applyAnchor,
  clearAnchorCache,
  emptyAnchorCache,
  loadAnchorCache,
  lookupAnchor,
  saveAnchorCache,
  type FmvAnchor,
  type FmvAnchorCache,
} from './fmv-anchor';
import {
  FEED_CONFIG_STORAGE_KEY,
  NoFeedSelectedError,
  saveFeedConfig,
} from './fiat-feeds';

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

function stubCustomUrlFeed(rate: number): typeof globalThis.fetch {
  return async () =>
    new Response(JSON.stringify({ rate }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
}

function pickCustomUrl() {
  saveFeedConfig({
    id: 'custom-url',
    customUrl: 'https://feed.example.com/r',
    selectedAt: '2026-04-18T00:00:00Z',
  });
}

describe('cache storage', () => {
  it('empty by default', () => {
    const c = loadAnchorCache();
    expect(c.version).toBe(1);
    expect(Object.keys(c.anchors)).toHaveLength(0);
  });

  it('save then load round-trips', () => {
    const cache: FmvAnchorCache = {
      version: 1,
      anchors: {
        'tx-1': {
          txHash: 'tx-1',
          base: 'SAP',
          quote: 'USD',
          rate: 0.1,
          txTimestamp: 1_700_000_000_000,
          anchoredAt: 1_710_000_000_000,
          source: 'custom-url',
          isHistorical: false,
        },
      },
    };
    saveAnchorCache(cache);
    expect(loadAnchorCache()).toEqual(cache);
  });

  it('clearAnchorCache empties storage', () => {
    saveAnchorCache({
      version: 1,
      anchors: {
        x: {
          txHash: 'x',
          base: 'SAP',
          quote: 'USD',
          rate: 1,
          txTimestamp: 0,
          anchoredAt: 0,
          source: 's',
          isHistorical: false,
        },
      },
    });
    clearAnchorCache();
    expect(loadAnchorCache().anchors).toEqual({});
  });

  it('ignores version mismatch in stored cache', () => {
    (globalThis.localStorage as unknown as MemStorage).setItem(
      FMV_ANCHOR_STORAGE_KEY,
      JSON.stringify({ version: 2, anchors: {} }),
    );
    expect(loadAnchorCache()).toEqual(emptyAnchorCache());
  });
});

describe('anchorFmv', () => {
  it('fetches and caches a new anchor when none exists', async () => {
    pickCustomUrl();
    const { anchor, cache } = await anchorFmv(
      {
        txHash: 'tx-1',
        base: 'SAP',
        quote: 'USD',
        txTimestamp: Date.now(),
      },
      undefined,
      stubCustomUrlFeed(0.12),
    );

    expect(anchor.rate).toBeCloseTo(0.12);
    expect(anchor.base).toBe('SAP');
    expect(anchor.quote).toBe('USD');
    expect(cache.anchors['tx-1'].rate).toBeCloseTo(0.12);
    // Persisted to storage.
    expect(loadAnchorCache().anchors['tx-1'].rate).toBeCloseTo(0.12);
  });

  it('returns existing anchor without re-fetching', async () => {
    pickCustomUrl();
    const existing: FmvAnchor = {
      txHash: 'tx-1',
      base: 'SAP',
      quote: 'USD',
      rate: 0.05,
      txTimestamp: 1_700_000_000_000,
      anchoredAt: 1_700_000_001_000,
      source: 'custom-url',
      isHistorical: true,
    };
    const cache: FmvAnchorCache = { version: 1, anchors: { 'tx-1': existing } };
    // A fetcher that would blow up if called.
    const fetcher: typeof globalThis.fetch = async () => {
      throw new Error('fetcher should not be invoked');
    };
    const result = await anchorFmv(
      { txHash: 'tx-1', base: 'SAP', quote: 'USD', txTimestamp: 1_700_000_000_000 },
      cache,
      fetcher,
    );
    expect(result.anchor).toEqual(existing);
  });

  it('throws NoFeedSelectedError when no feed configured', async () => {
    await expect(
      anchorFmv(
        { txHash: 'tx-1', base: 'SAP', quote: 'USD', txTimestamp: Date.now() },
        undefined,
        stubCustomUrlFeed(0.1),
      ),
    ).rejects.toBeInstanceOf(NoFeedSelectedError);
  });

  it('flags anchor as non-historical when asOf is far from txTimestamp', async () => {
    pickCustomUrl();
    const { anchor } = await anchorFmv(
      {
        txHash: 'old-tx',
        base: 'SAP',
        quote: 'USD',
        // 30 days ago — custom-url returns spot = now, so isHistorical=false.
        txTimestamp: Date.now() - 30 * 24 * 3600 * 1000,
      },
      undefined,
      stubCustomUrlFeed(0.1),
    );
    expect(anchor.isHistorical).toBe(false);
  });
});

describe('applyAnchor', () => {
  it('multiplies amount by rate', () => {
    const anchor: FmvAnchor = {
      txHash: 'tx-1',
      base: 'SAP',
      quote: 'ZAR',
      rate: 1.85,
      txTimestamp: 0,
      anchoredAt: 0,
      source: 's',
      isHistorical: false,
    };
    expect(applyAnchor(anchor, 100)).toBeCloseTo(185);
  });
});

describe('lookupAnchor', () => {
  it('returns cached anchor', () => {
    const a: FmvAnchor = {
      txHash: 'tx-1',
      base: 'SAP',
      quote: 'USD',
      rate: 0.1,
      txTimestamp: 0,
      anchoredAt: 0,
      source: 's',
      isHistorical: false,
    };
    const cache: FmvAnchorCache = { version: 1, anchors: { 'tx-1': a } };
    expect(lookupAnchor(cache, 'tx-1')).toEqual(a);
  });

  it('returns null when absent', () => {
    expect(lookupAnchor(emptyAnchorCache(), 'tx-missing')).toBeNull();
  });
});

describe('anchorBatch', () => {
  it('anchors successes and records failures', async () => {
    pickCustomUrl();
    // First call succeeds, second fails.
    let callCount = 0;
    const fetcher: typeof globalThis.fetch = async () => {
      callCount++;
      if (callCount === 1) {
        return new Response(JSON.stringify({ rate: 0.1 }), { status: 200 });
      }
      return new Response('{}', { status: 500 });
    };

    const { anchored, failed, cache } = await anchorBatch(
      [
        { txHash: 'ok-1', base: 'SAP', quote: 'USD', txTimestamp: Date.now() },
        { txHash: 'fail-1', base: 'SAP', quote: 'USD', txTimestamp: Date.now() },
      ],
      undefined,
      fetcher,
    );

    expect(anchored).toHaveLength(1);
    expect(anchored[0].txHash).toBe('ok-1');
    expect(failed).toHaveLength(1);
    expect(failed[0].txHash).toBe('fail-1');
    expect(cache.anchors['ok-1']).toBeDefined();
    expect(cache.anchors['fail-1']).toBeUndefined();
  });

  it('throws NoFeedSelectedError pre-flight when no feed selected', async () => {
    await expect(
      anchorBatch(
        [
          {
            txHash: 'tx-1',
            base: 'SAP',
            quote: 'USD',
            txTimestamp: Date.now(),
          },
        ],
        undefined,
        stubCustomUrlFeed(0.1),
      ),
    ).rejects.toBeInstanceOf(NoFeedSelectedError);
  });
});
