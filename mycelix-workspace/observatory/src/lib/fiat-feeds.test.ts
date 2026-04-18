// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
import { describe, it, expect, beforeEach } from 'vitest';
import {
  ALL_FEED_IDS,
  CustomUrlAdapter,
  FEED_CATALOG,
  FEED_CONFIG_STORAGE_KEY,
  FeedFetchError,
  NoFeedSelectedError,
  clearFeedConfig,
  loadFeedConfig,
  makeAdapter,
  requireAdapter,
  saveFeedConfig,
  validateFeedConfig,
  type FeedConfig,
} from './fiat-feeds';

// A minimal localStorage polyfill for Node test runs.
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

describe('catalog', () => {
  it('ships exactly the five planned feeds in stable order', () => {
    expect(ALL_FEED_IDS).toEqual([
      'coingecko',
      'kaiko',
      'chainlink',
      'coinmarketcap',
      'custom-url',
    ]);
  });

  it('catalog has an entry per feed id', () => {
    for (const id of ALL_FEED_IDS) {
      const entry = FEED_CATALOG.find((e) => e.id === id);
      expect(entry).toBeDefined();
      expect(entry!.displayName.length).toBeGreaterThan(0);
    }
  });

  it('kaiko and coinmarketcap require api keys; coingecko and chainlink do not', () => {
    expect(FEED_CATALOG.find((e) => e.id === 'kaiko')!.needsApiKey).toBe(true);
    expect(FEED_CATALOG.find((e) => e.id === 'coinmarketcap')!.needsApiKey).toBe(true);
    expect(FEED_CATALOG.find((e) => e.id === 'coingecko')!.needsApiKey).toBe(false);
    expect(FEED_CATALOG.find((e) => e.id === 'chainlink')!.needsApiKey).toBe(false);
  });
});

describe('validation', () => {
  const baseValid = (over: Partial<FeedConfig> = {}): FeedConfig => ({
    id: 'coingecko',
    selectedAt: '2026-04-18T00:00:00Z',
    ...over,
  });

  it('accepts a minimal coingecko config', () => {
    expect(validateFeedConfig(baseValid())).toBeNull();
  });

  it('requires api key for kaiko', () => {
    expect(validateFeedConfig(baseValid({ id: 'kaiko' }))).toMatch(/API key/);
    expect(
      validateFeedConfig(baseValid({ id: 'kaiko', apiKey: 'k1' })),
    ).toBeNull();
  });

  it('requires customUrl for custom-url', () => {
    expect(validateFeedConfig(baseValid({ id: 'custom-url' }))).toMatch(
      /customUrl/,
    );
  });

  it('rejects malformed custom-url', () => {
    expect(
      validateFeedConfig(baseValid({ id: 'custom-url', customUrl: 'not a url' })),
    ).toMatch(/valid URL/);
  });

  it('rejects non-http(s) custom-url', () => {
    expect(
      validateFeedConfig(
        baseValid({ id: 'custom-url', customUrl: 'ftp://foo.example' }),
      ),
    ).toMatch(/http\(s\)/);
  });

  it('accepts well-formed custom-url', () => {
    expect(
      validateFeedConfig(
        baseValid({ id: 'custom-url', customUrl: 'https://feed.example.com/rates' }),
      ),
    ).toBeNull();
  });

  it('rejects unknown feed id', () => {
    expect(
      validateFeedConfig({
        id: 'unknown' as unknown as FeedConfig['id'],
        selectedAt: '2026-04-18T00:00:00Z',
      }),
    ).toMatch(/unknown feed/);
  });
});

describe('storage round-trip', () => {
  it('save then load returns the same config', () => {
    const cfg: FeedConfig = {
      id: 'kaiko',
      apiKey: 'kkey',
      selectedAt: '2026-04-18T00:00:00Z',
      label: 'my kaiko',
    };
    saveFeedConfig(cfg);
    expect(loadFeedConfig()).toEqual(cfg);
  });

  it('loadFeedConfig returns null when nothing stored', () => {
    expect(loadFeedConfig()).toBeNull();
  });

  it('clearFeedConfig empties storage', () => {
    saveFeedConfig({
      id: 'coingecko',
      selectedAt: '2026-04-18T00:00:00Z',
    });
    clearFeedConfig();
    expect(loadFeedConfig()).toBeNull();
  });

  it('ignores stored config with unknown feed id', () => {
    (globalThis.localStorage as unknown as MemStorage).setItem(
      FEED_CONFIG_STORAGE_KEY,
      JSON.stringify({ id: 'bogus', selectedAt: 'x' }),
    );
    expect(loadFeedConfig()).toBeNull();
  });

  it('ignores stored config that fails to parse', () => {
    (globalThis.localStorage as unknown as MemStorage).setItem(
      FEED_CONFIG_STORAGE_KEY,
      'not valid json',
    );
    expect(loadFeedConfig()).toBeNull();
  });
});

describe('requireAdapter', () => {
  it('throws NoFeedSelectedError when no config stored', () => {
    expect(() => requireAdapter()).toThrow(NoFeedSelectedError);
  });

  it('returns an adapter once a config is stored', () => {
    saveFeedConfig({
      id: 'custom-url',
      customUrl: 'https://example.invalid/rates',
      selectedAt: '2026-04-18T00:00:00Z',
    });
    const a = requireAdapter();
    expect(a.id).toBe('custom-url');
  });
});

describe('custom-url adapter (live fetch with mock)', () => {
  it('fetches rate from the configured endpoint', async () => {
    const fakeFetch: typeof globalThis.fetch = async () =>
      new Response(JSON.stringify({ rate: 0.056 }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });

    const adapter = makeAdapter(
      {
        id: 'custom-url',
        customUrl: 'https://feed.example.com/rates',
        selectedAt: '2026-04-18T00:00:00Z',
      },
      fakeFetch,
    );
    const rate = await adapter.fetchRate('SAP', 'USD');
    expect(rate.rate).toBeCloseTo(0.056);
    expect(rate.base).toBe('SAP');
    expect(rate.quote).toBe('USD');
    expect(rate.source).toBe('custom-url');
  });

  it('raises FeedFetchError on bad response body', async () => {
    const fakeFetch: typeof globalThis.fetch = async () =>
      new Response(JSON.stringify({ not_rate: 1 }), { status: 200 });

    const adapter = new CustomUrlAdapter(
      { fetcher: fakeFetch },
      'https://feed.example.com/rates',
    );
    await expect(adapter.fetchRate('SAP', 'USD')).rejects.toBeInstanceOf(
      FeedFetchError,
    );
  });

  it('raises FeedFetchError on HTTP 500 with retriable=true', async () => {
    const fakeFetch: typeof globalThis.fetch = async () =>
      new Response('{}', { status: 500 });

    const adapter = new CustomUrlAdapter(
      { fetcher: fakeFetch },
      'https://feed.example.com/rates',
    );
    try {
      await adapter.fetchRate('SAP', 'USD');
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(FeedFetchError);
      expect((e as FeedFetchError).retriable).toBe(true);
    }
  });

  it('raises FeedFetchError on HTTP 400 with retriable=false', async () => {
    const fakeFetch: typeof globalThis.fetch = async () =>
      new Response('{}', { status: 400 });

    const adapter = new CustomUrlAdapter(
      { fetcher: fakeFetch },
      'https://feed.example.com/rates',
    );
    try {
      await adapter.fetchRate('SAP', 'USD');
      throw new Error('should have thrown');
    } catch (e) {
      expect(e).toBeInstanceOf(FeedFetchError);
      expect((e as FeedFetchError).retriable).toBe(false);
    }
  });
});

describe('stub adapters raise FeedFetchError as scaffold', () => {
  it('coingecko stub raises', async () => {
    const a = makeAdapter(
      { id: 'coingecko', selectedAt: 'x' },
      async () => new Response('{}'),
    );
    await expect(a.fetchRate('SAP', 'USD')).rejects.toBeInstanceOf(FeedFetchError);
  });

  it('chainlink stub raises', async () => {
    const a = makeAdapter(
      { id: 'chainlink', selectedAt: 'x' },
      async () => new Response('{}'),
    );
    await expect(a.fetchRate('SAP', 'USD')).rejects.toBeInstanceOf(FeedFetchError);
  });
});

describe('makeAdapter factory', () => {
  it('returns the right adapter class per id', () => {
    const fetcher: typeof globalThis.fetch = async () => new Response('{}');
    expect(
      makeAdapter({ id: 'coingecko', selectedAt: 'x' }, fetcher).id,
    ).toBe('coingecko');
    expect(
      makeAdapter(
        { id: 'kaiko', apiKey: 'k', selectedAt: 'x' },
        fetcher,
      ).id,
    ).toBe('kaiko');
    expect(
      makeAdapter({ id: 'chainlink', selectedAt: 'x' }, fetcher).id,
    ).toBe('chainlink');
    expect(
      makeAdapter(
        { id: 'coinmarketcap', apiKey: 'k', selectedAt: 'x' },
        fetcher,
      ).id,
    ).toBe('coinmarketcap');
    expect(
      makeAdapter(
        {
          id: 'custom-url',
          customUrl: 'https://x.example/r',
          selectedAt: 'x',
        },
        fetcher,
      ).id,
    ).toBe('custom-url');
  });

  it('throws if config invalid', () => {
    expect(() =>
      makeAdapter({ id: 'kaiko', selectedAt: 'x' }, async () => new Response('{}')),
    ).toThrow(FeedFetchError);
  });
});
