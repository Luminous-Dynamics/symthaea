// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Fiat-feed adapters for SAP → local-currency fair-market-value anchoring.
 *
 * IMPORTANT ARCHITECTURAL NOTE: by design, this module ships NO default
 * feed selection. Every adapter is optional. Onboarding MUST present
 * the user a choice and require active selection. No feed choice ever
 * reaches the Holochain DHT — the user's selection lives in browser
 * localStorage only, per MYCELIX_STATE_COEXISTENCE.md. Different users
 * can (and should) pick different feeds.
 *
 * The adapter interface is intentionally narrow: given a symbol pair
 * and an optional timestamp, return `{ rate, asOf, source }`. Feeds
 * that cannot honour a historical timestamp (most free tiers) return
 * the spot rate and set `asOf = Date.now()`; callers SHOULD treat
 * spot-rate FMV with more skepticism when tax-exporting old transactions.
 */

// ============================================================================
// Shared types
// ============================================================================

/** Canonical feed identifier used for user selection + telemetry. */
export type FeedId =
  | 'coingecko'
  | 'kaiko'
  | 'chainlink'
  | 'coinmarketcap'
  | 'custom-url';

/** Curated list of adapters. Keep stable across releases. */
export const ALL_FEED_IDS: readonly FeedId[] = [
  'coingecko',
  'kaiko',
  'chainlink',
  'coinmarketcap',
  'custom-url',
] as const;

/** Short, user-facing feed summary for the selector UI. */
export interface FeedCatalogEntry {
  id: FeedId;
  displayName: string;
  summary: string;
  /** Whether the feed requires an API key the user must supply. */
  needsApiKey: boolean;
  /** Whether the feed supports historical timestamps. */
  supportsHistorical: boolean;
  /** Public terms/TOS URL for due-diligence review. */
  termsUrl?: string;
}

/**
 * Catalog description of every adapter. UI presents these in the
 * onboarding selector; user MUST pick one (or decline and work with
 * hourly-rate-only exports).
 */
export const FEED_CATALOG: readonly FeedCatalogEntry[] = [
  {
    id: 'coingecko',
    displayName: 'CoinGecko',
    summary: 'Free spot rates, no API key needed for basic use.',
    needsApiKey: false,
    supportsHistorical: true,
    termsUrl: 'https://www.coingecko.com/en/terms',
  },
  {
    id: 'kaiko',
    displayName: 'Kaiko',
    summary: 'Institutional-grade, paid. Best for audit-defensibility.',
    needsApiKey: true,
    supportsHistorical: true,
    termsUrl: 'https://www.kaiko.com/pages/terms-of-service',
  },
  {
    id: 'chainlink',
    displayName: 'Chainlink',
    summary: 'On-chain oracle aggregates; free to query via RPC.',
    needsApiKey: false,
    supportsHistorical: false,
    termsUrl: 'https://chain.link/terms',
  },
  {
    id: 'coinmarketcap',
    displayName: 'CoinMarketCap',
    summary: 'Free tier with API key; paid for history.',
    needsApiKey: true,
    supportsHistorical: true,
    termsUrl: 'https://coinmarketcap.com/terms/',
  },
  {
    id: 'custom-url',
    displayName: 'Custom URL',
    summary:
      'Point at any JSON endpoint returning `{rate: number}`. For ' +
      'self-hosted or community-run feeds. Caveat emptor.',
    needsApiKey: false,
    supportsHistorical: false,
  },
] as const;

/** User's stored feed configuration. */
export interface FeedConfig {
  id: FeedId;
  /** API key if the chosen feed needs one. */
  apiKey?: string;
  /** Base URL for `custom-url`. Not used by other feeds. */
  customUrl?: string;
  /** ISO 8601 timestamp of the user's selection. */
  selectedAt: string;
  /** Free-form label for the user's records. */
  label?: string;
}

/** A rate returned by an adapter. */
export interface FeedRate {
  /** How many units of `quote` per 1 unit of `base`. */
  rate: number;
  /** Base symbol, e.g., `"SAP"`. */
  base: string;
  /** Quote symbol, e.g., `"USD"` or `"ZAR"`. */
  quote: string;
  /** JS time (ms since epoch) the rate is valid at. */
  asOf: number;
  /** Which feed produced it, for audit trail. */
  source: FeedId;
  /** Whether this rate was fetched live or served from cache. */
  fromCache: boolean;
}

/** Error raised by adapters when a rate cannot be produced. */
export class FeedFetchError extends Error {
  constructor(
    public readonly feed: FeedId,
    public readonly reason: string,
    public readonly retriable: boolean,
  ) {
    super(`[${feed}] ${reason}`);
    this.name = 'FeedFetchError';
  }
}

/** Adapter interface. All feeds implement this. */
export interface FiatFeedAdapter {
  readonly id: FeedId;
  /**
   * Fetch the rate of `base` in terms of `quote`. `atTimestamp` is in
   * JS ms since epoch; if omitted, return spot. If the feed cannot
   * honour a historical timestamp, it returns spot and sets
   * `result.asOf = now`; the caller must decide whether that is
   * acceptable for its use case (usually: old tx = no, recent = yes).
   */
  fetchRate(
    base: string,
    quote: string,
    atTimestamp?: number,
  ): Promise<FeedRate>;
}

// ============================================================================
// Selection + onboarding
// ============================================================================

/** localStorage key for the user's feed configuration. */
export const FEED_CONFIG_STORAGE_KEY = 'mycelix:tax-export:fiat-feed';

/** Raised when tax-export is invoked but the user hasn't picked a feed yet. */
export class NoFeedSelectedError extends Error {
  constructor() {
    super(
      'No fiat feed has been selected yet. Open onboarding to choose ' +
        'one of CoinGecko / Kaiko / Chainlink / CoinMarketCap / custom URL.',
    );
    this.name = 'NoFeedSelectedError';
  }
}

/**
 * Persist the user's feed configuration. Writes to localStorage when
 * available, a pass-through no-op under server-side tests.
 */
export function saveFeedConfig(config: FeedConfig): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(FEED_CONFIG_STORAGE_KEY, JSON.stringify(config));
}

/**
 * Load the user's feed configuration. Returns `null` if none set.
 */
export function loadFeedConfig(): FeedConfig | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(FEED_CONFIG_STORAGE_KEY);
  if (raw == null) return null;
  try {
    const parsed = JSON.parse(raw) as FeedConfig;
    if (!ALL_FEED_IDS.includes(parsed.id)) return null;
    return parsed;
  } catch {
    return null;
  }
}

/** Clear the user's feed configuration (for "unselect" or testing). */
export function clearFeedConfig(): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem(FEED_CONFIG_STORAGE_KEY);
}

/**
 * Validate a proposed feed configuration. Returns an error string if
 * invalid, or `null` if acceptable. UI surfaces the error inline.
 */
export function validateFeedConfig(config: FeedConfig): string | null {
  const catalogEntry = FEED_CATALOG.find((e) => e.id === config.id);
  if (!catalogEntry) return `unknown feed: ${config.id}`;
  if (catalogEntry.needsApiKey && !config.apiKey) {
    return `${catalogEntry.displayName} requires an API key`;
  }
  if (config.id === 'custom-url') {
    if (!config.customUrl) return 'custom-url requires customUrl';
    try {
      // URL() throws on invalid syntax.
      const u = new URL(config.customUrl);
      if (u.protocol !== 'https:' && u.protocol !== 'http:') {
        return 'customUrl must be http(s)';
      }
    } catch {
      return 'customUrl is not a valid URL';
    }
  }
  return null;
}

// ============================================================================
// Adapter implementations (stubs with docstrings for real fetch logic)
// ============================================================================

/**
 * Shared helper: return a spot-only rate using a user-injected fetch
 * function. All real-world adapters parameterize a `fetcher` so tests
 * can mock without pulling in network I/O.
 */
interface RestFetchOptions {
  fetcher: typeof globalThis.fetch;
  headers?: Record<string, string>;
}

/** CoinGecko — free, no API key for basic endpoints. */
export class CoinGeckoAdapter implements FiatFeedAdapter {
  readonly id: FeedId = 'coingecko';
  constructor(private readonly opts: RestFetchOptions) {}

  async fetchRate(base: string, quote: string, _atTimestamp?: number): Promise<FeedRate> {
    // Live impl would hit:
    //   /simple/price?ids=<base-coingecko-id>&vs_currencies=<quote>
    // and parse JSON response. This scaffold returns a placeholder.
    throw new FeedFetchError(
      this.id,
      `scaffold — wire up https://api.coingecko.com/api/v3/simple/price for ${base}/${quote}`,
      false,
    );
  }
}

/** Kaiko — institutional, paid, audit-grade. */
export class KaikoAdapter implements FiatFeedAdapter {
  readonly id: FeedId = 'kaiko';
  constructor(
    private readonly opts: RestFetchOptions,
    private readonly apiKey: string,
  ) {
    if (!apiKey) {
      throw new FeedFetchError(this.id, 'api key required', false);
    }
  }

  async fetchRate(base: string, quote: string, _atTimestamp?: number): Promise<FeedRate> {
    throw new FeedFetchError(
      this.id,
      `scaffold — wire up Kaiko v2 REST API (/data/trades.v1/...) for ${base}/${quote}`,
      false,
    );
  }
}

/** Chainlink — on-chain oracle aggregates via JSON-RPC. */
export class ChainlinkAdapter implements FiatFeedAdapter {
  readonly id: FeedId = 'chainlink';
  constructor(
    private readonly opts: RestFetchOptions,
    private readonly rpcUrl: string = 'https://ethereum-rpc.publicnode.com',
  ) {}

  async fetchRate(base: string, quote: string, _atTimestamp?: number): Promise<FeedRate> {
    throw new FeedFetchError(
      this.id,
      `scaffold — eth_call a Chainlink AggregatorV3 at the published ${base}/${quote} address via ${this.rpcUrl}`,
      false,
    );
  }
}

/** CoinMarketCap — free tier with API key. */
export class CoinMarketCapAdapter implements FiatFeedAdapter {
  readonly id: FeedId = 'coinmarketcap';
  constructor(
    private readonly opts: RestFetchOptions,
    private readonly apiKey: string,
  ) {
    if (!apiKey) {
      throw new FeedFetchError(this.id, 'api key required', false);
    }
  }

  async fetchRate(base: string, quote: string, _atTimestamp?: number): Promise<FeedRate> {
    throw new FeedFetchError(
      this.id,
      `scaffold — wire up https://pro-api.coinmarketcap.com/v2/... for ${base}/${quote}`,
      false,
    );
  }
}

/** Custom URL — arbitrary JSON endpoint returning `{rate: number}`. */
export class CustomUrlAdapter implements FiatFeedAdapter {
  readonly id: FeedId = 'custom-url';
  constructor(
    private readonly opts: RestFetchOptions,
    private readonly customUrl: string,
  ) {}

  async fetchRate(base: string, quote: string, _atTimestamp?: number): Promise<FeedRate> {
    const url = new URL(this.customUrl);
    url.searchParams.set('base', base);
    url.searchParams.set('quote', quote);
    const res = await this.opts.fetcher(url.toString(), {
      headers: this.opts.headers ?? {},
    });
    if (!res.ok) {
      throw new FeedFetchError(
        this.id,
        `HTTP ${res.status} from ${url.toString()}`,
        res.status >= 500,
      );
    }
    const body = (await res.json()) as { rate?: number };
    if (typeof body.rate !== 'number' || !Number.isFinite(body.rate) || body.rate <= 0) {
      throw new FeedFetchError(
        this.id,
        `custom-url response missing positive numeric "rate": ${JSON.stringify(body)}`,
        false,
      );
    }
    return {
      rate: body.rate,
      base,
      quote,
      asOf: Date.now(),
      source: 'custom-url',
      fromCache: false,
    };
  }
}

// ============================================================================
// Factory
// ============================================================================

/**
 * Build the configured adapter, throwing a descriptive error if the
 * config is missing or invalid.
 */
export function makeAdapter(
  config: FeedConfig,
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): FiatFeedAdapter {
  const invalid = validateFeedConfig(config);
  if (invalid) {
    throw new FeedFetchError(config.id, invalid, false);
  }
  const opts: RestFetchOptions = { fetcher };
  switch (config.id) {
    case 'coingecko':
      return new CoinGeckoAdapter(opts);
    case 'kaiko':
      return new KaikoAdapter(opts, config.apiKey!);
    case 'chainlink':
      return new ChainlinkAdapter(opts);
    case 'coinmarketcap':
      return new CoinMarketCapAdapter(opts, config.apiKey!);
    case 'custom-url':
      return new CustomUrlAdapter(opts, config.customUrl!);
  }
}

/**
 * Convenience: load the stored config and build an adapter, throwing
 * `NoFeedSelectedError` if nothing has been chosen yet. This is the
 * call-site tax-export should use before any SAP transaction is
 * anchored to a fiat value.
 */
export function requireAdapter(
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): FiatFeedAdapter {
  const cfg = loadFeedConfig();
  if (cfg == null) throw new NoFeedSelectedError();
  return makeAdapter(cfg, fetcher);
}
