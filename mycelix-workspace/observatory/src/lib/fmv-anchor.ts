// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Per-transaction fair-market-value anchor for SAP + marketplace
 * transactions. Computes and caches FMV at transaction time using the
 * user's chosen fiat feed (`fiat-feeds.ts`). Never writes to the DHT —
 * anchors live in browser localStorage so each user controls their
 * own record.
 *
 * Per plan decision 2026-04-18: no hardcoded default feed; the user
 * MUST have picked one via onboarding before calling `anchorFmv`.
 */

import {
  type FeedRate,
  FeedFetchError,
  NoFeedSelectedError,
  requireAdapter,
} from './fiat-feeds';

// ============================================================================
// Types
// ============================================================================

/** A single cached FMV snapshot for one transaction. */
export interface FmvAnchor {
  /** Opaque tx identifier matching `ClassificationEntry.txHash`. */
  txHash: string;
  /** Base currency of the underlying transaction (e.g., `"SAP"`). */
  base: string;
  /** Quote currency the user filed in (e.g., `"USD"`, `"ZAR"`). */
  quote: string;
  /** The rate: how many `quote` per 1 `base`. */
  rate: number;
  /** JS ms-since-epoch of the transaction (NOT of the anchor write). */
  txTimestamp: number;
  /** JS ms-since-epoch when this anchor was first computed/cached. */
  anchoredAt: number;
  /** Which feed produced the rate — audit trail. */
  source: string;
  /** Whether the source returned a historical rate (vs spot fallback). */
  isHistorical: boolean;
}

/** Cache of anchors keyed by `txHash`. */
export interface FmvAnchorCache {
  version: 1;
  anchors: Record<string, FmvAnchor>;
}

// ============================================================================
// Storage
// ============================================================================

export const FMV_ANCHOR_STORAGE_KEY = 'mycelix:tax-export:fmv-anchors';

export function emptyAnchorCache(): FmvAnchorCache {
  return { version: 1, anchors: {} };
}

export function loadAnchorCache(): FmvAnchorCache {
  if (typeof localStorage === 'undefined') return emptyAnchorCache();
  const raw = localStorage.getItem(FMV_ANCHOR_STORAGE_KEY);
  if (raw == null) return emptyAnchorCache();
  try {
    const parsed = JSON.parse(raw) as FmvAnchorCache;
    if (parsed.version !== 1 || typeof parsed.anchors !== 'object') {
      return emptyAnchorCache();
    }
    return parsed;
  } catch {
    return emptyAnchorCache();
  }
}

export function saveAnchorCache(cache: FmvAnchorCache): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.setItem(FMV_ANCHOR_STORAGE_KEY, JSON.stringify(cache));
}

export function clearAnchorCache(): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem(FMV_ANCHOR_STORAGE_KEY);
}

// ============================================================================
// Anchoring
// ============================================================================

/** Input for anchoring a single transaction. */
export interface AnchorRequest {
  txHash: string;
  base: string;
  quote: string;
  /** JS ms-since-epoch of the transaction itself (not of the anchor). */
  txTimestamp: number;
}

/**
 * Look up or compute an FMV anchor for a transaction. If an anchor
 * already exists in the cache for `txHash`, it is returned as-is
 * (anchors are immutable once set — this prevents re-valuing old
 * transactions against today's spot if the user re-runs export).
 *
 * Pre-condition: the user has selected a fiat feed via
 * `fiat-feeds::saveFeedConfig`. `NoFeedSelectedError` is thrown
 * otherwise.
 */
export async function anchorFmv(
  request: AnchorRequest,
  cache: FmvAnchorCache = loadAnchorCache(),
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): Promise<{ anchor: FmvAnchor; cache: FmvAnchorCache }> {
  const existing = cache.anchors[request.txHash];
  if (existing) {
    return { anchor: existing, cache };
  }

  const adapter = requireAdapter(fetcher);
  let rate: FeedRate;
  try {
    rate = await adapter.fetchRate(request.base, request.quote, request.txTimestamp);
  } catch (err) {
    if (err instanceof FeedFetchError) throw err;
    throw new FeedFetchError(
      adapter.id,
      `unexpected error: ${err instanceof Error ? err.message : String(err)}`,
      true,
    );
  }

  const isHistorical = Math.abs(rate.asOf - request.txTimestamp) < 24 * 3600 * 1000;
  const anchor: FmvAnchor = {
    txHash: request.txHash,
    base: request.base,
    quote: request.quote,
    rate: rate.rate,
    txTimestamp: request.txTimestamp,
    anchoredAt: Date.now(),
    source: rate.source,
    isHistorical,
  };

  const nextCache: FmvAnchorCache = {
    version: 1,
    anchors: { ...cache.anchors, [request.txHash]: anchor },
  };
  saveAnchorCache(nextCache);
  return { anchor, cache: nextCache };
}

/**
 * Apply an anchor: convert `amount` of `base` into `quote`. Throws if
 * the anchor's base/quote don't match.
 */
export function applyAnchor(anchor: FmvAnchor, amount: number): number {
  return amount * anchor.rate;
}

/** Lookup-only (no fetch) helper. Returns `null` if no anchor cached. */
export function lookupAnchor(
  cache: FmvAnchorCache,
  txHash: string,
): FmvAnchor | null {
  return cache.anchors[txHash] ?? null;
}

// ============================================================================
// Bulk anchor operations (for batch export)
// ============================================================================

/**
 * Anchor a batch of transactions. Returns a new cache, the successful
 * anchors, and the list of failed txHashes (with their errors) so the
 * UI can show the user what went wrong.
 */
export async function anchorBatch(
  requests: readonly AnchorRequest[],
  cache: FmvAnchorCache = loadAnchorCache(),
  fetcher: typeof globalThis.fetch = globalThis.fetch,
): Promise<{
  cache: FmvAnchorCache;
  anchored: FmvAnchor[];
  failed: Array<{ txHash: string; error: string }>;
}> {
  let currentCache = cache;
  const anchored: FmvAnchor[] = [];
  const failed: Array<{ txHash: string; error: string }> = [];

  // Pre-flight: if no feed is selected, surface that once rather than
  // failing every request individually.
  try {
    requireAdapter(fetcher);
  } catch (err) {
    if (err instanceof NoFeedSelectedError) throw err;
  }

  for (const req of requests) {
    try {
      const { anchor, cache: updated } = await anchorFmv(req, currentCache, fetcher);
      currentCache = updated;
      anchored.push(anchor);
    } catch (err) {
      failed.push({
        txHash: req.txHash,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
  return { cache: currentCache, anchored, failed };
}
