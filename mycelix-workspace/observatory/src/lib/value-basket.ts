/**
 * Value Basket Store — dual-layer price discovery for TEND purchasing power.
 *
 * Two data layers:
 * 1. **Consensus (DHT)**: Community-reported prices aggregated via trimmed median
 *    in the price-oracle zome. This is the "shared truth" of what things cost.
 * 2. **Personal (localStorage)**: Individual price estimates as fallback when
 *    conductor is unavailable or consensus doesn't exist yet for an item.
 *
 * The UI shows consensus prices when available, falling back to personal estimates.
 */

import { writable, derived, type Readable } from 'svelte/store';
import { browser } from '$app/environment';
import {
  getConsensusPrice,
  reportPrice,
  getBasketIndex,
  computeVolatility,
  type ConsensusResult,
  type BasketIndexResult,
  type VolatilityResult,
} from './resilience-client';
import { isConnected } from './conductor';
import {
  getCanonicalItems,
  getBasketWeights,
  getBasketName,
  type BasketItemConfig,
} from './community';

// ============================================================================
// Types
// ============================================================================

export interface BasketItem {
  name: string;
  unit: string;
  price_tend: number;
  updated_at: number;
}

export interface BasketSnapshot {
  timestamp: number;
  index: number;
}

/** A basket item enriched with consensus data when available. */
export interface EnrichedBasketItem extends BasketItem {
  key: string;
  /** Consensus price from DHT (null = no consensus yet) */
  consensus_price: number | null;
  /** Number of reporters behind consensus */
  reporter_count: number;
  /** Standard deviation of reports (price agreement indicator) */
  std_dev: number;
  /** Whether this price comes from DHT consensus or personal estimate */
  source: 'consensus' | 'personal';
  /** Average accuracy of reporters behind this consensus (0.0-1.0) */
  signal_integrity: number;
}

const STORAGE_KEY = 'mycelix-value-basket';
const HISTORY_KEY = 'mycelix-value-basket-history';
const MAX_HISTORY = 90;

// ============================================================================
// Canonical Items — loaded from community-config.json
// ============================================================================

/** Canonical basket items from community config. */
export const CANONICAL_ITEMS = getCanonicalItems();

/** Basket weights from community config (sum = 1.0). */
export const RESILIENCE_BASKET_WEIGHTS = getBasketWeights();

/** Basket name from community config. */
export const RESILIENCE_BASKET_NAME = getBasketName();

// ============================================================================
// localStorage Persistence (personal estimates fallback)
// ============================================================================

function loadBasket(): Map<string, BasketItem> {
  if (!browser) return new Map();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return new Map();
    const obj = JSON.parse(raw) as Record<string, BasketItem>;
    return new Map(Object.entries(obj));
  } catch {
    return new Map();
  }
}

function saveBasket(basket: Map<string, BasketItem>): void {
  if (!browser) return;
  const obj = Object.fromEntries(basket);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(obj));
}

function loadHistory(): BasketSnapshot[] {
  if (!browser) return [];
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as BasketSnapshot[];
  } catch {
    return [];
  }
}

function saveHistory(history: BasketSnapshot[]): void {
  if (!browser) return;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-MAX_HISTORY)));
}

// ============================================================================
// Stores
// ============================================================================

const _basket = writable<Map<string, BasketItem>>(loadBasket());
const _history = writable<BasketSnapshot[]>(loadHistory());
const _consensus = writable<Map<string, ConsensusResult>>(new Map());
const _basketIndex = writable<BasketIndexResult | null>(null);
const _volatility = writable<VolatilityResult | null>(null);
const _oracleLoading = writable<boolean>(false);
const _lastRefresh = writable<number>(0);

// Auto-persist personal basket on change
_basket.subscribe((b) => saveBasket(b));
_history.subscribe((h) => saveHistory(h));

export interface KeyedBasketItem extends BasketItem {
  key: string;
}

/** Reactive personal basket items as array with keys. */
export const basketItems: Readable<KeyedBasketItem[]> = derived(_basket, ($b) =>
  Array.from($b.entries())
    .map(([key, item]) => ({ ...item, key }))
    .sort((a, b) => a.name.localeCompare(b.name))
);

/** Enriched items: consensus prices overlaid on personal estimates. */
export const enrichedItems: Readable<EnrichedBasketItem[]> = derived(
  [_basket, _consensus],
  ([$basket, $consensus]) => {
    // Start from canonical items, overlay personal estimates, then consensus
    const result: EnrichedBasketItem[] = [];

    // Build a set of all known item keys
    const allKeys = new Set<string>();
    CANONICAL_ITEMS.forEach((c) => allKeys.add(c.key));
    for (const key of $basket.keys()) allKeys.add(key);
    for (const key of $consensus.keys()) allKeys.add(key);

    for (const key of allKeys) {
      const canonical = CANONICAL_ITEMS.find((c) => c.key === key);
      const personal = $basket.get(key);
      const consensus = $consensus.get(key);

      const name = personal?.name ?? canonical?.name ?? key;
      const unit = personal?.unit ?? canonical?.unit ?? 'unit';
      const personalPrice = personal?.price_tend ?? canonical?.default_price ?? 0;

      result.push({
        key,
        name,
        unit,
        price_tend: consensus?.median_price ?? personalPrice,
        updated_at: personal?.updated_at ?? Date.now(),
        consensus_price: consensus?.median_price ?? null,
        reporter_count: consensus?.reporter_count ?? 0,
        std_dev: consensus?.std_dev ?? 0,
        source: consensus ? 'consensus' : 'personal',
        signal_integrity: consensus?.signal_integrity ?? 1.0,
      });
    }

    return result.sort((a, b) => a.name.localeCompare(b.name));
  }
);

/** Community purchasing power index from DHT basket, falling back to local calc. */
export const purchasingPowerIndex: Readable<number> = derived(
  [_basketIndex, _basket],
  ([$idx, $basket]) => {
    if ($idx && $idx.index > 0) return $idx.index;
    // Fallback to personal basket average
    if ($basket.size === 0) return 0;
    let total = 0;
    for (const item of $basket.values()) total += item.price_tend;
    return total / $basket.size;
  }
);

/** Historical snapshots for sparkline. */
export const basketHistory: Readable<BasketSnapshot[]> = derived(_history, ($h) => $h);

/** Whether any basket data is available (personal or consensus). */
export const hasBasket: Readable<boolean> = derived(
  [_basket, _consensus],
  ([$b, $c]) => $b.size > 0 || $c.size > 0
);

/** Current volatility data from DHT. */
export const volatilityData: Readable<VolatilityResult | null> = derived(_volatility, ($v) => $v);

/** Whether oracle data is being fetched. */
export const oracleLoading: Readable<boolean> = derived(_oracleLoading, ($l) => $l);

/** DHT basket index result. */
export const basketIndexData: Readable<BasketIndexResult | null> = derived(_basketIndex, ($b) => $b);

// ============================================================================
// DHT Actions — Community Oracle
// ============================================================================

/**
 * Fetch consensus prices from DHT for all canonical items.
 * Silently falls back to personal estimates on failure.
 */
export async function refreshConsensus(): Promise<void> {
  _oracleLoading.set(true);
  const results = new Map<string, ConsensusResult>();

  // Fetch consensus for each canonical item in parallel
  const promises = CANONICAL_ITEMS.map(async (item) => {
    try {
      const consensus = await getConsensusPrice(item.key);
      results.set(item.key, consensus);
    } catch {
      // No consensus yet for this item — that's expected early on
    }
  });

  await Promise.all(promises);
  _consensus.set(results);

  // Try to fetch basket index
  try {
    const idx = await getBasketIndex(RESILIENCE_BASKET_NAME);
    _basketIndex.set(idx);
    // Record snapshot from DHT index
    if (idx.index > 0) {
      _history.update((h) => [...h, { timestamp: Date.now(), index: idx.index }]);
    }
  } catch {
    // Basket not defined yet on DHT — use local calculation
  }

  // Try to fetch volatility
  try {
    const vol = await computeVolatility(RESILIENCE_BASKET_NAME);
    _volatility.set(vol);
  } catch {
    // No volatility data yet
  }

  _oracleLoading.set(false);
  _lastRefresh.set(Date.now());
}

/**
 * Submit a price report to the DHT oracle.
 * Also updates the personal basket as immediate feedback.
 */
export async function submitPriceReport(
  key: string,
  priceTend: number,
  evidence: string,
): Promise<boolean> {
  // Update personal basket immediately for responsive UI
  const canonical = CANONICAL_ITEMS.find((c) => c.key === key);
  if (canonical) {
    addOrUpdateItem(key, {
      name: canonical.name,
      unit: canonical.unit,
      price_tend: priceTend,
      updated_at: Date.now(),
    });
  }

  // Submit to DHT
  try {
    await reportPrice({ item: key, price_tend: priceTend, evidence });
    // Refresh consensus for this item after reporting
    try {
      const consensus = await getConsensusPrice(key);
      _consensus.update((c) => {
        const next = new Map(c);
        next.set(key, consensus);
        return next;
      });
    } catch {
      // Consensus may not exist yet (need 2+ reporters)
    }
    return true;
  } catch (e) {
    console.warn('[ValueBasket] Price report failed (demo mode?):', e);
    return false;
  }
}

// ============================================================================
// Personal Basket Actions (localStorage)
// ============================================================================

export function addOrUpdateItem(key: string, item: BasketItem): void {
  _basket.update((b) => {
    const next = new Map(b);
    next.set(key, { ...item, updated_at: Date.now() });
    return next;
  });
  recordSnapshot();
}

export function removeItem(key: string): void {
  _basket.update((b) => {
    const next = new Map(b);
    next.delete(key);
    return next;
  });
  recordSnapshot();
}

export function clearBasket(): void {
  _basket.set(new Map());
  _history.set([]);
}

/** Record current index as a historical snapshot. */
function recordSnapshot(): void {
  let currentIndex = 0;
  _basket.subscribe(($b) => {
    if ($b.size === 0) return;
    let total = 0;
    for (const item of $b.values()) total += item.price_tend;
    currentIndex = total / $b.size;
  })();

  if (currentIndex > 0) {
    _history.update((h) => [...h, { timestamp: Date.now(), index: currentIndex }]);
  }
}

/** Export basket as JSON string for sharing. */
export function exportBasket(): string {
  let result = '';
  _basket.subscribe(($b) => {
    result = JSON.stringify(Object.fromEntries($b), null, 2);
  })();
  return result;
}

/** Import basket from JSON string. */
export function importBasket(json: string): boolean {
  try {
    const obj = JSON.parse(json) as Record<string, BasketItem>;
    const items = new Map(Object.entries(obj));
    for (const [, item] of items) {
      if (typeof item.name !== 'string' || typeof item.price_tend !== 'number') {
        return false;
      }
    }
    _basket.set(items);
    recordSnapshot();
    return true;
  } catch {
    return false;
  }
}

/**
 * Compute what a TEND balance can buy using best available prices.
 * Prefers consensus prices, falls back to personal estimates.
 */
export function purchasingPower(tendBalance: number): { name: string; quantity: string; unit: string }[] {
  const result: { name: string; quantity: string; unit: string }[] = [];
  let enriched: EnrichedBasketItem[] = [];
  enrichedItems.subscribe(($e) => { enriched = $e; })();

  for (const item of enriched) {
    const price = item.consensus_price ?? item.price_tend;
    if (price > 0) {
      const qty = tendBalance / price;
      result.push({
        name: item.name,
        quantity: qty >= 10 ? Math.round(qty).toString() : qty.toFixed(1),
        unit: item.unit,
      });
    }
  }
  return result.sort((a, b) => parseFloat(b.quantity) - parseFloat(a.quantity));
}
