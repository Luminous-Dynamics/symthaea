// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Transaction classification overlay for tax export.
 *
 * User-held. Never consensus. Never on the DHT. The overlay maps a
 * transaction hash (TEND exchange id, SAP payment hash, marketplace
 * transaction id, attribution receipt id) to one of the eight
 * tax-relevant classifications below. The user edits it freely; the
 * tax-export generator consumes it to decide whether an item is
 * income, expense, gift, or an internal transfer that does not count.
 *
 * This is the distinction between a *ledger* (what happened, written
 * immutably on chain) and a *classification* (what it means for the
 * user's tax situation, a subjective interpretation that may evolve).
 *
 * Part of the Mycelix × Nation-State coexistence extensions. See
 * MYCELIX_STATE_COEXISTENCE.md at the repo root.
 */

// ============================================================================
// Types
// ============================================================================

/**
 * The eight tax-relevant classifications. Callers SHOULD pick one per
 * transaction; unclassified transactions are treated as `Unclassified`
 * by the generator and surfaced to the user for review before export.
 */
export type Classification =
  | 'Gift'
  /** TEND hours provided/received against goods or services. */
  | 'BarterLabor'
  /** SAP received for work performed. */
  | 'Wage'
  /** SAP received as DAO distribution (pro-rata commons share). */
  | 'Distribution'
  /** Voluntary contribution to a commons pool (may be deductible). */
  | 'CommonsContribution'
  /** User moving value between their own addresses — not a taxable event. */
  | 'InternalTransfer'
  /** Marketplace sale (user as seller). */
  | 'Sale'
  /** Marketplace purchase (user as buyer). */
  | 'Purchase'
  /** User has not decided yet. Blocks export until resolved. */
  | 'Unclassified';

/** The full set of classifications in declaration order. */
export const ALL_CLASSIFICATIONS: readonly Classification[] = [
  'Gift',
  'BarterLabor',
  'Wage',
  'Distribution',
  'CommonsContribution',
  'InternalTransfer',
  'Sale',
  'Purchase',
  'Unclassified',
] as const;

/** Human-readable description for UI surfaces. */
export const CLASSIFICATION_LABELS: Record<Classification, string> = {
  Gift: 'Gift (not income in most jurisdictions)',
  BarterLabor: 'Barter / labor exchange (see novel-position for TEND)',
  Wage: 'Wage / salary income',
  Distribution: 'DAO distribution (pro-rata commons share)',
  CommonsContribution: 'Contribution to commons (may be deductible)',
  InternalTransfer: 'Internal transfer (not a taxable event)',
  Sale: 'Marketplace sale (revenue)',
  Purchase: 'Marketplace purchase (expense)',
  Unclassified: 'Unclassified — decide before export',
};

/** Which classifications count as income for the purposes of export totals. */
export const INCOME_CLASSIFICATIONS: readonly Classification[] = [
  'BarterLabor', // only if user's novel-position treats TEND as barter income
  'Wage',
  'Distribution',
  'Sale',
] as const;

/** Which classifications count as expenses. */
export const EXPENSE_CLASSIFICATIONS: readonly Classification[] = [
  'Purchase',
  'CommonsContribution',
] as const;

/**
 * A single transaction's classification. The overlay is an array of
 * these entries, indexed by `txHash`. Classification is plain data;
 * no signing or integrity beyond "it's on the user's device."
 */
export interface ClassificationEntry {
  /** Opaque transaction identifier. Format is per-source:
   *  - TEND:      ExchangeRecord.id
   *  - SAP:       payment action hash
   *  - Market:    transaction id
   *  - Attrib:    usage receipt id
   */
  txHash: string;
  /** Which source produced this transaction. */
  source: 'tend' | 'sap' | 'marketplace' | 'attribution';
  /** User's chosen classification. */
  classification: Classification;
  /** ISO 8601 timestamp of last user edit. */
  updatedAt: string;
  /** Optional user-entered note for audit trail. */
  note?: string;
}

/**
 * The overlay is a map-like structure for O(1) lookup by txHash.
 * Exposed as a plain object to simplify JSON serialization to the
 * user's local storage.
 */
export interface ClassificationOverlay {
  /** Schema version for forward-compat. */
  version: 1;
  /** Map: txHash → entry. */
  entries: Record<string, ClassificationEntry>;
}

// ============================================================================
// Overlay mutation
// ============================================================================

/** Create an empty overlay. */
export function emptyOverlay(): ClassificationOverlay {
  return { version: 1, entries: {} };
}

/** Look up a classification, defaulting to Unclassified if absent. */
export function classify(overlay: ClassificationOverlay, txHash: string): Classification {
  return overlay.entries[txHash]?.classification ?? 'Unclassified';
}

/**
 * Set a classification for a transaction. Returns a new overlay
 * (overlays are treated as immutable so UI reactivity works).
 */
export function setClassification(
  overlay: ClassificationOverlay,
  txHash: string,
  source: ClassificationEntry['source'],
  classification: Classification,
  note?: string,
): ClassificationOverlay {
  return {
    version: 1,
    entries: {
      ...overlay.entries,
      [txHash]: {
        txHash,
        source,
        classification,
        updatedAt: new Date().toISOString(),
        note,
      },
    },
  };
}

/** Remove a classification, reverting the transaction to Unclassified. */
export function unclassify(
  overlay: ClassificationOverlay,
  txHash: string,
): ClassificationOverlay {
  const { [txHash]: _removed, ...rest } = overlay.entries;
  return { version: 1, entries: rest };
}

// ============================================================================
// Queries
// ============================================================================

/** All txHashes currently marked as `Unclassified` (or missing from overlay). */
export function unclassifiedFromList(
  overlay: ClassificationOverlay,
  txHashes: readonly string[],
): string[] {
  return txHashes.filter((h) => classify(overlay, h) === 'Unclassified');
}

/** Count transactions per classification category. */
export function countByClassification(
  overlay: ClassificationOverlay,
): Record<Classification, number> {
  const counts: Record<Classification, number> = {
    Gift: 0,
    BarterLabor: 0,
    Wage: 0,
    Distribution: 0,
    CommonsContribution: 0,
    InternalTransfer: 0,
    Sale: 0,
    Purchase: 0,
    Unclassified: 0,
  };
  for (const entry of Object.values(overlay.entries)) {
    counts[entry.classification]++;
  }
  return counts;
}

/**
 * Whether `classification` is a classification the tax-export
 * generator should treat as income (subject to the user's novel
 * position on TEND if `BarterLabor`).
 */
export function isIncome(classification: Classification): boolean {
  return INCOME_CLASSIFICATIONS.includes(classification);
}

/**
 * Whether `classification` is an expense.
 */
export function isExpense(classification: Classification): boolean {
  return EXPENSE_CLASSIFICATIONS.includes(classification);
}

/**
 * Whether `classification` is neutral for tax purposes (not in totals).
 */
export function isNeutral(classification: Classification): boolean {
  return (
    classification === 'Gift' ||
    classification === 'InternalTransfer' ||
    classification === 'Unclassified'
  );
}

// ============================================================================
// Export-readiness check
// ============================================================================

/**
 * Result of checking whether the overlay is ready for tax export.
 * `ready` is true only if every transaction in `txHashes` has a
 * non-`Unclassified` classification.
 */
export interface ReadinessReport {
  ready: boolean;
  unclassifiedCount: number;
  unclassifiedSample: string[]; // up to 10 for UI display
  classified: number;
  total: number;
}

/**
 * Check whether the overlay covers every transaction in `txHashes`.
 * Returns a report the caller can surface in the UI before export.
 */
export function checkReadiness(
  overlay: ClassificationOverlay,
  txHashes: readonly string[],
): ReadinessReport {
  const unclassified = unclassifiedFromList(overlay, txHashes);
  return {
    ready: unclassified.length === 0,
    unclassifiedCount: unclassified.length,
    unclassifiedSample: unclassified.slice(0, 10),
    classified: txHashes.length - unclassified.length,
    total: txHashes.length,
  };
}

// ============================================================================
// Serialization
// ============================================================================

/** Serialize overlay to JSON string for storage (localStorage, IPC, file). */
export function serializeOverlay(overlay: ClassificationOverlay): string {
  return JSON.stringify(overlay);
}

/**
 * Deserialize overlay from JSON. Throws on schema mismatch so that
 * stale saves from an older release surface as errors rather than
 * silently reverting to empty.
 */
export function deserializeOverlay(json: string): ClassificationOverlay {
  const parsed = JSON.parse(json);
  if (parsed == null || typeof parsed !== 'object') {
    throw new Error('classification overlay: not an object');
  }
  if (parsed.version !== 1) {
    throw new Error(`classification overlay: unsupported version ${parsed.version}`);
  }
  if (typeof parsed.entries !== 'object' || parsed.entries === null) {
    throw new Error('classification overlay: entries missing');
  }
  return parsed as ClassificationOverlay;
}
