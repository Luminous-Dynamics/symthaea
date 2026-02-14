/**
 * LUCID Client Utilities
 */

import type { Record as HolochainRecord } from '@holochain/client';
import { decode } from '@msgpack/msgpack';

/**
 * Decode entry from a Holochain record
 */
export function decodeRecord<T>(record: HolochainRecord): T {
  const recordEntry = record.entry;
  if (!recordEntry) {
    throw new Error('Record has no entry');
  }

  // Check for Present variant which contains the entry
  if ('Present' in recordEntry) {
    const entry = recordEntry.Present;
    // Entry is an App entry with entry_type: 'App' and entry: Uint8Array
    if (entry.entry_type === 'App') {
      return decode(entry.entry as Uint8Array) as T;
    }
    throw new Error(`Unexpected entry type: ${entry.entry_type}`);
  }

  // Handle other variants
  if ('Hidden' in recordEntry) {
    throw new Error('Entry is hidden');
  }
  if ('NotApplicable' in recordEntry) {
    throw new Error('Entry not applicable');
  }
  if ('NotStored' in recordEntry) {
    throw new Error('Entry not stored');
  }

  throw new Error('Entry not present in record');
}

/**
 * Decode entries from multiple Holochain records
 */
export function decodeRecords<T>(records: HolochainRecord[]): T[] {
  return records.map((record) => decodeRecord<T>(record));
}

/**
 * Generate epistemic code string from classification
 */
export function epistemicCode(
  e: number,
  n: number,
  m: number,
  h: number
): string {
  return `E${e}N${n}M${m}H${h}`;
}

/**
 * Calculate overall epistemic strength
 * Weights: E=40%, N=25%, M=20%, H=15%
 */
export function calculateEpistemicStrength(
  e: number,
  n: number,
  m: number,
  h: number
): number {
  const eVal = e / 4; // E0-E4 -> 0-1
  const nVal = n / 3; // N0-N3 -> 0-1
  const mVal = m / 3; // M0-M3 -> 0-1
  const hVal = h / 4; // H0-H4 -> 0-1

  return 0.4 * eVal + 0.25 * nVal + 0.2 * mVal + 0.15 * hVal;
}

/**
 * Format timestamp for display
 */
export function formatTimestamp(timestamp: number): string {
  // Holochain timestamps are in microseconds
  const date = new Date(timestamp / 1000);
  return date.toISOString();
}

/**
 * Parse ISO date string to Holochain timestamp (microseconds)
 */
export function parseToTimestamp(isoDate: string): number {
  return new Date(isoDate).getTime() * 1000;
}

/**
 * Source type quality weights for confidence calculation
 */
export const SOURCE_QUALITY_WEIGHTS: Record<string, number> = {
  AcademicPaper: 1.0,
  OfficialDocument: 0.95,
  Dataset: 0.9,
  Book: 0.85,
  KnowledgeBase: 0.8,
  NewsArticle: 0.6,
  Video: 0.5,
  Audio: 0.5,
  BlogPost: 0.4,
  WebPage: 0.35,
  LucidThought: 0.5,
  Conversation: 0.3,
  PersonalExperience: 0.3,
  SocialMedia: 0.2,
  Other: 0.25,
};

/**
 * Get quality weight for a source type
 */
export function getSourceQualityWeight(sourceType: string): number {
  return SOURCE_QUALITY_WEIGHTS[sourceType] ?? 0.25;
}
