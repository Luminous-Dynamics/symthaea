#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Google Fact Check Tools API → Mycelix Claims Importer
 *
 * Searches the Google Fact Check Tools API for fact-checked claims
 * across multiple topics and converts them to Mycelix Claims with
 * verdict-based E/N/M classification.
 *
 * API: https://developers.google.com/fact-check/tools/api
 * Requires: Google API key with Fact Check Tools API enabled
 *
 * Verdicts are derived from the textualRating field using keyword matching.
 *
 * Usage:
 *   npx ts-node importers/factcheck-api-importer.ts --key API_KEY [--limit N]
 *
 * Output:
 *   seed-data/claims/factcheck-api.json
 */

import {
  buildClaim,
  filterValid,
  writeSeedFile,
  enm,
  type Claim,
  type EpistemicPosition,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const FACTCHECK_API = 'https://factchecktools.googleapis.com/v1alpha1/claims:search';
const DEFAULT_LIMIT = 2000;
const REQUEST_DELAY_MS = 200;
const PAGE_SIZE = 100; // Max per request

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const apiKey = (() => {
  const idx = process.argv.indexOf('--key');
  return idx !== -1 ? process.argv[idx + 1] : null;
})();

if (!apiKey) {
  console.error('Usage: npx ts-node importers/factcheck-api-importer.ts --key API_KEY [--limit N]');
  console.error();
  console.error('Get an API key at: https://console.cloud.google.com/apis/credentials');
  console.error('Enable: Fact Check Tools API');
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Topics to search
// ---------------------------------------------------------------------------

const SEARCH_TOPICS = [
  'climate change',
  'COVID-19 vaccine',
  'election fraud',
  'immigration',
  'economy inflation',
  'health nutrition',
  'technology AI',
  'environment pollution',
  'education',
  'energy renewable solar',
  'space exploration',
  'military defense',
  'science discovery',
  'water resources',
  'agriculture food',
  'ocean biodiversity',
  'poverty inequality',
  'nuclear energy',
  'cryptocurrency bitcoin',
  'population growth',
];

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

interface FactCheckClaim {
  text?: string;
  claimant?: string;
  claimDate?: string;
  claimReview?: Array<{
    publisher?: { name?: string; site?: string };
    url?: string;
    title?: string;
    reviewDate?: string;
    textualRating?: string;
    languageCode?: string;
  }>;
}

interface FactCheckResponse {
  claims?: FactCheckClaim[];
  nextPageToken?: string;
}

// ---------------------------------------------------------------------------
// Rating → E/N/M mapping
// ---------------------------------------------------------------------------

function ratingToVerdict(rating: string): 'true' | 'mixed' | 'false' | 'unknown' {
  const lower = rating.toLowerCase();

  // True patterns
  if (/\b(true|correct|accurate|confirmed|verified|supported)\b/.test(lower) &&
      !/\b(not|false|mostly false|half|partly|partially)\b/.test(lower)) {
    return 'true';
  }
  if (/\b(mostly true|largely true)\b/.test(lower)) {
    return 'true';
  }

  // False patterns
  if (/\b(false|incorrect|wrong|fabricated|fake|pants on fire|four pinocchios|debunked)\b/.test(lower) &&
      !/\b(not false|mostly true|half)\b/.test(lower)) {
    return 'false';
  }
  if (/\b(mostly false|largely false)\b/.test(lower)) {
    return 'false';
  }

  // Mixed patterns
  if (/\b(half true|mixture|misleading|exaggerated|partly|partially|needs context|out of context|distorts|cherry[- ]picks|two pinocchios|three pinocchios)\b/.test(lower)) {
    return 'mixed';
  }

  return 'unknown';
}

function verdictClassification(verdict: 'true' | 'mixed' | 'false' | 'unknown'): EpistemicPosition {
  switch (verdict) {
    case 'true':    return enm('E3', 'N0', 'M1');
    case 'mixed':   return enm('E1', 'N1', 'M1');
    case 'false':   return enm('E0', 'N0', 'M0');
    case 'unknown': return enm('E1', 'N0', 'M0');
  }
}

function verdictConfidence(verdict: 'true' | 'mixed' | 'false' | 'unknown'): number {
  switch (verdict) {
    case 'true':    return 0.80;
    case 'mixed':   return 0.55;
    case 'false':   return 0.80;
    case 'unknown': return 0.35;
  }
}

// ---------------------------------------------------------------------------
// API fetching
// ---------------------------------------------------------------------------

async function searchClaims(query: string, pageToken?: string): Promise<FactCheckResponse> {
  const params = new URLSearchParams({
    key: apiKey!,
    query,
    pageSize: String(PAGE_SIZE),
    languageCode: 'en',
  });
  if (pageToken) params.set('pageToken', pageToken);

  const url = `${FACTCHECK_API}?${params}`;
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`Fact Check API error: ${resp.status} ${resp.statusText}`);
  }

  return resp.json() as Promise<FactCheckResponse>;
}

// ---------------------------------------------------------------------------
// Conversion
// ---------------------------------------------------------------------------

function factCheckToClaim(fc: FactCheckClaim, topic: string): Claim | null {
  if (!fc.text || fc.text.length < 10) return null;

  // Use the first English review
  const review = fc.claimReview?.find((r) => !r.languageCode || r.languageCode === 'en');
  if (!review?.textualRating) return null;

  const verdict = ratingToVerdict(review.textualRating);
  const publisher = review.publisher?.name || review.publisher?.site || 'unknown';

  const tags = ['fact-check', 'google-factcheck-api', topic.split(' ')[0].toLowerCase()];
  switch (verdict) {
    case 'true':    tags.push('verified-true'); break;
    case 'mixed':   tags.push('mixed-verdict'); break;
    case 'false':   tags.push('verified-false'); break;
    case 'unknown': tags.push('unresolved'); break;
  }

  const sources: string[] = [];
  if (review.url) sources.push(review.url);

  return buildClaim({
    source: 'gfactcheck',
    content: fc.text,
    classification: verdictClassification(verdict),
    claimType: 'Fact',
    confidence: verdictConfidence(verdict),
    sources,
    tags,
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== Google Fact Check API → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  console.log(`Topics: ${SEARCH_TOPICS.length}`);
  console.log();

  const allClaims: Claim[] = [];
  const claimsPerTopic = Math.ceil(limit / SEARCH_TOPICS.length);

  for (const topic of SEARCH_TOPICS) {
    if (allClaims.length >= limit) break;

    console.log(`--- "${topic}" ---`);
    let topicClaims = 0;
    let pageToken: string | undefined;

    while (topicClaims < claimsPerTopic && allClaims.length < limit) {
      try {
        const resp = await searchClaims(topic, pageToken);
        const claims = resp.claims ?? [];

        if (claims.length === 0) break;

        for (const fc of claims) {
          const claim = factCheckToClaim(fc, topic);
          if (claim) {
            allClaims.push(claim);
            topicClaims++;
          }
        }

        pageToken = resp.nextPageToken;
        if (!pageToken) break;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`  [ERROR] ${msg}`);
        break;
      }

      await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
    }

    console.log(`  ${topicClaims} claims`);
  }

  console.log();

  // Breakdown
  const trueCount = allClaims.filter((c) => c.tags.includes('verified-true')).length;
  const mixedCount = allClaims.filter((c) => c.tags.includes('mixed-verdict')).length;
  const falseCount = allClaims.filter((c) => c.tags.includes('verified-false')).length;
  const unknownCount = allClaims.filter((c) => c.tags.includes('unresolved')).length;
  console.log(`Verdicts: TRUE=${trueCount}, MIXED=${mixedCount}, FALSE=${falseCount}, UNKNOWN=${unknownCount}`);
  console.log();

  const valid = filterValid(allClaims);
  const outFile = writeSeedFile('factcheck-api.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
