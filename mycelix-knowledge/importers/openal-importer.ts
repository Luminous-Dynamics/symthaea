#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * OpenAlex → Mycelix Claims Importer
 *
 * Fetches top-cited scientific works from the OpenAlex REST API and
 * converts their titles + key findings into Claims with E/N/M classification.
 *
 * OpenAlex API: https://docs.openalex.org/
 * License: CC0 (public domain)
 *
 * Classification:
 *   cited > 1000   → E4/N1/M2 (established, robust consensus)
 *   cited > 100    → E3/N1/M2 (replicated)
 *   cited > 10     → E2/N1/M2 (tested)
 *   else           → E1/N1/M1 (preliminary)
 *
 * Usage:
 *   npx ts-node importers/openal-importer.ts [--limit N] [--topic TOPIC]
 *
 * Options:
 *   --limit N       Max claims to generate (default: 2000)
 *   --topic TOPIC   Filter by OpenAlex concept (e.g., "physics", "biology")
 *
 * Output:
 *   seed-data/claims/openal-top.json
 */

import {
  buildClaim,
  filterValid,
  writeSeedFile,
  enm,
  type Claim,
  type ClaimType,
  type EpistemicPosition,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const OPENALEX_API = 'https://api.openalex.org';
const DEFAULT_LIMIT = 2000;
const PAGE_SIZE = 200; // OpenAlex max per_page
const REQUEST_DELAY_MS = 200; // Polite rate limiting

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const topicFilter = (() => {
  const idx = process.argv.indexOf('--topic');
  return idx !== -1 ? process.argv[idx + 1] : null;
})();

// ---------------------------------------------------------------------------
// OpenAlex types (subset of the full schema)
// ---------------------------------------------------------------------------

interface OpenAlexWork {
  id: string;
  doi?: string;
  title?: string;
  display_name?: string;
  cited_by_count: number;
  publication_year?: number;
  type?: string;
  concepts?: Array<{
    id: string;
    display_name: string;
    level: number;
    score: number;
  }>;
  authorships?: Array<{
    author: {
      id: string;
      display_name: string;
    };
  }>;
  primary_location?: {
    source?: {
      display_name: string;
    };
  };
}

interface OpenAlexResponse {
  meta: { count: number; per_page: number; page: number };
  results: OpenAlexWork[];
}

// ---------------------------------------------------------------------------
// E/N/M classification by citation count
// ---------------------------------------------------------------------------

function citationClassification(citedBy: number): EpistemicPosition {
  if (citedBy > 1000) return enm('E4', 'N1', 'M2');
  if (citedBy > 100) return enm('E3', 'N1', 'M2');
  if (citedBy > 10) return enm('E2', 'N1', 'M2');
  return enm('E1', 'N1', 'M1');
}

function citationConfidence(citedBy: number): number {
  if (citedBy > 1000) return 0.92;
  if (citedBy > 100) return 0.80;
  if (citedBy > 10) return 0.60;
  return 0.40;
}

// ---------------------------------------------------------------------------
// API fetching
// ---------------------------------------------------------------------------

async function fetchWorks(page: number, perPage: number): Promise<OpenAlexResponse> {
  let url = `${OPENALEX_API}/works?per_page=${perPage}&page=${page}&sort=cited_by_count:desc&filter=has_doi:true,type:article`;

  if (topicFilter) {
    url += `,concepts.display_name.search:${encodeURIComponent(topicFilter)}`;
  }

  const resp = await fetch(url, {
    headers: {
      Accept: 'application/json',
      'User-Agent': 'MycelixKnowledgeImporter/1.0 (mailto:tristan.stoltz@evolvingresonantcocreationism.com)',
    },
  });

  if (!resp.ok) {
    throw new Error(`OpenAlex API error: ${resp.status} ${resp.statusText}`);
  }

  return resp.json() as Promise<OpenAlexResponse>;
}

// ---------------------------------------------------------------------------
// Work → Claim conversion
// ---------------------------------------------------------------------------

function workToClaim(work: OpenAlexWork): Claim | null {
  const title = work.display_name || work.title;
  if (!title || title.length < 10) return null;

  // Build a human-readable claim from the work
  const authors = work.authorships?.slice(0, 3).map((a) => a.author.display_name) ?? [];
  const authorStr = authors.length > 0 ? authors.join(', ') : 'Unknown';
  const year = work.publication_year ?? 'n.d.';
  const journal = work.primary_location?.source?.display_name ?? '';

  // The claim content: "According to [authors] ([year]), [title]."
  let content = `According to ${authorStr} (${year}), ${title}`;
  if (!content.endsWith('.')) content += '.';

  // Build tags from concepts
  const tags = ['science', 'peer-reviewed'];
  if (work.concepts) {
    for (const concept of work.concepts.slice(0, 5)) {
      if (concept.score > 0.3) {
        tags.push(concept.display_name.toLowerCase().replace(/\s+/g, '-'));
      }
    }
  }
  if (journal) {
    tags.push('journal');
  }

  // Sources: DOI is canonical
  const sources: string[] = [];
  if (work.doi) sources.push(work.doi);
  if (work.id) sources.push(work.id);

  return buildClaim({
    source: 'openal',
    content,
    classification: citationClassification(work.cited_by_count),
    claimType: 'Fact',
    confidence: citationConfidence(work.cited_by_count),
    sources,
    tags,
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== OpenAlex → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  if (topicFilter) console.log(`Topic filter: ${topicFilter}`);
  console.log();

  const allClaims: Claim[] = [];
  const totalPages = Math.ceil(limit / PAGE_SIZE);

  for (let page = 1; page <= totalPages; page++) {
    const needed = Math.min(PAGE_SIZE, limit - allClaims.length);
    console.log(`--- Page ${page}/${totalPages} (fetching ${needed}) ---`);

    try {
      const resp = await fetchWorks(page, needed);
      const works = resp.results;

      if (works.length === 0) {
        console.log('  No more results.');
        break;
      }

      let converted = 0;
      for (const work of works) {
        const claim = workToClaim(work);
        if (claim) {
          allClaims.push(claim);
          converted++;
        }
      }

      console.log(`  Fetched ${works.length} works → ${converted} claims (total: ${allClaims.length})`);

      if (allClaims.length >= limit) break;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [ERROR] ${msg}`);
      // Continue to next page on error
    }

    await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
  }

  console.log();

  const valid = filterValid(allClaims);
  const outFile = writeSeedFile('openal-top.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
