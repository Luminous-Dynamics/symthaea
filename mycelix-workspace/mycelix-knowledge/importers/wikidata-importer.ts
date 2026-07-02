#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Wikidata SPARQL → Mycelix Claims Importer
 *
 * Queries Wikidata for curated facts (capitals, populations, scientific
 * constants, geographical features) and converts them to Claims with
 * E4/N0/M3 classification (established, objective, transformative).
 *
 * Usage:
 *   npx ts-node importers/wikidata-importer.ts [--limit N]
 *
 * Output:
 *   seed-data/claims/wikidata-core.json
 */

import {
  buildClaim,
  filterValid,
  writeSeedFile,
  SOURCE_ENM,
  type Claim,
} from './common';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const WIKIDATA_ENDPOINT = 'https://query.wikidata.org/sparql';
const DEFAULT_LIMIT = 2000;
const BATCH_DELAY_MS = 2000; // Be polite to Wikidata servers

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

// ---------------------------------------------------------------------------
// SPARQL queries — each returns { content, source_uri } rows
// ---------------------------------------------------------------------------

interface SparqlRow {
  content: string;
  source_uri: string;
  tags: string[];
}

/** Country capitals */
const QUERY_CAPITALS = (n: number) => `
SELECT ?countryLabel ?capitalLabel ?country WHERE {
  ?country wdt:P31 wd:Q6256 ;
           wdt:P36 ?capital .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT ${n}
`;

/** Country populations */
const QUERY_POPULATIONS = (n: number) => `
SELECT ?countryLabel ?population ?country WHERE {
  ?country wdt:P31 wd:Q6256 ;
           wdt:P1082 ?population .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?population)
LIMIT ${n}
`;

/** Chemical elements */
const QUERY_ELEMENTS = (n: number) => `
SELECT ?elementLabel ?symbol ?atomicNumber ?element WHERE {
  ?element wdt:P31 wd:Q11344 ;
           wdt:P246 ?symbol ;
           wdt:P1086 ?atomicNumber .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?atomicNumber
LIMIT ${n}
`;

/** Largest cities by population */
const QUERY_CITIES = (n: number) => `
SELECT ?cityLabel ?countryLabel ?population ?city WHERE {
  ?city wdt:P31/wdt:P279* wd:Q515 ;
        wdt:P17 ?country ;
        wdt:P1082 ?population .
  FILTER(?population > 1000000)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?population)
LIMIT ${n}
`;

/** Major rivers by length */
const QUERY_RIVERS = (n: number) => `
SELECT ?riverLabel ?length ?river WHERE {
  ?river wdt:P31 wd:Q4022 ;
         wdt:P2043 ?length .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?length)
LIMIT ${n}
`;

/** Nobel Prize winners (physics) */
const QUERY_NOBEL_PHYSICS = (n: number) => `
SELECT ?personLabel ?year ?person WHERE {
  ?person wdt:P166 wd:Q38104 .
  ?person p:P166 ?statement .
  ?statement ps:P166 wd:Q38104 ;
             pq:P585 ?date .
  BIND(YEAR(?date) AS ?year)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?year)
LIMIT ${n}
`;

// ---------------------------------------------------------------------------
// SPARQL execution
// ---------------------------------------------------------------------------

async function sparqlQuery(query: string): Promise<any[]> {
  const url = `${WIKIDATA_ENDPOINT}?query=${encodeURIComponent(query)}&format=json`;
  const resp = await fetch(url, {
    headers: {
      Accept: 'application/sparql-results+json',
      'User-Agent': 'MycelixKnowledgeImporter/1.0 (https://mycelix.net)',
    },
  });

  if (!resp.ok) {
    throw new Error(`SPARQL query failed: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.results?.bindings ?? [];
}

function wdUri(binding: any): string {
  return binding?.value ?? '';
}

// ---------------------------------------------------------------------------
// Row → Claim converters
// ---------------------------------------------------------------------------

function capitalsToClaims(rows: any[]): Claim[] {
  return rows.map((r) =>
    buildClaim({
      source: 'wikidata',
      content: `The capital of ${r.countryLabel.value} is ${r.capitalLabel.value}.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Fact',
      confidence: 0.98,
      sources: [wdUri(r.country)],
      tags: ['geography', 'capitals', 'countries'],
    }),
  );
}

function populationsToClaims(rows: any[]): Claim[] {
  return rows.map((r) => {
    const pop = parseInt(r.population.value, 10);
    const formatted = pop.toLocaleString('en-US');
    return buildClaim({
      source: 'wikidata',
      content: `The population of ${r.countryLabel.value} is approximately ${formatted}.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Fact',
      confidence: 0.90,
      sources: [wdUri(r.country)],
      tags: ['geography', 'population', 'demographics'],
    });
  });
}

function elementsToClaims(rows: any[]): Claim[] {
  return rows.map((r) =>
    buildClaim({
      source: 'wikidata',
      content: `${r.elementLabel.value} (${r.symbol.value}) has atomic number ${r.atomicNumber.value}.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Fact',
      confidence: 0.99,
      sources: [wdUri(r.element)],
      tags: ['chemistry', 'elements', 'science'],
    }),
  );
}

function citiesToClaims(rows: any[]): Claim[] {
  return rows.map((r) => {
    const pop = parseInt(r.population.value, 10);
    const formatted = pop.toLocaleString('en-US');
    return buildClaim({
      source: 'wikidata',
      content: `${r.cityLabel.value} in ${r.countryLabel.value} has a population of approximately ${formatted}.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Fact',
      confidence: 0.88,
      sources: [wdUri(r.city)],
      tags: ['geography', 'cities', 'population'],
    });
  });
}

function riversToClaims(rows: any[]): Claim[] {
  return rows.map((r) => {
    const km = parseFloat(r.length.value).toFixed(0);
    return buildClaim({
      source: 'wikidata',
      content: `The ${r.riverLabel.value} is approximately ${km} km long.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Fact',
      confidence: 0.92,
      sources: [wdUri(r.river)],
      tags: ['geography', 'rivers', 'hydrology'],
    });
  });
}

function nobelToClaims(rows: any[]): Claim[] {
  return rows.map((r) =>
    buildClaim({
      source: 'wikidata',
      content: `${r.personLabel.value} received the Nobel Prize in Physics in ${r.year.value}.`,
      classification: SOURCE_ENM.wikidata,
      claimType: 'Historical',
      confidence: 0.99,
      sources: [wdUri(r.person)],
      tags: ['science', 'nobel-prize', 'physics', 'history'],
    }),
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

interface QuerySpec {
  name: string;
  query: (n: number) => string;
  converter: (rows: any[]) => Claim[];
  count: number;
}

async function main() {
  console.log('=== Wikidata → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims total`);
  console.log();

  const perQuery = Math.max(50, Math.floor(limit / 6));

  const specs: QuerySpec[] = [
    { name: 'Country capitals', query: QUERY_CAPITALS, converter: capitalsToClaims, count: Math.min(perQuery, 300) },
    { name: 'Country populations', query: QUERY_POPULATIONS, converter: populationsToClaims, count: Math.min(perQuery, 300) },
    { name: 'Chemical elements', query: QUERY_ELEMENTS, converter: elementsToClaims, count: 120 },
    { name: 'Major cities', query: QUERY_CITIES, converter: citiesToClaims, count: perQuery },
    { name: 'Major rivers', query: QUERY_RIVERS, converter: riversToClaims, count: Math.min(perQuery, 200) },
    { name: 'Nobel Physics laureates', query: QUERY_NOBEL_PHYSICS, converter: nobelToClaims, count: Math.min(perQuery, 300) },
  ];

  const allClaims: Claim[] = [];

  for (const spec of specs) {
    console.log(`--- ${spec.name} (up to ${spec.count}) ---`);
    try {
      const rows = await sparqlQuery(spec.query(spec.count));
      const claims = spec.converter(rows);
      console.log(`  Fetched ${rows.length} rows → ${claims.length} claims`);
      allClaims.push(...claims);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [ERROR] ${msg}`);
    }

    // Rate-limit between queries
    await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
  }

  console.log();

  // Validate and write
  const valid = filterValid(allClaims);
  const outFile = writeSeedFile('wikidata-core.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
