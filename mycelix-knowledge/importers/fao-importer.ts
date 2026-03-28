#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * FAO FAOSTAT → Mycelix Claims Importer
 *
 * Fetches agricultural production and food supply data from the FAO
 * FAOSTAT API and converts to Claims with E4/N0/M3 classification.
 *
 * Also includes an embedded curated dataset of key food/agriculture facts.
 *
 * FAOSTAT: https://www.fao.org/faostat/en/
 * API: https://fenixservices.fao.org/faostat/api/v1/
 * License: CC BY-NC-SA 3.0 IGO
 *
 * Usage:
 *   npx ts-node importers/fao-importer.ts [--limit N] [--api]
 *
 * Without --api, uses the embedded curated dataset only.
 *
 * Output:
 *   seed-data/claims/fao-agriculture.json
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

const FAOSTAT_API = 'https://fenixservices.fao.org/faostat/api/v1';
const DEFAULT_LIMIT = 500;
const REQUEST_DELAY_MS = 500;

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const useApi = process.argv.includes('--api');

// ---------------------------------------------------------------------------
// Embedded curated agriculture & food facts
// ---------------------------------------------------------------------------

interface AgFact {
  content: string;
  confidence: number;
  tags: string[];
  source: string;
}

const CURATED_AG_FACTS: AgFact[] = [
  // Global production
  { content: 'Global cereal production was approximately 2.8 billion metric tons in 2022.', confidence: 0.93, tags: ['agriculture', 'cereals', 'production', 'global'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'China is the world\'s largest rice producer, producing approximately 212 million metric tons in 2022.', confidence: 0.93, tags: ['agriculture', 'rice', 'china', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'The United States is the world\'s largest corn (maize) producer at approximately 349 million metric tons in 2022.', confidence: 0.93, tags: ['agriculture', 'corn', 'maize', 'united-states', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'Global wheat production was approximately 808 million metric tons in 2022.', confidence: 0.92, tags: ['agriculture', 'wheat', 'production', 'global'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'Brazil is the world\'s largest soybean producer, followed by the United States and Argentina.', confidence: 0.92, tags: ['agriculture', 'soybean', 'brazil', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'Global sugar cane production was approximately 1.9 billion metric tons in 2022, with Brazil as the leading producer.', confidence: 0.91, tags: ['agriculture', 'sugar-cane', 'brazil', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },

  // Livestock
  { content: 'Global cattle population was approximately 1 billion head in 2022.', confidence: 0.90, tags: ['agriculture', 'livestock', 'cattle', 'global'], source: 'https://www.fao.org/faostat/en/#data/QA' },
  { content: 'China has the world\'s largest pig population at approximately 453 million head in 2022.', confidence: 0.90, tags: ['agriculture', 'livestock', 'pigs', 'china'], source: 'https://www.fao.org/faostat/en/#data/QA' },
  { content: 'Global milk production was approximately 930 million metric tons in 2022.', confidence: 0.91, tags: ['agriculture', 'dairy', 'milk', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },
  { content: 'Global egg production was approximately 93 million metric tons in 2022.', confidence: 0.90, tags: ['agriculture', 'eggs', 'poultry', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },

  // Food security
  { content: 'Approximately 735 million people were affected by hunger in 2022, up from 613 million in 2019.', confidence: 0.92, tags: ['food-security', 'hunger', 'malnutrition', 'global'], source: 'https://www.fao.org/state-of-food-security-nutrition' },
  { content: 'Africa has the highest prevalence of undernourishment at approximately 20% of the population.', confidence: 0.90, tags: ['food-security', 'hunger', 'africa', 'undernourishment'], source: 'https://www.fao.org/state-of-food-security-nutrition' },
  { content: 'Approximately one-third of food produced globally is lost or wasted, amounting to about 1.3 billion metric tons per year.', confidence: 0.88, tags: ['food-security', 'food-waste', 'food-loss', 'global'], source: 'https://www.fao.org/food-loss-and-food-waste/en/' },
  { content: 'Food loss and waste contribute approximately 8-10% of global greenhouse gas emissions.', confidence: 0.85, tags: ['food-security', 'food-waste', 'climate', 'emissions'], source: 'https://www.fao.org/food-loss-and-food-waste/en/' },

  // Land use
  { content: 'Approximately 37% of the global land surface is used for agriculture (cropland and pasture).', confidence: 0.91, tags: ['agriculture', 'land-use', 'global'], source: 'https://www.fao.org/faostat/en/#data/RL' },
  { content: 'Global agricultural land area has remained relatively stable at approximately 4.8 billion hectares since 2000.', confidence: 0.89, tags: ['agriculture', 'land-use', 'trends'], source: 'https://www.fao.org/faostat/en/#data/RL' },
  { content: 'Cropland covers approximately 1.6 billion hectares globally, or about 12% of the land surface.', confidence: 0.90, tags: ['agriculture', 'cropland', 'land-use'], source: 'https://www.fao.org/faostat/en/#data/RL' },
  { content: 'Deforestation for agriculture causes approximately 10% of global greenhouse gas emissions.', confidence: 0.85, tags: ['agriculture', 'deforestation', 'climate', 'land-use'], source: 'https://www.fao.org/state-of-forests/en/' },

  // Trade
  { content: 'Global agricultural trade was valued at approximately $1.9 trillion in 2022.', confidence: 0.88, tags: ['agriculture', 'trade', 'global', 'economics'], source: 'https://www.fao.org/faostat/en/#data/TP' },
  { content: 'The European Union is the world\'s largest agricultural importer and exporter combined.', confidence: 0.87, tags: ['agriculture', 'trade', 'eu'], source: 'https://www.fao.org/faostat/en/#data/TP' },
  { content: 'Coffee is one of the most traded agricultural commodities, with global production of approximately 10.8 million metric tons in 2022.', confidence: 0.89, tags: ['agriculture', 'coffee', 'trade', 'production'], source: 'https://www.fao.org/faostat/en/#data/QCL' },

  // Fisheries
  { content: 'Global fisheries and aquaculture production reached a record 214 million metric tons in 2020.', confidence: 0.91, tags: ['fisheries', 'aquaculture', 'production', 'global'], source: 'https://www.fao.org/state-of-fisheries-aquaculture' },
  { content: 'Aquaculture now provides approximately 49% of global fish production for human consumption.', confidence: 0.90, tags: ['fisheries', 'aquaculture', 'food'], source: 'https://www.fao.org/state-of-fisheries-aquaculture' },
  { content: 'China produces approximately 35% of global fisheries and aquaculture output.', confidence: 0.90, tags: ['fisheries', 'china', 'production'], source: 'https://www.fao.org/state-of-fisheries-aquaculture' },
];

// ---------------------------------------------------------------------------
// FAOSTAT API
// ---------------------------------------------------------------------------

interface FaoDataPoint {
  Area: string;
  Item: string;
  Element: string;
  Year: number;
  Unit: string;
  Value: number;
}

async function fetchFaoData(domainCode: string, areaCode: string, itemCode: string, elementCode: string, year: string): Promise<FaoDataPoint[]> {
  const url = `${FAOSTAT_API}/en/data/${domainCode}?area=${areaCode}&item=${itemCode}&element=${elementCode}&year=${year}`;

  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`FAOSTAT API error: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.data ?? [];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== FAO FAOSTAT → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  if (useApi) console.log('Mode: API (FAOSTAT)');
  else console.log('Mode: Curated dataset (pass --api for live FAOSTAT data)');
  console.log();

  const allClaims: Claim[] = [];

  // Curated facts
  console.log('--- Curated agriculture & food facts ---');
  for (const fact of CURATED_AG_FACTS) {
    allClaims.push(
      buildClaim({
        source: 'fao',
        content: fact.content,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: fact.confidence,
        sources: [fact.source],
        tags: fact.tags,
      }),
    );
  }
  console.log(`  ${CURATED_AG_FACTS.length} curated claims`);

  // API data
  if (useApi && allClaims.length < limit) {
    console.log();
    console.log('--- FAOSTAT API data ---');

    // Fetch top crop production by country for recent year
    try {
      // QCL domain = Production/Crops/Livestock
      // Area 5000 = World, Element 5510 = Production
      const data = await fetchFaoData('QCL', '5000', '15', '5510', '2022');
      let converted = 0;
      for (const dp of data.slice(0, limit - allClaims.length)) {
        if (dp.Value) {
          const valueStr = dp.Value >= 1_000_000
            ? `${(dp.Value / 1_000_000).toFixed(1)} million`
            : dp.Value.toLocaleString('en-US');

          allClaims.push(
            buildClaim({
              source: 'fao',
              content: `${dp.Area} produced ${valueStr} ${dp.Unit} of ${dp.Item} in ${dp.Year}, according to FAOSTAT.`,
              classification: SOURCE_ENM.institutional,
              claimType: 'Fact',
              confidence: 0.90,
              sources: [`https://www.fao.org/faostat/en/#data/QCL`],
              tags: ['agriculture', 'production', dp.Item.toLowerCase().replace(/\s+/g, '-')],
            }),
          );
          converted++;
        }
      }
      console.log(`  Crop production: ${data.length} data points → ${converted} claims`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [ERROR] ${msg}`);
    }
  }

  console.log();

  const valid = filterValid(allClaims.slice(0, limit));
  const outFile = writeSeedFile('fao-agriculture.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
