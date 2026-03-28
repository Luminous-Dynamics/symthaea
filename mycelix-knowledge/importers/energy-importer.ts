#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Energy Data → Mycelix Claims Importer (EIA + USGS)
 *
 * Fetches energy production and mineral resource data from the
 * US Energy Information Administration (EIA) API and includes curated
 * USGS mineral commodity facts.
 *
 * EIA API: https://www.eia.gov/opendata/
 * USGS: https://www.usgs.gov/centers/national-minerals-information-center
 * License: Public domain (US Government work)
 *
 * Usage:
 *   npx ts-node importers/energy-importer.ts [--api-key KEY] [--limit N]
 *
 * Without --api-key, uses the embedded curated dataset only.
 *
 * Output:
 *   seed-data/claims/energy-facts.json
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

const EIA_API = 'https://api.eia.gov/v2';
const DEFAULT_LIMIT = 1000;
const REQUEST_DELAY_MS = 300;

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const apiKey = (() => {
  const idx = process.argv.indexOf('--api-key');
  return idx !== -1 ? process.argv[idx + 1] : null;
})();

// ---------------------------------------------------------------------------
// Embedded curated energy & mineral facts
// ---------------------------------------------------------------------------

interface EnergyFact {
  content: string;
  confidence: number;
  tags: string[];
  source: string;
}

const CURATED_ENERGY_FACTS: EnergyFact[] = [
  // Global energy production
  { content: 'Global primary energy consumption was approximately 604 exajoules (EJ) in 2022.', confidence: 0.93, tags: ['energy', 'consumption', 'global'], source: 'https://www.eia.gov/international/data/world' },
  { content: 'Fossil fuels (oil, coal, natural gas) accounted for approximately 82% of global primary energy consumption in 2022.', confidence: 0.92, tags: ['energy', 'fossil-fuels', 'global'], source: 'https://www.eia.gov/international/data/world' },
  { content: 'Renewable energy sources accounted for approximately 14% of global primary energy consumption in 2022.', confidence: 0.90, tags: ['energy', 'renewables', 'global'], source: 'https://www.eia.gov/international/data/world' },
  { content: 'Global electricity generation was approximately 29,165 terawatt-hours (TWh) in 2022.', confidence: 0.92, tags: ['energy', 'electricity', 'generation'], source: 'https://www.eia.gov/international/data/world/electricity/electricity-generation' },

  // Oil
  { content: 'Global crude oil production was approximately 93.9 million barrels per day in 2023.', confidence: 0.93, tags: ['energy', 'oil', 'production'], source: 'https://www.eia.gov/international/data/world/petroleum-and-other-liquids/annual-crude-and-lease-condensate-reserves' },
  { content: 'The United States became the world\'s largest crude oil producer in 2018, surpassing Russia and Saudi Arabia.', confidence: 0.95, tags: ['energy', 'oil', 'united-states', 'production'], source: 'https://www.eia.gov/todayinenergy/' },
  { content: 'OPEC countries hold approximately 79.4% of the world\'s proven crude oil reserves.', confidence: 0.90, tags: ['energy', 'oil', 'opec', 'reserves'], source: 'https://www.eia.gov/international/data/world' },
  { content: 'Venezuela has the world\'s largest proven oil reserves at approximately 303 billion barrels.', confidence: 0.88, tags: ['energy', 'oil', 'venezuela', 'reserves'], source: 'https://www.eia.gov/international/data/world' },

  // Natural gas
  { content: 'Global natural gas production was approximately 4.1 trillion cubic meters in 2022.', confidence: 0.92, tags: ['energy', 'natural-gas', 'production'], source: 'https://www.eia.gov/international/data/world/natural-gas/dry-natural-gas-production' },
  { content: 'The United States is the world\'s largest natural gas producer, followed by Russia.', confidence: 0.94, tags: ['energy', 'natural-gas', 'united-states'], source: 'https://www.eia.gov/international/data/world/natural-gas/dry-natural-gas-production' },
  { content: 'Russia holds the world\'s largest proven natural gas reserves at approximately 48 trillion cubic meters.', confidence: 0.90, tags: ['energy', 'natural-gas', 'russia', 'reserves'], source: 'https://www.eia.gov/international/data/world' },

  // Coal
  { content: 'Global coal production was approximately 8.3 billion metric tons in 2022.', confidence: 0.92, tags: ['energy', 'coal', 'production'], source: 'https://www.eia.gov/international/data/world/coal-and-coke/coal-and-coke-production' },
  { content: 'China produces approximately 50% of the world\'s coal.', confidence: 0.93, tags: ['energy', 'coal', 'china', 'production'], source: 'https://www.eia.gov/international/data/world/coal-and-coke/coal-and-coke-production' },
  { content: 'The United States has the world\'s largest recoverable coal reserves at approximately 249 billion short tons.', confidence: 0.91, tags: ['energy', 'coal', 'united-states', 'reserves'], source: 'https://www.eia.gov/energyexplained/coal/how-much-coal-is-left.php' },

  // Nuclear
  { content: 'Nuclear energy provided approximately 10% of global electricity generation in 2022.', confidence: 0.93, tags: ['energy', 'nuclear', 'electricity'], source: 'https://www.eia.gov/international/data/world/electricity/electricity-generation' },
  { content: 'There are approximately 440 operational nuclear power reactors worldwide as of 2023.', confidence: 0.92, tags: ['energy', 'nuclear', 'reactors'], source: 'https://www.iaea.org/resources/databases/power-reactor-information-system-pris' },
  { content: 'France generates approximately 70% of its electricity from nuclear power, the highest share of any country.', confidence: 0.95, tags: ['energy', 'nuclear', 'france', 'electricity'], source: 'https://www.eia.gov/international/data/country/FRA/electricity/electricity-generation' },

  // Renewables
  { content: 'Global solar photovoltaic capacity exceeded 1,000 GW in 2022, growing by approximately 230 GW in a single year.', confidence: 0.92, tags: ['energy', 'solar', 'renewables', 'capacity'], source: 'https://www.irena.org/Statistics/View-Data-by-Topic/Capacity-and-Generation/Statistics-Time-Series' },
  { content: 'Global wind power capacity reached approximately 906 GW by the end of 2022.', confidence: 0.91, tags: ['energy', 'wind', 'renewables', 'capacity'], source: 'https://www.irena.org/Statistics/View-Data-by-Topic/Capacity-and-Generation/Statistics-Time-Series' },
  { content: 'Hydroelectric power is the largest source of renewable electricity, generating approximately 4,300 TWh in 2022.', confidence: 0.93, tags: ['energy', 'hydroelectric', 'renewables', 'electricity'], source: 'https://www.eia.gov/international/data/world/electricity/electricity-generation' },
  { content: 'The cost of solar photovoltaic electricity has declined by approximately 89% between 2010 and 2022.', confidence: 0.90, tags: ['energy', 'solar', 'cost', 'renewables'], source: 'https://www.irena.org/publications/2023/Aug/Renewable-Power-Generation-Costs-in-2022' },
  { content: 'The cost of onshore wind electricity has declined by approximately 69% between 2010 and 2022.', confidence: 0.90, tags: ['energy', 'wind', 'cost', 'renewables'], source: 'https://www.irena.org/publications/2023/Aug/Renewable-Power-Generation-Costs-in-2022' },

  // US energy
  { content: 'The United States consumed approximately 100.4 quadrillion BTU of primary energy in 2022.', confidence: 0.94, tags: ['energy', 'consumption', 'united-states'], source: 'https://www.eia.gov/totalenergy/data/monthly/' },
  { content: 'Natural gas is the largest source of US electricity generation at approximately 40% in 2022.', confidence: 0.94, tags: ['energy', 'natural-gas', 'electricity', 'united-states'], source: 'https://www.eia.gov/electricity/monthly/' },
  { content: 'The US transportation sector consumes approximately 28% of total US energy, primarily as petroleum.', confidence: 0.92, tags: ['energy', 'transportation', 'petroleum', 'united-states'], source: 'https://www.eia.gov/totalenergy/data/monthly/' },

  // USGS minerals
  { content: 'Lithium global production was approximately 130,000 metric tons in 2022, with Australia as the leading producer.', confidence: 0.90, tags: ['minerals', 'lithium', 'production', 'usgs'], source: 'https://pubs.usgs.gov/periodicals/mcs2023/mcs2023-lithium.pdf' },
  { content: 'Rare earth elements global production was approximately 300,000 metric tons in 2022, with China producing about 70%.', confidence: 0.90, tags: ['minerals', 'rare-earth', 'production', 'china', 'usgs'], source: 'https://pubs.usgs.gov/periodicals/mcs2023/mcs2023-rare-earths.pdf' },
  { content: 'Copper global mine production was approximately 22 million metric tons in 2022.', confidence: 0.91, tags: ['minerals', 'copper', 'production', 'usgs'], source: 'https://pubs.usgs.gov/periodicals/mcs2023/mcs2023-copper.pdf' },
  { content: 'Cobalt global mine production was approximately 190,000 metric tons in 2022, with the Democratic Republic of Congo producing about 73%.', confidence: 0.90, tags: ['minerals', 'cobalt', 'production', 'congo', 'usgs'], source: 'https://pubs.usgs.gov/periodicals/mcs2023/mcs2023-cobalt.pdf' },
  { content: 'Global iron ore production was approximately 2.6 billion metric tons in 2022.', confidence: 0.92, tags: ['minerals', 'iron-ore', 'production', 'usgs'], source: 'https://pubs.usgs.gov/periodicals/mcs2023/mcs2023-iron-ore.pdf' },
];

// ---------------------------------------------------------------------------
// EIA API v2
// ---------------------------------------------------------------------------

interface EiaDataPoint {
  period: string;
  value: number;
  seriesDescription?: string;
}

async function fetchEiaData(route: string, frequency: string, params: Record<string, string>): Promise<EiaDataPoint[]> {
  if (!apiKey) return [];

  const searchParams = new URLSearchParams({
    api_key: apiKey,
    frequency,
    ...params,
  });

  const url = `${EIA_API}/${route}/data?${searchParams}`;
  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`EIA API error: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.response?.data ?? [];
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== Energy Data → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  if (apiKey) console.log('Mode: API (EIA key provided)');
  else console.log('Mode: Curated dataset (pass --api-key KEY for live EIA data)');
  console.log();

  const allClaims: Claim[] = [];

  // Curated facts
  console.log('--- Curated energy & mineral facts ---');
  for (const fact of CURATED_ENERGY_FACTS) {
    allClaims.push(
      buildClaim({
        source: 'eia',
        content: fact.content,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: fact.confidence,
        sources: [fact.source],
        tags: fact.tags,
      }),
    );
  }
  console.log(`  ${CURATED_ENERGY_FACTS.length} curated claims`);

  // EIA API data
  if (apiKey && allClaims.length < limit) {
    console.log();
    console.log('--- EIA API data ---');

    try {
      // Fetch US electricity generation by source
      const elecData = await fetchEiaData(
        'electricity/electric-power-operational-data',
        'annual',
        { 'data[]': 'generation', start: '2020', end: '2023' },
      );

      for (const dp of elecData.slice(0, limit - allClaims.length)) {
        if (dp.value && dp.seriesDescription) {
          allClaims.push(
            buildClaim({
              source: 'eia',
              content: `${dp.seriesDescription}: ${dp.value.toLocaleString('en-US')} thousand MWh in ${dp.period}.`,
              classification: SOURCE_ENM.institutional,
              claimType: 'Fact',
              confidence: 0.92,
              sources: ['https://www.eia.gov/electricity/data.php'],
              tags: ['energy', 'electricity', 'united-states', 'generation'],
            }),
          );
        }
      }

      console.log(`  EIA electricity data: ${elecData.length} data points`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [ERROR] ${msg}`);
    }
  }

  console.log();

  const valid = filterValid(allClaims.slice(0, limit));
  const outFile = writeSeedFile('energy-facts.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
