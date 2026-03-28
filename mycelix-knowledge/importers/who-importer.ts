#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * WHO Global Health Observatory → Mycelix Claims Importer
 *
 * Fetches health indicators from the WHO GHO OData API and converts
 * to Claims with E4/N0/M3 classification (institutional health data).
 *
 * Also includes an embedded curated dataset of key global health facts.
 *
 * WHO GHO API: https://www.who.int/data/gho/info/gho-odata-api
 * License: CC BY-NC-SA 3.0 IGO
 *
 * Usage:
 *   npx ts-node importers/who-importer.ts [--limit N] [--api]
 *
 * Without --api, uses the embedded curated dataset only.
 *
 * Output:
 *   seed-data/claims/who-health.json
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

const GHO_API = 'https://ghoapi.azureedge.net/api';
const DEFAULT_LIMIT = 1000;
const REQUEST_DELAY_MS = 300;

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const useApi = process.argv.includes('--api');

// ---------------------------------------------------------------------------
// Embedded curated health facts
// ---------------------------------------------------------------------------

interface HealthFact {
  content: string;
  confidence: number;
  tags: string[];
  source: string;
}

const CURATED_HEALTH_FACTS: HealthFact[] = [
  // Life expectancy
  { content: 'Global life expectancy at birth increased from 66.8 years in 2000 to 73.3 years in 2019.', confidence: 0.95, tags: ['health', 'life-expectancy', 'global'], source: 'https://www.who.int/data/gho/data/indicators/indicator-details/GHO/life-expectancy-at-birth-(years)' },
  { content: 'Japan has one of the highest life expectancies in the world at approximately 84.3 years (2019).', confidence: 0.95, tags: ['health', 'life-expectancy', 'japan'], source: 'https://www.who.int/data/gho/data/indicators/indicator-details/GHO/life-expectancy-at-birth-(years)' },
  { content: 'The life expectancy gap between high-income and low-income countries is approximately 18 years.', confidence: 0.90, tags: ['health', 'life-expectancy', 'inequality'], source: 'https://www.who.int/data/gho/data/themes/mortality-and-global-health-estimates' },
  { content: 'Healthy life expectancy (HALE) at birth globally was 63.3 years in 2019.', confidence: 0.93, tags: ['health', 'hale', 'global'], source: 'https://www.who.int/data/gho/data/indicators/indicator-details/GHO/healthy-life-expectancy-(hale)-at-birth-(years)' },

  // Mortality and disease
  { content: 'Cardiovascular diseases are the leading cause of death globally, accounting for approximately 17.9 million deaths per year.', confidence: 0.95, tags: ['health', 'cardiovascular', 'mortality', 'ncd'], source: 'https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)' },
  { content: 'Cancer caused approximately 10 million deaths worldwide in 2020.', confidence: 0.94, tags: ['health', 'cancer', 'mortality'], source: 'https://www.who.int/news-room/fact-sheets/detail/cancer' },
  { content: 'Diabetes prevalence has risen from 4.7% of adults in 1980 to 8.5% in 2014 globally.', confidence: 0.92, tags: ['health', 'diabetes', 'prevalence', 'ncd'], source: 'https://www.who.int/news-room/fact-sheets/detail/diabetes' },
  { content: 'Malaria caused an estimated 619,000 deaths in 2021, with 95% occurring in the WHO African Region.', confidence: 0.93, tags: ['health', 'malaria', 'africa', 'infectious'], source: 'https://www.who.int/news-room/fact-sheets/detail/malaria' },
  { content: 'Tuberculosis (TB) caused an estimated 1.3 million deaths among HIV-negative people in 2022.', confidence: 0.92, tags: ['health', 'tuberculosis', 'infectious', 'mortality'], source: 'https://www.who.int/news-room/fact-sheets/detail/tuberculosis' },
  { content: 'HIV/AIDS has claimed approximately 40.4 million lives since the beginning of the epidemic.', confidence: 0.93, tags: ['health', 'hiv', 'aids', 'mortality'], source: 'https://www.who.int/data/gho/data/themes/hiv-aids' },

  // Maternal and child health
  { content: 'The global under-5 mortality rate declined from 93 per 1,000 live births in 1990 to 37 in 2022.', confidence: 0.94, tags: ['health', 'child-mortality', 'progress'], source: 'https://www.who.int/data/gho/data/themes/topics/topic-details/GHO/child-mortality-and-causes-of-death' },
  { content: 'The global maternal mortality ratio was approximately 223 per 100,000 live births in 2020.', confidence: 0.92, tags: ['health', 'maternal-mortality', 'pregnancy'], source: 'https://www.who.int/data/gho/data/themes/topics/sdg-target-3-1-maternal-mortality' },
  { content: 'Approximately 5 million children under age 5 died in 2021, mostly from preventable causes.', confidence: 0.93, tags: ['health', 'child-mortality', 'preventable'], source: 'https://www.who.int/news-room/fact-sheets/detail/children-reducing-mortality' },
  { content: 'Global immunization coverage for the third dose of diphtheria-tetanus-pertussis (DTP3) was 84% in 2022.', confidence: 0.94, tags: ['health', 'vaccination', 'immunization'], source: 'https://www.who.int/data/gho/data/themes/topics/immunization' },

  // Mental health
  { content: 'Depression is the leading cause of disability worldwide, affecting approximately 280 million people.', confidence: 0.90, tags: ['health', 'depression', 'mental-health', 'disability'], source: 'https://www.who.int/news-room/fact-sheets/detail/depression' },
  { content: 'Approximately 700,000 people die by suicide every year worldwide.', confidence: 0.92, tags: ['health', 'suicide', 'mental-health', 'mortality'], source: 'https://www.who.int/news-room/fact-sheets/detail/suicide' },
  { content: 'One in every eight people globally lives with a mental health condition.', confidence: 0.88, tags: ['health', 'mental-health', 'prevalence'], source: 'https://www.who.int/news-room/fact-sheets/detail/mental-disorders' },

  // Nutrition
  { content: 'Approximately 735 million people faced chronic hunger in 2022.', confidence: 0.90, tags: ['health', 'hunger', 'nutrition', 'food-security'], source: 'https://www.who.int/data/gho/data/themes/topics/joint-child-malnutrition-estimates-unicef-who-wb' },
  { content: 'Over 1.9 billion adults worldwide were overweight in 2016, of whom over 650 million were obese.', confidence: 0.92, tags: ['health', 'obesity', 'nutrition', 'ncd'], source: 'https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight' },
  { content: 'An estimated 149 million children under 5 were stunted (too short for age) in 2020.', confidence: 0.91, tags: ['health', 'stunting', 'child-nutrition', 'malnutrition'], source: 'https://www.who.int/data/gho/data/themes/topics/joint-child-malnutrition-estimates-unicef-who-wb' },

  // Environmental health
  { content: 'Ambient air pollution caused an estimated 4.2 million premature deaths worldwide in 2019.', confidence: 0.90, tags: ['health', 'air-pollution', 'environment', 'mortality'], source: 'https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health' },
  { content: 'Household air pollution from cooking with solid fuels caused approximately 3.2 million deaths per year.', confidence: 0.88, tags: ['health', 'air-pollution', 'household', 'cooking'], source: 'https://www.who.int/news-room/fact-sheets/detail/household-air-pollution-and-health' },
  { content: 'Approximately 2 billion people globally use a drinking water source contaminated with feces.', confidence: 0.88, tags: ['health', 'water', 'sanitation', 'wash'], source: 'https://www.who.int/news-room/fact-sheets/detail/drinking-water' },
  { content: 'Unsafe water, sanitation, and hygiene cause approximately 829,000 diarrheal deaths annually.', confidence: 0.89, tags: ['health', 'wash', 'diarrhea', 'mortality'], source: 'https://www.who.int/data/gho/data/themes/topics/water-sanitation-and-hygiene' },

  // Health systems
  { content: 'At least half of the world\'s population lacks access to essential health services.', confidence: 0.88, tags: ['health', 'access', 'uhc', 'health-systems'], source: 'https://www.who.int/data/gho/data/themes/topics/universal-health-coverage' },
  { content: 'There is a global shortage of approximately 15 million health workers, primarily in low- and middle-income countries.', confidence: 0.85, tags: ['health', 'workforce', 'health-systems', 'inequality'], source: 'https://www.who.int/data/gho/data/themes/topics/health-workforce' },
];

// ---------------------------------------------------------------------------
// WHO GHO OData API
// ---------------------------------------------------------------------------

interface GhoIndicator {
  IndicatorCode: string;
  IndicatorName: string;
}

interface GhoDataValue {
  Id: number;
  IndicatorCode: string;
  SpatialDim: string;     // country code
  TimeDim: string;        // year
  Dim1: string | null;    // e.g., sex
  NumericValue: number | null;
  Value: string;
}

const KEY_INDICATORS = [
  { code: 'WHOSIS_000001', name: 'Life expectancy at birth', unit: 'years' },
  { code: 'WHOSIS_000015', name: 'Neonatal mortality rate', unit: 'per 1000 live births' },
  { code: 'MDG_0000000001', name: 'Under-5 mortality rate', unit: 'per 1000 live births' },
  { code: 'WHS2_131', name: 'Maternal mortality ratio', unit: 'per 100,000 live births' },
];

async function fetchIndicatorData(indicatorCode: string, topN: number): Promise<GhoDataValue[]> {
  const url = `${GHO_API}/${indicatorCode}?$filter=TimeDim eq 2019 and Dim1 eq 'BTSX'&$top=${topN}&$orderby=NumericValue desc`;

  const resp = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`GHO API error: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.value ?? [];
}

function ghoValueToClaim(
  indicatorName: string,
  unit: string,
  row: GhoDataValue,
): Claim | null {
  if (row.NumericValue == null) return null;

  const country = row.SpatialDim;
  const year = row.TimeDim;
  const value = row.NumericValue;

  return buildClaim({
    source: 'who',
    content: `The ${indicatorName.toLowerCase()} in ${country} was ${value.toFixed(1)} ${unit} in ${year}, according to WHO Global Health Observatory data.`,
    classification: SOURCE_ENM.institutional,
    claimType: 'Fact',
    confidence: 0.90,
    sources: [`https://www.who.int/data/gho/data/indicators/indicator-details/GHO/${row.IndicatorCode}`],
    tags: ['health', 'who', indicatorName.toLowerCase().replace(/\s+/g, '-'), country.toLowerCase()],
  });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== WHO Global Health Observatory → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  if (useApi) console.log('Mode: API (GHO OData)');
  else console.log('Mode: Curated dataset (pass --api for live data)');
  console.log();

  const allClaims: Claim[] = [];

  // Curated facts
  console.log('--- Curated health facts ---');
  for (const fact of CURATED_HEALTH_FACTS) {
    allClaims.push(
      buildClaim({
        source: 'who',
        content: fact.content,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: fact.confidence,
        sources: [fact.source],
        tags: fact.tags,
      }),
    );
  }
  console.log(`  ${CURATED_HEALTH_FACTS.length} curated claims`);

  // API data
  if (useApi && allClaims.length < limit) {
    console.log();
    console.log('--- WHO GHO API indicators ---');

    const perIndicator = Math.ceil((limit - allClaims.length) / KEY_INDICATORS.length);

    for (const indicator of KEY_INDICATORS) {
      if (allClaims.length >= limit) break;

      console.log(`  ${indicator.name}...`);
      try {
        const rows = await fetchIndicatorData(indicator.code, perIndicator);
        let converted = 0;
        for (const row of rows) {
          const claim = ghoValueToClaim(indicator.name, indicator.unit, row);
          if (claim) {
            allClaims.push(claim);
            converted++;
          }
        }
        console.log(`    ${rows.length} rows → ${converted} claims`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(`    [ERROR] ${msg}`);
      }

      await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
    }
  }

  console.log();

  const valid = filterValid(allClaims.slice(0, limit));
  const outFile = writeSeedFile('who-health.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
