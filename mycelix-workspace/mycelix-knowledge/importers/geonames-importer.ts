#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * GeoNames → Mycelix Claims Importer
 *
 * Parses GeoNames data (cities, countries) and converts geographic facts
 * into Claims with E4/N0/M3 classification (established, objective, transformative).
 *
 * GeoNames data: https://download.geonames.org/export/dump/
 * License: Creative Commons Attribution 4.0
 *
 * Uses the cities15000.txt file (all cities with population > 15,000).
 * Tab-separated, columns defined at: https://download.geonames.org/export/dump/readme.txt
 *
 * Usage:
 *   npx ts-node importers/geonames-importer.ts <path-to-cities15000.txt> [--limit N]
 *
 * If no file is provided, generates claims from an embedded curated dataset
 * of major world cities, countries, and geographic features.
 *
 * Output:
 *   seed-data/claims/geonames-major.json
 */

import * as fs from 'fs';
import * as readline from 'readline';
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

const DEFAULT_LIMIT = 5000;

const geonamesPath = process.argv[2] && !process.argv[2].startsWith('--')
  ? process.argv[2]
  : null;

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

// ---------------------------------------------------------------------------
// GeoNames TSV columns (cities15000.txt)
// ---------------------------------------------------------------------------
// 0: geonameid, 1: name, 2: asciiname, 3: alternatenames, 4: latitude,
// 5: longitude, 6: feature class, 7: feature code, 8: country code,
// 9: cc2, 10: admin1 code, 11: admin2 code, 12: admin3 code, 13: admin4 code,
// 14: population, 15: elevation, 16: dem, 17: timezone, 18: modification date

const COL = {
  GEONAMEID: 0,
  NAME: 1,
  LATITUDE: 4,
  LONGITUDE: 5,
  FEATURE_CLASS: 6,
  FEATURE_CODE: 7,
  COUNTRY_CODE: 8,
  POPULATION: 14,
  ELEVATION: 15,
  TIMEZONE: 17,
} as const;

// ---------------------------------------------------------------------------
// Country code → name mapping (ISO 3166-1 alpha-2, common subset)
// ---------------------------------------------------------------------------

const COUNTRY_NAMES: Record<string, string> = {
  AF: 'Afghanistan', AL: 'Albania', DZ: 'Algeria', AR: 'Argentina', AU: 'Australia',
  AT: 'Austria', BD: 'Bangladesh', BE: 'Belgium', BR: 'Brazil', CA: 'Canada',
  CL: 'Chile', CN: 'China', CO: 'Colombia', CG: 'Congo', HR: 'Croatia',
  CU: 'Cuba', CZ: 'Czechia', DK: 'Denmark', EG: 'Egypt', ET: 'Ethiopia',
  FI: 'Finland', FR: 'France', DE: 'Germany', GH: 'Ghana', GR: 'Greece',
  HU: 'Hungary', IN: 'India', ID: 'Indonesia', IR: 'Iran', IQ: 'Iraq',
  IE: 'Ireland', IL: 'Israel', IT: 'Italy', JP: 'Japan', JO: 'Jordan',
  KE: 'Kenya', KR: 'South Korea', KW: 'Kuwait', MY: 'Malaysia', MX: 'Mexico',
  MA: 'Morocco', MM: 'Myanmar', NP: 'Nepal', NL: 'Netherlands', NZ: 'New Zealand',
  NG: 'Nigeria', NO: 'Norway', PK: 'Pakistan', PE: 'Peru', PH: 'Philippines',
  PL: 'Poland', PT: 'Portugal', RO: 'Romania', RU: 'Russia', SA: 'Saudi Arabia',
  SG: 'Singapore', ZA: 'South Africa', ES: 'Spain', SE: 'Sweden', CH: 'Switzerland',
  TW: 'Taiwan', TH: 'Thailand', TR: 'Turkey', UA: 'Ukraine', AE: 'United Arab Emirates',
  GB: 'United Kingdom', US: 'United States', VN: 'Vietnam',
};

function countryName(code: string): string {
  return COUNTRY_NAMES[code] || code;
}

// ---------------------------------------------------------------------------
// GeoNames TSV parser
// ---------------------------------------------------------------------------

async function parseGeoNamesTsv(filepath: string, maxClaims: number): Promise<Claim[]> {
  const claims: Claim[] = [];

  const fileStream = fs.createReadStream(filepath, { encoding: 'utf-8' });
  const rl = readline.createInterface({
    input: fileStream,
    crlfDelay: Infinity,
  });

  let parsed = 0;

  for await (const line of rl) {
    if (parsed >= maxClaims) break;

    const cols = line.split('\t');
    if (cols.length < 19) continue;

    const name = cols[COL.NAME];
    const cc = cols[COL.COUNTRY_CODE];
    const pop = parseInt(cols[COL.POPULATION], 10);
    const lat = parseFloat(cols[COL.LATITUDE]);
    const lon = parseFloat(cols[COL.LONGITUDE]);
    const elev = cols[COL.ELEVATION] ? parseInt(cols[COL.ELEVATION], 10) : null;
    const tz = cols[COL.TIMEZONE];
    const geoId = cols[COL.GEONAMEID];

    if (!name || !cc || isNaN(pop)) continue;

    const country = countryName(cc);
    const sourceUri = `https://www.geonames.org/${geoId}`;

    // Population claim
    if (pop > 0) {
      claims.push(
        buildClaim({
          source: 'geonames',
          content: `${name} is a city in ${country} with a population of approximately ${pop.toLocaleString('en-US')}.`,
          classification: SOURCE_ENM.geonames,
          claimType: 'Fact',
          confidence: 0.92,
          sources: [sourceUri],
          tags: ['geography', 'cities', 'population', cc.toLowerCase()],
        }),
      );
      parsed++;
    }

    // Coordinates claim (for cities > 500K)
    if (pop > 500000 && parsed < maxClaims) {
      claims.push(
        buildClaim({
          source: 'geonames',
          content: `${name} (${country}) is located at approximately ${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E.`,
          classification: SOURCE_ENM.geonames,
          claimType: 'Fact',
          confidence: 0.98,
          sources: [sourceUri],
          tags: ['geography', 'coordinates', cc.toLowerCase()],
        }),
      );
      parsed++;
    }

    // Timezone claim (for cities > 1M)
    if (pop > 1000000 && tz && parsed < maxClaims) {
      claims.push(
        buildClaim({
          source: 'geonames',
          content: `${name} (${country}) is in the ${tz} timezone.`,
          classification: SOURCE_ENM.geonames,
          claimType: 'Fact',
          confidence: 0.95,
          sources: [sourceUri],
          tags: ['geography', 'timezone', cc.toLowerCase()],
        }),
      );
      parsed++;
    }

    if (parsed % 2000 === 0 && parsed > 0) {
      console.log(`  Parsed ${parsed} claims...`);
    }
  }

  return claims;
}

// ---------------------------------------------------------------------------
// Embedded curated dataset (used when no file provided)
// ---------------------------------------------------------------------------

interface CuratedCity {
  name: string;
  country: string;
  population: number;
  lat: number;
  lon: number;
}

const CURATED_CITIES: CuratedCity[] = [
  { name: 'Tokyo', country: 'Japan', population: 13960000, lat: 35.68, lon: 139.69 },
  { name: 'Delhi', country: 'India', population: 32940000, lat: 28.61, lon: 77.23 },
  { name: 'Shanghai', country: 'China', population: 28520000, lat: 31.23, lon: 121.47 },
  { name: 'São Paulo', country: 'Brazil', population: 22430000, lat: -23.55, lon: -46.63 },
  { name: 'Mexico City', country: 'Mexico', population: 21920000, lat: 19.43, lon: -99.13 },
  { name: 'Cairo', country: 'Egypt', population: 21750000, lat: 30.04, lon: 31.24 },
  { name: 'Mumbai', country: 'India', population: 21670000, lat: 19.08, lon: 72.88 },
  { name: 'Beijing', country: 'China', population: 21540000, lat: 39.90, lon: 116.40 },
  { name: 'Dhaka', country: 'Bangladesh', population: 22480000, lat: 23.81, lon: 90.41 },
  { name: 'Osaka', country: 'Japan', population: 19060000, lat: 34.69, lon: 135.50 },
  { name: 'New York City', country: 'United States', population: 18820000, lat: 40.71, lon: -74.01 },
  { name: 'Karachi', country: 'Pakistan', population: 16840000, lat: 24.86, lon: 67.01 },
  { name: 'Buenos Aires', country: 'Argentina', population: 15370000, lat: -34.60, lon: -58.38 },
  { name: 'Istanbul', country: 'Turkey', population: 15640000, lat: 41.01, lon: 28.98 },
  { name: 'Lagos', country: 'Nigeria', population: 15390000, lat: 6.52, lon: 3.38 },
  { name: 'Kinshasa', country: 'DR Congo', population: 14970000, lat: -4.44, lon: 15.27 },
  { name: 'Manila', country: 'Philippines', population: 14400000, lat: 14.60, lon: 120.98 },
  { name: 'Rio de Janeiro', country: 'Brazil', population: 13540000, lat: -22.91, lon: -43.17 },
  { name: 'Guangzhou', country: 'China', population: 13640000, lat: 23.13, lon: 113.26 },
  { name: 'Moscow', country: 'Russia', population: 12640000, lat: 55.76, lon: 37.62 },
  { name: 'Paris', country: 'France', population: 11020000, lat: 48.86, lon: 2.35 },
  { name: 'London', country: 'United Kingdom', population: 9540000, lat: 51.51, lon: -0.13 },
  { name: 'Lima', country: 'Peru', population: 10880000, lat: -12.05, lon: -77.04 },
  { name: 'Bangkok', country: 'Thailand', population: 10720000, lat: 13.76, lon: 100.50 },
  { name: 'Seoul', country: 'South Korea', population: 9776000, lat: 37.57, lon: 126.98 },
  { name: 'Jakarta', country: 'Indonesia', population: 10560000, lat: -6.21, lon: 106.85 },
  { name: 'Tehran', country: 'Iran', population: 9260000, lat: 35.69, lon: 51.39 },
  { name: 'Berlin', country: 'Germany', population: 3645000, lat: 52.52, lon: 13.41 },
  { name: 'Sydney', country: 'Australia', population: 5310000, lat: -33.87, lon: 151.21 },
  { name: 'Toronto', country: 'Canada', population: 6200000, lat: 43.65, lon: -79.38 },
  { name: 'Singapore', country: 'Singapore', population: 5690000, lat: 1.35, lon: 103.82 },
  { name: 'Nairobi', country: 'Kenya', population: 4920000, lat: -1.29, lon: 36.82 },
  { name: 'Johannesburg', country: 'South Africa', population: 5780000, lat: -26.20, lon: 28.04 },
  { name: 'Rome', country: 'Italy', population: 4260000, lat: 41.90, lon: 12.50 },
  { name: 'Madrid', country: 'Spain', population: 6640000, lat: 40.42, lon: -3.70 },
  { name: 'Riyadh', country: 'Saudi Arabia', population: 7680000, lat: 24.69, lon: 46.72 },
  { name: 'Taipei', country: 'Taiwan', population: 2650000, lat: 25.03, lon: 121.57 },
  { name: 'Bogotá', country: 'Colombia', population: 11340000, lat: 4.71, lon: -74.07 },
  { name: 'Warsaw', country: 'Poland', population: 1790000, lat: 52.23, lon: 21.01 },
  { name: 'Hanoi', country: 'Vietnam', population: 8050000, lat: 21.03, lon: 105.85 },
];

function curatedToClaims(): Claim[] {
  const claims: Claim[] = [];

  for (const city of CURATED_CITIES) {
    // Population claim
    claims.push(
      buildClaim({
        source: 'geonames',
        content: `${city.name} is a major city in ${city.country} with a metropolitan population of approximately ${city.population.toLocaleString('en-US')}.`,
        classification: SOURCE_ENM.geonames,
        claimType: 'Fact',
        confidence: 0.90,
        sources: [`https://www.geonames.org/search?q=${encodeURIComponent(city.name)}`],
        tags: ['geography', 'cities', 'population'],
      }),
    );

    // Coordinates claim
    claims.push(
      buildClaim({
        source: 'geonames',
        content: `${city.name} (${city.country}) is located at approximately ${city.lat.toFixed(2)}° latitude, ${city.lon.toFixed(2)}° longitude.`,
        classification: SOURCE_ENM.geonames,
        claimType: 'Fact',
        confidence: 0.98,
        sources: [`https://www.geonames.org/search?q=${encodeURIComponent(city.name)}`],
        tags: ['geography', 'coordinates'],
      }),
    );
  }

  return claims;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== GeoNames → Mycelix Claims Importer ===');

  let claims: Claim[];

  if (geonamesPath) {
    console.log(`Input:  ${geonamesPath}`);
    console.log(`Limit:  ${limit}`);
    console.log();

    if (!fs.existsSync(geonamesPath)) {
      console.error(`File not found: ${geonamesPath}`);
      console.error('Download from: https://download.geonames.org/export/dump/cities15000.zip');
      process.exit(1);
    }

    console.log('--- Parsing GeoNames TSV ---');
    claims = await parseGeoNamesTsv(geonamesPath, limit);
  } else {
    console.log('No input file — using embedded curated dataset (40 major cities)');
    console.log();
    claims = curatedToClaims();
  }

  console.log(`  Generated ${claims.length} raw claims`);
  console.log();

  const valid = filterValid(claims);
  const outFile = writeSeedFile('geonames-major.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
