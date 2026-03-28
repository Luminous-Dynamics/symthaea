#!/usr/bin/env ts-node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * NOAA Climate → Mycelix Claims Importer
 *
 * Fetches climate data from NOAA's National Centers for Environmental
 * Information (NCEI) Climate Data Online API and converts to Claims.
 *
 * Also includes an embedded curated dataset of key climate facts that
 * can be used without an API token.
 *
 * NOAA CDO API: https://www.ncei.noaa.gov/cdo-web/webservices/v2
 * License: Public domain (US Government work)
 *
 * Usage:
 *   npx ts-node importers/noaa-importer.ts [--token TOKEN] [--limit N]
 *
 * Without --token, uses the embedded curated dataset only.
 *
 * Output:
 *   seed-data/claims/noaa-climate.json
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

const NOAA_API = 'https://www.ncei.noaa.gov/cdo-web/api/v2';
const DEFAULT_LIMIT = 1000;
const REQUEST_DELAY_MS = 300;

const limit = (() => {
  const idx = process.argv.indexOf('--limit');
  return idx !== -1 ? parseInt(process.argv[idx + 1], 10) : DEFAULT_LIMIT;
})();

const apiToken = (() => {
  const idx = process.argv.indexOf('--token');
  return idx !== -1 ? process.argv[idx + 1] : null;
})();

// ---------------------------------------------------------------------------
// Embedded curated climate facts
// ---------------------------------------------------------------------------

interface ClimateFact {
  content: string;
  confidence: number;
  tags: string[];
  source: string;
}

const CURATED_CLIMATE_FACTS: ClimateFact[] = [
  // Global temperature records
  { content: 'The global average surface temperature has increased by approximately 1.1°C since the pre-industrial era (1850-1900).', confidence: 0.95, tags: ['climate', 'temperature', 'global-warming'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series' },
  { content: 'The year 2023 was the warmest year on record according to NOAA global temperature data.', confidence: 0.98, tags: ['climate', 'temperature', 'records'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series' },
  { content: 'The 10 warmest years on record have all occurred since 2010.', confidence: 0.97, tags: ['climate', 'temperature', 'trends'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series' },
  { content: 'Global sea level has risen approximately 21-24 cm since 1880, with the rate of rise accelerating.', confidence: 0.93, tags: ['climate', 'sea-level', 'oceans'], source: 'https://www.climate.gov/news-features/understanding-climate/climate-change-global-sea-level' },
  { content: 'Arctic sea ice extent has declined approximately 13% per decade since satellite records began in 1979.', confidence: 0.94, tags: ['climate', 'arctic', 'sea-ice'], source: 'https://nsidc.org/arcticseaicenews/' },

  // CO2 and greenhouse gases
  { content: 'Atmospheric CO2 concentration reached 421 ppm in 2023, the highest in at least 800,000 years.', confidence: 0.98, tags: ['climate', 'co2', 'greenhouse-gases'], source: 'https://gml.noaa.gov/ccgg/trends/' },
  { content: 'Pre-industrial atmospheric CO2 concentration was approximately 280 ppm.', confidence: 0.97, tags: ['climate', 'co2', 'historical'], source: 'https://gml.noaa.gov/ccgg/trends/' },
  { content: 'Methane (CH4) concentration in the atmosphere reached 1,922 ppb in 2023.', confidence: 0.95, tags: ['climate', 'methane', 'greenhouse-gases'], source: 'https://gml.noaa.gov/ccgg/trends_ch4/' },
  { content: 'The ocean absorbs approximately 30% of anthropogenic CO2 emissions, leading to ocean acidification.', confidence: 0.92, tags: ['climate', 'oceans', 'co2', 'acidification'], source: 'https://oceanservice.noaa.gov/facts/acidification.html' },

  // Extreme weather
  { content: 'The number of billion-dollar weather and climate disasters in the US has been increasing, with 28 events in 2023.', confidence: 0.96, tags: ['climate', 'extreme-weather', 'disasters', 'united-states'], source: 'https://www.ncei.noaa.gov/access/billions/' },
  { content: 'Global ocean heat content has been increasing consistently since the 1950s and reached record levels in 2023.', confidence: 0.95, tags: ['climate', 'ocean-heat', 'trends'], source: 'https://www.ncei.noaa.gov/access/global-ocean-heat-content/' },
  { content: 'The frequency of Category 4 and 5 hurricanes in the Atlantic has increased since the 1980s.', confidence: 0.85, tags: ['climate', 'hurricanes', 'extreme-weather'], source: 'https://www.gfdl.noaa.gov/global-warming-and-hurricanes/' },

  // Regional climate
  { content: 'The contiguous United States has warmed approximately 1.0°C since the beginning of the 20th century.', confidence: 0.94, tags: ['climate', 'temperature', 'united-states'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series' },
  { content: 'Alaska has warmed at approximately twice the rate of the contiguous United States.', confidence: 0.92, tags: ['climate', 'temperature', 'alaska', 'arctic'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/50' },
  { content: 'Annual precipitation in the contiguous US has increased by approximately 4% since 1901.', confidence: 0.88, tags: ['climate', 'precipitation', 'united-states'], source: 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/national/time-series' },

  // Ice and cryosphere
  { content: 'The Greenland ice sheet has been losing mass at an accelerating rate, losing approximately 270 billion metric tons per year since 2002.', confidence: 0.93, tags: ['climate', 'greenland', 'ice-sheet', 'sea-level'], source: 'https://climate.nasa.gov/vital-signs/ice-sheets/' },
  { content: 'Antarctic ice sheet mass loss has tripled in the last decade compared to the previous decade.', confidence: 0.90, tags: ['climate', 'antarctica', 'ice-sheet'], source: 'https://climate.nasa.gov/vital-signs/ice-sheets/' },
  { content: 'Global glacier volume has decreased significantly since the 1970s, with the rate of loss accelerating.', confidence: 0.92, tags: ['climate', 'glaciers', 'cryosphere'], source: 'https://wgms.ch/global-glacier-state/' },

  // Ocean patterns
  { content: 'El Niño events cause temporary increases of 0.1-0.3°C in global average surface temperature.', confidence: 0.90, tags: ['climate', 'enso', 'el-nino', 'oceans'], source: 'https://www.climate.gov/enso' },
  { content: 'The Atlantic Meridional Overturning Circulation (AMOC) has shown signs of weakening since the mid-20th century.', confidence: 0.80, tags: ['climate', 'amoc', 'ocean-circulation'], source: 'https://www.nature.com/articles/s41558-021-01097-4' },

  // Historical climate
  { content: 'The Medieval Warm Period (approximately 900-1300 CE) was characterized by regional warmth in parts of the Northern Hemisphere.', confidence: 0.85, tags: ['climate', 'historical', 'medieval'], source: 'https://www.ncei.noaa.gov/access/paleo-search/' },
  { content: 'The Little Ice Age (approximately 1300-1850 CE) was a period of regional cooling, particularly in Europe and North America.', confidence: 0.87, tags: ['climate', 'historical', 'little-ice-age'], source: 'https://www.ncei.noaa.gov/access/paleo-search/' },
  { content: 'Ice core records from Antarctica show CO2 levels fluctuated between 180 and 280 ppm over the past 800,000 years before industrialization.', confidence: 0.96, tags: ['climate', 'ice-cores', 'co2', 'paleoclimate'], source: 'https://www.ncei.noaa.gov/access/paleo-search/' },

  // Biodiversity impacts
  { content: 'Coral bleaching events have become approximately 5 times more frequent since the 1980s due to ocean warming.', confidence: 0.88, tags: ['climate', 'coral-reefs', 'biodiversity', 'oceans'], source: 'https://coralreefwatch.noaa.gov/' },
  { content: 'Spring plant blooming dates have advanced by an average of 2-3 days per decade across the Northern Hemisphere since the 1970s.', confidence: 0.85, tags: ['climate', 'phenology', 'biodiversity', 'seasons'], source: 'https://www.usanpn.org/' },
];

// ---------------------------------------------------------------------------
// NOAA CDO API (used when token is provided)
// ---------------------------------------------------------------------------

interface NoaaStation {
  id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation: number;
  mindate: string;
  maxdate: string;
}

async function fetchNoaaStations(token: string, maxStations: number): Promise<NoaaStation[]> {
  const url = `${NOAA_API}/stations?datasetid=GHCND&limit=${Math.min(maxStations, 1000)}&sortfield=name`;
  const resp = await fetch(url, {
    headers: { token, Accept: 'application/json' },
  });

  if (!resp.ok) {
    throw new Error(`NOAA API error: ${resp.status} ${resp.statusText}`);
  }

  const data = await resp.json();
  return data.results ?? [];
}

interface NoaaDataPoint {
  date: string;
  datatype: string;
  station: string;
  value: number;
}

async function fetchStationData(
  token: string,
  stationId: string,
  startDate: string,
  endDate: string,
): Promise<NoaaDataPoint[]> {
  const url = `${NOAA_API}/data?datasetid=GHCND&stationid=${stationId}&startdate=${startDate}&enddate=${endDate}&datatypeid=TMAX,TMIN,PRCP&limit=365`;
  const resp = await fetch(url, {
    headers: { token, Accept: 'application/json' },
  });

  if (!resp.ok) return [];

  const data = await resp.json();
  return data.results ?? [];
}

function stationDataToClaims(station: NoaaStation, dataPoints: NoaaDataPoint[]): Claim[] {
  const claims: Claim[] = [];

  // Compute annual averages
  const tmax = dataPoints.filter((d) => d.datatype === 'TMAX');
  const tmin = dataPoints.filter((d) => d.datatype === 'TMIN');
  const prcp = dataPoints.filter((d) => d.datatype === 'PRCP');

  if (tmax.length > 30) {
    const avgMax = tmax.reduce((s, d) => s + d.value, 0) / tmax.length / 10; // GHCND stores in tenths of °C
    claims.push(
      buildClaim({
        source: 'noaa',
        content: `The average daily maximum temperature at ${station.name} is approximately ${avgMax.toFixed(1)}°C based on recent GHCND records.`,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: 0.90,
        sources: [`https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/${station.id}/detail`],
        tags: ['climate', 'temperature', 'weather-station'],
      }),
    );
  }

  if (tmin.length > 30) {
    const avgMin = tmin.reduce((s, d) => s + d.value, 0) / tmin.length / 10;
    claims.push(
      buildClaim({
        source: 'noaa',
        content: `The average daily minimum temperature at ${station.name} is approximately ${avgMin.toFixed(1)}°C based on recent GHCND records.`,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: 0.90,
        sources: [`https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/${station.id}/detail`],
        tags: ['climate', 'temperature', 'weather-station'],
      }),
    );
  }

  if (prcp.length > 30) {
    const totalPrcp = prcp.reduce((s, d) => s + d.value, 0) / 10; // tenths of mm → mm
    claims.push(
      buildClaim({
        source: 'noaa',
        content: `The total annual precipitation at ${station.name} is approximately ${totalPrcp.toFixed(0)} mm based on recent GHCND records.`,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: 0.88,
        sources: [`https://www.ncei.noaa.gov/cdo-web/datasets/GHCND/stations/${station.id}/detail`],
        tags: ['climate', 'precipitation', 'weather-station'],
      }),
    );
  }

  return claims;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== NOAA Climate → Mycelix Claims Importer ===');
  console.log(`Target: ~${limit} claims`);
  if (apiToken) console.log('Mode: API (CDO token provided)');
  else console.log('Mode: Curated dataset (no API token — pass --token TOKEN for live data)');
  console.log();

  const allClaims: Claim[] = [];

  // Always include curated facts
  console.log('--- Curated climate facts ---');
  for (const fact of CURATED_CLIMATE_FACTS) {
    allClaims.push(
      buildClaim({
        source: 'noaa',
        content: fact.content,
        classification: SOURCE_ENM.institutional,
        claimType: 'Fact',
        confidence: fact.confidence,
        sources: [fact.source],
        tags: fact.tags,
      }),
    );
  }
  console.log(`  ${CURATED_CLIMATE_FACTS.length} curated claims`);

  // Fetch from API if token provided and we need more
  if (apiToken && allClaims.length < limit) {
    console.log();
    console.log('--- NOAA CDO API station data ---');

    try {
      const stationsNeeded = Math.ceil((limit - allClaims.length) / 3);
      const stations = await fetchNoaaStations(apiToken, stationsNeeded);
      console.log(`  Found ${stations.length} stations`);

      const endDate = '2024-12-31';
      const startDate = '2024-01-01';

      for (const station of stations) {
        if (allClaims.length >= limit) break;

        try {
          const data = await fetchStationData(apiToken, station.id, startDate, endDate);
          const claims = stationDataToClaims(station, data);
          allClaims.push(...claims);

          if (claims.length > 0) {
            console.log(`  ${station.name}: ${claims.length} claims`);
          }
        } catch {
          // Continue on per-station errors
        }

        await new Promise((r) => setTimeout(r, REQUEST_DELAY_MS));
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`  [ERROR] ${msg}`);
    }
  }

  console.log();

  const valid = filterValid(allClaims.slice(0, limit));
  const outFile = writeSeedFile('noaa-climate.json', valid);

  console.log();
  console.log(`=== Done: ${valid.length} valid claims → ${outFile} ===`);
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
