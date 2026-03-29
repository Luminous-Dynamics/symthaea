#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * CSV Adapter for Mycelix Supply Chain
 *
 * Reads CSV files with supply chain events and ingests them via the API
 */

import { SupplyChainClient, SupplyEventVC, EventType } from '@mycelix/supplychain-sdk';
import { parse } from 'csv-parse';
import { createReadStream } from 'fs';
import { program } from 'commander';

interface CsvRow {
  eventType: EventType;
  issuer: string;
  productId: string;
  batchId: string;
  prevBatchIds?: string;
  quantity: string;
  unit: string;
  facilityId: string;
  facilityName: string;
  timestamp?: string;
}

program
  .name('csv-adapter')
  .description('Ingest supply chain events from CSV files')
  .requiredOption('-f, --file <path>', 'CSV file to ingest')
  .option('-u, --url <url>', 'API base URL', 'http://localhost:8080')
  .option('-d, --dry-run', 'Parse CSV without sending to API')
  .parse();

const options = program.opts();

async function ingestCsv(filePath: string, apiUrl: string, dryRun = false) {
  const client = new SupplyChainClient({ baseUrl: apiUrl });

  // Test connection
  if (!dryRun) {
    try {
      const health = await client.health();
      console.log(`Connected to API (version ${health.version})`);
    } catch (error) {
      console.error('Failed to connect to API:', error);
      process.exit(1);
    }
  }

  const parser = createReadStream(filePath).pipe(
    parse({
      columns: true,
      skip_empty_lines: true,
      trim: true,
    })
  );

  let processed = 0;
  let errors = 0;

  for await (const row of parser as AsyncIterable<CsvRow>) {
    try {
      const event: SupplyEventVC = {
        '@context': [
          'https://www.w3.org/2018/credentials/v1',
          'https://mycelix.org/contexts/supply-chain/v1',
        ],
        type: ['VerifiableCredential', 'SupplyChainEvent'],
        issuer: row.issuer,
        issuanceDate: new Date().toISOString(),
        credentialSubject: {
          eventType: row.eventType,
          productId: row.productId,
          batchId: row.batchId,
          prevBatchIds: row.prevBatchIds ? row.prevBatchIds.split(',').map(s => s.trim()) : undefined,
          quantity: parseFloat(row.quantity),
          unit: row.unit,
          facility: {
            id: row.facilityId,
            name: row.facilityName,
          },
          timestamp: row.timestamp || new Date().toISOString(),
        },
      };

      if (dryRun) {
        console.log(`[DRY RUN] Would ingest: ${row.eventType} for batch ${row.batchId}`);
      } else {
        const result = await client.ingestEvent(event);
        console.log(`✓ Ingested ${row.eventType} for batch ${row.batchId} → claim ${result.claim_id}`);
      }

      processed++;
    } catch (error) {
      console.error(`✗ Failed to ingest row:`, row, error);
      errors++;
    }
  }

  console.log(`\nSummary: ${processed} processed, ${errors} errors`);
}

ingestCsv(options.file, options.url, options.dryRun).catch(console.error);
