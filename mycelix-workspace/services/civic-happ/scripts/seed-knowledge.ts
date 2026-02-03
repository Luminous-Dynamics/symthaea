#!/usr/bin/env npx tsx
/**
 * Seed Knowledge Script
 *
 * Reads all JSON files from seed-data/ and creates entries in the
 * civic hApp via the Holochain conductor.
 *
 * Usage:
 *   npx tsx scripts/seed-knowledge.ts [--url ws://localhost:4445]
 */

import { AppWebsocket } from '@holochain/client';
import { readFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const SEED_DIR = join(__dirname, '..', 'seed-data');
const ROLE_NAME = 'civic';
const ZOME_NAME = 'civic_knowledge';

interface SeedEntry {
  domain: string;
  knowledge_type: string;
  title: string;
  content: string;
  geographic_scope?: string;
  keywords: string[];
  source?: string;
  expires_at?: number;
  links: string[];
  contact_phone?: string;
  address?: string;
}

async function main() {
  const url = process.argv.find((a) => a.startsWith('--url='))?.split('=')[1]
    ?? process.argv[process.argv.indexOf('--url') + 1]
    ?? 'ws://localhost:4445';

  console.log(`Connecting to conductor at ${url}...`);
  const client = await AppWebsocket.connect(url);
  console.log('Connected.');

  const files = readdirSync(SEED_DIR).filter((f) => f.endsWith('.json'));
  let total = 0;
  let created = 0;

  for (const file of files) {
    const domain = file.replace('.json', '');
    const entries: SeedEntry[] = JSON.parse(
      readFileSync(join(SEED_DIR, file), 'utf-8'),
    );

    console.log(`\nSeeding ${domain} (${entries.length} entries)...`);

    for (const entry of entries) {
      total++;
      try {
        const hash = await client.callZome({
          role_name: ROLE_NAME,
          zome_name: ZOME_NAME,
          fn_name: 'create_knowledge',
          payload: entry,
        });
        created++;
        console.log(`  ✓ ${entry.title}`);
      } catch (err: any) {
        console.error(`  ✗ ${entry.title}: ${err.message ?? err}`);
      }
    }
  }

  console.log(`\nDone. Seeded ${created}/${total} entries.`);
  process.exit(0);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
