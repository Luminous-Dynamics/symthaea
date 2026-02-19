#!/usr/bin/env ts-node
/**
 * Seed Support Knowledge Base
 *
 * Seeds the Mycelix support knowledge base with pre-written articles
 * from the seed-data/support/ JSON files. Idempotent: skips articles
 * that already exist (matched by title).
 *
 * Usage:
 *   npx ts-node scripts/seed-support-knowledge.ts [conductor-url]
 *
 * Arguments:
 *   conductor-url  WebSocket URL of the Holochain conductor
 *                  (default: ws://localhost:8888)
 *
 * Example:
 *   npx ts-node scripts/seed-support-knowledge.ts
 *   npx ts-node scripts/seed-support-knowledge.ts ws://192.168.1.10:8888
 */

import * as fs from 'fs';
import * as path from 'path';
import { SupportClient } from '../mycelix-workspace/sdk-ts/src/clients/support';
import type { KnowledgeArticleInput, SupportCategory, ArticleSource, DifficultyLevel } from '../mycelix-workspace/sdk-ts/src/clients/support/types';

// ---------------------------------------------------------------------------
// Types for the seed data JSON format
// ---------------------------------------------------------------------------

interface SeedArticle {
  title: string;
  content: string;
  category: SupportCategory;
  tags: string[];
  source: ArticleSource;
  difficultyLevel: DifficultyLevel;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CONDUCTOR_URL = process.argv[2] || 'ws://localhost:8888';

const SEED_DATA_DIR = path.resolve(__dirname, '..', 'seed-data', 'support');

const SEED_FILES = [
  'holochain-basics.json',
  'networking-101.json',
  'mycelix-troubleshooting.json',
  'common-it-issues.json',
  'security-basics.json',
  'linux-essentials.json',
  'nix-troubleshooting.json',
  'community-patterns.json',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadSeedFile(filename: string): SeedArticle[] {
  const filepath = path.join(SEED_DATA_DIR, filename);
  if (!fs.existsSync(filepath)) {
    console.error(`  [MISSING] ${filepath}`);
    return [];
  }
  const raw = fs.readFileSync(filepath, 'utf-8');
  const articles: SeedArticle[] = JSON.parse(raw);
  return articles;
}

function buildArticleInput(
  seed: SeedArticle,
  authorPubKey: Uint8Array,
): KnowledgeArticleInput {
  return {
    title: seed.title,
    content: seed.content,
    category: seed.category,
    tags: seed.tags,
    author: authorPubKey,
    source: seed.source,
    difficultyLevel: seed.difficultyLevel,
    upvotes: 0,
    verified: true,       // Pre-seeded articles are considered verified
    deprecated: false,
    deprecationReason: null,
    version: 1,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('=== Mycelix Support Knowledge Base Seeder ===');
  console.log(`Conductor URL: ${CONDUCTOR_URL}`);
  console.log(`Seed data dir: ${SEED_DATA_DIR}`);
  console.log();

  // Connect to the conductor
  console.log('Connecting to Holochain conductor...');
  let support: SupportClient;
  try {
    support = await SupportClient.connect({
      url: CONDUCTOR_URL,
      config: {
        roleName: 'commons',
        debug: false,
        timeout: 30000,
      },
    });
  } catch (err) {
    console.error(`Failed to connect to conductor at ${CONDUCTOR_URL}`);
    console.error(err instanceof Error ? err.message : String(err));
    console.error();
    console.error('Make sure the Holochain conductor is running and the Mycelix');
    console.error('unified hApp is installed. See: seed-data/support/README for details.');
    process.exit(1);
  }

  const connected = await support.isConnected();
  if (!connected) {
    console.error('Connected but app info is unavailable. Is the hApp installed?');
    process.exit(1);
  }

  const agentPubKey = support.getAgentPubKey();
  console.log(`Connected as agent: ${Buffer.from(agentPubKey).toString('hex').slice(0, 16)}...`);
  console.log();

  // Fetch existing articles to check for duplicates (by title)
  console.log('Fetching existing articles for deduplication...');
  const existingArticles = await support.knowledge.listRecentArticles(500);
  const existingTitles = new Set(existingArticles.map((a) => a.title));
  console.log(`Found ${existingTitles.size} existing articles.`);
  console.log();

  // Process each seed file
  let totalSeeded = 0;
  let totalSkipped = 0;
  let totalErrors = 0;

  for (const filename of SEED_FILES) {
    console.log(`--- ${filename} ---`);
    const articles = loadSeedFile(filename);

    if (articles.length === 0) {
      console.log('  No articles found (file missing or empty).');
      console.log();
      continue;
    }

    for (const seed of articles) {
      // Deduplication check by title
      if (existingTitles.has(seed.title)) {
        console.log(`  [SKIP] "${seed.title}" (already exists)`);
        totalSkipped++;
        continue;
      }

      try {
        const input = buildArticleInput(seed, agentPubKey);
        const created = await support.knowledge.createArticle(input);
        console.log(`  [SEED] "${seed.title}" (${seed.category}, ${seed.difficultyLevel})`);
        // Add to our dedup set so we don't double-create within the same run
        existingTitles.add(seed.title);
        totalSeeded++;
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error(`  [ERROR] "${seed.title}": ${errMsg}`);
        totalErrors++;
      }
    }

    console.log();
  }

  // Summary
  console.log('=== Summary ===');
  console.log(`  Seeded:  ${totalSeeded}`);
  console.log(`  Skipped: ${totalSkipped} (already existed)`);
  console.log(`  Errors:  ${totalErrors}`);
  console.log();

  if (totalErrors > 0) {
    console.log('Some articles failed to seed. Check the errors above.');
    process.exit(1);
  }

  console.log('Done.');
}

main().catch((err) => {
  console.error('Unexpected error:', err);
  process.exit(1);
});
