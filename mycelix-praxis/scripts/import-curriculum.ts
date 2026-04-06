#!/usr/bin/env -S npx tsx
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Import a curriculum JSON file into the Holochain knowledge_zome.
//
// Usage:
//   npx tsx scripts/import-curriculum.ts <curriculum.json> [--conductor-url ws://localhost:8888]
//
// This script connects to a running Holochain conductor and calls
// import_curriculum() to populate the DHT with nodes and edges.

import { readFileSync } from 'fs';

// Types matching the Rust CurriculumImportInput
interface ImportNode {
  id: string;
  title: string;
  description: string;
  node_type: string;
  difficulty: string;
  domain: string;
  subdomain: string;
  tags: string[];
  estimated_hours: number;
  grade_levels: string[];
  bloom_level: string;
  subject_area: string;
}

interface ImportEdge {
  from: string;
  to: string;
  edge_type: string;
  strength_permille: number;
  rationale: string;
}

interface CurriculumImportInput {
  nodes: ImportNode[];
  edges: ImportEdge[];
}

interface CurriculumImportResult {
  nodes_created: number;
  edges_created: number;
  node_hashes: [string, Uint8Array][];
  errors: string[];
}

// Parse command line
const args = process.argv.slice(2);
const jsonFile = args.find(a => !a.startsWith('--'));
const conductorUrl = args.find(a => a.startsWith('--conductor-url='))?.split('=')[1] || 'ws://localhost:8888';

if (!jsonFile) {
  console.error('Usage: npx tsx scripts/import-curriculum.ts <curriculum.json> [--conductor-url=ws://localhost:8888]');
  console.error('');
  console.error('The curriculum JSON file should be in the unified graph format');
  console.error('produced by: edunet-standards-ingest merge ... -o unified.json');
  process.exit(1);
}

async function main() {
  console.log(`Loading ${jsonFile}...`);
  const raw = readFileSync(jsonFile, 'utf-8');
  const doc = JSON.parse(raw);

  console.log(`Loaded: ${doc.nodes?.length || 0} nodes, ${doc.edges?.length || 0} edges`);
  console.log(`Title: ${doc.metadata?.title || 'unknown'}`);

  // Prepare import input (strip optional fields not needed by the zome)
  const input: CurriculumImportInput = {
    nodes: (doc.nodes || []).map((n: any) => ({
      id: n.id,
      title: n.title,
      description: n.description || '',
      node_type: n.node_type || 'Concept',
      difficulty: n.difficulty || 'Intermediate',
      domain: n.domain || '',
      subdomain: n.subdomain || '',
      tags: n.tags || [],
      estimated_hours: n.estimated_hours || 0,
      grade_levels: n.grade_levels || [],
      bloom_level: n.bloom_level || 'Understand',
      subject_area: n.subject_area || '',
    })),
    edges: (doc.edges || []).map((e: any) => ({
      from: e.from,
      to: e.to,
      edge_type: e.edge_type || 'Recommends',
      strength_permille: e.strength_permille || 500,
      rationale: e.rationale || '',
    })),
  };

  console.log(`\nPrepared import: ${input.nodes.length} nodes, ${input.edges.length} edges`);
  console.log(`Conductor URL: ${conductorUrl}`);
  console.log('');

  // Try to connect to Holochain conductor
  try {
    // Dynamic import to avoid hard dependency
    const { AppWebsocket } = await import('@holochain/client');

    console.log('Connecting to Holochain conductor...');
    const client = await AppWebsocket.connect(conductorUrl);
    console.log('Connected!');

    console.log('Calling import_curriculum()...');
    const result: CurriculumImportResult = await client.callZome({
      role_name: 'mycelix-edunet',
      zome_name: 'knowledge_coordinator',
      fn_name: 'import_curriculum',
      payload: input,
    });

    console.log('\n=== IMPORT RESULT ===');
    console.log(`  Nodes created: ${result.nodes_created}`);
    console.log(`  Edges created: ${result.edges_created}`);
    console.log(`  Errors: ${result.errors.length}`);

    if (result.errors.length > 0) {
      console.log('\nErrors:');
      for (const err of result.errors.slice(0, 10)) {
        console.log(`  - ${err}`);
      }
      if (result.errors.length > 10) {
        console.log(`  ... and ${result.errors.length - 10} more`);
      }
    }

    console.log(`\nNode hash mapping saved (${result.node_hashes.length} entries)`);

  } catch (e: any) {
    if (e.code === 'MODULE_NOT_FOUND' || e.message?.includes('Cannot find module')) {
      console.log('⚠ @holochain/client not available. Generating dry-run report instead.\n');
      dryRun(input);
    } else {
      console.error(`Failed to connect to conductor at ${conductorUrl}:`);
      console.error(`  ${e.message || e}`);
      console.error('\nMake sure the Holochain conductor is running:');
      console.error('  hc sandbox --piped generate -a 9999 happ/mycelix-edunet.happ --run=8888');
      console.error('\nOr run in dry-run mode (no conductor needed):');
      dryRun(input);
    }
  }
}

function dryRun(input: CurriculumImportInput) {
  console.log('=== DRY RUN (no conductor) ===\n');
  console.log(`Would create ${input.nodes.length} nodes:`);

  // Group by grade level
  const byGrade: Record<string, number> = {};
  for (const node of input.nodes) {
    for (const g of node.grade_levels) {
      byGrade[g] = (byGrade[g] || 0) + 1;
    }
  }
  for (const [grade, count] of Object.entries(byGrade).sort((a, b) => a[0].localeCompare(b[0]))) {
    console.log(`  ${grade.padEnd(20)} ${count} nodes`);
  }

  // Group by subject
  const bySubject: Record<string, number> = {};
  for (const node of input.nodes) {
    bySubject[node.subject_area] = (bySubject[node.subject_area] || 0) + 1;
  }
  console.log(`\nBy subject:`);
  for (const [subj, count] of Object.entries(bySubject).sort((a, b) => b[1] - a[1])) {
    console.log(`  ${subj.padEnd(40)} ${count}`);
  }

  console.log(`\nWould create ${input.edges.length} edges:`);
  const byType: Record<string, number> = {};
  for (const edge of input.edges) {
    byType[edge.edge_type] = (byType[edge.edge_type] || 0) + 1;
  }
  for (const [type_, count] of Object.entries(byType)) {
    console.log(`  ${type_.padEnd(15)} ${count}`);
  }

  console.log('\nTo import for real, start the conductor first.');
}

main().catch(console.error);
