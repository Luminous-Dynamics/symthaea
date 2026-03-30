#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Multi-Level Site Clustering Script
 *
 * Creates hierarchical clusters for dynamic zoom-based display.
 * Generates 3 levels: World (1°), Regional (0.5°), Local (0.1°)
 *
 * Usage:
 *   node scripts/cluster-sites-multi-level.js
 */

const fs = require('fs');
const path = require('path');

const INPUT_FILE = path.join(__dirname, '../public/data/demo-sites.json');
const OUTPUT_FILE = path.join(__dirname, '../public/data/sites-multi-level.json');

function clusterSites(sites, gridSize, levelName) {
  console.log(`  Creating ${levelName} clusters (${gridSize}° grid)...`);

  const grid = new Map();

  for (const site of sites) {
    const gridLat = Math.floor(site.latitude / gridSize);
    const gridLng = Math.floor(site.longitude / gridSize);
    const key = `${gridLat},${gridLng}`;

    if (!grid.has(key)) {
      grid.set(key, {
        sites: [],
        sumLat: 0,
        sumLng: 0,
        totalCapacity: 0,
        totalIRR: 0,
        totalScore: 0,
        states: new Set(),
        types: new Set(),
      });
    }

    const cell = grid.get(key);
    cell.sites.push(site);
    cell.sumLat += site.latitude;
    cell.sumLng += site.longitude;
    cell.totalCapacity += site.estimated_capacity_mw || 0;
    cell.totalIRR += site.estimated_irr || 0;
    cell.totalScore += site.score || 0;
    cell.states.add(site.state);
    cell.types.add(site.type);
  }

  const clusters = [];
  let clusterId = 1;

  for (const [key, cell] of grid) {
    const count = cell.sites.length;
    const latitude = cell.sumLat / count;
    const longitude = cell.sumLng / count;

    const representative = cell.sites.reduce((best, site) =>
      (site.score || 0) > (best.score || 0) ? site : best
    );

    clusters.push({
      id: `${levelName}-${clusterId++}`,
      name: count === 1
        ? representative.name
        : `${count} sites near ${representative.name}`,
      latitude,
      longitude,
      type: cell.types.size === 1 ? [...cell.types][0] : 'mixed',
      site_count: count,
      total_capacity_mw: Math.round(cell.totalCapacity * 100) / 100,
      avg_irr: Math.round((cell.totalIRR / count) * 100) / 100,
      avg_score: Math.round((cell.totalScore / count) * 100) / 100,
      states: [...cell.states].sort(),
      representative_id: representative.id,
      grid_key: key,
      grid_size: gridSize,
      // Store site IDs for lookup
      site_ids: cell.sites.map(s => s.id),
    });
  }

  clusters.sort((a, b) => b.total_capacity_mw - a.total_capacity_mw);

  console.log(`    → ${clusters.length} clusters created`);
  return clusters;
}

async function main() {
  console.log('🌍 Terra Atlas Multi-Level Clustering\n');

  // Load sites
  console.log('Loading sites...');
  const sites = JSON.parse(fs.readFileSync(INPUT_FILE, 'utf8'));
  console.log(`Loaded ${sites.length.toLocaleString()} sites\n`);

  // Create 3 zoom levels
  const levels = [
    { name: 'world', gridSize: 1.0, description: 'Far view (zoomed out)' },
    { name: 'regional', gridSize: 0.5, description: 'Medium view' },
    { name: 'local', gridSize: 0.1, description: 'Close view (zoomed in)' },
  ];

  const output = {
    metadata: {
      generatedAt: new Date().toISOString(),
      totalSites: sites.length,
      levels: levels.map(l => ({ name: l.name, gridSize: l.gridSize })),
      totalCapacityMw: sites.reduce((sum, s) => sum + (s.estimated_capacity_mw || 0), 0),
    },
    levels: {},
  };

  console.log('Creating zoom levels:\n');

  for (const level of levels) {
    console.log(`📊 ${level.name.toUpperCase()} Level (${level.description})`);
    const clusters = clusterSites(sites, level.gridSize, level.name);

    output.levels[level.name] = {
      gridSize: level.gridSize,
      clusterCount: clusters.length,
      avgClusterSize: (sites.length / clusters.length).toFixed(1),
      clusters,
    };

    console.log(`    Avg cluster size: ${output.levels[level.name].avgClusterSize} sites`);
    console.log(`    Reduction: ${(sites.length / clusters.length).toFixed(0)}x\n`);
  }

  // Calculate file size
  const jsonOutput = JSON.stringify(output, null, 2);
  const fileSizeMB = (Buffer.byteLength(jsonOutput) / 1024 / 1024).toFixed(2);

  // Summary
  console.log('📈 Summary:');
  console.log('─'.repeat(50));
  console.log(`Total Sites:         ${sites.length.toLocaleString()}`);
  console.log(`World Clusters:      ${output.levels.world.clusterCount.toLocaleString()}`);
  console.log(`Regional Clusters:   ${output.levels.regional.clusterCount.toLocaleString()}`);
  console.log(`Local Clusters:      ${output.levels.local.clusterCount.toLocaleString()}`);
  console.log(`Total Capacity:      ${(output.metadata.totalCapacityMw / 1000).toFixed(1)} GW`);
  console.log(`File Size:           ${fileSizeMB} MB`);

  // Write output
  console.log(`\n📁 Writing to: ${OUTPUT_FILE}`);
  fs.writeFileSync(OUTPUT_FILE, jsonOutput);

  console.log('\n✅ Multi-level clustering complete!\n');
}

main().catch(err => {
  console.error('Clustering failed:', err);
  process.exit(1);
});
