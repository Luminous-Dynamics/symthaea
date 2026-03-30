#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Site Clustering Script
 *
 * Creates a clustered summary of sites for efficient globe rendering.
 * Uses grid-based spatial clustering for O(n) performance.
 *
 * Usage:
 *   node scripts/cluster-sites.js [options]
 *
 * Options:
 *   --grid-size=N    Clustering grid size in degrees (default: 1)
 *   --output=PATH    Custom output path
 */

const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2).reduce((acc, arg) => {
  const [key, value] = arg.replace('--', '').split('=');
  acc[key] = value === undefined ? true : isNaN(value) ? value : Number(value);
  return acc;
}, {});

const INPUT_FILE = path.join(__dirname, '../public/data/demo-sites.json');
const DEFAULT_OUTPUT = path.join(__dirname, '../public/data/sites-clustered.json');
const GRID_SIZE = args['grid-size'] || 1; // 1 degree = ~111km at equator

function clusterSites(sites, gridSize) {
  console.log(`Clustering ${sites.length.toLocaleString()} sites with ${gridSize}° grid...`);

  // Grid-based clustering: O(n) complexity
  const grid = new Map();

  for (const site of sites) {
    // Calculate grid cell
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

  // Convert grid to cluster array
  const clusters = [];
  let clusterId = 1;

  for (const [key, cell] of grid) {
    const count = cell.sites.length;

    // Calculate centroid
    const latitude = cell.sumLat / count;
    const longitude = cell.sumLng / count;

    // Find representative site (highest score)
    const representative = cell.sites.reduce((best, site) =>
      (site.score || 0) > (best.score || 0) ? site : best
    );

    clusters.push({
      id: `cluster-${clusterId++}`,
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
      // Include full site list for small clusters
      sites: count <= 5 ? cell.sites.map(s => s.id) : undefined,
    });
  }

  // Sort by total capacity descending
  clusters.sort((a, b) => b.total_capacity_mw - a.total_capacity_mw);

  return clusters;
}

async function main() {
  console.log('🔄 Terra Atlas Site Clustering\n');

  // Load sites
  console.log('Loading sites...');
  const sites = JSON.parse(fs.readFileSync(INPUT_FILE, 'utf8'));
  console.log(`Loaded ${sites.length.toLocaleString()} sites\n`);

  // Cluster
  const clusters = clusterSites(sites, GRID_SIZE);

  // Statistics
  const totalCapacity = clusters.reduce((sum, c) => sum + c.total_capacity_mw, 0);
  const avgClusterSize = sites.length / clusters.length;

  const sizeDistribution = {
    '1 site': clusters.filter(c => c.site_count === 1).length,
    '2-5 sites': clusters.filter(c => c.site_count >= 2 && c.site_count <= 5).length,
    '6-20 sites': clusters.filter(c => c.site_count >= 6 && c.site_count <= 20).length,
    '21-100 sites': clusters.filter(c => c.site_count >= 21 && c.site_count <= 100).length,
    '100+ sites': clusters.filter(c => c.site_count > 100).length,
  };

  console.log('\n📊 Clustering Results:');
  console.log('─'.repeat(40));
  console.log(`Total clusters:      ${clusters.length.toLocaleString()}`);
  console.log(`Total sites:         ${sites.length.toLocaleString()}`);
  console.log(`Avg cluster size:    ${avgClusterSize.toFixed(1)} sites`);
  console.log(`Total capacity:      ${(totalCapacity / 1000).toFixed(1)} GW`);
  console.log(`Reduction ratio:     ${(sites.length / clusters.length).toFixed(0)}x`);

  console.log('\n📦 Cluster Size Distribution:');
  Object.entries(sizeDistribution).forEach(([range, count]) => {
    const pct = ((count / clusters.length) * 100).toFixed(1);
    console.log(`  ${range}: ${count.toLocaleString()} (${pct}%)`);
  });

  // Top clusters by capacity
  console.log('\n🏆 Top 10 Clusters by Capacity:');
  clusters.slice(0, 10).forEach((c, i) => {
    console.log(`  ${i + 1}. ${c.name.substring(0, 40)}: ${c.total_capacity_mw.toFixed(1)} MW (${c.site_count} sites)`);
  });

  // Write output
  const outputPath = args.output || DEFAULT_OUTPUT;
  console.log(`\n📁 Writing to: ${outputPath}`);

  const output = {
    metadata: {
      generatedAt: new Date().toISOString(),
      totalSites: sites.length,
      totalClusters: clusters.length,
      gridSizeDegrees: GRID_SIZE,
      totalCapacityMw: totalCapacity,
    },
    clusters,
  };

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2));

  const fileSizeKB = (Buffer.byteLength(JSON.stringify(output)) / 1024).toFixed(1);
  console.log(`File size: ${fileSizeKB} KB`);
  console.log('\n✅ Clustering complete!\n');
}

main().catch(err => {
  console.error('Clustering failed:', err);
  process.exit(1);
});
