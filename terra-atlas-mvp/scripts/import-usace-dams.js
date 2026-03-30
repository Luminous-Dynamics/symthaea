#!/usr/bin/env node

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * USACE Dam Data Import Script
 *
 * Transforms USACE National Inventory of Dams data into Terra Atlas site format
 * with calculated investment metrics and validation.
 *
 * Usage:
 *   node scripts/import-usace-dams.js [options]
 *
 * Options:
 *   --limit=N     Only process first N records (for testing)
 *   --min-capacity=N  Filter dams with capacity >= N MW
 *   --output=PATH  Custom output path
 *   --stats       Show statistics only, don't write file
 */

const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2).reduce((acc, arg) => {
  const [key, value] = arg.replace('--', '').split('=');
  acc[key] = value === undefined ? true : isNaN(value) ? value : Number(value);
  return acc;
}, {});

const INPUT_FILE = path.join(__dirname, '../data/usace-dams.json');
const DEFAULT_OUTPUT = path.join(__dirname, '../public/data/demo-sites.json');

// IRR calculation based on dam characteristics
function calculateIRR(dam) {
  let baseIRR = 8.0; // Base IRR for hydro projects

  // Capacity bonus (larger = better economies of scale)
  if (dam.capacityMw >= 50) baseIRR += 2.0;
  else if (dam.capacityMw >= 20) baseIRR += 1.5;
  else if (dam.capacityMw >= 10) baseIRR += 1.0;
  else if (dam.capacityMw >= 5) baseIRR += 0.5;

  // Retrofit potential bonus
  const potential = dam.metadata?.retrofitPotential;
  if (potential === 'high') baseIRR += 2.5;
  else if (potential === 'medium') baseIRR += 1.0;

  // Storage bonus (more storage = more reliable)
  const storage = dam.metadata?.storage || 0;
  if (storage >= 100000) baseIRR += 1.0;
  else if (storage >= 50000) baseIRR += 0.5;

  // Water flow bonus
  const waterFlow = dam.environmental?.waterFlow || 0;
  if (waterFlow >= 50) baseIRR += 0.5;
  else if (waterFlow >= 25) baseIRR += 0.25;

  // Age adjustment (newer = better infrastructure)
  const yearBuilt = dam.metadata?.yearBuilt || 1970;
  if (yearBuilt >= 2000) baseIRR += 0.5;
  else if (yearBuilt >= 1980) baseIRR += 0.25;
  else if (yearBuilt < 1950) baseIRR -= 0.5;

  // Add some variance to make it realistic
  const variance = (Math.random() - 0.5) * 1.5;

  return Math.max(6.0, Math.min(16.0, baseIRR + variance));
}

// Investment score calculation (0-100)
function calculateScore(dam, irr) {
  let score = 50; // Base score

  // IRR contribution (40 points max)
  score += Math.min(40, (irr - 6) * 4);

  // Capacity contribution (20 points max)
  if (dam.capacityMw >= 100) score += 20;
  else if (dam.capacityMw >= 50) score += 15;
  else if (dam.capacityMw >= 20) score += 10;
  else if (dam.capacityMw >= 10) score += 5;

  // Hazard class (lower = better, 10 points max)
  const hazard = dam.metadata?.hazardClass;
  if (hazard === 'Low') score += 10;
  else if (hazard === 'Significant') score += 5;
  // High hazard = no bonus

  // Retrofit potential (10 points max)
  const potential = dam.metadata?.retrofitPotential;
  if (potential === 'high') score += 10;
  else if (potential === 'medium') score += 5;

  return Math.max(0, Math.min(100, score));
}

// Transform USACE dam to Terra Atlas site format
function transformDam(dam) {
  const irr = calculateIRR(dam);
  const score = calculateScore(dam, irr);

  return {
    id: dam.id,
    name: dam.name,
    latitude: dam.location.latitude,
    longitude: dam.location.longitude,
    type: 'hydro',
    estimated_capacity_mw: dam.capacityMw,
    estimated_irr: Number(irr.toFixed(2)),
    country: 'USA',
    state: dam.location.state,
    status: dam.status,
    score: Number(score.toFixed(2)),
    // Additional metadata for enhanced features
    metadata: {
      county: dam.location.county,
      terrain: dam.location.terrain,
      developer: dam.developer,
      yearBuilt: dam.metadata?.yearBuilt,
      damHeight: dam.metadata?.damHeight,
      storage: dam.metadata?.storage,
      hazardClass: dam.metadata?.hazardClass,
      retrofitPotential: dam.metadata?.retrofitPotential,
      waterFlow: dam.environmental?.waterFlow
    }
  };
}

// Validate transformed site
function validateSite(site) {
  const errors = [];

  if (!site.id) errors.push('Missing id');
  if (!site.name) errors.push('Missing name');
  if (typeof site.latitude !== 'number' || site.latitude < -90 || site.latitude > 90) {
    errors.push(`Invalid latitude: ${site.latitude}`);
  }
  if (typeof site.longitude !== 'number' || site.longitude < -180 || site.longitude > 180) {
    errors.push(`Invalid longitude: ${site.longitude}`);
  }
  if (typeof site.estimated_capacity_mw !== 'number' || site.estimated_capacity_mw <= 0) {
    errors.push(`Invalid capacity: ${site.estimated_capacity_mw}`);
  }
  if (!site.state || site.state.length !== 2) {
    errors.push(`Invalid state: ${site.state}`);
  }

  return errors;
}

// Main import function
async function main() {
  console.log('🏗️  Terra Atlas USACE Dam Import\n');

  // Load source data
  console.log('Loading USACE data...');
  const sourceData = JSON.parse(fs.readFileSync(INPUT_FILE, 'utf8'));
  const projects = sourceData.projects;

  console.log(`Found ${projects.length.toLocaleString()} dam projects\n`);

  // Apply filters
  let filtered = projects;

  if (args['min-capacity']) {
    filtered = filtered.filter(d => d.capacityMw >= args['min-capacity']);
    console.log(`Filtered to ${filtered.length.toLocaleString()} with capacity >= ${args['min-capacity']} MW`);
  }

  if (args.limit) {
    filtered = filtered.slice(0, args.limit);
    console.log(`Limited to first ${args.limit.toLocaleString()} records`);
  }

  // Transform all records
  console.log('\nTransforming records...');
  const sites = [];
  const errors = [];
  let validationErrors = 0;

  for (const dam of filtered) {
    const site = transformDam(dam);
    const siteErrors = validateSite(site);

    if (siteErrors.length > 0) {
      validationErrors++;
      if (validationErrors <= 10) {
        errors.push({ id: dam.id, errors: siteErrors });
      }
    } else {
      sites.push(site);
    }
  }

  // Calculate statistics
  const totalCapacity = sites.reduce((sum, s) => sum + s.estimated_capacity_mw, 0);
  const avgIRR = sites.reduce((sum, s) => sum + s.estimated_irr, 0) / sites.length;
  const avgScore = sites.reduce((sum, s) => sum + s.score, 0) / sites.length;

  const byState = sites.reduce((acc, s) => {
    acc[s.state] = (acc[s.state] || 0) + 1;
    return acc;
  }, {});

  const topStates = Object.entries(byState)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  // Print statistics
  console.log('\n📊 Import Statistics:');
  console.log('─'.repeat(40));
  console.log(`Total sites:         ${sites.length.toLocaleString()}`);
  console.log(`Validation errors:   ${validationErrors.toLocaleString()}`);
  console.log(`Total capacity:      ${(totalCapacity / 1000).toFixed(1)} GW`);
  console.log(`Average IRR:         ${avgIRR.toFixed(2)}%`);
  console.log(`Average score:       ${avgScore.toFixed(1)}/100`);

  console.log('\n🗺️  Top 10 States by Site Count:');
  topStates.forEach(([state, count], i) => {
    console.log(`  ${i + 1}. ${state}: ${count.toLocaleString()}`);
  });

  // Capacity distribution
  const capacityRanges = {
    '0-5 MW': sites.filter(s => s.estimated_capacity_mw < 5).length,
    '5-20 MW': sites.filter(s => s.estimated_capacity_mw >= 5 && s.estimated_capacity_mw < 20).length,
    '20-50 MW': sites.filter(s => s.estimated_capacity_mw >= 20 && s.estimated_capacity_mw < 50).length,
    '50-100 MW': sites.filter(s => s.estimated_capacity_mw >= 50 && s.estimated_capacity_mw < 100).length,
    '100+ MW': sites.filter(s => s.estimated_capacity_mw >= 100).length,
  };

  console.log('\n⚡ Capacity Distribution:');
  Object.entries(capacityRanges).forEach(([range, count]) => {
    const pct = ((count / sites.length) * 100).toFixed(1);
    console.log(`  ${range}: ${count.toLocaleString()} (${pct}%)`);
  });

  // Show validation errors if any
  if (errors.length > 0) {
    console.log('\n⚠️  Sample Validation Errors:');
    errors.forEach(({ id, errors }) => {
      console.log(`  ${id}: ${errors.join(', ')}`);
    });
    if (validationErrors > 10) {
      console.log(`  ... and ${validationErrors - 10} more`);
    }
  }

  // Write output
  if (!args.stats) {
    const outputPath = args.output || DEFAULT_OUTPUT;
    console.log(`\n📁 Writing to: ${outputPath}`);

    // Create output directory if needed
    const outputDir = path.dirname(outputPath);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Write with formatting for smaller files, compact for large
    const jsonOutput = sites.length > 10000
      ? JSON.stringify(sites)
      : JSON.stringify(sites, null, 2);

    fs.writeFileSync(outputPath, jsonOutput);

    const fileSizeMB = (Buffer.byteLength(jsonOutput) / 1024 / 1024).toFixed(2);
    console.log(`File size: ${fileSizeMB} MB`);
  }

  console.log('\n✅ Import complete!\n');
}

main().catch(err => {
  console.error('Import failed:', err);
  process.exit(1);
});
