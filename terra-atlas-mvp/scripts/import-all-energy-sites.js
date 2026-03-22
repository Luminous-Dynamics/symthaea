#!/usr/bin/env node
/**
 * Unified Energy Sites Import
 *
 * Combines multiple data sources into a single unified dataset:
 * - USACE dams (103,650 hydro retrofit opportunities)
 * - SMR nuclear pipeline (26 advanced reactor projects)
 *
 * Usage:
 *   node scripts/import-all-energy-sites.js
 */

const fs = require('fs');
const path = require('path');

const USACE_FILE = path.join(__dirname, '../data/usace-dams.json');
const SMR_FILE = path.join(__dirname, '../data/smr-pipeline-projects.json');
const OUTPUT_FILE = path.join(__dirname, '../public/data/demo-sites.json');

// Calculate IRR for SMR projects based on characteristics
function calculateSMR_IRR(project) {
  let baseIRR = 9.5; // Base IRR for nuclear

  // Licensing phase bonus (further along = higher IRR)
  const phase = project.licensing_phase || '';
  if (phase.includes('Combined License Application')) baseIRR += 2.0;
  else if (phase.includes('Early Site Permit')) baseIRR += 1.5;
  else if (phase.includes('Design Certification')) baseIRR += 1.0;

  // Capacity factor bonus (higher = better)
  const capacityFactor = project.capacity_factor || 0.85;
  if (capacityFactor >= 0.90) baseIRR += 1.5;
  else if (capacityFactor >= 0.85) baseIRR += 1.0;

  // Developer track record (established = better)
  const developer = project.developer || '';
  if (developer.includes('TerraPower') || developer.includes('NuScale') ||
      developer.includes('X-energy') || developer.includes('Rolls-Royce')) {
    baseIRR += 1.0;
  }

  // Size bonus (larger projects = better economies of scale)
  const capacity = project.total_capacity_mw || project.capacity_mw || 0;
  if (capacity >= 1000) baseIRR += 1.5;
  else if (capacity >= 500) baseIRR += 1.0;

  // Add variance
  const variance = (Math.random() - 0.5) * 1.0;

  return Math.max(7.0, Math.min(15.0, baseIRR + variance));
}

// Calculate investment score for SMR (0-100)
function calculateSMR_Score(project, irr) {
  let score = 50;

  // IRR contribution (35 points)
  score += Math.min(35, (irr - 7) * 4.5);

  // Licensing phase (20 points)
  const phase = project.licensing_phase || '';
  if (phase.includes('Combined License')) score += 20;
  else if (phase.includes('Construction Permit')) score += 18;
  else if (phase.includes('Early Site Permit')) score += 15;
  else if (phase.includes('Design Certification')) score += 12;

  // Capacity (15 points)
  const capacity = project.total_capacity_mw || project.capacity_mw || 0;
  if (capacity >= 1000) score += 15;
  else if (capacity >= 500) score += 12;
  else if (capacity >= 300) score += 10;
  else if (capacity >= 100) score += 7;

  // Developer reputation (15 points)
  const developer = project.developer || '';
  if (developer.includes('TerraPower') || developer.includes('NuScale')) score += 15;
  else if (developer.includes('X-energy') || developer.includes('Rolls-Royce')) score += 12;
  else score += 8;

  // Capacity factor (10 points)
  const cf = project.capacity_factor || 0.85;
  score += Math.min(10, (cf - 0.80) * 200);

  // Timeline bonus (5 points)
  if (project.estimated_construction_start) {
    const year = parseInt(project.estimated_construction_start.split('-')[0]);
    if (year <= 2026) score += 5;
    else if (year <= 2028) score += 3;
    else if (year <= 2030) score += 1;
  }

  return Math.max(0, Math.min(100, score));
}

// Transform SMR project to unified format
function transformSMR(project) {
  const irr = calculateSMR_IRR(project);
  const score = calculateSMR_Score(project, irr);

  return {
    id: project.project_id,
    name: project.project_name,
    latitude: project.latitude,
    longitude: project.longitude,
    type: 'nuclear',
    estimated_capacity_mw: project.total_capacity_mw || project.capacity_mw || 0,
    estimated_irr: Number(irr.toFixed(2)),
    country: 'USA',
    state: project.state,
    status: project.status || 'In Development',
    score: Number(score.toFixed(2)),
    metadata: {
      developer: project.developer,
      reactorType: project.reactor_type,
      numberOfModules: project.number_of_modules,
      licensingPhase: project.licensing_phase,
      estimatedCOD: project.estimated_commercial_operation,
      capacityFactor: project.capacity_factor,
      estimatedCost: project.estimated_project_cost,
      jobs: {
        construction: project.construction_jobs,
        permanent: project.permanent_jobs
      }
    }
  };
}

// Calculate IRR for USACE dams (from previous script)
function calculateDamIRR(dam) {
  let baseIRR = 8.0;

  if (dam.capacityMw >= 50) baseIRR += 2.0;
  else if (dam.capacityMw >= 20) baseIRR += 1.5;
  else if (dam.capacityMw >= 10) baseIRR += 1.0;
  else if (dam.capacityMw >= 5) baseIRR += 0.5;

  const potential = dam.metadata?.retrofitPotential;
  if (potential === 'high') baseIRR += 2.5;
  else if (potential === 'medium') baseIRR += 1.0;

  const storage = dam.metadata?.storage || 0;
  if (storage >= 100000) baseIRR += 1.0;
  else if (storage >= 50000) baseIRR += 0.5;

  const waterFlow = dam.environmental?.waterFlow || 0;
  if (waterFlow >= 50) baseIRR += 0.5;
  else if (waterFlow >= 25) baseIRR += 0.25;

  const yearBuilt = dam.metadata?.yearBuilt || 1970;
  if (yearBuilt >= 2000) baseIRR += 0.5;
  else if (yearBuilt >= 1980) baseIRR += 0.25;
  else if (yearBuilt < 1950) baseIRR -= 0.5;

  const variance = (Math.random() - 0.5) * 1.5;
  return Math.max(6.0, Math.min(16.0, baseIRR + variance));
}

// Calculate score for USACE dams
function calculateDamScore(dam, irr) {
  let score = 50;
  score += Math.min(40, (irr - 6) * 4);

  if (dam.capacityMw >= 100) score += 20;
  else if (dam.capacityMw >= 50) score += 15;
  else if (dam.capacityMw >= 20) score += 10;
  else if (dam.capacityMw >= 10) score += 5;

  const hazard = dam.metadata?.hazardClass;
  if (hazard === 'Low') score += 10;
  else if (hazard === 'Significant') score += 5;

  const potential = dam.metadata?.retrofitPotential;
  if (potential === 'high') score += 10;
  else if (potential === 'medium') score += 5;

  return Math.max(0, Math.min(100, score));
}

// Transform USACE dam to unified format
function transformDam(dam) {
  const irr = calculateDamIRR(dam);
  const score = calculateDamScore(dam, irr);

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

async function main() {
  console.log('🌍 Terra Atlas Unified Energy Import\n');

  // Load USACE dams
  console.log('Loading USACE dam data...');
  const usaceData = JSON.parse(fs.readFileSync(USACE_FILE, 'utf8'));
  const dams = usaceData.projects || [];
  console.log(`  ✓ ${dams.length.toLocaleString()} hydro retrofit opportunities\n`);

  // Load SMR projects
  console.log('Loading SMR nuclear pipeline...');
  const smrProjects = JSON.parse(fs.readFileSync(SMR_FILE, 'utf8'));
  console.log(`  ✓ ${smrProjects.length.toLocaleString()} advanced reactor projects\n`);

  // Transform all data
  console.log('Transforming data...');
  const allSites = [
    ...dams.map(transformDam),
    ...smrProjects.map(transformSMR)
  ];

  console.log(`  ✓ ${allSites.length.toLocaleString()} total sites unified\n`);

  // Calculate statistics
  const stats = {
    total: allSites.length,
    byType: {
      hydro: allSites.filter(s => s.type === 'hydro').length,
      nuclear: allSites.filter(s => s.type === 'nuclear').length,
    },
    totalCapacity: allSites.reduce((sum, s) => sum + s.estimated_capacity_mw, 0),
    avgIRR: allSites.reduce((sum, s) => sum + s.estimated_irr, 0) / allSites.length,
    avgScore: allSites.reduce((sum, s) => sum + s.score, 0) / allSites.length,
  };

  console.log('📊 Unified Dataset Statistics:');
  console.log('─'.repeat(50));
  console.log(`Total Sites:         ${stats.total.toLocaleString()}`);
  console.log(`  Hydro:             ${stats.byType.hydro.toLocaleString()}`);
  console.log(`  Nuclear:           ${stats.byType.nuclear.toLocaleString()}`);
  console.log(`Total Capacity:      ${(stats.totalCapacity / 1000).toFixed(1)} GW`);
  console.log(`Average IRR:         ${stats.avgIRR.toFixed(2)}%`);
  console.log(`Average Score:       ${stats.avgScore.toFixed(1)}/100`);

  // Write output
  console.log(`\n📁 Writing to: ${OUTPUT_FILE}`);
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allSites));

  const fileSizeMB = (Buffer.byteLength(JSON.stringify(allSites)) / 1024 / 1024).toFixed(2);
  console.log(`File size: ${fileSizeMB} MB`);

  console.log('\n✅ Unified import complete!\n');
}

main().catch(err => {
  console.error('Import failed:', err);
  process.exit(1);
});
