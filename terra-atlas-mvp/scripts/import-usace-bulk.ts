// Bulk Import USACE NID Dams to Supabase energy_projects table
//
// Reads from data/usace-dams.json (103,650 dams) and inserts into
// the energy_projects table using Drizzle ORM in batches.
//
// Usage:
//   DATABASE_URL="postgresql://..." npx tsx scripts/import-usace-bulk.ts
//
// Options:
//   --dry-run      Print stats without inserting
//   --limit N      Only import first N dams
//   --state XX     Only import dams from state XX
//   --hydro-only   Only import dams with hydropower potential > 0 MW

import * as fs from 'fs'
import * as path from 'path'
import { db } from '../lib/drizzle/db'
import { energyProjects } from '../lib/drizzle/schema-energy'

const BATCH_SIZE = 500

interface USACEProject {
  id: string
  name: string
  type: string
  location: {
    latitude: number
    longitude: number
    state: string
    county: string
    terrain: string
  }
  capacityMw: number
  developer: string
  status: string
  environmental?: {
    waterFlow?: number
  }
  metadata?: {
    source?: string
    nidId?: string
    isExistingDam?: boolean
    hasHydropower?: boolean
    damHeight?: number
    storage?: number
    yearBuilt?: number
    hazardClass?: string
    retrofitPotential?: string
  }
}

function mapToEnergyProject(dam: USACEProject) {
  const capacityMw = dam.capacityMw ?? 0
  const investmentMillion = capacityMw * 3.5 // ~$3.5M/MW for retrofits
  const annualGenerationGwh = (capacityMw * 8760 * 0.45) / 1000 // 45% capacity factor
  const co2AvoidedTons = capacityMw * 8760 * 0.45 * 0.5 // 0.5 tons CO2/MWh

  return {
    projectType: 'dams' as const,
    subType: dam.metadata?.isExistingDam ? 'existing_dam' : 'dam',
    name: dam.name,
    description: [
      dam.metadata?.hasHydropower ? 'Existing hydropower' : 'Retrofit opportunity',
      dam.metadata?.damHeight ? `${dam.metadata.damHeight}ft height` : null,
      dam.metadata?.yearBuilt ? `built ${dam.metadata.yearBuilt}` : null,
      dam.metadata?.hazardClass ? `${dam.metadata.hazardClass} hazard` : null,
    ]
      .filter(Boolean)
      .join('. '),
    latitude: dam.location.latitude.toFixed(7),
    longitude: dam.location.longitude.toFixed(7),
    state: dam.location.state,
    country: 'US',
    capacityMw: capacityMw > 0 ? capacityMw.toFixed(2) : '0.00',
    annualGenerationGwh: annualGenerationGwh > 0 ? annualGenerationGwh.toFixed(2) : null,
    capacityFactor: capacityMw > 0 ? '45.00' : null,
    status: mapStatus(dam.status, dam.metadata?.hasHydropower ?? false),
    developer: dam.developer || 'Unknown',
    owner: 'US Army Corps of Engineers',
    totalCostMillion: capacityMw > 0 ? investmentMillion.toFixed(2) : null,
    lcoePerMwh: capacityMw > 0 ? '55.00' : null,
    expectedIrr: capacityMw > 0 ? '12.00' : null,
    co2AvoidedTonsYear: co2AvoidedTons > 0 ? co2AvoidedTons.toFixed(0) : null,
    dataSourceId: 'USACE_NID',
    dataSourceName: 'USACE National Inventory of Dams',
    externalId: dam.id || dam.metadata?.nidId || null,
    properties: {
      dam_height_ft: dam.metadata?.damHeight ?? null,
      storage_af: dam.metadata?.storage ?? null,
      year_built: dam.metadata?.yearBuilt ?? null,
      hazard_class: dam.metadata?.hazardClass ?? null,
      retrofit_potential: dam.metadata?.retrofitPotential ?? null,
      has_hydropower: dam.metadata?.hasHydropower ?? false,
      terrain: dam.location.terrain ?? null,
      water_flow: dam.environmental?.waterFlow ?? null,
    },
    isPublic: true,
    isVerified: false,
    createdAt: new Date(),
    updatedAt: new Date(),
  }
}

function mapStatus(status: string | undefined, hasHydropower: boolean): string {
  if (hasHydropower) return 'operational'
  if (!status) return 'potential'
  const lower = status.toLowerCase()
  if (lower.includes('retrofit')) return 'potential'
  if (lower.includes('operational')) return 'operational'
  if (lower.includes('construction')) return 'construction'
  return 'potential'
}

async function main() {
  const args = process.argv.slice(2)
  const dryRun = args.includes('--dry-run')
  const limitIdx = args.indexOf('--limit')
  const importLimit = limitIdx >= 0 ? parseInt(args[limitIdx + 1], 10) : Infinity
  const stateIdx = args.indexOf('--state')
  const stateFilter = stateIdx >= 0 ? args[stateIdx + 1]?.toUpperCase() : null
  const hydroOnly = args.includes('--hydro-only')

  console.log('=== USACE NID Bulk Import ===')
  console.log(`  Dry run: ${dryRun}`)
  console.log(`  Limit: ${importLimit === Infinity ? 'all' : importLimit}`)
  console.log(`  State filter: ${stateFilter || 'none'}`)
  console.log(`  Hydro only: ${hydroOnly}`)
  console.log()

  // Load data
  const dataFile = path.join(process.cwd(), 'data', 'usace-dams.json')
  if (!fs.existsSync(dataFile)) {
    console.error(`File not found: ${dataFile}`)
    process.exit(1)
  }

  console.log('Loading USACE data...')
  const raw = JSON.parse(fs.readFileSync(dataFile, 'utf-8'))
  let projects: USACEProject[] = raw.projects || []
  console.log(`  Total projects in file: ${projects.length}`)

  // Apply filters
  if (stateFilter) {
    projects = projects.filter((p) => p.location?.state === stateFilter)
    console.log(`  After state filter (${stateFilter}): ${projects.length}`)
  }

  if (hydroOnly) {
    projects = projects.filter((p) => (p.capacityMw ?? 0) > 0)
    console.log(`  After hydro-only filter: ${projects.length}`)
  }

  if (importLimit < projects.length) {
    projects = projects.slice(0, importLimit)
    console.log(`  After limit: ${projects.length}`)
  }

  // Stats
  const totalCapacity = projects.reduce((sum, p) => sum + (p.capacityMw ?? 0), 0)
  const states = new Set(projects.map((p) => p.location?.state).filter(Boolean))
  const withHydro = projects.filter((p) => (p.capacityMw ?? 0) > 0).length

  console.log()
  console.log('=== Pre-import Statistics ===')
  console.log(`  Dams to import: ${projects.length}`)
  console.log(`  With hydropower: ${withHydro}`)
  console.log(`  Total capacity: ${totalCapacity.toFixed(1)} MW`)
  console.log(`  States: ${states.size}`)
  console.log(`  Batches: ${Math.ceil(projects.length / BATCH_SIZE)}`)
  console.log()

  if (dryRun) {
    console.log('DRY RUN - no data inserted.')
    return
  }

  // Insert in batches
  let inserted = 0
  let errors = 0
  const startTime = Date.now()

  for (let i = 0; i < projects.length; i += BATCH_SIZE) {
    const batch = projects.slice(i, i + BATCH_SIZE)
    const records = batch.map(mapToEnergyProject)

    try {
      await db.insert(energyProjects).values(records)
      inserted += batch.length

      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
      const rate = (inserted / ((Date.now() - startTime) / 1000)).toFixed(0)
      process.stdout.write(
        `\r  Inserted ${inserted}/${projects.length} (${rate}/s, ${elapsed}s elapsed)`
      )
    } catch (err: any) {
      errors += batch.length
      console.error(`\n  Batch ${Math.floor(i / BATCH_SIZE)} failed:`, err.message?.slice(0, 200))
    }
  }

  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log('\n')
  console.log('=== Import Complete ===')
  console.log(`  Inserted: ${inserted}`)
  console.log(`  Errors: ${errors}`)
  console.log(`  Time: ${totalTime}s`)
  console.log(`  Rate: ${(inserted / parseFloat(totalTime)).toFixed(0)} records/s`)
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('Fatal error:', err)
    process.exit(1)
  })
