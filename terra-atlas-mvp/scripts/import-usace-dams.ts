#!/usr/bin/env ts-node
/**
 * USACE National Inventory of Dams Data Import Script
 *
 * Downloads and imports dam data from the USACE National Inventory of Dams
 * into the Terra Atlas sites table.
 *
 * Data source: https://nid.sec.usace.army.mil/
 * Dataset size: ~87,000 dams in the United States
 *
 * Usage:
 *   ts-node scripts/import-usace-dams.ts
 *
 * Environment variables required:
 *   NEXT_PUBLIC_SUPABASE_URL - Supabase project URL
 *   SUPABASE_SERVICE_ROLE_KEY - Supabase service role key (for write access)
 */

import { createClient } from '@supabase/supabase-js'
import fetch from 'node-fetch'
import * as fs from 'fs'
import * as path from 'path'
import * as csv from 'csv-parse/sync'

// Load environment variables from .env.local
const envPath = path.join(process.cwd(), '.env.local')
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf-8')
  envContent.split('\n').forEach(line => {
    const match = line.match(/^([^=:#]+)=(.*)$/)
    if (match) {
      const key = match[1].trim()
      const value = match[2].trim().replace(/^["']|["']$/g, '')
      process.env[key] = value
    }
  })
}

// Configuration
const USACE_API_URL = 'https://nid.sec.usace.army.mil/api/nation/csv'
const BATCH_SIZE = 1000 // Insert 1000 records at a time
const DATA_DIR = path.join(process.cwd(), 'data')
const CACHE_FILE = path.join(DATA_DIR, 'usace-dams.csv')

// Initialize Supabase client
const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!
)

interface USACEDam {
  damName: string
  nidId: string
  latitude: number
  longitude: number
  state: string
  county: string
  river: string
  damHeight: number
  nidHeight: number
  hydraulicHeight: number
  nidStorage: number
  maxStorage: number
  normalStorage: number
  surfaceArea: number
  yearCompleted: number
  ownerType: string
  ownerName: string
  damType: string
  purposes: string
  hazard: string
  eapLastRevDate: string
  inspectionDate: string
}

interface SiteRecord {
  name: string
  type: string
  status: string
  latitude: number
  longitude: number
  power_mw: number
  country: string
  state_province: string
  city: string
  commissioning_date: Date | null
  owner: string
  data_source: string
  data_quality: string
  description: string
}

/**
 * Download USACE dam data if not cached
 */
async function downloadUSACEData(): Promise<string> {
  // Create data directory if it doesn't exist
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true })
  }

  // Check if we have cached data
  if (fs.existsSync(CACHE_FILE)) {
    const stats = fs.statSync(CACHE_FILE)
    const age = Date.now() - stats.mtimeMs
    const maxAge = 7 * 24 * 60 * 60 * 1000 // 7 days in milliseconds

    if (age < maxAge) {
      console.log('✅ Using cached USACE data (age: ' + Math.round(age / (24 * 60 * 60 * 1000)) + ' days)')
      return fs.readFileSync(CACHE_FILE, 'utf-8')
    }
  }

  console.log('📥 Downloading USACE National Inventory of Dams...')

  try {
    const response = await fetch(USACE_API_URL)
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`)
    }

    const csvData = await response.text()

    // Cache the data
    fs.writeFileSync(CACHE_FILE, csvData)
    console.log(`✅ Downloaded and cached ${(csvData.length / 1024 / 1024).toFixed(2)} MB of data`)

    return csvData
  } catch (error) {
    console.error('❌ Failed to download USACE data:', error)

    // Try to use cached data even if old
    if (fs.existsSync(CACHE_FILE)) {
      console.log('⚠️  Using old cached data as fallback')
      return fs.readFileSync(CACHE_FILE, 'utf-8')
    }

    throw error
  }
}

/**
 * Parse CSV data into dam records
 */
function parseCSVData(csvData: string): USACEDam[] {
  const records = csv.parse(csvData, {
    columns: (header) => header.map((col: string) => {
      // Map CSV column names to our interface property names
      const mapping: Record<string, string> = {
        'Dam Name': 'damName',
        'NID ID': 'nidId',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'State': 'state',
        'County': 'county',
        'River or Stream Name': 'river',
        'Dam Height (Ft)': 'damHeight',
        'NID Height (Ft)': 'nidHeight',
        'Hydraulic Height (Ft)': 'hydraulicHeight',
        'NID Storage (Acre-Ft)': 'nidStorage',
        'Max Storage (Acre-Ft)': 'maxStorage',
        'Normal Storage (Acre-Ft)': 'normalStorage',
        'Surface Area (Acres)': 'surfaceArea',
        'Year Completed': 'yearCompleted',
        'Primary Owner Type': 'ownerType',
        'Owner Names': 'ownerName',
        'Primary Dam Type': 'damType',
        'Purposes': 'purposes',
        'Hazard Potential Classification': 'hazard',
        'EAP Last Revision Date': 'eapLastRevDate',
        'Last Inspection Date': 'inspectionDate'
      }
      return mapping[col] || col
    }),
    skip_empty_lines: true,
    cast: true,
    cast_date: false,
    from_line: 2  // Skip the metadata header on line 1
  }) as USACEDam[]

  console.log(`📊 Parsed ${records.length} dam records`)
  return records
}

/**
 * Estimate hydroelectric capacity based on dam characteristics
 * This is a rough approximation since the USACE data doesn't include power capacity
 */
function estimateHydroCapacity(dam: USACEDam): number {
  // Only estimate for dams with hydroelectric purpose
  if (!dam.purposes || !dam.purposes.toLowerCase().includes('h')) {
    return 0
  }

  // Rough estimation formula based on head and storage
  // P (MW) ≈ 9.81 * Q (m³/s) * H (m) * efficiency / 1000
  // We'll use storage and height as proxies

  const height = dam.hydraulicHeight || dam.nidHeight || dam.damHeight || 0
  const storage = dam.maxStorage || dam.nidStorage || 0

  if (height === 0 || storage === 0) {
    return 0.1 // Minimal capacity for small hydro
  }

  // Very rough estimation: capacity in MW
  // This is intentionally conservative to avoid overestimating
  const estimatedMW = Math.min(
    Math.sqrt(storage / 1000) * (height / 10),
    5000 // Cap at 5000 MW (Three Gorges is ~22,500 MW)
  )

  return Math.max(0.1, estimatedMW)
}

/**
 * Convert USACE dam record to site record
 */
function damToSite(dam: USACEDam): SiteRecord | null {
  // Skip dams without location data
  if (!dam.latitude || !dam.longitude) {
    return null
  }

  // Skip dams outside reasonable US bounds
  if (dam.latitude < 24 || dam.latitude > 72 || dam.longitude < -180 || dam.longitude > -65) {
    return null
  }

  const powerMW = estimateHydroCapacity(dam)

  // Determine status based on completion year
  let status = 'operational'
  if (!dam.yearCompleted || dam.yearCompleted > new Date().getFullYear()) {
    status = 'development'
  }

  // Build description
  const purposes = parseDamPurposes(dam.purposes || '')
  const description = [
    dam.river ? `Located on ${dam.river} River` : null,
    `${dam.damHeight} ft high`,
    purposes.length > 0 ? `Purposes: ${purposes.join(', ')}` : null,
    dam.hazard ? `Hazard classification: ${dam.hazard}` : null
  ].filter(Boolean).join('. ')

  return {
    name: dam.damName || `Dam ${dam.nidId}`,
    type: 'hydro',
    status,
    latitude: dam.latitude,
    longitude: dam.longitude,
    power_mw: powerMW,
    country: 'United States',
    state_province: dam.state,
    city: dam.county,
    commissioning_date: dam.yearCompleted ? new Date(dam.yearCompleted, 0, 1) : null,
    owner: dam.ownerName || 'Unknown',
    data_source: 'usace',
    data_quality: 'medium',
    description
    // Note: metadata field temporarily removed - will be added in future update
    // metadata: { nid_id, dam_height_ft, max_storage_acre_feet, etc. }
  }
}

/**
 * Parse USACE dam purposes code
 */
function parseDamPurposes(purposesCode: string): string[] {
  const purposes: Record<string, string> = {
    'C': 'Flood Control',
    'D': 'Debris Control',
    'F': 'Fish and Wildlife',
    'G': 'Fire Protection',
    'H': 'Hydroelectric',
    'I': 'Irrigation',
    'N': 'Navigation',
    'O': 'Other',
    'P': 'Recreation',
    'R': 'Water Supply',
    'S': 'Stock/Farm Pond',
    'T': 'Tailings'
  }

  const codes = purposesCode.toUpperCase().split('')
  return codes.map(code => purposes[code]).filter(Boolean)
}

/**
 * Insert sites into database in batches
 */
async function insertSites(sites: SiteRecord[]): Promise<void> {
  console.log(`📤 Inserting ${sites.length} sites into database...`)

  let inserted = 0
  let skipped = 0
  let errors = 0

  for (let i = 0; i < sites.length; i += BATCH_SIZE) {
    const batch = sites.slice(i, i + BATCH_SIZE)
    const progress = ((i + batch.length) / sites.length * 100).toFixed(1)

    try {
      const { error, count } = await supabase
        .from('sites')
        .insert(batch, { count: 'exact' })

      if (error) {
        console.error(`❌ Error inserting batch ${i}-${i + batch.length}:`, error.message)
        errors += batch.length
      } else {
        inserted += count || batch.length
        process.stdout.write(`\r✅ Progress: ${progress}% (${inserted} inserted, ${skipped} skipped, ${errors} errors)`)
      }
    } catch (error) {
      console.error(`❌ Exception inserting batch ${i}-${i + batch.length}:`, error)
      errors += batch.length
    }
  }

  console.log(`\n\n✅ Import complete:`)
  console.log(`   - Inserted: ${inserted}`)
  console.log(`   - Skipped: ${skipped}`)
  console.log(`   - Errors: ${errors}`)
}

/**
 * Main import function
 */
async function main() {
  console.log('🚀 Starting USACE Dam Data Import')
  console.log('=' + '='.repeat(50))

  try {
    // Step 1: Download data
    const csvData = await downloadUSACEData()

    // Step 2: Parse CSV
    const dams = parseCSVData(csvData)

    // Step 3: Convert to site records
    console.log('🔄 Converting dam records to site format...')
    const sites = dams
      .map(damToSite)
      .filter((site): site is SiteRecord => site !== null)
      .filter(site => site.power_mw > 0) // Only include dams with hydroelectric capacity

    console.log(`📊 Converted ${sites.length} dams with hydroelectric capacity`)

    // Step 4: Insert into database
    await insertSites(sites)

    // Step 5: Refresh materialized view
    console.log('🔄 Refreshing materialized view...')
    await supabase.rpc('refresh_sites_globe_data')
    console.log('✅ Materialized view refreshed')

    console.log('\n🎉 Import completed successfully!')
    process.exit(0)

  } catch (error) {
    console.error('\n❌ Import failed:', error)
    process.exit(1)
  }
}

// Run the import
if (require.main === module) {
  main()
}

export { main, downloadUSACEData, parseCSVData, damToSite }
