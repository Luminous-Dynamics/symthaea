// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Create Curated Homepage Dataset
 * Generates a balanced selection of top energy sites for the globe visualization
 *
 * Usage: npx tsx scripts/data-fetchers/create-homepage-dataset.ts
 */

import * as fs from 'fs'
import * as path from 'path'

const DATA_DIR = path.join(process.cwd(), 'data')
const PUBLIC_DATA_DIR = path.join(process.cwd(), 'public', 'data')
const OUTPUT_FILE = path.join(PUBLIC_DATA_DIR, 'demo-sites.json')

interface ProcessedSite {
  id: string
  name: string
  latitude: number
  longitude: number
  type: string
  estimated_capacity_mw: number
  estimated_irr: number
  country: string
  state?: string
  status?: string
  score?: number
}

async function main() {
  console.log('Creating Curated Homepage Dataset')
  console.log('='.repeat(50))

  // Ensure output directory exists
  if (!fs.existsSync(PUBLIC_DATA_DIR)) {
    fs.mkdirSync(PUBLIC_DATA_DIR, { recursive: true })
  }

  const allSites: ProcessedSite[] = []

  // 1. Load USACE dam data - select top high-potential retrofits
  const usaceFile = path.join(DATA_DIR, 'usace-dams.json')
  if (fs.existsSync(usaceFile)) {
    console.log('\n📍 Loading USACE dam data...')
    const usaceData = JSON.parse(fs.readFileSync(usaceFile, 'utf-8'))

    // Filter to high potential only and sample across states
    const highPotential = usaceData.projects
      .filter((p: any) => p.metadata.retrofitPotential === 'high')
      .sort((a: any, b: any) => b.capacityMw - a.capacityMw)

    // Take top sites per state for geographic diversity
    const byState: Record<string, any[]> = {}
    for (const dam of highPotential) {
      const state = dam.location.state
      if (!byState[state]) byState[state] = []
      if (byState[state].length < 8) { // Max 8 per state
        byState[state].push(dam)
      }
    }

    // Flatten and convert
    const selectedDams = Object.values(byState).flat().slice(0, 300)
    console.log(`  → Selected ${selectedDams.length} high-potential dam retrofits`)

    for (const dam of selectedDams) {
      allSites.push({
        id: dam.id,
        name: dam.name,
        latitude: dam.location.latitude,
        longitude: dam.location.longitude,
        type: 'hydro',
        estimated_capacity_mw: dam.capacityMw,
        estimated_irr: 8 + Math.random() * 6, // 8-14% IRR estimate
        country: 'USA',
        state: dam.location.state,
        status: dam.status,
        score: dam.metadata.retrofitPotential === 'high' ? 75 + Math.random() * 20 : 50 + Math.random() * 25,
      })
    }
  }

  // 2. Load SMR nuclear data - all projects are valuable
  const smrFile = path.join(DATA_DIR, 'smr-pipeline.json')
  if (fs.existsSync(smrFile)) {
    console.log('\n⚛️ Loading SMR nuclear data...')
    const smrData = JSON.parse(fs.readFileSync(smrFile, 'utf-8'))

    console.log(`  → Adding all ${smrData.projects.length} SMR projects`)

    for (const project of smrData.projects) {
      allSites.push({
        id: project.id,
        name: project.name,
        latitude: project.location.latitude,
        longitude: project.location.longitude,
        type: 'nuclear',
        estimated_capacity_mw: project.capacityMw,
        estimated_irr: 10 + Math.random() * 4, // 10-14% IRR estimate
        country: 'USA',
        state: project.location.state,
        status: project.status,
        score: project.status.includes('Construction') ? 85 :
               project.status.includes('Approved') ? 78 :
               project.status.includes('Application') ? 65 : 55,
      })
    }
  }

  // 3. Generate synthetic solar/wind to balance the dataset
  console.log('\n☀️ Adding synthetic solar projects...')
  const solarStates = [
    { state: 'CA', lat: 34.5, lon: -118.2, count: 15 },
    { state: 'AZ', lat: 33.4, lon: -111.9, count: 12 },
    { state: 'TX', lat: 31.8, lon: -99.5, count: 20 },
    { state: 'NV', lat: 36.2, lon: -115.1, count: 8 },
    { state: 'FL', lat: 27.6, lon: -81.5, count: 10 },
    { state: 'NC', lat: 35.5, lon: -79.0, count: 8 },
    { state: 'GA', lat: 32.2, lon: -83.5, count: 6 },
    { state: 'CO', lat: 39.0, lon: -105.5, count: 6 },
  ]

  let solarCount = 0
  for (const region of solarStates) {
    for (let i = 0; i < region.count; i++) {
      const capacity = 50 + Math.random() * 450 // 50-500 MW
      allSites.push({
        id: `solar-${region.state.toLowerCase()}-${i}`,
        name: `${region.state} Solar Project ${i + 1}`,
        latitude: region.lat + (Math.random() - 0.5) * 4,
        longitude: region.lon + (Math.random() - 0.5) * 6,
        type: 'solar',
        estimated_capacity_mw: Math.round(capacity),
        estimated_irr: 9 + Math.random() * 5,
        country: 'USA',
        state: region.state,
        status: Math.random() > 0.3 ? 'In Development' : 'Permitted',
        score: 60 + Math.random() * 30,
      })
      solarCount++
    }
  }
  console.log(`  → Added ${solarCount} solar projects`)

  console.log('\n💨 Adding synthetic wind projects...')
  const windStates = [
    { state: 'TX', lat: 32.5, lon: -100.5, count: 18 },
    { state: 'IA', lat: 42.0, lon: -93.5, count: 10 },
    { state: 'OK', lat: 35.5, lon: -97.5, count: 12 },
    { state: 'KS', lat: 38.5, lon: -98.8, count: 10 },
    { state: 'CA', lat: 35.0, lon: -118.5, count: 8 },
    { state: 'IL', lat: 40.0, lon: -89.0, count: 6 },
    { state: 'MN', lat: 45.0, lon: -94.0, count: 6 },
    { state: 'CO', lat: 40.0, lon: -104.0, count: 6 },
  ]

  let windCount = 0
  for (const region of windStates) {
    for (let i = 0; i < region.count; i++) {
      const capacity = 100 + Math.random() * 400 // 100-500 MW
      allSites.push({
        id: `wind-${region.state.toLowerCase()}-${i}`,
        name: `${region.state} Wind Farm ${i + 1}`,
        latitude: region.lat + (Math.random() - 0.5) * 3,
        longitude: region.lon + (Math.random() - 0.5) * 5,
        type: 'wind',
        estimated_capacity_mw: Math.round(capacity),
        estimated_irr: 8 + Math.random() * 6,
        country: 'USA',
        state: region.state,
        status: Math.random() > 0.4 ? 'In Development' : 'Permitted',
        score: 55 + Math.random() * 35,
      })
      windCount++
    }
  }
  console.log(`  → Added ${windCount} wind projects`)

  // 4. Add some geothermal sites
  console.log('\n🌋 Adding geothermal projects...')
  const geothermalSites = [
    { state: 'CA', name: 'The Geysers Expansion', lat: 38.8, lon: -122.8, capacity: 120 },
    { state: 'NV', name: 'Dixie Valley', lat: 39.9, lon: -117.9, capacity: 85 },
    { state: 'NV', name: 'Steamboat Springs', lat: 39.4, lon: -119.7, capacity: 65 },
    { state: 'UT', name: 'Roosevelt Hot Springs', lat: 38.5, lon: -112.8, capacity: 50 },
    { state: 'OR', name: 'Newberry Volcano', lat: 43.7, lon: -121.2, capacity: 40 },
    { state: 'ID', name: 'Raft River', lat: 42.1, lon: -113.2, capacity: 30 },
  ]

  for (let i = 0; i < geothermalSites.length; i++) {
    const site = geothermalSites[i]
    allSites.push({
      id: `geothermal-${i}`,
      name: site.name,
      latitude: site.lat + (Math.random() - 0.5) * 0.3,
      longitude: site.lon + (Math.random() - 0.5) * 0.3,
      type: 'geothermal',
      estimated_capacity_mw: site.capacity,
      estimated_irr: 11 + Math.random() * 4,
      country: 'USA',
      state: site.state,
      status: 'In Development',
      score: 70 + Math.random() * 20,
    })
  }
  console.log(`  → Added ${geothermalSites.length} geothermal projects`)

  // 5. Add battery storage projects
  console.log('\n🔋 Adding battery storage projects...')
  const storageStates = [
    { state: 'CA', lat: 34.0, lon: -118.2, count: 8 },
    { state: 'TX', lat: 29.8, lon: -95.4, count: 6 },
    { state: 'AZ', lat: 33.5, lon: -112.1, count: 4 },
    { state: 'FL', lat: 26.1, lon: -80.1, count: 4 },
  ]

  let storageCount = 0
  for (const region of storageStates) {
    for (let i = 0; i < region.count; i++) {
      const capacity = 50 + Math.random() * 250 // 50-300 MW
      allSites.push({
        id: `storage-${region.state.toLowerCase()}-${i}`,
        name: `${region.state} Battery Storage ${i + 1}`,
        latitude: region.lat + (Math.random() - 0.5) * 2,
        longitude: region.lon + (Math.random() - 0.5) * 3,
        type: 'storage',
        estimated_capacity_mw: Math.round(capacity),
        estimated_irr: 10 + Math.random() * 5,
        country: 'USA',
        state: region.state,
        status: 'In Development',
        score: 65 + Math.random() * 25,
      })
      storageCount++
    }
  }
  console.log(`  → Added ${storageCount} battery storage projects`)

  // Calculate statistics
  const stats = {
    total: allSites.length,
    byType: {} as Record<string, number>,
    totalCapacity: 0,
    avgScore: 0,
  }

  for (const site of allSites) {
    stats.byType[site.type] = (stats.byType[site.type] || 0) + 1
    stats.totalCapacity += site.estimated_capacity_mw
    stats.avgScore += site.score || 0
  }
  stats.avgScore = Math.round(stats.avgScore / allSites.length)

  // Save the dataset
  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(allSites, null, 2))

  console.log('\n' + '='.repeat(50))
  console.log('Summary:')
  console.log(`  Total sites: ${stats.total}`)
  console.log(`  Total capacity: ${Math.round(stats.totalCapacity).toLocaleString()} MW`)
  console.log(`  Average score: ${stats.avgScore}`)
  console.log('\nBy Energy Type:')
  Object.entries(stats.byType)
    .sort((a, b) => b[1] - a[1])
    .forEach(([type, count]) => {
      console.log(`  ${type}: ${count}`)
    })
  console.log(`\nData saved to: ${OUTPUT_FILE}`)
}

main()
