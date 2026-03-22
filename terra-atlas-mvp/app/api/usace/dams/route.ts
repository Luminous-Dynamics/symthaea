import { NextResponse } from 'next/server'
import * as fs from 'fs'
import * as path from 'path'

interface USACEDam {
  id: string
  name: string
  type: string
  subtype: string
  latitude: number
  longitude: number
  capacity_mw: number
  status: string
  owner: string
  state: string
  county: string
  river: string
  year_built: number | null
  height_ft: number
  length_ft: number
  storage_af: number
  drainage_area_sqmi: number
  purposes: string[]
  hazard_potential: string
  source: string
}

interface USACEStatistics {
  total_dams: number
  hydro_dams: number
  total_capacity_mw: number
  states_covered: number
  average_capacity_mw: number
  hazard_breakdown: {
    high: number
    significant: number
    low: number
  }
}

// Cache parsed USACE data to avoid re-reading 78MB file
let cachedDams: USACEDam[] | null = null
let cachedStats: USACEStatistics | null = null
let cacheTime = 0
const CACHE_TTL = 5 * 60 * 1000 // 5 minutes

function mapHazardClass(hazardClass: string | undefined): string {
  if (!hazardClass) return 'low'
  const lower = hazardClass.toLowerCase()
  if (lower.includes('high')) return 'high'
  if (lower.includes('significant')) return 'significant'
  return 'low'
}

function mapPurposes(hasHydropower: boolean, status: string): string[] {
  const purposes: string[] = []
  if (hasHydropower) purposes.push('Hydroelectric')
  // All dams in USACE NID serve at least water control
  purposes.push('Water Supply')
  if (status === 'Retrofit Opportunity') purposes.push('Retrofit Candidate')
  return purposes.length > 0 ? purposes : ['Water Supply']
}

function loadUSACEData(): { dams: USACEDam[]; stats: USACEStatistics } {
  const now = Date.now()
  if (cachedDams && cachedStats && now - cacheTime < CACHE_TTL) {
    return { dams: cachedDams, stats: cachedStats }
  }

  const dataDir = path.join(process.cwd(), 'data')
  const usaceFile = path.join(dataDir, 'usace-dams.json')

  if (!fs.existsSync(usaceFile)) {
    return { dams: [], stats: emptyStats() }
  }

  const raw = JSON.parse(fs.readFileSync(usaceFile, 'utf-8'))
  const projects: any[] = raw.projects || []

  const dams: USACEDam[] = projects.map((p: any) => ({
    id: p.id || p.metadata?.nidId || `NID-${Math.random().toString(36).slice(2)}`,
    name: p.name,
    type: 'hydro',
    subtype: p.metadata?.isExistingDam ? 'existing_dam' : 'dam',
    latitude: p.location?.latitude ?? 0,
    longitude: p.location?.longitude ?? 0,
    capacity_mw: p.capacityMw ?? 0,
    status: mapStatus(p.status, p.metadata?.hasHydropower),
    owner: p.developer || 'Unknown',
    state: p.location?.state || '',
    county: p.location?.county || '',
    river: '', // Not available in USACE JSON
    year_built: p.metadata?.yearBuilt ?? null,
    height_ft: p.metadata?.damHeight ?? 0,
    length_ft: 0, // Not available in USACE JSON
    storage_af: p.metadata?.storage ?? 0,
    drainage_area_sqmi: 0, // Not available in USACE JSON
    purposes: mapPurposes(p.metadata?.hasHydropower ?? false, p.status),
    hazard_potential: mapHazardClass(p.metadata?.hazardClass),
    source: 'USACE NID',
  }))

  const stats = calculateStatistics(dams)

  cachedDams = dams
  cachedStats = stats
  cacheTime = now

  return { dams, stats }
}

function mapStatus(status: string | undefined, hasHydropower: boolean | undefined): string {
  if (!status) return 'unknown'
  if (hasHydropower) return 'operational'
  if (status.toLowerCase().includes('retrofit')) return 'retrofit_opportunity'
  return status.toLowerCase().replace(/\s+/g, '_')
}

function calculateStatistics(dams: USACEDam[]): USACEStatistics {
  const hydroDams = dams.filter((d) => d.capacity_mw > 0)
  const totalCapacity = dams.reduce((sum, d) => sum + d.capacity_mw, 0)
  const states = new Set(dams.map((d) => d.state).filter(Boolean))

  return {
    total_dams: dams.length,
    hydro_dams: hydroDams.length,
    total_capacity_mw: totalCapacity,
    states_covered: states.size,
    average_capacity_mw: hydroDams.length > 0 ? totalCapacity / hydroDams.length : 0,
    hazard_breakdown: {
      high: dams.filter((d) => d.hazard_potential === 'high').length,
      significant: dams.filter((d) => d.hazard_potential === 'significant').length,
      low: dams.filter((d) => d.hazard_potential === 'low').length,
    },
  }
}

function emptyStats(): USACEStatistics {
  return {
    total_dams: 0,
    hydro_dams: 0,
    total_capacity_mw: 0,
    states_covered: 0,
    average_capacity_mw: 0,
    hazard_breakdown: { high: 0, significant: 0, low: 0 },
  }
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)

  const limit = Math.min(parseInt(searchParams.get('limit') || '500', 10), 10000)
  const offset = parseInt(searchParams.get('offset') || '0', 10)
  const state = searchParams.get('state')
  const minCapacity = parseFloat(searchParams.get('min_capacity') || '0')
  const hazard = searchParams.get('hazard')
  const search = searchParams.get('search')
  const hydroOnly = searchParams.get('hydro_only') === 'true'

  const { dams: allDams, stats: fullStats } = loadUSACEData()

  let filtered = allDams

  if (state) {
    filtered = filtered.filter((d) => d.state === state.toUpperCase())
  }
  if (minCapacity > 0) {
    filtered = filtered.filter((d) => d.capacity_mw >= minCapacity)
  }
  if (hazard) {
    filtered = filtered.filter((d) => d.hazard_potential === hazard.toLowerCase())
  }
  if (hydroOnly) {
    filtered = filtered.filter((d) => d.capacity_mw > 0)
  }
  if (search) {
    const searchLower = search.toLowerCase()
    filtered = filtered.filter(
      (d) =>
        d.name.toLowerCase().includes(searchLower) ||
        d.state.toLowerCase().includes(searchLower) ||
        d.county.toLowerCase().includes(searchLower) ||
        d.owner.toLowerCase().includes(searchLower)
    )
  }

  const filteredStats = calculateStatistics(filtered)
  const paginated = filtered.slice(offset, offset + limit)

  return NextResponse.json(
    {
      dams: paginated,
      statistics: filteredStats,
      pagination: {
        total: filtered.length,
        limit,
        offset,
        has_more: offset + limit < filtered.length,
      },
    },
    {
      headers: {
        'Cache-Control': 'public, s-maxage=300, stale-while-revalidate=600',
      },
    }
  )
}
