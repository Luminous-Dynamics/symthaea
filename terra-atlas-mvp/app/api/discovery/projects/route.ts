import { NextRequest, NextResponse } from 'next/server'
import path from 'path'
import fs from 'fs'
import logger from '@/lib/logger'

type DiscoveryRequest = {
  type: string
  data_source: string
  limit: number
  offset: number
  state?: string | null
  status?: string | null
  source?: string | null
  developer?: string | null
  sort: string
  order: 'asc' | 'desc'
  min_capacity: number
  max_capacity: number
}

const dataDir = path.join(process.cwd(), 'data')
const STATS_RESPONSE_TTL = 60 * 1000
const statsResponseCache = new Map<string, { expiresAt: number; payload: any; status: number }>()

export async function GET(request: NextRequest) {
  return handleRequest(request, 'GET')
}

export async function POST(request: NextRequest) {
  return handleRequest(request, 'POST')
}

async function handleRequest(request: NextRequest, method: 'GET' | 'POST') {
  try {
    const params = await parseParams(request, method)
    const result = buildResponse(params)
    return NextResponse.json(result.body, { status: result.status })
  } catch (error) {
    logger.error('Discovery Projects API error', error)
    return NextResponse.json(
      {
        success: false,
        error: 'Failed to fetch discovery data',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    )
  }
}

async function parseParams(request: NextRequest, method: 'GET' | 'POST'): Promise<DiscoveryRequest> {
  const baseValues =
    method === 'GET'
      ? Object.fromEntries(request.nextUrl.searchParams)
      : await request.json().catch(() => ({}))

  const get = (key: string, fallback: string) => {
    const value = baseValues[key]
    if (value === undefined || value === null || value === '') return fallback
    return value
  }

  const toNumber = (value: string, fallback: number) => {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }

  return {
    type: get('type', 'projects'),
    data_source: get('data_source', 'ferc'),
    limit: Math.min(toNumber(get('limit', '100'), 100), 1000),
    offset: Math.max(toNumber(get('offset', '0'), 0), 0),
    state: get('state', '') || null,
    status: get('status', '') || null,
    source: get('source', '') || null,
    developer: get('developer', '') || null,
    sort: get('sort', 'queue_date'),
    order: get('order', 'desc').toLowerCase() === 'asc' ? 'asc' : 'desc',
    min_capacity: Math.max(toNumber(get('min_capacity', '0'), 0), 0),
    max_capacity: Math.max(toNumber(get('max_capacity', '10000'), 10000), 0)
  }
}

function buildResponse(params: DiscoveryRequest): { status: number; body: any } {
  const {
    type,
    data_source,
    limit,
    offset,
    state,
    status,
    source,
    developer,
    sort,
    order,
    min_capacity,
    max_capacity
  } = params

  switch (type) {
    case 'stats':
      return cacheStatsResponse('stats', () =>
        success({
          success: true,
          data: loadStats(),
          message: 'FERC queue statistics loaded'
        })
      )
    case 'corridors':
      return handleCorridors(limit, offset)
    case 'usace-stats':
      return cacheStatsResponse('usace-stats', () =>
        success({
          success: true,
          data: loadUSACEStats(),
          message: 'USACE dam retrofit statistics loaded'
        })
      )
    case 'smr-stats':
      return cacheStatsResponse('smr-stats', () =>
        success({
          success: true,
          data: loadSMRStats(),
          message: 'SMR pipeline statistics loaded'
        })
      )
    case 'smr':
    case 'projects':
    default:
      break
  }

  if (type === 'smr' || data_source === 'smr') {
    return cacheStatsResponse(
      `smr-${state ?? 'any'}-${limit}-${offset}-${sort}-${order}-${min_capacity}-${max_capacity}`,
      () =>
        handleSMRProjects({
          limit,
          offset,
          state,
          sort,
          order,
          min_capacity,
          max_capacity
        })
    )
  }

  if (type === 'usace' || data_source === 'usace') {
    return handleUSACEProjects({
      limit,
      offset,
      state,
      sort,
      order,
      min_capacity,
      max_capacity
    })
  }

  return handleFERCProjects({
    limit,
    offset,
    state,
    status,
    source,
    developer,
    sort,
    order,
    min_capacity,
    max_capacity
  })
}

function handleCorridors(limit: number, offset: number) {
  return cacheStatsResponse(`corridors-${limit}-${offset}`, () => {
    const corridors = loadCorridorOpportunities()
    const paginated = corridors.slice(offset, offset + limit)
    return success({
      success: true,
      data: paginated,
      pagination: {
        total: corridors.length,
        limit,
        offset,
        has_more: offset + limit < corridors.length
      },
      message: `Found ${corridors.length} corridor opportunities with $47.6B potential savings`
    })
  })
}

function handleSMRProjects({
  limit,
  offset,
  state,
  sort,
  order,
  min_capacity,
  max_capacity
}: {
  limit: number
  offset: number
  state: string | null
  sort: string
  order: 'asc' | 'desc'
  min_capacity: number
  max_capacity: number
}) {
  let smrProjects = loadSMRData()

  if (state) {
    smrProjects = smrProjects.filter((p: any) => p.state === state.toUpperCase())
  }

  smrProjects = smrProjects.filter(
    (p: any) => p.total_capacity_mw >= min_capacity && p.total_capacity_mw <= max_capacity
  )

  smrProjects.sort((a: any, b: any) => {
    let sortField = sort
    if (sortField === 'queue_date') sortField = 'estimated_construction_start'
    const aVal = a[sortField]
    const bVal = b[sortField]
    return order === 'asc' ? (aVal > bVal ? 1 : -1) : aVal < bVal ? 1 : -1
  })

  const totalCapacity = smrProjects.reduce((sum: number, p: any) => sum + p.total_capacity_mw, 0)
  const totalInvestment = smrProjects.reduce((sum: number, p: any) => sum + p.estimated_project_cost, 0)
  const totalJobs = smrProjects.reduce(
    (sum: number, p: any) => sum + p.construction_jobs + p.permanent_jobs,
    0
  )

  const paginated = smrProjects.slice(offset, offset + limit)

  return success({
    success: true,
    data: paginated,
    data_source: 'smr',
    pagination: {
      total: smrProjects.length,
      limit,
      offset,
      has_more: offset + limit < smrProjects.length
    },
    aggregates: {
      total_projects: smrProjects.length,
      total_capacity_mw: Math.round(totalCapacity),
      total_capacity_gw: (totalCapacity / 1000).toFixed(1),
      total_investment_required: totalInvestment,
      total_investment_billions: (totalInvestment / 1_000_000_000).toFixed(1),
      total_jobs_created: totalJobs,
      avg_capacity_mw:
        smrProjects.length > 0 ? (totalCapacity / smrProjects.length).toFixed(1) : '0.0'
    },
    message: `Found ${smrProjects.length} SMR projects totaling ${(totalCapacity / 1000).toFixed(
      1
    )} GW`
  })
}

function handleUSACEProjects({
  limit,
  offset,
  state,
  sort,
  order,
  min_capacity,
  max_capacity
}: {
  limit: number
  offset: number
  state: string | null
  sort: string
  order: 'asc' | 'desc'
  min_capacity: number
  max_capacity: number
}) {
  let dams = loadUSACEData()

  if (state) {
    dams = dams.filter((d: any) => d.state === state.toUpperCase())
  }

  dams = dams.filter(
    (d: any) => d.retrofit_potential_mw >= min_capacity && d.retrofit_potential_mw <= max_capacity
  )

  dams.sort((a: any, b: any) => {
    let sortField = sort
    if (sortField === 'queue_date') sortField = 'year_completed'
    const aVal = a[sortField]
    const bVal = b[sortField]
    return order === 'asc' ? (aVal > bVal ? 1 : -1) : aVal < bVal ? 1 : -1
  })

  const totalPotential = dams.reduce((sum: number, d: any) => sum + d.retrofit_potential_mw, 0)
  const totalInvestment = dams.reduce((sum: number, d: any) => sum + d.retrofit_cost, 0)
  const totalGeneration = dams.reduce(
    (sum: number, d: any) => sum + d.estimated_annual_generation_mwh,
    0
  )
  const totalJobs = dams.reduce((sum: number, d: any) => sum + d.jobs_created, 0)

  const paginated = dams.slice(offset, offset + limit)

  return success({
    success: true,
    data: paginated,
    data_source: 'usace',
    pagination: {
      total: dams.length,
      limit,
      offset,
      has_more: offset + limit < dams.length
    },
    aggregates: {
      total_dams: dams.length,
      total_retrofit_potential_mw: Math.round(totalPotential),
      total_retrofit_potential_gw: (totalPotential / 1000).toFixed(1),
      total_investment_required: totalInvestment,
      total_investment_billions: (totalInvestment / 1_000_000_000).toFixed(1),
      total_annual_generation_gwh: (totalGeneration / 1000).toFixed(1),
      total_jobs_potential: totalJobs,
      avg_payback_years:
        dams.length > 0
          ? (dams.reduce((sum: number, d: any) => sum + d.payback_period_years, 0) / dams.length).toFixed(1)
          : '0.0'
    },
    message: `Found ${dams.length} viable dam retrofit opportunities totaling ${(totalPotential / 1000).toFixed(
      1
    )} GW`
  })
}

function handleFERCProjects({
  limit,
  offset,
  state,
  status,
  source,
  developer,
  sort,
  order,
  min_capacity,
  max_capacity
}: {
  limit: number
  offset: number
  state: string | null
  status: string | null
  source: string | null
  developer: string | null
  sort: string
  order: 'asc' | 'desc'
  min_capacity: number
  max_capacity: number
}) {
  let projects = loadFERCData()

  if (state) {
    projects = projects.filter((p: any) => p.state === state.toUpperCase())
  }

  if (status) {
    if (status === 'withdrawn') {
      projects = projects.filter((p: any) => p.withdrawn === true)
    } else if (status === 'active') {
      projects = projects.filter((p: any) => !p.withdrawn && !p.operational)
    } else if (status === 'operational') {
      projects = projects.filter((p: any) => p.operational === true)
    }
  }

  if (source) {
    projects = projects.filter((p: any) =>
      p.energy_source.toLowerCase().includes(source.toLowerCase())
    )
  }

  if (developer) {
    projects = projects.filter((p: any) => p.developer.toLowerCase().includes(developer.toLowerCase()))
  }

  projects = projects.filter(
    (p: any) => p.capacity_mw >= min_capacity && p.capacity_mw <= max_capacity
  )

  projects.sort((a: any, b: any) => {
    let aVal = a[sort]
    let bVal = b[sort]

    if (['queue_date', 'withdrawn_date', 'in_service_date'].includes(sort)) {
      aVal = new Date(aVal).getTime()
      bVal = new Date(bVal).getTime()
    }

    return order === 'asc' ? (aVal > bVal ? 1 : -1) : aVal < bVal ? 1 : -1
  })

  const totalCapacity = projects.reduce((sum: number, p: any) => sum + p.capacity_mw, 0)
  const totalCost = projects.reduce((sum: number, p: any) => sum + p.total_cost, 0)
  const statesRepresented = new Set(projects.map((p: any) => p.state)).size
  const developersRepresented = new Set(projects.map((p: any) => p.developer)).size
  const withdrawnCount = projects.filter((p: any) => p.withdrawn).length

  const paginated = projects.slice(offset, offset + limit)

  return success({
    success: true,
    data: paginated,
    pagination: {
      total: projects.length,
      limit,
      offset,
      has_more: offset + limit < projects.length
    },
    aggregates: {
      total_projects: projects.length,
      total_capacity_mw: Math.round(totalCapacity),
      total_capacity_gw: (totalCapacity / 1000).toFixed(1),
      total_interconnection_cost: totalCost,
      total_cost_billions: (totalCost / 1_000_000_000).toFixed(1),
      states_represented: statesRepresented,
      developers_represented: developersRepresented,
      withdrawn_count: withdrawnCount,
      withdrawn_rate:
        projects.length > 0 ? ((withdrawnCount / projects.length) * 100).toFixed(1) : '0.0'
    },
    message: `Found ${projects.length.toLocaleString()} FERC queue projects totaling ${(totalCapacity / 1000).toFixed(
      1
    )} GW`
  })
}

function success(body: any, status = 200) {
  return { status, body }
}

function cacheStatsResponse(
  key: string,
  builder: () => { status: number; body: any }
) {
  const cached = statsResponseCache.get(key)
  if (cached && cached.expiresAt > Date.now()) {
    return { status: cached.status, body: cached.payload }
  }

  const fresh = builder()
  statsResponseCache.set(key, {
    expiresAt: Date.now() + STATS_RESPONSE_TTL,
    payload: fresh.body,
    status: fresh.status,
  })

  return fresh
}

type CacheEntry = { mtimeMs: number; data: any }
const jsonCache = new Map<string, CacheEntry>()

function loadJson(filename: string, fallback: any) {
  try {
    const filePath = path.join(dataDir, filename)
    const { mtimeMs } = fs.statSync(filePath)
    const cached = jsonCache.get(filePath)
    if (cached && cached.mtimeMs === mtimeMs) {
      return cached.data
    }
    const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'))
    jsonCache.set(filePath, { mtimeMs, data })
    return data
  } catch (error) {
    logger.error('Error loading discovery data file', { filename, error })
    return fallback
  }
}

const loadFERCData = () => loadJson('ferc-queue-2024.json', [])
const loadUSACEData = () => loadJson('usace-retrofit-opportunities.json', [])
const loadSMRData = () => loadJson('smr-pipeline-projects.json', [])
const loadSMRStats = () => loadJson('smr-stats.json', null)
const loadUSACEStats = () => loadJson('usace-stats.json', null)
const loadCorridorOpportunities = () => loadJson('corridor-opportunities.json', [])
const loadStats = () => loadJson('ferc-stats.json', null)
