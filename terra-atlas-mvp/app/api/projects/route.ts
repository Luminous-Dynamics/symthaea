// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository rootexport const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server'
import logger from '@/lib/logger'
import { ProjectsQuerySchema } from '@/lib/schemas/projects'
import { db } from '@/lib/drizzle/db'
import { energyProjects } from '@/lib/drizzle/schema-energy'
import { and, eq, ilike, or, sql } from 'drizzle-orm'
import { createClient } from '@supabase/supabase-js'
import * as fs from 'fs'
import * as path from 'path'

// Load USACE and SMR data from files
function loadFileData() {
  const dataDir = path.join(process.cwd(), 'data')
  const usaceFile = path.join(dataDir, 'usace-dams.json')
  const smrFile = path.join(dataDir, 'smr-pipeline.json')

  let usaceProjects: any[] = []
  let smrProjects: any[] = []

  try {
    if (fs.existsSync(usaceFile)) {
      const usaceData = JSON.parse(fs.readFileSync(usaceFile, 'utf-8'))
      usaceProjects = usaceData.projects?.map((p: any) => ({
        id: p.id,
        name: p.name,
        type: 'dams',
        developer: p.developer || 'USACE',
        owner_type: 'Federal',
        state: p.location?.state,
        county: p.location?.county,
        region: p.location?.state,
        country: 'United States',
        latitude: p.location?.latitude,
        longitude: p.location?.longitude,
        capacity_mw: p.capacityMw,
        energy_source: 'Hydroelectric',
        technology_type: p.damType || 'Dam Retrofit',
        status: p.status || 'Retrofit Opportunity',
        operational: p.status === 'Operational',
        year_completed: null,
        investment: (p.capacityMw || 0) * 2.5, // $2.5M per MW estimate
        annual_revenue_potential: (p.capacityMw || 0) * 0.3, // $300K per MW
        carbon_avoided_tons_per_year: (p.capacityMw || 0) * 1500,
        aiScore: p.metadata?.retrofitPotential === 'high' ? 85 :
                 p.metadata?.retrofitPotential === 'medium' ? 65 : 45,
      })) || []
    }
  } catch (e) {
    logger.error('Failed to load USACE data', e)
  }

  try {
    if (fs.existsSync(smrFile)) {
      const smrData = JSON.parse(fs.readFileSync(smrFile, 'utf-8'))
      smrProjects = smrData.projects?.map((p: any) => ({
        id: p.id,
        name: p.name,
        type: 'smr',
        developer: p.developer,
        owner_type: p.ownerType || 'Private',
        state: p.location?.state,
        county: p.location?.county,
        region: p.location?.state,
        country: 'United States',
        latitude: p.location?.latitude,
        longitude: p.location?.longitude,
        capacity_mw: p.capacityMw,
        energy_source: 'Nuclear',
        technology_type: p.reactorType || 'Small Modular Reactor',
        status: p.status,
        operational: p.status === 'Operational',
        year_completed: p.expectedCod,
        investment: p.estimatedCost || (p.capacityMw || 0) * 8, // $8M per MW
        annual_revenue_potential: (p.capacityMw || 0) * 0.8, // $800K per MW
        carbon_avoided_tons_per_year: (p.capacityMw || 0) * 3000,
        aiScore: p.status?.includes('Construction') ? 90 :
                 p.status?.includes('Approved') ? 80 :
                 p.status?.includes('Application') ? 70 : 60,
      })) || []
    }
  } catch (e) {
    logger.error('Failed to load SMR data', e)
  }

  return { usaceProjects, smrProjects }
}

// Cache file data to avoid repeated reads
let fileDataCache: { usaceProjects: any[]; smrProjects: any[] } | null = null
let cacheTime = 0
const CACHE_TTL = 60000 // 1 minute

function getFileData() {
  const now = Date.now()
  if (!fileDataCache || now - cacheTime > CACHE_TTL) {
    fileDataCache = loadFileData()
    cacheTime = now
  }
  return fileDataCache
}

const RATE_LIMIT_WINDOW_MS = 60 * 1000
const RATE_LIMIT_MAX_REQUESTS = 120
type RateLimitEntry = { count: number; expiresAt: number }
const rateLimitStore = new Map<string, RateLimitEntry>()

function getClientKey(request: Request) {
  const forwardedFor = request.headers.get('x-forwarded-for')
  if (forwardedFor) {
    return forwardedFor.split(',')[0]?.trim() || forwardedFor
  }
  return request.headers.get('x-real-ip') ?? request.headers.get('cf-connecting-ip') ?? 'unknown'
}

function checkRateLimit(key: string) {
  const now = Date.now()
  const current = rateLimitStore.get(key)

  if (!current || current.expiresAt <= now) {
    rateLimitStore.set(key, { count: 1, expiresAt: now + RATE_LIMIT_WINDOW_MS })
    return { allowed: true }
  }

  if (current.count >= RATE_LIMIT_MAX_REQUESTS) {
    return { allowed: false, retryAfterMs: current.expiresAt - now }
  }

  current.count += 1
  rateLimitStore.set(key, current)
  return { allowed: true }
}

function mapProjectRecord(record: Record<string, unknown>) {
  const toNumber = (value: unknown) => (value === null || value === undefined ? null : Number(value))

  return {
    id: record.id,
    name: record.name,
    type: record.type,
    developer: record.developer,
    owner_type: record.owner_type,
    state: record.state,
    county: record.county,
    region: record.region,
    country: record.country,
    latitude: toNumber(record.latitude),
    longitude: toNumber(record.longitude),
    capacity_mw: toNumber(record.capacity_mw),
    energy_source: record.energy_source,
    technology_type: record.technology_type,
    status: record.status,
    operational: record.operational,
    year_completed: record.year_completed,
    investment: record.investment,
    annual_revenue_potential: record.annual_revenue_potential,
    carbon_avoided_tons_per_year: record.carbon_avoided_tons_per_year,
  }
}

async function fetchProjects(params: {
  limit: number
  offset: number
  type?: string
  status?: string
  country?: string
  search?: string
}) {
  const { limit, offset, type, status, country, search } = params
  const conditions = []

  if (type) conditions.push(eq(energyProjects.projectType, type))
  if (status) conditions.push(eq(energyProjects.status, status))
  if (country) conditions.push(eq(energyProjects.country, country))
  if (search) {
    const searchPattern = `%${search}%`
    conditions.push(
      or(
        ilike(energyProjects.name, searchPattern),
        ilike(energyProjects.state, searchPattern),
        ilike(energyProjects.developer, searchPattern)
      )
    )
  }

  const whereClause = conditions.length ? and(...conditions) : undefined

  const projectsQuery = await db
    .select({
      id: energyProjects.id,
      name: energyProjects.name,
      type: energyProjects.projectType,
      developer: energyProjects.developer,
      owner_type: energyProjects.owner,
      state: energyProjects.state,
      county: energyProjects.city,
      region: energyProjects.state,
      country: energyProjects.country,
      latitude: energyProjects.latitude,
      longitude: energyProjects.longitude,
      capacity_mw: energyProjects.capacityMw,
      energy_source: energyProjects.projectType,
      technology_type: energyProjects.subType,
      status: energyProjects.status,
      operational: energyProjects.status,
      year_completed: energyProjects.codDate,
      investment: energyProjects.totalCostMillion,
      annual_revenue_potential: energyProjects.annualGenerationGwh,
      carbon_avoided_tons_per_year: energyProjects.co2AvoidedTonsYear,
    })
    .from(energyProjects)
    .where(whereClause)
    .limit(limit)
    .offset(offset)

  const countResult = await db
    .select({
      count: sql<number>`COUNT(*)`,
    })
    .from(energyProjects)
    .where(whereClause)

  const metadataTypes = await db
    .selectDistinct({ type: energyProjects.projectType })
    .from(energyProjects)
    .where(sql`${energyProjects.projectType} IS NOT NULL`)

  const metadataStatuses = await db
    .selectDistinct({ status: energyProjects.status })
    .from(energyProjects)
    .where(sql`${energyProjects.status} IS NOT NULL`)

  const metadataCountries = await db
    .selectDistinct({ country: energyProjects.country })
    .from(energyProjects)
    .where(sql`${energyProjects.country} IS NOT NULL`)
    .limit(50)

  return {
    projects: projectsQuery.map(mapProjectRecord),
    total: countResult[0]?.count ?? 0,
    metadata: {
      types: metadataTypes.map((row) => row.type).filter(Boolean),
      statuses: metadataStatuses.map((row) => row.status).filter(Boolean),
      countries: metadataCountries.map((row) => row.country).filter(Boolean),
    },
  }
}

/**
 * Enrich projects with trust scores from Supabase and sort by
 * discovery score (harmony-weighted). Higher-trust projects surface first.
 *
 * Discovery score = phi * 0.3 + harmony * 0.3 + co2_norm * 0.2 + trust_norm * 0.2
 */
async function enrichWithTrustScores(projects: Record<string, unknown>[]) {
  try {
    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
    const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
    if (!supabaseUrl || !supabaseKey) return projects

    const supabase = createClient(supabaseUrl, supabaseKey)

    // Batch fetch trust scores for all project IDs
    const projectIds = projects.map((p) => String(p.id)).filter(Boolean)
    if (projectIds.length === 0) return projects

    const { data: scores } = await supabase
      .from('consciousness_scores')
      .select('project_id, phi_score, harmony_alignment')
      .in('project_id', projectIds)
      .order('assessed_at', { ascending: false })

    // Build lookup (latest score per project)
    const scoreMap = new Map<string, { phi: number; harmony: number }>()
    for (const s of scores || []) {
      if (!scoreMap.has(s.project_id)) {
        scoreMap.set(s.project_id, {
          phi: Number(s.phi_score) || 0,
          harmony: Number(s.harmony_alignment) || 0,
        })
      }
    }

    // Enrich and compute discovery score
    const enriched = projects.map((p) => {
      const id = String(p.id)
      const cs = scoreMap.get(id)
      const phi = cs?.phi ?? 0
      const harmony = cs?.harmony ?? 0
      const co2 = Number(p.carbon_avoided_tons_per_year) || 0
      const co2Norm = Math.min(co2 / 10000, 1)

      const discoveryScore = phi * 0.3 + harmony * 0.3 + co2Norm * 0.2 + 0.5 * 0.2

      return {
        ...p,
        consciousness: cs ? { phi_score: phi, harmony_alignment: harmony } : null,
        discovery_score: Math.round(discoveryScore * 1000) / 1000,
      }
    })

    // Sort by discovery score descending
    enriched.sort((a, b) => (b.discovery_score ?? 0) - (a.discovery_score ?? 0))

    return enriched
  } catch {
    // If consciousness table doesn't exist yet, return projects unchanged
    return projects
  }
}

export async function GET(request: Request) {
  const clientKey = getClientKey(request)
  const rateLimit = checkRateLimit(clientKey)

  if (!rateLimit.allowed) {
    const headers: Record<string, string> = {}
    if (rateLimit.retryAfterMs !== undefined) {
      headers['Retry-After'] = Math.ceil(rateLimit.retryAfterMs / 1000).toString()
    }

    return NextResponse.json(
      { error: 'Too many requests. Please slow down.' },
      {
        status: 429,
        headers,
      }
    )
  }

  const { searchParams } = new URL(request.url)
  const sortBy = searchParams.get('sort') // 'harmony' for trust-weighted discovery
  const rawQuery = {
    limit: searchParams.get('limit') ?? undefined,
    offset: searchParams.get('offset') ?? undefined,
    type: searchParams.get('type') ?? undefined,
    status: searchParams.get('status') ?? undefined,
    country: searchParams.get('country') ?? undefined,
    search: searchParams.get('search') ?? undefined,
  }
  const parsedQuery = ProjectsQuerySchema.safeParse(rawQuery)

  if (!parsedQuery.success) {
    logger.warn({
      message: 'Invalid /api/projects query',
      context: { issues: parsedQuery.error.issues },
    })

    return NextResponse.json(
      {
        error: 'Invalid query parameters',
        issues: parsedQuery.error.issues,
      },
      { status: 400 }
    )
  }

  const { limit, offset, type, status, country, search } = parsedQuery.data

  // For dams and smr types, use file data directly
  if (type === 'dams' || type === 'smr') {
    const fileData = getFileData()
    let projects = type === 'dams' ? fileData.usaceProjects : fileData.smrProjects

    // Apply search filter
    if (search) {
      const searchLower = search.toLowerCase()
      projects = projects.filter((p: any) =>
        p.name?.toLowerCase().includes(searchLower) ||
        p.state?.toLowerCase().includes(searchLower) ||
        p.developer?.toLowerCase().includes(searchLower)
      )
    }

    // Apply status filter
    if (status) {
      projects = projects.filter((p: any) => p.status === status)
    }

    // Apply pagination
    const total = projects.length
    const paginatedProjects = projects.slice(offset, offset + limit)

    return NextResponse.json({
      projects: paginatedProjects,
      total,
      metadata: {
        types: ['dams', 'smr', 'solar', 'wind', 'hydro', 'battery', 'geothermal'],
        statuses: [...new Set(projects.map((p: any) => p.status).filter(Boolean))],
        countries: ['United States'],
      },
    })
  }

  try {
    const payload = await fetchProjects({
      limit,
      offset,
      type,
      status,
      country,
      search,
    })

    // Enrich with trust scores and sort by harmony if requested
    if (sortBy === 'harmony') {
      const enriched = await enrichWithTrustScores(payload.projects)
      return NextResponse.json({
        ...payload,
        projects: enriched,
        sort: 'harmony',
      })
    }

    return NextResponse.json(payload)
  } catch (error) {
    logger.error('Projects API error', error)

    // Fallback to file data when database is unavailable
    const fileData = getFileData()
    let fallbackProjects = [...fileData.usaceProjects, ...fileData.smrProjects]

    // Apply type filter
    if (type) {
      fallbackProjects = fallbackProjects.filter((p: any) => p.type === type)
    }

    // Apply search filter
    if (search) {
      const searchLower = search.toLowerCase()
      fallbackProjects = fallbackProjects.filter((p: any) =>
        p.name?.toLowerCase().includes(searchLower) ||
        p.state?.toLowerCase().includes(searchLower) ||
        p.developer?.toLowerCase().includes(searchLower)
      )
    }

    // Apply pagination
    const total = fallbackProjects.length
    const paginatedProjects = fallbackProjects.slice(offset, offset + limit)

    return NextResponse.json({
      projects: paginatedProjects,
      total,
      metadata: {
        types: ['dams', 'smr', 'solar', 'wind', 'hydro', 'battery', 'geothermal'],
        statuses: ['Retrofit Opportunity', 'Construction', 'NRC Review', 'Site Preparation'],
        countries: ['United States'],
      },
      source: 'file-fallback',
    })
  }
}
