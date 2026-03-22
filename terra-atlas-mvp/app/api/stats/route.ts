import { NextResponse } from 'next/server'
import logger from '@/lib/logger'
import { StatsResponseSchema, createDefaultStatsResponse } from '@/lib/schemas/stats'
import { db } from '@/lib/drizzle/db'
import { sql } from 'drizzle-orm'

// Force this route to be dynamic (not pre-rendered)
export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

const STATS_CACHE_TTL = 60 * 1000
let statsCache: { expiresAt: number; payload: ReturnType<typeof createDefaultStatsResponse> } | null = null

async function fetchSupabaseStats() {
  const overviewResult = await db.execute(sql`
    SELECT
      COUNT(*)::int AS total_projects,
      COUNT(DISTINCT state)::int AS states,
      COUNT(DISTINCT state)::int AS regions,
      COUNT(DISTINCT developer)::int AS developers,
      COUNT(DISTINCT project_type)::int AS project_types,
      COALESCE(SUM(capacity_mw)::float, 0) AS total_capacity_mw,
      COALESCE(AVG(capacity_mw)::float, 0) AS avg_capacity_mw,
      COALESCE(MIN(capacity_mw)::float, 0) AS min_capacity_mw,
      COALESCE(MAX(capacity_mw)::float, 0) AS max_capacity_mw
    FROM energy_projects
  `)

  const overviewRow = overviewResult.rows[0] as Record<string, unknown>

  const byTypeResult = await db.execute(sql`
    SELECT
      project_type AS type,
      COUNT(*)::int AS count,
      COALESCE(SUM(capacity_mw)::float, 0) AS total_capacity
    FROM energy_projects
    WHERE project_type IS NOT NULL
    GROUP BY project_type
    ORDER BY count DESC
    LIMIT 10
  `)

  const byStatusResult = await db.execute(sql`
    SELECT
      status,
      COUNT(*)::int AS count
    FROM energy_projects
    WHERE status IS NOT NULL
    GROUP BY status
  `)

  const topRegionsResult = await db.execute(sql`
    SELECT
      state AS region,
      COUNT(*)::int AS count,
      COALESCE(SUM(capacity_mw)::float, 0) AS total_capacity
    FROM energy_projects
    WHERE state IS NOT NULL
    GROUP BY state
    ORDER BY count DESC
    LIMIT 10
  `)

  const topDevelopersResult = await db.execute(sql`
    SELECT
      developer,
      COUNT(*)::int AS projects,
      COALESCE(SUM(capacity_mw)::float, 0) AS total_capacity
    FROM energy_projects
    WHERE developer IS NOT NULL
    GROUP BY developer
    ORDER BY projects DESC
    LIMIT 10
  `)

  const recentProjectsResult = await db.execute(sql`
    SELECT
      id,
      name,
      project_type,
      developer,
      state,
      country,
      capacity_mw,
      status,
      created_at
    FROM energy_projects
    ORDER BY created_at DESC NULLS LAST
    LIMIT 5
  `)

  const totalCapacityMw = Number(overviewRow?.total_capacity_mw ?? 0)
  const totalProjects = Number(overviewRow?.total_projects ?? 0)

  return {
    overview: {
      total_projects: totalProjects,
      regions: Number(overviewRow?.regions ?? 0),
      states: Number(overviewRow?.states ?? 0),
      developers: Number(overviewRow?.developers ?? 0),
      project_types: Number(overviewRow?.project_types ?? 0),
      total_capacity_mw: Math.round(totalCapacityMw),
      total_capacity_gw: Math.round(totalCapacityMw / 1000),
      avg_capacity_mw: Number(overviewRow?.avg_capacity_mw ?? 0),
      min_capacity_mw: Number(overviewRow?.min_capacity_mw ?? 0),
      max_capacity_mw: Number(overviewRow?.max_capacity_mw ?? 0),
      total_investment_opportunity: totalCapacityMw * 1_000_000,
      estimated_jobs: Math.round(totalProjects * 17.4),
      estimated_homes_powered: Math.round(totalCapacityMw * 750),
    },
    by_type: byTypeResult.rows,
    by_status: byStatusResult.rows,
    top_regions: topRegionsResult.rows,
    top_developers: topDevelopersResult.rows,
    recent_projects: recentProjectsResult.rows,
    timestamp: new Date().toISOString(),
  }
}

export async function GET() {
  try {
    // Return cached response if valid
    if (statsCache && statsCache.expiresAt > Date.now()) {
      return NextResponse.json(statsCache.payload)
    }

    const payload = await fetchSupabaseStats()

    const parsed = StatsResponseSchema.safeParse(payload)
    if (!parsed.success) {
      logger.error('Stats schema validation failed', parsed.error)
      const fallback = createDefaultStatsResponse()
      statsCache = {
        expiresAt: Date.now() + STATS_CACHE_TTL,
        payload: fallback,
      }
      return NextResponse.json(fallback, { status: 200 })
    }

    statsCache = {
      expiresAt: Date.now() + STATS_CACHE_TTL,
      payload: parsed.data,
    }

    return NextResponse.json(parsed.data, {
      headers: {
        'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
      },
    })
  } catch (error) {
    logger.error('Stats API error', error)

    const fallback = createDefaultStatsResponse()
    statsCache = {
      expiresAt: Date.now() + STATS_CACHE_TTL,
      payload: fallback,
    }

    // Return placeholder stats on error
    return NextResponse.json(fallback, { status: 200 })
  }
}
