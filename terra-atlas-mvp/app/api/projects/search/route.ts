export const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server'
import logger from '@/lib/logger'
import { db } from '@/lib/drizzle/db'
import { energyProjects } from '@/lib/drizzle/schema-energy'
import { and, desc, ilike, inArray, or, sql } from 'drizzle-orm'

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('q') || ''
  const types = searchParams.get('types')?.split(',').filter(Boolean) || []
  const statuses = searchParams.get('statuses')?.split(',').filter(Boolean) || []
  const countries = searchParams.get('countries')?.split(',').filter(Boolean) || []
  const limit = parseInt(searchParams.get('limit') || '20')

  try {
    const conditions = []

    // Text search across multiple fields
    if (query) {
      const searchPattern = `%${query}%`
      conditions.push(
        or(
          ilike(energyProjects.name, searchPattern),
          ilike(energyProjects.developer, searchPattern),
          ilike(energyProjects.state, searchPattern),
          ilike(energyProjects.country, searchPattern),
          ilike(energyProjects.projectType, searchPattern)
        )
      )
    }

    // Type filter
    if (types.length > 0) {
      conditions.push(inArray(energyProjects.projectType, types))
    }

    // Status filter
    if (statuses.length > 0) {
      conditions.push(inArray(energyProjects.status, statuses))
    }

    // Country filter
    if (countries.length > 0) {
      conditions.push(inArray(energyProjects.country, countries))
    }

    const whereClause = conditions.length ? and(...conditions) : undefined

    // Build the query with ordering
    const results = await db
      .select({
        id: energyProjects.id,
        name: energyProjects.name,
        type: energyProjects.projectType,
        developer: energyProjects.developer,
        state: energyProjects.state,
        country: energyProjects.country,
        latitude: energyProjects.latitude,
        longitude: energyProjects.longitude,
        capacity_mw: energyProjects.capacityMw,
        status: energyProjects.status,
        expected_irr: energyProjects.expectedIrr,
        total_cost_million: energyProjects.totalCostMillion,
      })
      .from(energyProjects)
      .where(whereClause)
      .orderBy(
        query
          ? sql`CASE
              WHEN ${energyProjects.name} ILIKE ${`${query}%`} THEN 1
              WHEN ${energyProjects.developer} ILIKE ${`${query}%`} THEN 2
              ELSE 3
            END`
          : desc(energyProjects.capacityMw)
      )
      .limit(limit)

    return NextResponse.json({
      results,
      total: results.length,
      query,
      filters: { types, statuses, countries },
    })
  } catch (error) {
    logger.error('Search error', error)
    return NextResponse.json(
      {
        error: 'Search failed',
        results: [],
      },
      { status: 500 }
    )
  }
}
