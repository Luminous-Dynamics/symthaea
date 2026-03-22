export const dynamic = 'force-dynamic';
import { NextResponse } from 'next/server'
import logger from '@/lib/logger'
import { db } from '@/lib/drizzle/db'
import { energyProjects } from '@/lib/drizzle/schema-energy'
import { and, desc, eq, gte, lte, sql } from 'drizzle-orm'

function convertToCSV(data: Record<string, unknown>[], columns: string[]): string {
  // Create header row
  const header = columns.join(',')

  // Create data rows
  const rows = data.map((row) => {
    return columns
      .map((col) => {
        const value = row[col]
        // Handle values that contain commas or quotes
        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
          return `"${value.replace(/"/g, '""')}"`
        }
        return value ?? ''
      })
      .join(',')
  })

  return [header, ...rows].join('\n')
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const format = searchParams.get('format') || 'csv'
  const dataset = searchParams.get('dataset') || 'projects'
  const limit = parseInt(searchParams.get('limit') || '10000')
  const type = searchParams.get('type')
  const status = searchParams.get('status')
  const minCapacity = searchParams.get('minCapacity')
  const maxCapacity = searchParams.get('maxCapacity')
  const metricParam = searchParams.get('metric') || 'capacity_mw'
  const allowedMetrics = new Set(['capacity_mw', 'investment', 'jobs_created'])

  try {
    let data: Record<string, unknown>[] = []
    let columns: string[] = []

    switch (dataset) {
      case 'projects': {
        const conditions = []

        if (type) conditions.push(eq(energyProjects.projectType, type))
        if (status) conditions.push(eq(energyProjects.status, status))
        if (minCapacity) conditions.push(gte(energyProjects.capacityMw, minCapacity))
        if (maxCapacity) conditions.push(lte(energyProjects.capacityMw, maxCapacity))

        const whereClause = conditions.length ? and(...conditions) : undefined

        const results = await db
          .select({
            id: energyProjects.id,
            name: energyProjects.name,
            type: energyProjects.projectType,
            developer: energyProjects.developer,
            owner: energyProjects.owner,
            state: energyProjects.state,
            country: energyProjects.country,
            latitude: energyProjects.latitude,
            longitude: energyProjects.longitude,
            status: energyProjects.status,
            capacity_mw: energyProjects.capacityMw,
            annual_generation_gwh: energyProjects.annualGenerationGwh,
            investment: energyProjects.totalCostMillion,
            co2_offset_tons: energyProjects.co2AvoidedTonsYear,
          })
          .from(energyProjects)
          .where(whereClause)
          .orderBy(desc(energyProjects.capacityMw))
          .limit(limit)

        data = results as Record<string, unknown>[]
        columns = [
          'id',
          'name',
          'type',
          'developer',
          'owner',
          'state',
          'country',
          'latitude',
          'longitude',
          'status',
          'capacity_mw',
          'annual_generation_gwh',
          'investment',
          'co2_offset_tons',
        ]
        break
      }

      case 'statistics': {
        // Export aggregated statistics by type
        const statsResults = await db.execute(sql`
          SELECT
            project_type as type,
            COUNT(*)::int as project_count,
            COALESCE(SUM(capacity_mw)::float, 0) as total_capacity_mw,
            COALESCE(AVG(capacity_mw)::float, 0) as avg_capacity_mw,
            COALESCE(SUM(total_cost_million)::float, 0) as total_investment,
            COALESCE(SUM(co2_avoided_tons_year)::float, 0) as total_co2_offset
          FROM energy_projects
          WHERE project_type IS NOT NULL
          GROUP BY project_type
          ORDER BY project_count DESC
        `)

        data = statsResults.rows as Record<string, unknown>[]
        columns = [
          'type',
          'project_count',
          'total_capacity_mw',
          'avg_capacity_mw',
          'total_investment',
          'total_co2_offset',
        ]
        break
      }

      case 'summary': {
        // Export state-by-state summary
        const summaryResults = await db.execute(sql`
          SELECT
            state,
            country,
            COUNT(*)::int as project_count,
            COUNT(DISTINCT project_type)::int as technology_types,
            COALESCE(SUM(capacity_mw)::float, 0) as total_capacity_mw,
            COALESCE(SUM(total_cost_million)::float, 0) as total_investment,
            COALESCE(SUM(co2_avoided_tons_year)::float, 0) as total_co2_offset
          FROM energy_projects
          WHERE state IS NOT NULL
          GROUP BY state, country
          ORDER BY total_capacity_mw DESC
          LIMIT ${limit}
        `)

        data = summaryResults.rows as Record<string, unknown>[]
        columns = [
          'state',
          'country',
          'project_count',
          'technology_types',
          'total_capacity_mw',
          'total_investment',
          'total_co2_offset',
        ]
        break
      }

      case 'top-projects': {
        const metric = allowedMetrics.has(metricParam) ? metricParam : 'capacity_mw'
        const orderColumn =
          metric === 'capacity_mw'
            ? energyProjects.capacityMw
            : metric === 'investment'
              ? energyProjects.totalCostMillion
              : energyProjects.capacityMw

        const topResults = await db
          .select({
            id: energyProjects.id,
            name: energyProjects.name,
            type: energyProjects.projectType,
            developer: energyProjects.developer,
            state: energyProjects.state,
            country: energyProjects.country,
            capacity_mw: energyProjects.capacityMw,
            investment: energyProjects.totalCostMillion,
            status: energyProjects.status,
          })
          .from(energyProjects)
          .orderBy(desc(orderColumn))
          .limit(Math.min(limit, 100))

        data = topResults as Record<string, unknown>[]
        columns = [
          'id',
          'name',
          'type',
          'developer',
          'state',
          'country',
          'capacity_mw',
          'investment',
          'status',
        ]
        break
      }

      default:
        return NextResponse.json({ error: 'Invalid dataset' }, { status: 400 })
    }

    // Format based on requested format
    if (format === 'json') {
      return NextResponse.json({
        dataset,
        timestamp: new Date().toISOString(),
        count: data.length,
        filters: { type, status, minCapacity, maxCapacity },
        data,
      })
    } else if (format === 'csv') {
      const csv = convertToCSV(data, columns)

      return new NextResponse(csv, {
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': `attachment; filename="terra-atlas-${dataset}-${Date.now()}.csv"`,
        },
      })
    } else {
      return NextResponse.json({ error: 'Invalid format. Use csv or json' }, { status: 400 })
    }
  } catch (error) {
    logger.error('Export error', error)
    return NextResponse.json(
      {
        error: 'Export failed',
        message: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    )
  }
}
