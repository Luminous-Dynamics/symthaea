// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/drizzle/db'
import { sql } from 'drizzle-orm'

/**
 * GET /api/coordinator/discover
 *
 * National Energy Transition Coordinator — Location Discovery API
 *
 * Finds all energy projects (USACE dams, FERC queue, SMR) within a radius
 * of a given location, scored and ranked by viability.
 *
 * Query params:
 *   lat       - Latitude (required)
 *   lon       - Longitude (required)
 *   radius_km - Search radius in km (default: 50, max: 500)
 *   types     - Comma-separated project types (default: all)
 *   limit     - Max results (default: 50, max: 500)
 *   sort      - Sort by: capacity, distance, trust (default: capacity)
 *
 * Uses Haversine distance on local PostgreSQL (no PostGIS required).
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const lat = parseFloat(searchParams.get('lat') || '')
    const lon = parseFloat(searchParams.get('lon') || '')
    const radiusKm = Math.min(parseFloat(searchParams.get('radius_km') || '50'), 500)
    const types = searchParams.get('types')?.split(',').filter(Boolean) || []
    const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 500)
    const sort = searchParams.get('sort') || 'capacity'

    if (isNaN(lat) || isNaN(lon)) {
      return NextResponse.json(
        { error: 'lat and lon are required numeric parameters' },
        { status: 400 }
      )
    }

    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      return NextResponse.json(
        { error: 'lat must be -90..90, lon must be -180..180' },
        { status: 400 }
      )
    }

    // Haversine distance in km (computed in SQL for efficiency)
    // Only compute for rows within a bounding box first (index-friendly pre-filter)
    const latDelta = radiusKm / 111.0 // ~111km per degree latitude
    const lonDelta = radiusKm / (111.0 * Math.cos(lat * Math.PI / 180))
    const latMin = lat - latDelta
    const latMax = lat + latDelta
    const lonMin = lon - lonDelta
    const lonMax = lon + lonDelta

    let typeFilter = ''
    if (types.length > 0) {
      const escaped = types.map(t => `'${t.replace(/'/g, "''")}'`).join(',')
      typeFilter = `AND project_type IN (${escaped})`
    }

    const sortColumn = sort === 'distance' ? 'distance_km'
      : sort === 'trust' ? 'trust_score DESC NULLS LAST, capacity_mw'
      : 'capacity_mw'
    const sortDir = sort === 'distance' ? 'ASC' : 'DESC'

    const query = sql.raw(`
      SELECT
        id, name, project_type, developer, state, county,
        latitude, longitude, capacity_mw, status,
        energy_source, technology_type, data_source,
        investment_million, co2_avoided_tons_year,
        construction_jobs, permanent_jobs,
        retrofit_potential, dam_height, year_built, hazard_class,
        interconnection_status, interconnection_cost, queue_position,
        licensing_phase, nrc_docket, estimated_cod, number_of_modules,
        capacity_factor, trust_score, harmony_alignment,
        (6371 * acos(
          LEAST(1.0, GREATEST(-1.0,
            cos(radians(${lat})) * cos(radians(latitude))
            * cos(radians(longitude) - radians(${lon}))
            + sin(radians(${lat})) * sin(radians(latitude))
          ))
        )) AS distance_km
      FROM energy_projects
      WHERE latitude BETWEEN ${latMin} AND ${latMax}
        AND longitude BETWEEN ${lonMin} AND ${lonMax}
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
        ${typeFilter}
      ORDER BY ${sortColumn} ${sortDir}
      LIMIT ${limit}
    `)

    const results = await db.execute(query)
    const rows = results.rows || results

    // Filter by actual Haversine distance (bounding box was approximate)
    const filtered = (rows as Record<string, unknown>[]).filter(
      (r) => (r.distance_km as number) <= radiusKm
    )

    // Compute summary stats
    const totalCapacity = filtered.reduce((s, r) => s + (Number(r.capacity_mw) || 0), 0)
    const totalInvestment = filtered.reduce((s, r) => s + (Number(r.investment_million) || 0), 0)
    const totalCO2 = filtered.reduce((s, r) => s + (Number(r.co2_avoided_tons_year) || 0), 0)
    const byType: Record<string, number> = {}
    const bySource: Record<string, number> = {}
    for (const r of filtered) {
      const t = String(r.project_type || 'Unknown')
      const s = String(r.data_source || 'Unknown')
      byType[t] = (byType[t] || 0) + 1
      bySource[s] = (bySource[s] || 0) + 1
    }

    return NextResponse.json({
      query: { lat, lon, radius_km: radiusKm, types: types.length ? types : 'all', sort, limit },
      results: filtered.length,
      summary: {
        total_capacity_mw: Math.round(totalCapacity * 10) / 10,
        total_investment_million: Math.round(totalInvestment * 10) / 10,
        total_co2_avoided_tons_year: Math.round(totalCO2),
        by_type: byType,
        by_source: bySource,
      },
      projects: filtered.map((r) => ({
        id: r.id,
        name: r.name,
        project_type: r.project_type,
        developer: r.developer,
        state: r.state,
        county: r.county,
        latitude: r.latitude,
        longitude: r.longitude,
        capacity_mw: r.capacity_mw,
        status: r.status,
        data_source: r.data_source,
        distance_km: Math.round((r.distance_km as number) * 10) / 10,
        investment_million: r.investment_million,
        co2_avoided_tons_year: r.co2_avoided_tons_year,
        retrofit_potential: r.retrofit_potential,
        dam_height: r.dam_height,
        year_built: r.year_built,
        hazard_class: r.hazard_class,
        interconnection_status: r.interconnection_status,
        interconnection_cost: r.interconnection_cost,
        queue_position: r.queue_position,
        licensing_phase: r.licensing_phase,
        estimated_cod: r.estimated_cod,
        trust_score: r.trust_score,
        harmony_alignment: r.harmony_alignment,
      })),
    })
  } catch (error) {
    console.error('Discovery API error:', error)

    // Fallback: query local PostgreSQL directly if Drizzle fails
    try {
      const { Client } = await import('pg')
      const client = new Client({ database: 'terra_atlas', user: 'postgres' })
      await client.connect()

      const { searchParams } = new URL(request.url)
      const lat = parseFloat(searchParams.get('lat') || '0')
      const lon = parseFloat(searchParams.get('lon') || '0')
      const radiusKm = Math.min(parseFloat(searchParams.get('radius_km') || '50'), 500)
      const limit = Math.min(parseInt(searchParams.get('limit') || '50'), 500)

      const latDelta = radiusKm / 111.0
      const lonDelta = radiusKm / (111.0 * Math.cos(lat * Math.PI / 180))

      const result = await client.query(`
        SELECT id, name, project_type, state, capacity_mw, status, data_source,
          latitude, longitude, trust_score, harmony_alignment,
          (6371 * acos(
            LEAST(1.0, GREATEST(-1.0,
              cos(radians($1)) * cos(radians(latitude))
              * cos(radians(longitude) - radians($2))
              + sin(radians($1)) * sin(radians(latitude))
            ))
          )) AS distance_km
        FROM energy_projects
        WHERE latitude BETWEEN $3 AND $4
          AND longitude BETWEEN $5 AND $6
          AND latitude IS NOT NULL
        ORDER BY capacity_mw DESC
        LIMIT $7
      `, [lat, lon, lat - latDelta, lat + latDelta, lon - lonDelta, lon + lonDelta, limit])

      await client.end()

      const filtered = result.rows.filter(r => r.distance_km <= radiusKm)

      return NextResponse.json({
        query: { lat, lon, radius_km: radiusKm },
        results: filtered.length,
        source: 'local-postgresql',
        projects: filtered,
      })
    } catch (fallbackError) {
      return NextResponse.json(
        { error: 'Discovery service unavailable', detail: String(error) },
        { status: 500 }
      )
    }
  }
}
