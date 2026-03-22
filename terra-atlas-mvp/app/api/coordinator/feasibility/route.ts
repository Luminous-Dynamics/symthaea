// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { analyzeFeasibility } from '@/lib/ferc/queue-analyzer'

/**
 * GET /api/coordinator/feasibility
 *
 * National Energy Transition Coordinator — Interconnection Feasibility API
 *
 * Analyzes grid interconnection feasibility for a proposed energy project
 * using real FERC queue data (11,547 entries with costs, outcomes, positions).
 *
 * Query params:
 *   lat          - Project latitude (required)
 *   lon          - Project longitude (required)
 *   type         - Project type: Solar, Wind, Nuclear, Storage, Hydro (required)
 *   capacity_mw  - Project capacity in MW (required)
 *   state        - State code (optional, auto-detected from nearby FERC entries)
 *
 * Returns:
 *   - Queue position estimate
 *   - Historical success rate for this region + type
 *   - Time-to-interconnection estimate (months range)
 *   - Interconnection + network upgrade cost estimates (25th/50th/75th percentile)
 *   - Similar nearby projects for comparison
 *   - Feasibility factors with impact assessment
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const lat = parseFloat(searchParams.get('lat') || '')
    const lon = parseFloat(searchParams.get('lon') || '')
    const type = searchParams.get('type') || ''
    const capacityMw = parseFloat(searchParams.get('capacity_mw') || '')
    const state = searchParams.get('state') || undefined

    if (isNaN(lat) || isNaN(lon) || !type || isNaN(capacityMw)) {
      return NextResponse.json(
        { error: 'Required: lat, lon, type, capacity_mw' },
        { status: 400 }
      )
    }

    // Connect to local PostgreSQL
    const { Client } = await import('pg')
    const client = new Client({ database: 'terra_atlas', user: 'postgres' })
    await client.connect()

    const result = await analyzeFeasibility(
      { latitude: lat, longitude: lon, project_type: type, capacity_mw: capacityMw, state },
      client
    )

    await client.end()

    return NextResponse.json({
      query: { lat, lon, type, capacity_mw: capacityMw, state },
      feasibility: result,
    })
  } catch (error) {
    console.error('Feasibility API error:', error)
    return NextResponse.json(
      { error: 'Feasibility analysis failed', detail: String(error) },
      { status: 500 }
    )
  }
}
