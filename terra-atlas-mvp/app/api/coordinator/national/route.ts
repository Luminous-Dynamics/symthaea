// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'

/**
 * GET /api/coordinator/national
 *
 * National Energy Transition Coordinator — National Aggregation API
 *
 * Aggregates 115K+ real energy projects by state, technology, status,
 * and data source. Returns the full national picture of the clean
 * energy transition pipeline.
 *
 * Query params:
 *   breakdown - Comma-separated: state,type,status,source (default: all)
 *   state     - Filter to a specific state
 *   type      - Filter to a specific technology
 */
export async function GET(request: NextRequest) {
  try {
    const { Client } = await import('pg')
    const client = new Client({ database: 'terra_atlas', user: 'postgres' })
    await client.connect()

    const { searchParams } = new URL(request.url)
    const stateFilter = searchParams.get('state')
    const typeFilter = searchParams.get('type')

    const where: string[] = []
    const params: unknown[] = []
    let paramIdx = 1

    if (stateFilter) {
      where.push(`state = $${paramIdx++}`)
      params.push(stateFilter)
    }
    if (typeFilter) {
      where.push(`project_type ILIKE $${paramIdx++}`)
      params.push(`%${typeFilter}%`)
    }

    const whereClause = where.length > 0 ? `WHERE ${where.join(' AND ')}` : ''

    // Total summary
    const totalResult = await client.query(`
      SELECT
        COUNT(*) as total_projects,
        COALESCE(SUM(capacity_mw), 0) as total_capacity_mw,
        COALESCE(SUM(investment_million), 0) as total_investment_million,
        COALESCE(SUM(co2_avoided_tons_year), 0) as total_co2_avoided,
        COALESCE(SUM(construction_jobs), 0) as total_construction_jobs,
        COALESCE(SUM(permanent_jobs), 0) as total_permanent_jobs,
        COUNT(DISTINCT state) as states_covered
      FROM energy_projects ${whereClause}
    `, params)

    const total = totalResult.rows[0]

    // By state (top 20)
    const stateResult = await client.query(`
      SELECT state,
        COUNT(*) as projects,
        COALESCE(SUM(capacity_mw), 0)::bigint as capacity_mw,
        COALESCE(SUM(co2_avoided_tons_year), 0)::bigint as co2_avoided,
        COUNT(DISTINCT project_type) as technology_types
      FROM energy_projects
      ${whereClause}
      ${stateFilter ? '' : 'AND state IS NOT NULL AND state != \'\''}
      GROUP BY state
      ORDER BY SUM(capacity_mw) DESC
      LIMIT 20
    `, params)

    // By technology
    const typeResult = await client.query(`
      SELECT project_type,
        COUNT(*) as projects,
        COALESCE(SUM(capacity_mw), 0)::bigint as capacity_mw,
        COALESCE(SUM(investment_million), 0)::bigint as investment_million,
        COALESCE(AVG(capacity_mw), 0)::int as avg_capacity_mw
      FROM energy_projects ${whereClause}
      GROUP BY project_type
      ORDER BY SUM(capacity_mw) DESC
    `, params)

    // By status
    const statusResult = await client.query(`
      SELECT status,
        COUNT(*) as projects,
        COALESCE(SUM(capacity_mw), 0)::bigint as capacity_mw
      FROM energy_projects ${whereClause}
      GROUP BY status
      ORDER BY COUNT(*) DESC
      LIMIT 15
    `, params)

    // By data source
    const sourceResult = await client.query(`
      SELECT data_source,
        COUNT(*) as projects,
        COALESCE(SUM(capacity_mw), 0)::bigint as capacity_mw
      FROM energy_projects ${whereClause}
      GROUP BY data_source
      ORDER BY COUNT(*) DESC
    `, params)

    // Trust-scored projects (top 20 by trust score)
    const trustedResult = await client.query(`
      SELECT id, name, project_type, capacity_mw, state, status, data_source,
             trust_score, harmony_alignment
      FROM energy_projects
      WHERE trust_score IS NOT NULL AND trust_score > 0
      ${stateFilter ? `AND state = $${params.length + 1}` : ''}
      ORDER BY trust_score DESC
      LIMIT 20
    `, stateFilter ? [...params, stateFilter] : params)

    // National targets (2035 clean electricity goal)
    const TARGET_CLEAN_GW = 1200 // ~1,200 GW needed for 100% clean by 2035
    const currentCleanGw = Number(total.total_capacity_mw) / 1000
    const progressPct = Math.min((currentCleanGw / TARGET_CLEAN_GW) * 100, 100)

    await client.end()

    return NextResponse.json({
      summary: {
        total_projects: Number(total.total_projects),
        total_capacity_mw: Number(total.total_capacity_mw),
        total_capacity_gw: Math.round(Number(total.total_capacity_mw) / 100) / 10,
        total_capacity_tw: Math.round(Number(total.total_capacity_mw) / 1e4) / 100,
        total_investment_billion: Math.round(Number(total.total_investment_million) / 100) / 10,
        total_co2_avoided_million_tons: Math.round(Number(total.total_co2_avoided) / 1e4) / 100,
        total_construction_jobs: Number(total.total_construction_jobs),
        total_permanent_jobs: Number(total.total_permanent_jobs),
        states_covered: Number(total.states_covered),
      },
      scorecard: {
        target_clean_gw: TARGET_CLEAN_GW,
        pipeline_gw: Math.round(currentCleanGw * 10) / 10,
        progress_pct: Math.round(progressPct * 10) / 10,
        gap_gw: Math.round((TARGET_CLEAN_GW - currentCleanGw) * 10) / 10,
        note: 'Pipeline includes all projects (proposed through operational). Not all will be built.',
      },
      by_state: stateResult.rows.map(r => ({
        state: r.state,
        projects: Number(r.projects),
        capacity_mw: Number(r.capacity_mw),
        co2_avoided: Number(r.co2_avoided),
        technology_types: Number(r.technology_types),
      })),
      by_technology: typeResult.rows.map(r => ({
        type: r.project_type,
        projects: Number(r.projects),
        capacity_mw: Number(r.capacity_mw),
        investment_million: Number(r.investment_million),
        avg_capacity_mw: Number(r.avg_capacity_mw),
      })),
      by_status: statusResult.rows.map(r => ({
        status: r.status,
        projects: Number(r.projects),
        capacity_mw: Number(r.capacity_mw),
      })),
      by_source: sourceResult.rows.map(r => ({
        source: r.data_source,
        projects: Number(r.projects),
        capacity_mw: Number(r.capacity_mw),
      })),
      top_trust_scored: trustedResult.rows.map(r => ({
        id: r.id,
        name: r.name,
        type: r.project_type,
        capacity_mw: Number(r.capacity_mw),
        state: r.state,
        status: r.status,
        trust_score: Number(r.trust_score),
        harmony_alignment: Number(r.harmony_alignment),
      })),
      filters_applied: {
        state: stateFilter || 'all',
        type: typeFilter || 'all',
      },
    })
  } catch (error) {
    console.error('National API error:', error)
    return NextResponse.json(
      { error: 'National aggregation failed', detail: String(error) },
      { status: 500 }
    )
  }
}
