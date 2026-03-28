// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

/**
 * GET /api/projects/:id/impact
 *
 * Returns QOL impact timeline for a project, synced from Holochain.
 * Includes per-period metrics and aggregated totals.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params

    if (!id) {
      return NextResponse.json({ error: 'Project ID is required' }, { status: 400 })
    }

    const supabase = createClient(supabaseUrl, supabaseAnonKey)

    // Get impact records for this project
    const { data: records, error: recordsError } = await supabase
      .from('impact_metrics')
      .select('*')
      .eq('project_id', id)
      .order('period_end', { ascending: false })
      .limit(50)

    if (recordsError) {
      if (recordsError.code === '42P01') {
        return NextResponse.json({
          project_id: id,
          records: [],
          summary: null,
          message: 'Impact metrics not yet configured'
        })
      }
      throw recordsError
    }

    // Compute summary
    const summary = (records || []).reduce((acc: {
      total_co2_avoided: number
      total_jobs_created: number
      total_trust_delta: number
      peak_households: number
      report_count: number
    }, r: Record<string, unknown>) => {
      acc.total_co2_avoided += Number(r.co2_avoided_tonnes) || 0
      acc.total_jobs_created += Number(r.jobs_created) || 0
      acc.total_trust_delta += Number(r.community_trust_delta) || 0
      acc.peak_households = Math.max(acc.peak_households, Number(r.energy_access_households) || 0)
      acc.report_count += 1
      return acc
    }, {
      total_co2_avoided: 0,
      total_jobs_created: 0,
      total_trust_delta: 0,
      peak_households: 0,
      report_count: 0,
    })

    // Net Humanity Benefit (matches Holochain coordinator formula)
    const nhb = (summary.total_co2_avoided / 1000) * 0.3
      + (summary.total_jobs_created / 100) * 0.25
      + Math.min(Math.max(summary.total_trust_delta, -1), 1) * 0.25
      + (summary.peak_households / 1000) * 0.2

    return NextResponse.json({
      project_id: id,
      records: (records || []).map((r: Record<string, unknown>) => ({
        id: r.id,
        period_start: r.period_start,
        period_end: r.period_end,
        co2_avoided_tonnes: r.co2_avoided_tonnes,
        jobs_created: r.jobs_created,
        community_trust_delta: r.community_trust_delta,
        energy_access_households: r.energy_access_households,
        biodiversity_index_delta: r.biodiversity_index_delta,
        reporter_did: r.reporter_did,
        verified_by: r.verified_by,
        created_at: r.created_at,
      })),
      summary: {
        ...summary,
        net_humanity_benefit: Math.round(nhb * 1000) / 1000,
      },
    })
  } catch (error) {
    console.error('Error fetching impact metrics:', error)
    return NextResponse.json(
      { error: 'Failed to fetch impact metrics' },
      { status: 500 }
    )
  }
}
