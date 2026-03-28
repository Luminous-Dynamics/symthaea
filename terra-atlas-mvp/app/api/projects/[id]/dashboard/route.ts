// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

/**
 * GET /api/projects/:id/dashboard
 *
 * Combined consciousness + impact + allocation view for a project.
 * This is the primary endpoint for the Conscious Allocation Network UI.
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

    // Fetch consciousness and impact in parallel
    const [consciousnessResult, impactResult] = await Promise.all([
      supabase
        .from('consciousness_scores')
        .select('phi_score, harmony_alignment, per_harmony_scores, care_activation, meta_awareness, assessed_at, scorer_did')
        .eq('project_id', id)
        .order('assessed_at', { ascending: false })
        .limit(1)
        .maybeSingle(),
      supabase
        .from('impact_metrics')
        .select('co2_avoided_tonnes, jobs_created, community_trust_delta, energy_access_households, biodiversity_index_delta')
        .eq('project_id', id),
    ])

    // Handle missing tables gracefully
    const consciousness = consciousnessResult.error?.code === '42P01' ? null : consciousnessResult.data
    const impacts = impactResult.error?.code === '42P01' ? [] : (impactResult.data || [])

    // Aggregate impact
    const impact = impacts.reduce((acc: {
      total_co2_avoided: number
      total_jobs_created: number
      total_trust_delta: number
      peak_households: number
      total_biodiversity_delta: number
      report_count: number
    }, r: Record<string, unknown>) => {
      acc.total_co2_avoided += Number(r.co2_avoided_tonnes) || 0
      acc.total_jobs_created += Number(r.jobs_created) || 0
      acc.total_trust_delta += Number(r.community_trust_delta) || 0
      acc.peak_households = Math.max(acc.peak_households, Number(r.energy_access_households) || 0)
      acc.total_biodiversity_delta += Number(r.biodiversity_index_delta) || 0
      acc.report_count += 1
      return acc
    }, {
      total_co2_avoided: 0,
      total_jobs_created: 0,
      total_trust_delta: 0,
      peak_households: 0,
      total_biodiversity_delta: 0,
      report_count: 0,
    })

    // Net Humanity Benefit
    const nhb = (impact.total_co2_avoided / 1000) * 0.3
      + (impact.total_jobs_created / 100) * 0.25
      + Math.min(Math.max(impact.total_trust_delta, -1), 1) * 0.25
      + (impact.peak_households / 1000) * 0.2

    // Discovery score (for ranking)
    const phi = consciousness?.phi_score ?? 0
    const harmony = consciousness?.harmony_alignment ?? 0
    const co2Norm = Math.min(impact.total_co2_avoided / 10000, 1)
    const trustNorm = (impact.total_trust_delta + 1) / 2
    const discoveryScore = phi * 0.3 + harmony * 0.3 + co2Norm * 0.2 + trustNorm * 0.2

    return NextResponse.json({
      project_id: id,
      consciousness: consciousness ? {
        phi_score: consciousness.phi_score,
        harmony_alignment: consciousness.harmony_alignment,
        per_harmony_scores: consciousness.per_harmony_scores,
        care_activation: consciousness.care_activation,
        meta_awareness: consciousness.meta_awareness,
        assessed_at: consciousness.assessed_at,
      } : null,
      impact,
      net_humanity_benefit: Math.round(nhb * 1000) / 1000,
      discovery_score: Math.round(discoveryScore * 1000) / 1000,
      has_consciousness_data: !!consciousness,
      has_impact_data: impact.report_count > 0,
    })
  } catch (error) {
    console.error('Error fetching project dashboard:', error)
    return NextResponse.json(
      { error: 'Failed to fetch project dashboard' },
      { status: 500 }
    )
  }
}
