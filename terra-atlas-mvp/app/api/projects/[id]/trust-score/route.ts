// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
export const dynamic = 'force-dynamic'
import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

// Server-side Supabase client (uses service role for reading public data)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

/**
 * GET /api/projects/:id/trust-score
 *
 * Returns trust scores for a project, synced from Symthaea via Holochain.
 * Public endpoint — trust scores are part of the project's public profile.
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params

    if (!id || id.length === 0) {
      return NextResponse.json(
        { error: 'Project ID is required' },
        { status: 400 }
      )
    }

    const supabase = createClient(supabaseUrl, supabaseAnonKey)

    // Get the latest consciousness score for this project
    const { data: latest, error: latestError } = await supabase
      .from('consciousness_scores')
      .select('*')
      .eq('project_id', id)
      .order('assessed_at', { ascending: false })
      .limit(1)
      .maybeSingle()

    if (latestError) {
      // Table may not exist yet — return empty response gracefully
      if (latestError.code === '42P01') {
        return NextResponse.json({
          project_id: id,
          latest: null,
          history: [],
          message: 'Consciousness scoring not yet configured'
        })
      }
      throw latestError
    }

    // Get recent history (last 10 assessments)
    const { data: history, error: historyError } = await supabase
      .from('consciousness_scores')
      .select('id, phi_score, harmony_alignment, care_activation, meta_awareness, assessed_at, scorer_did')
      .eq('project_id', id)
      .order('assessed_at', { ascending: false })
      .limit(10)

    if (historyError && historyError.code !== '42P01') {
      throw historyError
    }

    return NextResponse.json({
      project_id: id,
      latest: latest ? {
        phi_score: latest.phi_score,
        harmony_alignment: latest.harmony_alignment,
        per_harmony_scores: latest.per_harmony_scores,
        care_activation: latest.care_activation,
        meta_awareness: latest.meta_awareness,
        scorer_did: latest.scorer_did,
        assessment_cycle: latest.assessment_cycle,
        assessed_at: latest.assessed_at,
      } : null,
      history: (history || []).map((h: Record<string, unknown>) => ({
        id: h.id,
        phi_score: h.phi_score,
        harmony_alignment: h.harmony_alignment,
        care_activation: h.care_activation,
        meta_awareness: h.meta_awareness,
        assessed_at: h.assessed_at,
        scorer_did: h.scorer_did,
      })),
    })
  } catch (error) {
    console.error('Error fetching consciousness scores:', error)
    return NextResponse.json(
      { error: 'Failed to fetch consciousness scores' },
      { status: 500 }
    )
  }
}
