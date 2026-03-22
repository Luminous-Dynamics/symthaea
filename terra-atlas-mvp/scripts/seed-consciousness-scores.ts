// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root/**
 * Seed consciousness scores for existing Terra Atlas energy sites.
 *
 * Uses a deterministic scoring model based on project attributes
 * (type, capacity, status) to generate realistic initial Phi and
 * Harmony Alignment scores. These will be replaced by live Symthaea
 * evaluations once the runtime bridge is connected.
 *
 * Usage:
 *   npx tsx scripts/seed-consciousness-scores.ts
 *
 * Requires NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY env vars.
 */

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

if (!supabaseUrl || !serviceKey) {
  console.error('Missing NEXT_PUBLIC_SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY')
  process.exit(1)
}

const supabase = createClient(supabaseUrl, serviceKey)

// Harmony scores by project type (Eight Harmonies)
const HARMONY_PROFILES: Record<string, Record<string, number>> = {
  Solar: {
    'Resonant Coherence': 0.82, 'Pan-Sentient Flourishing': 0.75,
    'Ecological Reciprocity': 0.90, 'Epistemic Humility': 0.60,
    'Radical Translucency': 0.70, 'Compassionate Action': 0.72,
    'Temporal Stewardship': 0.85, 'Sacred Stillness': 0.55,
  },
  Wind: {
    'Resonant Coherence': 0.78, 'Pan-Sentient Flourishing': 0.70,
    'Ecological Reciprocity': 0.85, 'Epistemic Humility': 0.62,
    'Radical Translucency': 0.68, 'Compassionate Action': 0.65,
    'Temporal Stewardship': 0.82, 'Sacred Stillness': 0.58,
  },
  Hydro: {
    'Resonant Coherence': 0.75, 'Pan-Sentient Flourishing': 0.68,
    'Ecological Reciprocity': 0.72, 'Epistemic Humility': 0.65,
    'Radical Translucency': 0.72, 'Compassionate Action': 0.70,
    'Temporal Stewardship': 0.88, 'Sacred Stillness': 0.62,
  },
  Nuclear: {
    'Resonant Coherence': 0.70, 'Pan-Sentient Flourishing': 0.55,
    'Ecological Reciprocity': 0.60, 'Epistemic Humility': 0.75,
    'Radical Translucency': 0.80, 'Compassionate Action': 0.50,
    'Temporal Stewardship': 0.90, 'Sacred Stillness': 0.45,
  },
  Storage: {
    'Resonant Coherence': 0.80, 'Pan-Sentient Flourishing': 0.72,
    'Ecological Reciprocity': 0.78, 'Epistemic Humility': 0.58,
    'Radical Translucency': 0.65, 'Compassionate Action': 0.68,
    'Temporal Stewardship': 0.75, 'Sacred Stillness': 0.52,
  },
  Geothermal: {
    'Resonant Coherence': 0.85, 'Pan-Sentient Flourishing': 0.78,
    'Ecological Reciprocity': 0.88, 'Epistemic Humility': 0.70,
    'Radical Translucency': 0.72, 'Compassionate Action': 0.75,
    'Temporal Stewardship': 0.92, 'Sacred Stillness': 0.65,
  },
}

// Deterministic consciousness score from project attributes
function computeScores(project: Record<string, unknown>) {
  const type = String(project.type || project.project_type || 'Solar')
  const capacityMw = Number(project.capacity_mw || project.power_mw || 0)
  const status = String(project.status || 'planning')

  // Base Phi from project characteristics
  // Higher capacity = more integration potential, but diminishing returns
  const capacityFactor = Math.min(Math.log10(Math.max(capacityMw, 1) + 1) / 4, 1.0)

  // Status maturity factor
  const statusFactors: Record<string, number> = {
    'operational': 0.85, 'Operational': 0.85,
    'construction': 0.70, 'Construction': 0.70,
    'development': 0.55, 'Development': 0.55,
    'planning': 0.40, 'Planning': 0.40,
    'funding': 0.45, 'Funding': 0.45,
    'Retrofit Opportunity': 0.50,
    'NRC Review': 0.60,
    'Site Preparation': 0.55,
  }
  const statusFactor = statusFactors[status] || 0.45

  // Phi = capacity contribution + status contribution + type bonus
  const typeBonus: Record<string, number> = {
    Solar: 0.08, Wind: 0.06, Geothermal: 0.10, Hydro: 0.05,
    Nuclear: 0.03, Storage: 0.07,
  }
  const phi = Math.min(
    0.3 * capacityFactor + 0.5 * statusFactor + (typeBonus[type] || 0.04) + 0.1,
    0.95
  )

  // Harmony alignment from type profile
  const profile = HARMONY_PROFILES[type] || HARMONY_PROFILES['Solar']
  const harmonyValues = Object.values(profile)
  const harmonyAlignment = harmonyValues.reduce((s, v) => s + v, 0) / harmonyValues.length

  // Care activation correlates with community benefit potential
  const careActivation = Math.min(0.4 + capacityFactor * 0.3 + statusFactor * 0.2, 0.90)

  // Meta-awareness from project transparency/maturity
  const metaAwareness = Math.min(statusFactor * 0.8 + 0.15, 0.85)

  return {
    phi_score: Math.round(phi * 1000) / 1000,
    harmony_alignment: Math.round(harmonyAlignment * 1000) / 1000,
    per_harmony_scores: profile,
    care_activation: Math.round(careActivation * 1000) / 1000,
    meta_awareness: Math.round(metaAwareness * 1000) / 1000,
  }
}

async function main() {
  console.log('Fetching existing sites from Supabase...')

  // Try to get sites from the sites table
  const { data: sites, error: sitesError } = await supabase
    .from('sites')
    .select('id, name, type, power_mw, status, country, state')
    .limit(100)

  if (sitesError) {
    console.log('Sites table not available, trying energy_projects...')
    const { data: projects, error: projError } = await supabase
      .from('energy_projects')
      .select('id, name, project_type, capacity_mw, status, country, state')
      .limit(100)

    if (projError) {
      console.error('No project tables found:', projError.message)
      process.exit(1)
    }

    await seedScores(projects || [], 'energy_projects')
    return
  }

  await seedScores(sites || [], 'sites')
}

async function seedScores(projects: Record<string, unknown>[], source: string) {
  console.log(`Found ${projects.length} projects from ${source}`)

  if (projects.length === 0) {
    console.log('No projects to seed. Seeding demo scores instead...')
    await seedDemoScores()
    return
  }

  const scores = projects.map((p) => {
    const computed = computeScores(p)
    return {
      project_id: String(p.id),
      ...computed,
      per_harmony_scores: computed.per_harmony_scores,
      scorer_did: 'did:mycelix:symthaea-seed',
      assessment_cycle: 0,
    }
  })

  console.log(`Inserting ${scores.length} consciousness scores...`)

  const { error } = await supabase
    .from('consciousness_scores')
    .upsert(scores, { onConflict: 'project_id' })
    .select()

  if (error) {
    // Table might not exist — try creating it
    if (error.code === '42P01') {
      console.error('consciousness_scores table does not exist. Run migration 005 first.')
      process.exit(1)
    }
    // upsert conflict column might not be unique — insert instead
    console.log('Upsert failed, trying insert...', error.message)
    const { error: insertError } = await supabase
      .from('consciousness_scores')
      .insert(scores)

    if (insertError) {
      console.error('Insert failed:', insertError.message)
      process.exit(1)
    }
  }

  console.log('Seeded consciousness scores:')
  for (const s of scores) {
    console.log(`  ${s.project_id}: Phi=${s.phi_score} Harmony=${s.harmony_alignment} Care=${s.care_activation}`)
  }
  console.log('Done.')
}

async function seedDemoScores() {
  const demoProjects = [
    { id: 'demo-solar-1', type: 'Solar', capacity_mw: 50, status: 'operational' },
    { id: 'demo-wind-1', type: 'Wind', capacity_mw: 120, status: 'construction' },
    { id: 'demo-hydro-1', type: 'Hydro', capacity_mw: 200, status: 'operational' },
    { id: 'demo-nuclear-1', type: 'Nuclear', capacity_mw: 300, status: 'NRC Review' },
    { id: 'demo-geo-1', type: 'Geothermal', capacity_mw: 30, status: 'development' },
    { id: 'demo-storage-1', type: 'Storage', capacity_mw: 80, status: 'planning' },
  ]

  const scores = demoProjects.map((p) => {
    const computed = computeScores(p)
    return {
      project_id: p.id,
      ...computed,
      per_harmony_scores: computed.per_harmony_scores,
      scorer_did: 'did:mycelix:symthaea-seed-demo',
      assessment_cycle: 0,
    }
  })

  const { error } = await supabase.from('consciousness_scores').insert(scores)
  if (error) {
    console.error('Demo seed failed:', error.message)
    process.exit(1)
  }

  console.log('Seeded demo consciousness scores:')
  for (const s of scores) {
    console.log(`  ${s.project_id}: Phi=${s.phi_score} Harmony=${s.harmony_alignment}`)
  }
}

main().catch(console.error)
