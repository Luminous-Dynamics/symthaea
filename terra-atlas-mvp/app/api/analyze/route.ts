/**
 * Project Analysis API
 * Uses the Sacred Seven AI system for intelligent site selection
 */

import { NextRequest, NextResponse } from 'next/server'
import { getIntelligenceService } from '@/lib/ai/intelligence-service'
import type { ProjectInput, SearchOptions } from '@/lib/ai/types'

// POST /api/analyze - Analyze a single project
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { project, action } = body as {
      project?: ProjectInput
      projects?: ProjectInput[]
      action?: 'analyze' | 'batch' | 'search' | 'stats'
      options?: SearchOptions
    }

    const intelligence = getIntelligenceService()

    // Single project analysis
    if (action === 'analyze' || (!action && body.project)) {
      if (!project) {
        return NextResponse.json(
          { error: 'Project data required' },
          { status: 400 }
        )
      }

      const analysis = await intelligence.analyzeProject(project)
      const explanation = intelligence.explainScore(analysis)

      return NextResponse.json({
        success: true,
        analysis,
        explanation,
      })
    }

    // Batch analysis
    if (action === 'batch' && body.projects) {
      const analyses = await intelligence.analyzeProjects(body.projects)
      return NextResponse.json({
        success: true,
        analyses,
        count: analyses.length,
      })
    }

    // Search and rank
    if (action === 'search' && body.projects) {
      const { projects, analyses } = await intelligence.searchAndRank(
        body.projects,
        body.options || {}
      )

      // Convert Map to object for JSON
      const analysisMap: Record<string, unknown> = {}
      analyses.forEach((value, key) => {
        analysisMap[key] = value
      })

      return NextResponse.json({
        success: true,
        projects,
        analyses: analysisMap,
        count: projects.length,
      })
    }

    // Portfolio stats
    if (action === 'stats' && body.projects) {
      const stats = await intelligence.getPortfolioStats(body.projects)
      return NextResponse.json({
        success: true,
        stats,
      })
    }

    return NextResponse.json(
      { error: 'Invalid action or missing data' },
      { status: 400 }
    )
  } catch (error) {
    console.error('Analysis API error:', error)
    return NextResponse.json(
      { error: 'Analysis failed', details: String(error) },
      { status: 500 }
    )
  }
}

// GET /api/analyze?id=xxx - Quick score for a project
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')

    if (!id) {
      return NextResponse.json(
        { error: 'Project ID required' },
        { status: 400 }
      )
    }

    // This would typically fetch the project from DB
    // For now, return info about the API
    return NextResponse.json({
      message: 'Use POST to analyze projects',
      endpoints: {
        analyze: 'POST with { project: ProjectInput }',
        batch: 'POST with { action: "batch", projects: ProjectInput[] }',
        search: 'POST with { action: "search", projects: ProjectInput[], options: SearchOptions }',
        stats: 'POST with { action: "stats", projects: ProjectInput[] }',
      },
      analyzers: [
        { code: 'HELIOS', name: 'Solar Optimization', types: ['solar', 'hybrid'] },
        { code: 'AEOLUS', name: 'Wind Analysis', types: ['wind', 'hybrid'] },
        { code: 'HYDRA', name: 'Hydroelectric Assessment', types: ['hydro'] },
        { code: 'VULCAN', name: 'Geothermal Evaluation', types: ['geothermal'] },
        { code: 'POSEIDON', name: 'Ocean Energy Potential', types: ['ocean'] },
        { code: 'APOLLO', name: 'Nuclear (SMR) Feasibility', types: ['nuclear'] },
        { code: 'ATLAS', name: 'Grid & Storage Optimization', types: ['all'] },
      ],
    })
  } catch (error) {
    console.error('Analysis API error:', error)
    return NextResponse.json(
      { error: 'Request failed' },
      { status: 500 }
    )
  }
}
