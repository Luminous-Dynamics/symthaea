// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Terra Atlas Intelligence Service
 * Orchestrates the Sacred Seven analyzers for intelligent site selection
 */

import type {
  ProjectInput,
  SiteAnalysis,
  AnalyzerResult,
  InvestmentReadiness,
  Risk,
  Opportunity,
  SearchOptions,
  PortfolioStats,
  ProjectType,
} from './types'

import { HeliosAnalyzer } from './analyzers/helios'
import { AeolusAnalyzer } from './analyzers/aeolus'
import { HydraAnalyzer } from './analyzers/hydra'
import { VulcanAnalyzer } from './analyzers/vulcan'
import { PoseidonAnalyzer } from './analyzers/poseidon'
import { ApolloAnalyzer } from './analyzers/apollo'
import { AtlasAnalyzer } from './analyzers/atlas'

// Singleton instance
let instance: IntelligenceService | null = null

/**
 * Main Intelligence Service
 * Coordinates all Sacred Seven analyzers for comprehensive site analysis
 */
export class IntelligenceService {
  private analyzers = {
    helios: new HeliosAnalyzer(),
    aeolus: new AeolusAnalyzer(),
    hydra: new HydraAnalyzer(),
    vulcan: new VulcanAnalyzer(),
    poseidon: new PoseidonAnalyzer(),
    apollo: new ApolloAnalyzer(),
    atlas: new AtlasAnalyzer(),
  }

  // Cache for analysis results
  private analysisCache = new Map<string, SiteAnalysis>()
  private cacheTimeout = 1000 * 60 * 60 // 1 hour

  /**
   * Get singleton instance
   */
  static getInstance(): IntelligenceService {
    if (!instance) {
      instance = new IntelligenceService()
    }
    return instance
  }

  /**
   * Analyze a project using applicable Sacred Seven analyzers
   */
  async analyzeProject(project: ProjectInput): Promise<SiteAnalysis> {
    // Check cache
    const cacheKey = `${project.id}-${JSON.stringify(project)}`
    const cached = this.analysisCache.get(cacheKey)
    if (cached && Date.now() - cached.timestamp.getTime() < this.cacheTimeout) {
      return cached
    }

    // Run applicable analyzers in parallel
    const analyses: SiteAnalysis['analyses'] = {}
    const analyzerPromises: Promise<void>[] = []

    // Determine which analyzers to run based on project type
    const applicableAnalyzers = this.getApplicableAnalyzers(project.type)

    for (const [key, analyzer] of Object.entries(this.analyzers)) {
      if (applicableAnalyzers.includes(key)) {
        analyzerPromises.push(
          analyzer.analyze(project).then((result) => {
            analyses[key as keyof typeof analyses] = result
          })
        )
      }
    }

    // Always run Atlas (grid/storage) for all projects
    if (!applicableAnalyzers.includes('atlas')) {
      analyzerPromises.push(
        this.analyzers.atlas.analyze(project).then((result) => {
          analyses.atlas = result
        })
      )
    }

    await Promise.all(analyzerPromises)

    // Calculate overall score
    const overallScore = this.calculateOverallScore(analyses, project.type)

    // Generate combined recommendations
    const combinedRecommendations = this.combineRecommendations(analyses)

    // Assess investment readiness
    const investmentReadiness = this.assessInvestmentReadiness(project, analyses, overallScore)

    const result: SiteAnalysis = {
      projectId: project.id,
      timestamp: new Date(),
      overallScore,
      viabilityRating: this.getViabilityRating(overallScore),
      analyses,
      combinedRecommendations,
      investmentReadiness,
    }

    // Cache result
    this.analysisCache.set(cacheKey, result)

    return result
  }

  /**
   * Batch analyze multiple projects
   */
  async analyzeProjects(projects: ProjectInput[]): Promise<SiteAnalysis[]> {
    return Promise.all(projects.map((p) => this.analyzeProject(p)))
  }

  /**
   * Get quick score without full analysis (for list views)
   */
  async getQuickScore(project: ProjectInput): Promise<number> {
    // Use primary analyzer for quick scoring
    const primaryAnalyzer = this.getPrimaryAnalyzer(project.type)
    if (!primaryAnalyzer) return 50

    const result = await this.analyzers[primaryAnalyzer].analyze(project)
    return result.score
  }

  /**
   * Search and rank projects by AI score
   */
  async searchAndRank(
    projects: ProjectInput[],
    options: SearchOptions
  ): Promise<{ projects: ProjectInput[]; analyses: Map<string, SiteAnalysis> }> {
    // Filter projects
    let filtered = projects

    if (options.types?.length) {
      filtered = filtered.filter((p) => options.types!.includes(p.type))
    }

    if (options.states?.length) {
      filtered = filtered.filter((p) => options.states!.includes(p.location.state))
    }

    if (options.minCapacity !== undefined) {
      filtered = filtered.filter((p) => p.capacityMw >= options.minCapacity!)
    }

    if (options.maxCapacity !== undefined) {
      filtered = filtered.filter((p) => p.capacityMw <= options.maxCapacity!)
    }

    // Analyze all filtered projects
    const analyses = new Map<string, SiteAnalysis>()
    const analysisPromises = filtered.map(async (p) => {
      const analysis = await this.analyzeProject(p)
      analyses.set(p.id, analysis)
      return { project: p, analysis }
    })

    const results = await Promise.all(analysisPromises)

    // Filter by minimum score
    if (options.minScore !== undefined) {
      const minScore = options.minScore
      filtered = results
        .filter((r) => r.analysis.overallScore >= minScore)
        .map((r) => r.project)
    }

    // Sort
    const sortBy = options.sortBy || 'score'
    const sortOrder = options.sortOrder || 'desc'

    filtered.sort((a, b) => {
      let comparison = 0

      switch (sortBy) {
        case 'score':
          comparison = (analyses.get(a.id)?.overallScore || 0) - (analyses.get(b.id)?.overallScore || 0)
          break
        case 'capacity':
          comparison = a.capacityMw - b.capacityMw
          break
        case 'name':
          comparison = a.name.localeCompare(b.name)
          break
        default:
          comparison = 0
      }

      return sortOrder === 'desc' ? -comparison : comparison
    })

    // Pagination
    if (options.offset !== undefined || options.limit !== undefined) {
      const start = options.offset || 0
      const end = options.limit ? start + options.limit : undefined
      filtered = filtered.slice(start, end)
    }

    return { projects: filtered, analyses }
  }

  /**
   * Generate portfolio statistics
   */
  async getPortfolioStats(projects: ProjectInput[]): Promise<PortfolioStats> {
    const analyses = await this.analyzeProjects(projects)

    const stats: PortfolioStats = {
      totalProjects: projects.length,
      totalCapacityMw: projects.reduce((sum, p) => sum + p.capacityMw, 0),
      averageScore: analyses.reduce((sum, a) => sum + a.overallScore, 0) / analyses.length || 0,
      byType: {} as PortfolioStats['byType'],
      byState: {},
      byViability: {},
    }

    // Group by type
    for (const project of projects) {
      const analysis = analyses.find((a) => a.projectId === project.id)
      const score = analysis?.overallScore || 0

      if (!stats.byType[project.type]) {
        stats.byType[project.type] = { count: 0, capacityMw: 0, avgScore: 0 }
      }
      stats.byType[project.type].count++
      stats.byType[project.type].capacityMw += project.capacityMw

      // Running average for score
      const prev = stats.byType[project.type]
      prev.avgScore = (prev.avgScore * (prev.count - 1) + score) / prev.count
    }

    // Group by state
    for (const project of projects) {
      const state = project.location.state
      if (!stats.byState[state]) {
        stats.byState[state] = { count: 0, capacityMw: 0 }
      }
      stats.byState[state].count++
      stats.byState[state].capacityMw += project.capacityMw
    }

    // Group by viability
    for (const analysis of analyses) {
      const rating = analysis.viabilityRating
      stats.byViability[rating] = (stats.byViability[rating] || 0) + 1
    }

    return stats
  }

  /**
   * Get natural language explanation for a score
   */
  explainScore(analysis: SiteAnalysis): string {
    const { overallScore, viabilityRating, analyses, investmentReadiness } = analysis

    let explanation = `This project scores ${overallScore}/100 (${viabilityRating}). `

    // Highlight top factors
    const factors: { name: string; score: number }[] = []
    for (const [key, result] of Object.entries(analyses)) {
      if (result) {
        factors.push({ name: this.getAnalyzerDisplayName(key), score: result.score })
      }
    }

    factors.sort((a, b) => b.score - a.score)

    if (factors.length > 0) {
      const top = factors[0]
      const bottom = factors[factors.length - 1]

      if (top.score > 70) {
        explanation += `Strong ${top.name.toLowerCase()} potential (${top.score}/100). `
      }

      if (bottom.score < 50 && factors.length > 1) {
        explanation += `${bottom.name} needs attention (${bottom.score}/100). `
      }
    }

    // Investment readiness
    explanation += `Investment stage: ${investmentReadiness.stage}. `
    explanation += `Estimated time to revenue: ${investmentReadiness.estimatedTimeToRevenue}.`

    return explanation
  }

  // Private helper methods

  private getApplicableAnalyzers(type: ProjectType): string[] {
    const mapping: Record<ProjectType, string[]> = {
      solar: ['helios', 'atlas'],
      wind: ['aeolus', 'atlas'],
      hydro: ['hydra', 'atlas'],
      geothermal: ['vulcan', 'atlas'],
      ocean: ['poseidon', 'atlas'],
      nuclear: ['apollo', 'atlas'],
      storage: ['atlas'],
      hybrid: ['helios', 'aeolus', 'atlas'],
    }
    return mapping[type] || ['atlas']
  }

  private getPrimaryAnalyzer(type: ProjectType): string | null {
    const mapping: Record<ProjectType, string> = {
      solar: 'helios',
      wind: 'aeolus',
      hydro: 'hydra',
      geothermal: 'vulcan',
      ocean: 'poseidon',
      nuclear: 'apollo',
      storage: 'atlas',
      hybrid: 'atlas',
    }
    return mapping[type] || null
  }

  private calculateOverallScore(
    analyses: SiteAnalysis['analyses'],
    type: ProjectType
  ): number {
    const results = Object.values(analyses).filter(Boolean) as AnalyzerResult[]
    if (results.length === 0) return 0

    // Weight primary analyzer more heavily
    const primary = this.getPrimaryAnalyzer(type)
    let totalWeight = 0
    let weightedSum = 0

    for (const [key, result] of Object.entries(analyses)) {
      if (!result) continue

      const weight = key === primary ? 2 : 1
      totalWeight += weight
      weightedSum += result.score * weight
    }

    return Math.round(weightedSum / totalWeight)
  }

  private combineRecommendations(analyses: SiteAnalysis['analyses']): string[] {
    const all: string[] = []
    for (const result of Object.values(analyses)) {
      if (result) {
        all.push(...result.recommendations)
      }
    }
    // Deduplicate and limit
    return [...new Set(all)].slice(0, 5)
  }

  private assessInvestmentReadiness(
    project: ProjectInput,
    analyses: SiteAnalysis['analyses'],
    overallScore: number
  ): InvestmentReadiness {
    // Determine stage based on project status
    let stage: InvestmentReadiness['stage'] = 'early'
    const status = project.status?.toLowerCase() || ''

    if (status.includes('operational') || status.includes('online')) {
      stage = 'operational'
    } else if (status.includes('construction') || status.includes('building')) {
      stage = 'construction'
    } else if (status.includes('permit') || status.includes('approved')) {
      stage = 'permitted'
    } else if (status.includes('development') || status.includes('active')) {
      stage = 'development'
    }

    // Estimate time to revenue based on stage
    const timeToRevenue: Record<InvestmentReadiness['stage'], string> = {
      early: '36-48 months',
      development: '24-36 months',
      permitted: '18-24 months',
      construction: '6-12 months',
      operational: 'Generating revenue',
    }

    // Collect risks and opportunities from analyses
    const risks: Risk[] = []
    const opportunities: Opportunity[] = []

    for (const result of Object.values(analyses)) {
      if (!result) continue

      // Convert warnings to risks
      for (const warning of result.warnings) {
        risks.push({
          category: 'technical',
          severity: result.confidence === 'low' ? 'high' : 'medium',
          description: warning,
        })
      }

      // High-scoring factors become opportunities
      for (const factor of result.factors) {
        if (factor.value > 75) {
          opportunities.push({
            category: 'location',
            impact: factor.value > 85 ? 'high' : 'medium',
            description: `Strong ${factor.name}: ${factor.description}`,
          })
        }
      }
    }

    // Calculate readiness score
    const stageScores: Record<InvestmentReadiness['stage'], number> = {
      early: 20,
      development: 40,
      permitted: 60,
      construction: 80,
      operational: 100,
    }

    const readinessScore = Math.round((stageScores[stage] + overallScore) / 2)

    return {
      score: readinessScore,
      stage,
      estimatedTimeToRevenue: timeToRevenue[stage],
      keyRisks: risks.slice(0, 3),
      keyOpportunities: opportunities.slice(0, 3),
    }
  }

  private getViabilityRating(score: number): SiteAnalysis['viabilityRating'] {
    if (score >= 80) return 'excellent'
    if (score >= 65) return 'good'
    if (score >= 50) return 'fair'
    if (score >= 30) return 'poor'
    return 'not_viable'
  }

  private getAnalyzerDisplayName(key: string): string {
    const names: Record<string, string> = {
      helios: 'Solar potential',
      aeolus: 'Wind resource',
      hydra: 'Hydro capacity',
      vulcan: 'Geothermal',
      poseidon: 'Ocean energy',
      apollo: 'Nuclear feasibility',
      atlas: 'Grid access',
    }
    return names[key] || key
  }
}

// Export singleton getter
export function getIntelligenceService(): IntelligenceService {
  return IntelligenceService.getInstance()
}
