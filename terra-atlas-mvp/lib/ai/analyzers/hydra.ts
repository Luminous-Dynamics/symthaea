/**
 * HYDRA - Hydroelectric Assessment Analyzer
 * Evaluates hydroelectric potential including dam retrofits, run-of-river, and pumped storage
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class HydraAnalyzer implements Analyzer {
  name = 'HYDRA'
  code = 'HYDRA'
  description = 'Hydroelectric assessment including water flow, head height, and environmental impact'
  applicableTypes = ['hydro'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Water Resource (40% weight)
    const waterFlow = project.environmental?.waterFlow || this.estimateWaterFlow(project)
    const waterScore = this.scoreWaterResource(waterFlow, project.capacityMw)
    factors.push({
      name: 'Water Resource',
      value: waterScore,
      weight: 0.4,
      description: `${waterFlow.toFixed(0)} m³/s - ${this.getWaterRating(waterScore)}`,
      dataSource: project.environmental?.waterFlow ? 'Project data' : 'USGS estimate',
    })

    // Factor 2: Infrastructure (25% weight)
    const infraScore = this.scoreInfrastructure(project)
    factors.push({
      name: 'Infrastructure',
      value: infraScore,
      weight: 0.25,
      description: this.getInfraDescription(infraScore, project),
    })

    // Factor 3: Grid Access (20% weight)
    const gridScore = this.scoreGridAccess(project)
    factors.push({
      name: 'Grid Connection',
      value: gridScore,
      weight: 0.2,
      description: this.getGridDescription(gridScore, project),
    })

    // Factor 4: Environmental/Regulatory (15% weight)
    const envScore = this.scoreEnvironmental(project)
    factors.push({
      name: 'Environmental',
      value: envScore,
      weight: 0.15,
      description: 'Fish passage, water rights, FERC licensing requirements',
    })

    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Recommendations
    if (project.metadata?.isExistingDam) {
      recommendations.push('Dam retrofit project - faster permitting timeline possible')
    }

    if (waterScore > 70 && infraScore > 60) {
      recommendations.push('Strong hydro fundamentals - consider baseload PPA structure')
    }

    if (envScore < 50) {
      warnings.push('Environmental constraints may extend permitting timeline')
      recommendations.push('Early engagement with fish & wildlife agencies recommended')
    }

    if (project.capacityMw > 30) {
      recommendations.push('Large hydro may qualify for capacity market revenues')
    }

    const hasDirectData = !!project.environmental?.waterFlow
    const confidence = hasDirectData ? 'high' : 'medium'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: this.estimateCapacityFactor(waterScore),
        projectType: this.determineProjectType(project),
        estimatedAnnualGeneration: this.estimateAnnualGeneration(project.capacityMw, waterScore),
      },
    }
  }

  private estimateWaterFlow(project: ProjectInput): number {
    // Estimate based on capacity - typical 10-20 m³/s per MW for run-of-river
    return project.capacityMw * 15 + Math.random() * project.capacityMw * 5
  }

  private scoreWaterResource(flow: number, capacity: number): number {
    // Check if flow is adequate for stated capacity
    const flowPerMw = flow / capacity
    if (flowPerMw >= 20) return 90
    if (flowPerMw >= 15) return 75
    if (flowPerMw >= 10) return 60
    if (flowPerMw >= 5) return 45
    return 30
  }

  private getWaterRating(score: number): string {
    if (score >= 75) return 'Excellent flow'
    if (score >= 60) return 'Good flow'
    if (score >= 45) return 'Adequate flow'
    return 'Limited flow'
  }

  private scoreInfrastructure(project: ProjectInput): number {
    // Existing dam retrofits score higher
    if (project.metadata?.isExistingDam) return 85
    if (project.status?.toLowerCase().includes('existing')) return 80
    return 60
  }

  private getInfraDescription(score: number, project: ProjectInput): string {
    if (score >= 80) return 'Existing dam infrastructure - retrofit opportunity'
    if (score >= 60) return 'Some existing infrastructure present'
    return 'New construction required'
  }

  private scoreGridAccess(project: ProjectInput): number {
    if (!project.grid?.nearestSubstation) return 65 // Hydro often near existing grid
    const distance = project.grid.nearestSubstation.distance
    if (distance < 10) return 90
    if (distance < 25) return 75
    return 55
  }

  private getGridDescription(score: number, project: ProjectInput): string {
    const distance = project.grid?.nearestSubstation?.distance
    if (!distance) return 'Grid access typically favorable for hydro sites'
    return `${distance.toFixed(0)} km to substation`
  }

  private scoreEnvironmental(project: ProjectInput): number {
    // Base score - would need real environmental assessment
    const state = project.location.state?.toUpperCase()
    // Pacific Northwest has more stringent fish protection
    if (['WA', 'OR', 'ID', 'MT'].includes(state)) return 55
    return 65
  }

  private estimateCapacityFactor(waterScore: number): number {
    // Hydro typically 30-60% capacity factor
    if (waterScore >= 75) return 0.55
    if (waterScore >= 60) return 0.45
    return 0.35
  }

  private determineProjectType(project: ProjectInput): string {
    if (project.metadata?.isExistingDam) return 'Dam retrofit'
    if (project.capacityMw > 100) return 'Conventional hydro'
    if (project.capacityMw < 10) return 'Small hydro'
    return 'Run-of-river'
  }

  private estimateAnnualGeneration(capacityMw: number, waterScore: number): number {
    const cf = this.estimateCapacityFactor(waterScore)
    return Math.round(capacityMw * 8760 * cf)
  }
}
