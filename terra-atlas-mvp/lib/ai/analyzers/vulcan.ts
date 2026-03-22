/**
 * VULCAN - Geothermal Evaluation Analyzer
 * Evaluates geothermal energy potential based on thermal gradients and geological factors
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class VulcanAnalyzer implements Analyzer {
  name = 'VULCAN'
  code = 'VULCAN'
  description = 'Geothermal evaluation including thermal gradient, reservoir quality, and drilling feasibility'
  applicableTypes = ['geothermal'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Thermal Resource (45% weight)
    const gradient = project.environmental?.geothermalGradient || this.estimateGradient(project.location)
    const thermalScore = this.scoreThermalResource(gradient)
    factors.push({
      name: 'Thermal Resource',
      value: thermalScore,
      weight: 0.45,
      description: `${gradient.toFixed(0)} °C/km gradient - ${this.getThermalRating(thermalScore)}`,
      dataSource: project.environmental?.geothermalGradient ? 'Project data' : 'USGS estimate',
    })

    // Factor 2: Geological Favorability (25% weight)
    const geoScore = this.scoreGeology(project)
    factors.push({
      name: 'Geology',
      value: geoScore,
      weight: 0.25,
      description: this.getGeoDescription(geoScore, project),
    })

    // Factor 3: Grid/Market Access (20% weight)
    const gridScore = this.scoreGridAccess(project)
    factors.push({
      name: 'Market Access',
      value: gridScore,
      weight: 0.2,
      description: this.getGridDescription(gridScore, project),
    })

    // Factor 4: Development Risk (10% weight)
    const riskScore = this.scoreRisk(project)
    factors.push({
      name: 'Development Risk',
      value: riskScore,
      weight: 0.1,
      description: 'Drilling success probability and resource confirmation',
    })

    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Recommendations
    if (thermalScore > 75) {
      recommendations.push('High-grade resource - conventional hydrothermal development viable')
    } else if (thermalScore > 50) {
      recommendations.push('Moderate resource - consider Enhanced Geothermal Systems (EGS)')
    } else {
      warnings.push('Limited thermal resource - EGS or heat pump applications only')
    }

    if (project.capacityMw < 20) {
      recommendations.push('Small-scale project - evaluate direct-use heating applications')
    }

    // Geothermal-specific states
    const state = project.location.state?.toUpperCase()
    if (['CA', 'NV', 'UT', 'ID', 'OR'].includes(state)) {
      recommendations.push('Located in geothermal-favorable region with established supply chains')
    }

    const hasDirectData = !!project.environmental?.geothermalGradient
    const confidence = hasDirectData ? 'high' : 'low'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: 0.90, // Geothermal baseload
        resourceType: this.determineResourceType(gradient),
        estimatedAnnualGeneration: Math.round(project.capacityMw * 8760 * 0.90),
      },
    }
  }

  private estimateGradient(location: { latitude: number; state?: string }): number {
    const state = location.state?.toUpperCase()

    // Western US geothermal corridor
    const highGradientStates = ['NV', 'CA', 'UT', 'ID', 'OR', 'AZ', 'NM']
    const moderateStates = ['WY', 'CO', 'MT', 'WA', 'HI', 'AK']

    if (highGradientStates.includes(state)) return 40 + Math.random() * 30
    if (moderateStates.includes(state)) return 30 + Math.random() * 15
    return 20 + Math.random() * 10
  }

  private scoreThermalResource(gradient: number): number {
    // High enthalpy >60°C/km, Medium 40-60, Low <40
    if (gradient >= 60) return 90
    if (gradient >= 45) return 75
    if (gradient >= 35) return 55
    if (gradient >= 25) return 40
    return 25
  }

  private getThermalRating(score: number): string {
    if (score >= 75) return 'High enthalpy'
    if (score >= 55) return 'Medium enthalpy'
    return 'Low enthalpy'
  }

  private scoreGeology(project: ProjectInput): number {
    const state = project.location.state?.toUpperCase()
    // Known geothermal provinces
    if (['NV', 'CA'].includes(state)) return 85
    if (['UT', 'ID', 'OR', 'HI'].includes(state)) return 75
    return 50
  }

  private getGeoDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Located in established geothermal province'
    if (score >= 50) return 'Geological setting shows geothermal potential'
    return 'Limited geological data - exploration required'
  }

  private scoreGridAccess(project: ProjectInput): number {
    if (!project.grid?.nearestSubstation) return 55
    const distance = project.grid.nearestSubstation.distance
    if (distance < 15) return 85
    if (distance < 40) return 65
    return 45
  }

  private getGridDescription(score: number, project: ProjectInput): string {
    if (score >= 70) return 'Good grid access - geothermal provides valuable baseload'
    return 'Remote location typical for geothermal sites'
  }

  private scoreRisk(project: ProjectInput): number {
    // Geothermal has exploration risk
    if (project.status?.toLowerCase().includes('producing')) return 90
    if (project.status?.toLowerCase().includes('confirmed')) return 70
    return 50
  }

  private determineResourceType(gradient: number): string {
    if (gradient >= 60) return 'Conventional hydrothermal'
    if (gradient >= 40) return 'Binary cycle'
    return 'Enhanced Geothermal System (EGS)'
  }
}
