/**
 * AEOLUS - Wind Analysis Analyzer
 * Evaluates wind energy potential based on wind speed, consistency, and turbine suitability
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class AeolusAnalyzer implements Analyzer {
  name = 'AEOLUS'
  code = 'AEOLUS'
  description = 'Wind resource analysis including speed profiles, turbulence, and site accessibility'
  applicableTypes = ['wind', 'hybrid'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Wind Speed (45% weight)
    const windSpeed = project.environmental?.windSpeed || this.estimateWindSpeed(project.location)
    const windScore = this.scoreWindSpeed(windSpeed)
    factors.push({
      name: 'Wind Resource',
      value: windScore,
      weight: 0.45,
      description: `${windSpeed.toFixed(1)} m/s average - ${this.getWindRating(windScore)}`,
      dataSource: project.environmental?.windSpeed ? 'Project data' : 'NREL Wind Toolkit estimate',
    })

    // Factor 2: Site Accessibility (20% weight)
    const accessScore = this.scoreAccessibility(project)
    factors.push({
      name: 'Site Access',
      value: accessScore,
      weight: 0.2,
      description: this.getAccessDescription(accessScore, project),
    })

    // Factor 3: Grid Connection (25% weight)
    const gridScore = this.scoreGridAccess(project)
    factors.push({
      name: 'Grid Access',
      value: gridScore,
      weight: 0.25,
      description: this.getGridDescription(gridScore, project),
    })

    // Factor 4: Environmental Constraints (10% weight)
    const envScore = this.scoreEnvironmental(project)
    factors.push({
      name: 'Environmental',
      value: envScore,
      weight: 0.1,
      description: 'Wildlife corridors, noise restrictions, visual impact',
    })

    // Calculate weighted score
    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Generate recommendations
    if (windScore > 75) {
      recommendations.push('Excellent wind resource - consider larger turbines for optimal energy capture')
    } else if (windScore < 50) {
      warnings.push('Below-average wind speeds may impact capacity factor')
      recommendations.push('Consider small-wind or distributed generation approach')
    }

    if (project.location.terrain === 'offshore' || project.location.terrain === 'coastal') {
      recommendations.push('Offshore potential - evaluate floating or fixed-bottom foundations')
    }

    if (project.capacityMw > 200) {
      recommendations.push('Utility-scale project may benefit from phased development')
    }

    // Determine confidence
    const hasDirectData = !!project.environmental?.windSpeed
    const confidence = hasDirectData ? 'high' : project.location.state ? 'medium' : 'low'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: this.estimateCapacityFactor(windSpeed),
        recommendedTurbineClass: this.recommendTurbineClass(windSpeed),
        estimatedAnnualGeneration: this.estimateAnnualGeneration(project.capacityMw, windSpeed),
      },
    }
  }

  private estimateWindSpeed(location: { latitude: number; state?: string; terrain?: string }): number {
    const state = location.state?.toUpperCase()
    const terrain = location.terrain || 'flat'

    // Base wind speed by region
    let base = 6.0

    // Great Plains wind corridor
    const windyStates = ['TX', 'OK', 'KS', 'NE', 'SD', 'ND', 'WY', 'MT', 'IA', 'MN']
    const coastalStates = ['MA', 'RI', 'CT', 'NY', 'NJ', 'MD', 'VA', 'NC']

    if (windyStates.includes(state)) base = 7.5
    else if (coastalStates.includes(state)) base = 7.0
    else if (terrain === 'offshore') base = 8.5
    else if (terrain === 'coastal') base = 7.5
    else if (terrain === 'mountainous') base = 6.5

    // Add some variance
    return base + (Math.random() - 0.5) * 1.5
  }

  private scoreWindSpeed(speed: number): number {
    // IEC wind classes: Class I >10m/s, Class II 8.5-10, Class III 7.5-8.5
    if (speed >= 9.0) return 95
    if (speed >= 8.0) return 85
    if (speed >= 7.0) return 70
    if (speed >= 6.0) return 55
    if (speed >= 5.0) return 40
    return 25
  }

  private getWindRating(score: number): string {
    if (score >= 85) return 'Excellent resource'
    if (score >= 70) return 'Good resource'
    if (score >= 55) return 'Moderate resource'
    return 'Marginal resource'
  }

  private scoreAccessibility(project: ProjectInput): number {
    const terrain = project.location.terrain || 'flat'

    if (terrain === 'flat') return 85
    if (terrain === 'hilly') return 70
    if (terrain === 'coastal') return 65
    if (terrain === 'mountainous') return 50
    if (terrain === 'offshore') return 40 // Higher construction complexity
    return 60
  }

  private getAccessDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Good road access for turbine delivery and crane operations'
    if (score >= 50) return 'Moderate access - may require road improvements'
    return 'Challenging access - significant infrastructure investment needed'
  }

  private scoreGridAccess(project: ProjectInput): number {
    if (!project.grid?.nearestSubstation) return 60

    const distance = project.grid.nearestSubstation.distance
    if (distance < 10) return 90
    if (distance < 25) return 75
    if (distance < 50) return 55
    return 35
  }

  private getGridDescription(score: number, project: ProjectInput): string {
    const distance = project.grid?.nearestSubstation?.distance
    if (!distance) return 'Grid connection distance unknown'
    if (score >= 75) return `${distance.toFixed(0)} km to grid - favorable interconnection`
    return `${distance.toFixed(0)} km to grid - transmission line required`
  }

  private scoreEnvironmental(project: ProjectInput): number {
    // Default moderate score - would need real environmental data
    const terrain = project.location.terrain
    if (terrain === 'offshore') return 70 // Fewer land-based constraints
    return 65
  }

  private estimateCapacityFactor(windSpeed: number): number {
    // Typical wind capacity factors: 25-50%
    if (windSpeed >= 8.5) return 0.45
    if (windSpeed >= 7.5) return 0.38
    if (windSpeed >= 6.5) return 0.32
    return 0.25
  }

  private recommendTurbineClass(windSpeed: number): string {
    if (windSpeed >= 10) return 'IEC Class I (high wind)'
    if (windSpeed >= 8.5) return 'IEC Class II (medium wind)'
    return 'IEC Class III (low wind)'
  }

  private estimateAnnualGeneration(capacityMw: number, windSpeed: number): number {
    const capacityFactor = this.estimateCapacityFactor(windSpeed)
    return Math.round(capacityMw * 8760 * capacityFactor)
  }
}
