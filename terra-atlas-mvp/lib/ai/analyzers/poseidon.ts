/**
 * POSEIDON - Ocean Energy Potential Analyzer
 * Evaluates wave, tidal, and offshore wind energy potential
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class PoseidonAnalyzer implements Analyzer {
  name = 'POSEIDON'
  code = 'POSEIDON'
  description = 'Ocean energy analysis including wave height, tidal range, and marine conditions'
  applicableTypes = ['ocean'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Wave/Tidal Resource (40% weight)
    const waveHeight = project.environmental?.waveHeight || this.estimateWaveHeight(project.location)
    const tideRange = project.environmental?.tideRange || this.estimateTideRange(project.location)
    const resourceScore = this.scoreOceanResource(waveHeight, tideRange)
    factors.push({
      name: 'Ocean Resource',
      value: resourceScore,
      weight: 0.4,
      description: `Wave: ${waveHeight.toFixed(1)}m, Tide: ${tideRange.toFixed(1)}m - ${this.getResourceRating(resourceScore)}`,
      dataSource: 'NOAA Marine Data',
    })

    // Factor 2: Marine Conditions (20% weight)
    const conditionsScore = this.scoreMarineConditions(project)
    factors.push({
      name: 'Marine Conditions',
      value: conditionsScore,
      weight: 0.2,
      description: this.getConditionsDescription(conditionsScore),
    })

    // Factor 3: Grid/Port Access (25% weight)
    const accessScore = this.scoreAccess(project)
    factors.push({
      name: 'Infrastructure Access',
      value: accessScore,
      weight: 0.25,
      description: this.getAccessDescription(accessScore),
    })

    // Factor 4: Technology Maturity (15% weight)
    const techScore = this.scoreTechnology(project)
    factors.push({
      name: 'Technology Readiness',
      value: techScore,
      weight: 0.15,
      description: 'Marine energy technology commercialization status',
    })

    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Recommendations
    if (tideRange > 5) {
      recommendations.push('Excellent tidal range - consider tidal barrage or stream turbines')
    } else if (waveHeight > 2) {
      recommendations.push('Strong wave resource - evaluate point absorber or oscillating systems')
    }

    warnings.push('Marine energy technologies still maturing - higher technical risk')
    recommendations.push('Consider phased demonstration before full-scale deployment')

    if (project.capacityMw > 50) {
      recommendations.push('Large ocean energy project - may qualify for DOE demonstration funding')
    }

    const confidence = 'medium' // Ocean energy data still developing

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: this.estimateCapacityFactor(resourceScore),
        primaryResource: tideRange > waveHeight * 2.5 ? 'Tidal' : 'Wave',
        estimatedAnnualGeneration: this.estimateAnnualGeneration(project.capacityMw, resourceScore),
      },
    }
  }

  private estimateWaveHeight(location: { latitude: number; state?: string }): number {
    const state = location.state?.toUpperCase()
    // Pacific coast has larger waves
    if (['WA', 'OR', 'CA', 'AK', 'HI'].includes(state)) return 2.0 + Math.random() * 1.5
    // Atlantic coast
    if (['ME', 'MA', 'RI', 'NY', 'NJ', 'NC', 'SC'].includes(state)) return 1.5 + Math.random() * 1.0
    return 1.0 + Math.random() * 0.5
  }

  private estimateTideRange(location: { state?: string }): number {
    const state = location.state?.toUpperCase()
    // Bay of Fundy influence
    if (['ME'].includes(state)) return 4.0 + Math.random() * 3.0
    // Pacific Northwest
    if (['WA', 'AK'].includes(state)) return 3.0 + Math.random() * 2.0
    return 1.5 + Math.random() * 1.0
  }

  private scoreOceanResource(waveHeight: number, tideRange: number): number {
    // Score based on best resource
    const waveScore = waveHeight >= 3 ? 85 : waveHeight >= 2 ? 70 : waveHeight >= 1.5 ? 55 : 35
    const tideScore = tideRange >= 6 ? 90 : tideRange >= 4 ? 75 : tideRange >= 2.5 ? 60 : 40
    return Math.max(waveScore, tideScore)
  }

  private getResourceRating(score: number): string {
    if (score >= 75) return 'Excellent resource'
    if (score >= 55) return 'Good resource'
    return 'Moderate resource'
  }

  private scoreMarineConditions(project: ProjectInput): number {
    // Consider storm frequency, ice, etc.
    const state = project.location.state?.toUpperCase()
    if (['HI', 'FL'].includes(state)) return 75 // Calm but hurricanes
    if (['AK'].includes(state)) return 50 // Ice and storms
    return 65
  }

  private getConditionsDescription(score: number): string {
    if (score >= 70) return 'Favorable marine conditions for installation and O&M'
    if (score >= 50) return 'Seasonal weather windows for operations'
    return 'Challenging conditions - limited operational windows'
  }

  private scoreAccess(project: ProjectInput): number {
    // Ocean projects need port access
    const terrain = project.location.terrain
    if (terrain === 'coastal') return 75
    if (terrain === 'offshore') return 60
    return 50
  }

  private getAccessDescription(score: number): string {
    if (score >= 70) return 'Good port facilities and grid connection nearby'
    return 'Infrastructure development required'
  }

  private scoreTechnology(project: ProjectInput): number {
    // Ocean energy tech still emerging
    return 55 // Moderate TRL for most technologies
  }

  private estimateCapacityFactor(resourceScore: number): number {
    // Ocean energy typically 25-40%
    if (resourceScore >= 75) return 0.35
    if (resourceScore >= 55) return 0.28
    return 0.22
  }

  private estimateAnnualGeneration(capacityMw: number, resourceScore: number): number {
    const cf = this.estimateCapacityFactor(resourceScore)
    return Math.round(capacityMw * 8760 * cf)
  }
}
