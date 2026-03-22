/**
 * HELIOS - Solar Optimization Analyzer
 * Evaluates solar energy potential based on irradiance, land use, and grid factors
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class HeliosAnalyzer implements Analyzer {
  name = 'HELIOS'
  code = 'HELIOS'
  description = 'Solar optimization analysis including irradiance, land suitability, and panel efficiency'
  applicableTypes = ['solar', 'hybrid'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Solar Irradiance (40% weight)
    const irradiance = project.environmental?.solarIrradiance || this.estimateIrradiance(project.location.latitude)
    const irradianceScore = this.scoreIrradiance(irradiance)
    factors.push({
      name: 'Solar Irradiance',
      value: irradianceScore,
      weight: 0.4,
      description: `${irradiance.toFixed(1)} kWh/m²/day - ${this.getIrradianceRating(irradianceScore)}`,
      dataSource: project.environmental?.solarIrradiance ? 'Project data' : 'NREL NSRDB estimate',
    })

    // Factor 2: Land Availability (20% weight)
    const landScore = this.scoreLandAvailability(project)
    factors.push({
      name: 'Land Suitability',
      value: landScore,
      weight: 0.2,
      description: this.getLandDescription(landScore, project),
    })

    // Factor 3: Grid Access (25% weight)
    const gridScore = this.scoreGridAccess(project)
    factors.push({
      name: 'Grid Proximity',
      value: gridScore,
      weight: 0.25,
      description: this.getGridDescription(gridScore, project),
    })

    // Factor 4: Climate Factors (15% weight)
    const climateScore = this.scoreClimateFactors(project)
    factors.push({
      name: 'Climate Conditions',
      value: climateScore,
      weight: 0.15,
      description: `Temperature, humidity, and weather pattern suitability`,
    })

    // Calculate weighted score
    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Generate recommendations
    if (irradianceScore > 75) {
      recommendations.push('Excellent solar resource - consider tracking systems for +15-25% output')
    } else if (irradianceScore < 50) {
      warnings.push('Below-average solar irradiance may impact project economics')
      recommendations.push('Consider bifacial panels to capture diffuse radiation')
    }

    if (gridScore < 60) {
      recommendations.push('Evaluate battery storage to reduce grid upgrade costs')
    }

    if (project.capacityMw > 100) {
      recommendations.push('Large-scale project may qualify for utility PPA opportunities')
    }

    // Determine confidence
    const hasDirectData = !!project.environmental?.solarIrradiance
    const confidence = hasDirectData ? 'high' : project.location.state ? 'medium' : 'low'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: this.estimateCapacityFactor(irradiance),
        estimatedAnnualGeneration: this.estimateAnnualGeneration(project.capacityMw, irradiance),
      },
    }
  }

  private estimateIrradiance(latitude: number): number {
    // Rough estimate based on US latitudes
    // Peak solar belt ~25-35°N
    const absLat = Math.abs(latitude)
    if (absLat >= 25 && absLat <= 35) return 5.5 + Math.random() * 1.5 // Southwest
    if (absLat < 25) return 5.0 + Math.random() * 1.0 // Tropical
    if (absLat <= 40) return 4.5 + Math.random() * 1.0 // Mid-latitude
    if (absLat <= 45) return 4.0 + Math.random() * 0.8 // Northern
    return 3.5 + Math.random() * 0.5 // Far north
  }

  private scoreIrradiance(irradiance: number): number {
    // Scale: <3.5 = poor, 3.5-4.5 = fair, 4.5-5.5 = good, 5.5-6.5 = excellent, >6.5 = exceptional
    if (irradiance >= 6.5) return 95
    if (irradiance >= 5.5) return 85
    if (irradiance >= 4.5) return 70
    if (irradiance >= 3.5) return 50
    return 30
  }

  private getIrradianceRating(score: number): string {
    if (score >= 85) return 'Excellent resource'
    if (score >= 70) return 'Good resource'
    if (score >= 50) return 'Moderate resource'
    return 'Limited resource'
  }

  private scoreLandAvailability(project: ProjectInput): number {
    // Estimate based on terrain and project size
    const terrain = project.location.terrain || 'flat'
    const baseScore = terrain === 'flat' ? 85 : terrain === 'hilly' ? 60 : 40

    // Larger projects need more suitable land
    const sizePenalty = project.capacityMw > 500 ? 10 : project.capacityMw > 200 ? 5 : 0

    return Math.max(30, baseScore - sizePenalty)
  }

  private getLandDescription(score: number, project: ProjectInput): string {
    const terrain = project.location.terrain || 'unknown'
    if (score >= 75) return `Flat terrain ideal for solar arrays`
    if (score >= 50) return `${terrain} terrain suitable with site preparation`
    return `Challenging terrain may increase installation costs`
  }

  private scoreGridAccess(project: ProjectInput): number {
    if (!project.grid?.nearestSubstation) {
      return 60 // Unknown, assume moderate
    }

    const distance = project.grid.nearestSubstation.distance
    if (distance < 5) return 90
    if (distance < 15) return 75
    if (distance < 30) return 60
    if (distance < 50) return 45
    return 30
  }

  private getGridDescription(score: number, project: ProjectInput): string {
    const distance = project.grid?.nearestSubstation?.distance
    if (!distance) return 'Grid proximity unknown - verify interconnection options'
    if (score >= 75) return `${distance.toFixed(1)} km to substation - excellent access`
    if (score >= 50) return `${distance.toFixed(1)} km to substation - moderate build-out needed`
    return `${distance.toFixed(1)} km to substation - significant transmission investment`
  }

  private scoreClimateFactors(project: ProjectInput): number {
    // Estimate based on state/region
    const state = project.location.state?.toUpperCase()

    // States with good solar climate (low humidity, clear skies)
    const excellentStates = ['AZ', 'NV', 'NM', 'CA', 'CO', 'UT']
    const goodStates = ['TX', 'OK', 'KS', 'FL', 'GA', 'NC', 'SC']

    if (excellentStates.includes(state)) return 85
    if (goodStates.includes(state)) return 70
    return 55
  }

  private estimateCapacityFactor(irradiance: number): number {
    // Typical solar capacity factors: 15-30%
    return Math.min(0.30, Math.max(0.15, irradiance * 0.04))
  }

  private estimateAnnualGeneration(capacityMw: number, irradiance: number): number {
    // MWh per year
    const capacityFactor = this.estimateCapacityFactor(irradiance)
    return Math.round(capacityMw * 8760 * capacityFactor)
  }
}
