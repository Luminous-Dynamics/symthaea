// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * APOLLO - Nuclear (SMR) Feasibility Analyzer
 * Evaluates nuclear energy potential with focus on Small Modular Reactors
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class ApolloAnalyzer implements Analyzer {
  name = 'APOLLO'
  code = 'APOLLO'
  description = 'Nuclear feasibility analysis including site suitability, licensing pathway, and cooling requirements'
  applicableTypes = ['nuclear'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Site Suitability (35% weight)
    const siteScore = this.scoreSiteSuitability(project)
    factors.push({
      name: 'Site Suitability',
      value: siteScore,
      weight: 0.35,
      description: this.getSiteDescription(siteScore, project),
      dataSource: 'NRC Site Evaluation Criteria',
    })

    // Factor 2: Regulatory Pathway (30% weight)
    const regScore = this.scoreRegulatoryPathway(project)
    factors.push({
      name: 'Licensing Pathway',
      value: regScore,
      weight: 0.3,
      description: this.getRegDescription(regScore, project),
    })

    // Factor 3: Grid/Market (20% weight)
    const gridScore = this.scoreGridMarket(project)
    factors.push({
      name: 'Grid & Market',
      value: gridScore,
      weight: 0.2,
      description: this.getGridDescription(gridScore, project),
    })

    // Factor 4: Community/Political (15% weight)
    const socialScore = this.scoreSocialFactors(project)
    factors.push({
      name: 'Community Factors',
      value: socialScore,
      weight: 0.15,
      description: 'Local support, workforce, and political environment',
    })

    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // SMR-specific recommendations
    const isSMR = project.capacityMw <= 300
    if (isSMR) {
      recommendations.push('SMR scale - may benefit from NRC streamlined licensing')
      if (project.metadata?.coalRetirement) {
        recommendations.push('Coal-to-nuclear conversion - existing infrastructure and workforce')
      }
    } else {
      recommendations.push('Large nuclear project - traditional NRC Part 52 licensing')
      warnings.push('Long development timeline (10-15 years typical)')
    }

    if (siteScore < 60) {
      warnings.push('Site may require additional NRC safety reviews')
    }

    if (gridScore > 70) {
      recommendations.push('Strong grid demand - nuclear baseload highly valuable')
    }

    // High confidence for nuclear due to extensive regulatory requirements
    const confidence = project.status?.toLowerCase().includes('licensed') ? 'high' : 'medium'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedCapacityFactor: 0.92, // Nuclear industry average
        reactorType: isSMR ? 'Small Modular Reactor' : 'Large Light Water Reactor',
        estimatedLicensingTimeline: isSMR ? '5-8 years' : '8-12 years',
        estimatedAnnualGeneration: Math.round(project.capacityMw * 8760 * 0.92),
      },
    }
  }

  private scoreSiteSuitability(project: ProjectInput): number {
    const state = project.location.state?.toUpperCase()
    const terrain = project.location.terrain

    let score = 70 // Base score

    // Seismic considerations
    if (project.environmental?.seismicRisk === 'high') score -= 20
    else if (project.environmental?.seismicRisk === 'low') score += 10

    // Cooling water availability
    if (terrain === 'coastal') score += 10
    if (project.environmental?.waterFlow && project.environmental.waterFlow > 50) score += 5

    // Existing nuclear-friendly states
    const nuclearStates = ['IL', 'PA', 'SC', 'TN', 'TX', 'GA', 'NC', 'VA', 'AZ']
    if (nuclearStates.includes(state)) score += 5

    // Population density (lower is better for nuclear)
    // Would need real data - using state as proxy
    const highDensityStates = ['NJ', 'RI', 'MA', 'CT', 'MD']
    if (highDensityStates.includes(state)) score -= 10

    return Math.min(95, Math.max(30, score))
  }

  private getSiteDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Favorable site characteristics for nuclear development'
    if (score >= 55) return 'Site viable with standard NRC safety reviews'
    return 'Site may face additional regulatory scrutiny'
  }

  private scoreRegulatoryPathway(project: ProjectInput): number {
    const status = project.status?.toLowerCase() || ''

    if (status.includes('licensed') || status.includes('approved')) return 90
    if (status.includes('application') || status.includes('review')) return 70
    if (status.includes('pre-application')) return 55

    // SMR pathway may be faster
    if (project.capacityMw <= 300) return 50
    return 40
  }

  private getRegDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Advanced licensing status - NRC approval pathway clear'
    if (score >= 55) return 'Standard NRC licensing timeline expected'
    return 'Early-stage - full NRC review process ahead'
  }

  private scoreGridMarket(project: ProjectInput): number {
    // Nuclear provides baseload - valuable in capacity-constrained regions
    const state = project.location.state?.toUpperCase()

    // High electricity price / capacity constrained states
    const highValueStates = ['CA', 'NY', 'MA', 'CT', 'NJ', 'IL', 'TX']
    if (highValueStates.includes(state)) return 80

    // Growing markets
    const growthStates = ['TX', 'FL', 'AZ', 'NC', 'GA', 'VA']
    if (growthStates.includes(state)) return 75

    if (project.grid?.nearestSubstation?.availableCapacity) {
      const available = project.grid.nearestSubstation.availableCapacity
      if (available >= project.capacityMw) return 85
    }

    return 65
  }

  private getGridDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Strong grid demand and favorable market conditions'
    if (score >= 60) return 'Adequate grid capacity for nuclear baseload'
    return 'Grid upgrades may be required'
  }

  private scoreSocialFactors(project: ProjectInput): number {
    const state = project.location.state?.toUpperCase()

    // States with existing nuclear industry tend to be more supportive
    const nuclearStates = ['SC', 'TN', 'IL', 'PA', 'TX', 'NC', 'GA', 'VA', 'AZ', 'AL']
    if (nuclearStates.includes(state)) return 70

    // States with anti-nuclear policies
    const challengingStates = ['CA', 'VT', 'MA', 'OR']
    if (challengingStates.includes(state)) return 40

    return 55
  }
}
