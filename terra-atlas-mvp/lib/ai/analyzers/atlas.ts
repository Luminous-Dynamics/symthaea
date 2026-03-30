// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * ATLAS - Grid & Storage Optimization Analyzer
 * Evaluates grid connectivity, storage needs, and infrastructure requirements
 */

import type { Analyzer, AnalyzerResult, ProjectInput, ScoringFactor } from '../types'

export class AtlasAnalyzer implements Analyzer {
  name = 'ATLAS'
  code = 'ATLAS'
  description = 'Grid and storage optimization including interconnection, transmission, and energy storage potential'
  applicableTypes = ['solar', 'wind', 'hydro', 'geothermal', 'ocean', 'nuclear', 'storage', 'hybrid'] as const

  async analyze(project: ProjectInput): Promise<AnalyzerResult> {
    const factors: ScoringFactor[] = []
    const recommendations: string[] = []
    const warnings: string[] = []

    // Factor 1: Interconnection Access (35% weight)
    const interconnectionScore = this.scoreInterconnection(project)
    factors.push({
      name: 'Interconnection',
      value: interconnectionScore,
      weight: 0.35,
      description: this.getInterconnectionDescription(interconnectionScore, project),
      dataSource: project.grid?.nearestSubstation ? 'Project data' : 'EIA Form 860',
    })

    // Factor 2: Transmission Capacity (25% weight)
    const transmissionScore = this.scoreTransmission(project)
    factors.push({
      name: 'Transmission',
      value: transmissionScore,
      weight: 0.25,
      description: this.getTransmissionDescription(transmissionScore, project),
    })

    // Factor 3: Market Value (25% weight)
    const marketScore = this.scoreMarketValue(project)
    factors.push({
      name: 'Market Value',
      value: marketScore,
      weight: 0.25,
      description: this.getMarketDescription(marketScore, project),
    })

    // Factor 4: Storage Synergy (15% weight)
    const storageScore = this.scoreStorageSynergy(project)
    factors.push({
      name: 'Storage Potential',
      value: storageScore,
      weight: 0.15,
      description: this.getStorageDescription(storageScore, project),
    })

    const score = Math.round(
      factors.reduce((sum, f) => sum + f.value * f.weight, 0)
    )

    // Grid-specific recommendations
    if (interconnectionScore < 60) {
      warnings.push('Grid interconnection may be challenging')
      recommendations.push('Evaluate battery storage to reduce grid upgrade requirements')
    }

    if (project.grid?.curtailmentRisk === 'high') {
      warnings.push('High curtailment risk in this region')
      recommendations.push('Co-located storage can capture otherwise curtailed energy')
    }

    if (marketScore > 70) {
      recommendations.push('Strong market fundamentals - consider merchant or hybrid PPA')
    }

    // Interconnection queue position
    if (project.grid?.interconnectionQueue) {
      const position = project.grid.interconnectionQueue.position
      if (position > 100) {
        warnings.push(`Queue position ${position} - expect 3-5 year interconnection timeline`)
      } else {
        recommendations.push(`Queue position ${position} - favorable interconnection timeline`)
      }
    }

    // Storage project specific
    if (project.type === 'storage') {
      recommendations.push('Evaluate ancillary services and capacity market revenues')
      if (marketScore > 65) {
        recommendations.push('High-value market - consider 4-hour duration for capacity credit')
      }
    }

    const hasGridData = !!project.grid?.nearestSubstation
    const confidence = hasGridData ? 'high' : 'medium'

    return {
      score,
      confidence,
      factors,
      recommendations,
      warnings,
      metadata: {
        estimatedInterconnectionCost: this.estimateInterconnectionCost(project),
        recommendedStorageMw: this.recommendStorage(project),
        marketPriceEstimate: this.estimateMarketPrice(project),
      },
    }
  }

  private scoreInterconnection(project: ProjectInput): number {
    if (!project.grid?.nearestSubstation) return 55

    const distance = project.grid.nearestSubstation.distance
    const capacity = project.grid.nearestSubstation.availableCapacity || 0

    let score = 70

    // Distance scoring
    if (distance < 5) score += 20
    else if (distance < 15) score += 10
    else if (distance > 40) score -= 20

    // Capacity headroom
    if (capacity >= project.capacityMw * 1.2) score += 10
    else if (capacity < project.capacityMw * 0.5) score -= 15

    // Queue position
    if (project.grid.interconnectionQueue) {
      const position = project.grid.interconnectionQueue.position
      if (position < 50) score += 5
      else if (position > 200) score -= 10
    }

    return Math.min(95, Math.max(25, score))
  }

  private getInterconnectionDescription(score: number, project: ProjectInput): string {
    const distance = project.grid?.nearestSubstation?.distance
    if (!distance) return 'Interconnection data not available - verify with utility'
    if (score >= 75) return `${distance.toFixed(0)} km to substation - excellent grid access`
    if (score >= 55) return `${distance.toFixed(0)} km to substation - standard interconnection`
    return `${distance.toFixed(0)} km to substation - significant grid investment required`
  }

  private scoreTransmission(project: ProjectInput): number {
    const access = project.grid?.transmissionAccess || 'nearby'

    if (access === 'direct') return 90
    if (access === 'nearby') return 70
    return 45
  }

  private getTransmissionDescription(score: number, project: ProjectInput): string {
    if (score >= 80) return 'Direct transmission line access'
    if (score >= 60) return 'Transmission available with minimal upgrade'
    return 'Transmission line extension required'
  }

  private scoreMarketValue(project: ProjectInput): number {
    const state = project.location.state?.toUpperCase()

    // High electricity price markets
    const premiumMarkets = ['CA', 'NY', 'MA', 'CT', 'NJ', 'NH', 'VT', 'ME', 'RI']
    const goodMarkets = ['TX', 'IL', 'PA', 'OH', 'MI', 'AZ', 'NV', 'CO']

    if (premiumMarkets.includes(state)) return 85
    if (goodMarkets.includes(state)) return 70
    return 55
  }

  private getMarketDescription(score: number, project: ProjectInput): string {
    if (score >= 75) return 'Premium electricity market with strong pricing'
    if (score >= 60) return 'Competitive wholesale market'
    return 'Lower electricity prices - focus on low-cost development'
  }

  private scoreStorageSynergy(project: ProjectInput): number {
    // Storage complements variable renewables
    if (project.type === 'storage') return 90
    if (project.type === 'solar' || project.type === 'wind') return 75
    if (project.type === 'hydro') return 80 // Pumped storage potential
    return 50 // Baseload doesn't need storage as much
  }

  private getStorageDescription(score: number, project: ProjectInput): string {
    if (project.type === 'storage') return 'Battery storage project - strong market demand'
    if (score >= 70) return 'Co-located storage recommended for value optimization'
    return 'Storage not critical for this resource type'
  }

  private estimateInterconnectionCost(project: ProjectInput): string {
    const distance = project.grid?.nearestSubstation?.distance || 20

    // Rough estimates: $1-3M per mile for transmission
    const costPerKm = 1.5 // Million USD
    const baseCost = 5 // Million USD for substation upgrades
    const totalCost = baseCost + distance * costPerKm

    if (totalCost < 10) return `$${totalCost.toFixed(0)}M (estimated)`
    if (totalCost < 50) return `$${totalCost.toFixed(0)}M (estimated)`
    return `$${totalCost.toFixed(0)}M+ (significant investment)`
  }

  private recommendStorage(project: ProjectInput): number {
    // Recommend storage as % of project capacity
    if (project.type === 'storage') return project.capacityMw
    if (project.type === 'solar') return Math.round(project.capacityMw * 0.5)
    if (project.type === 'wind') return Math.round(project.capacityMw * 0.3)
    return 0
  }

  private estimateMarketPrice(project: ProjectInput): string {
    const state = project.location.state?.toUpperCase()

    // Very rough wholesale price estimates ($/MWh)
    const prices: Record<string, number> = {
      CA: 65,
      NY: 55,
      TX: 40,
      IL: 35,
      PA: 38,
      DEFAULT: 35,
    }

    const price = prices[state] || prices.DEFAULT
    return `$${price}/MWh (estimated wholesale)`
  }
}
