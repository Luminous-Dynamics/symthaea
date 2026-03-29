// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Cross-Domain Metabolic Dashboard
 *
 * Provides a unified view of resource flows across all 5 civilizational domains:
 * - Food: Nutrition, security, distribution
 * - Water: Quality, availability, sustainability
 * - Energy: Generation, consumption, grid stability
 * - Shelter: Housing security, community resources
 * - Medicine: Healthcare access, treatment outcomes
 *
 * The dashboard aggregates metrics from domain-specific hApps and bridges
 * to provide holistic health indicators for communities and households.
 *
 * @packageDocumentation
 * @module metabolism/cross-domain-dashboard
 */

import type { AgentPubKey } from '@holochain/client';

// ============================================================================
// Domain Types
// ============================================================================

/**
 * The 5 civilizational resource domains
 */
export type ResourceDomain = 'food' | 'water' | 'energy' | 'shelter' | 'medicine';

/**
 * Health status levels for domains
 */
export type DomainHealthStatus = 'healthy' | 'adequate' | 'stressed' | 'critical' | 'unknown';

/**
 * Trend direction for metrics
 */
export type TrendDirection = 'improving' | 'stable' | 'declining' | 'volatile';

// ============================================================================
// Domain Metrics
// ============================================================================

/**
 * Base metrics common to all domains
 */
export interface BaseDomainMetrics {
  /** Domain identifier */
  domain: ResourceDomain;
  /** Overall health score (0-1) */
  healthScore: number;
  /** Health status category */
  status: DomainHealthStatus;
  /** 7-day trend */
  trend: TrendDirection;
  /** Confidence in the metrics (0-1) */
  confidence: number;
  /** Last update timestamp */
  lastUpdated: number;
  /** Number of active data sources */
  activeDataSources: number;
}

/**
 * Food domain metrics
 */
export interface FoodMetrics extends BaseDomainMetrics {
  domain: 'food';
  /** Food security level */
  securityLevel: 'high' | 'marginal' | 'low' | 'very_low';
  /** Days of food reserves */
  reserveDays: number;
  /** Nutrition adequacy score (0-1) */
  nutritionScore: number;
  /** Local production percentage */
  localProductionPercent: number;
  /** Active community food programs */
  communityPrograms: number;
  /** Seasonal availability index */
  seasonalIndex: number;
}

/**
 * Water domain metrics
 */
export interface WaterMetrics extends BaseDomainMetrics {
  domain: 'water';
  /** Water quality index (0-100) */
  qualityIndex: number;
  /** Days of water reserves */
  reserveDays: number;
  /** Per capita daily consumption (liters) */
  perCapitaConsumption: number;
  /** Infrastructure health (0-1) */
  infrastructureHealth: number;
  /** Groundwater level status */
  groundwaterStatus: 'normal' | 'low' | 'critical';
  /** Recycling/reuse percentage */
  recyclingPercent: number;
}

/**
 * Energy domain metrics
 */
export interface EnergyMetrics extends BaseDomainMetrics {
  domain: 'energy';
  /** Grid reliability (uptime percentage) */
  gridReliability: number;
  /** Renewable percentage */
  renewablePercent: number;
  /** Peak load vs capacity ratio */
  peakLoadRatio: number;
  /** Storage capacity (hours of backup) */
  storageHours: number;
  /** Average cost per kWh */
  avgCostPerKwh: number;
  /** Carbon intensity (g CO2/kWh) */
  carbonIntensity: number;
}

/**
 * Shelter domain metrics
 */
export interface ShelterMetrics extends BaseDomainMetrics {
  domain: 'shelter';
  /** Housing security level */
  securityLevel: 'stable' | 'at_risk' | 'insecure' | 'unhoused';
  /** Affordability index (income to housing cost ratio) */
  affordabilityIndex: number;
  /** Occupancy rate in community */
  occupancyRate: number;
  /** Available emergency beds */
  emergencyBeds: number;
  /** Average condition score (0-1) */
  conditionScore: number;
  /** Community resource hubs */
  resourceHubs: number;
}

/**
 * Medicine/Healthcare domain metrics
 */
export interface MedicineMetrics extends BaseDomainMetrics {
  domain: 'medicine';
  /** Healthcare access score (0-1) */
  accessScore: number;
  /** Average wait time (hours) for non-emergency */
  avgWaitHours: number;
  /** Preventive care coverage (0-1) */
  preventiveCoverage: number;
  /** Essential medicine availability (0-1) */
  medicineAvailability: number;
  /** Provider density (per 1000 population) */
  providerDensity: number;
  /** Mental health resource score (0-1) */
  mentalHealthScore: number;
}

/**
 * Union of all domain metrics
 */
export type DomainMetrics =
  | FoodMetrics
  | WaterMetrics
  | EnergyMetrics
  | ShelterMetrics
  | MedicineMetrics;

// ============================================================================
// Cross-Domain Analysis
// ============================================================================

/**
 * Cross-domain dependency relationship
 */
export interface DomainDependency {
  /** Source domain */
  from: ResourceDomain;
  /** Target domain */
  to: ResourceDomain;
  /** Dependency strength (0-1) */
  strength: number;
  /** Type of dependency */
  type: 'critical' | 'important' | 'supportive';
  /** Description of the dependency */
  description: string;
}

/**
 * Cross-domain alert
 */
export interface CrossDomainAlert {
  /** Alert identifier */
  alertId: string;
  /** Severity level */
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  /** Affected domains */
  affectedDomains: ResourceDomain[];
  /** Alert title */
  title: string;
  /** Detailed description */
  description: string;
  /** Recommended actions */
  recommendations: string[];
  /** When the alert was triggered */
  triggeredAt: number;
  /** Whether the alert has been acknowledged */
  acknowledged: boolean;
  /** Source of the alert */
  source: 'automatic' | 'manual' | 'predictive';
}

/**
 * Cross-domain recommendation
 */
export interface CrossDomainRecommendation {
  /** Recommendation identifier */
  recommendationId: string;
  /** Priority (1 = highest) */
  priority: number;
  /** Affected domains */
  domains: ResourceDomain[];
  /** Action to take */
  action: string;
  /** Expected impact description */
  expectedImpact: string;
  /** Estimated resources needed */
  resourcesNeeded: string;
  /** Confidence in recommendation (0-1) */
  confidence: number;
}

/**
 * Cascade risk assessment
 */
export interface CascadeRisk {
  /** Origin domain where failure starts */
  originDomain: ResourceDomain;
  /** Domains that would be affected */
  cascadePath: ResourceDomain[];
  /** Probability of cascade occurring (0-1) */
  probability: number;
  /** Potential impact severity (0-1) */
  impactSeverity: number;
  /** Risk score (probability * severity) */
  riskScore: number;
  /** Mitigation strategies */
  mitigations: string[];
}

// ============================================================================
// Dashboard State
// ============================================================================

/**
 * Complete dashboard state
 */
export interface DashboardState {
  /** Timestamp of state */
  timestamp: number;
  /** Agent viewing the dashboard */
  agentId?: AgentPubKey;
  /** Scope of data (household, community, region) */
  scope: 'household' | 'community' | 'region' | 'global';
  /** Domain metrics */
  metrics: {
    food: FoodMetrics;
    water: WaterMetrics;
    energy: EnergyMetrics;
    shelter: ShelterMetrics;
    medicine: MedicineMetrics;
  };
  /** Overall health score (0-1) */
  overallHealth: number;
  /** Overall status */
  overallStatus: DomainHealthStatus;
  /** Active alerts */
  alerts: CrossDomainAlert[];
  /** Current recommendations */
  recommendations: CrossDomainRecommendation[];
  /** Cascade risks */
  cascadeRisks: CascadeRisk[];
  /** Domain dependencies */
  dependencies: DomainDependency[];
}

/**
 * Dashboard summary for quick view
 */
export interface DashboardSummary {
  /** Timestamp */
  timestamp: number;
  /** Overall health score */
  overallHealth: number;
  /** Domain scores */
  domainScores: Record<ResourceDomain, number>;
  /** Number of active alerts by severity */
  alertCounts: {
    info: number;
    warning: number;
    critical: number;
    emergency: number;
  };
  /** Top 3 priorities */
  topPriorities: string[];
  /** Domains needing attention */
  domainsNeedingAttention: ResourceDomain[];
}

// ============================================================================
// Dashboard Configuration
// ============================================================================

/**
 * Dashboard configuration
 */
export interface DashboardConfig {
  /** Update interval in milliseconds */
  updateIntervalMs: number;
  /** Alert thresholds by domain */
  alertThresholds: Record<ResourceDomain, number>;
  /** Enable predictive alerts */
  enablePredictiveAlerts: boolean;
  /** Cascade risk calculation enabled */
  enableCascadeRiskAnalysis: boolean;
  /** Data sources to include */
  dataSources: {
    hApps: boolean;
    bridges: boolean;
    external: boolean;
  };
  /** Historical data retention (days) */
  historyRetentionDays: number;
}

/**
 * Default dashboard configuration
 */
export const DEFAULT_DASHBOARD_CONFIG: DashboardConfig = {
  updateIntervalMs: 60000, // 1 minute
  alertThresholds: {
    food: 0.4,
    water: 0.4,
    energy: 0.3,
    shelter: 0.4,
    medicine: 0.35,
  },
  enablePredictiveAlerts: true,
  enableCascadeRiskAnalysis: true,
  dataSources: {
    hApps: true,
    bridges: true,
    external: false,
  },
  historyRetentionDays: 30,
};

// ============================================================================
// Dashboard Service
// ============================================================================

/**
 * Cross-Domain Metabolic Dashboard Service
 *
 * Aggregates data from all 5 resource domains and provides unified analysis.
 */
export class CrossDomainDashboard {
  private config: DashboardConfig;
  private currentState: DashboardState | null = null;
  private historyBuffer: DashboardState[] = [];
  private alertHandlers: Array<(alert: CrossDomainAlert) => void> = [];

  /**
   * Standard domain dependencies
   * These represent the interconnected nature of civilizational resources
   */
  private static readonly STANDARD_DEPENDENCIES: DomainDependency[] = [
    // Energy dependencies
    {
      from: 'energy',
      to: 'water',
      strength: 0.9,
      type: 'critical',
      description: 'Water pumping and treatment requires electricity',
    },
    {
      from: 'energy',
      to: 'shelter',
      strength: 0.7,
      type: 'important',
      description: 'Heating, cooling, and lighting depend on energy',
    },
    {
      from: 'energy',
      to: 'medicine',
      strength: 0.95,
      type: 'critical',
      description: 'Medical equipment and refrigeration require power',
    },
    {
      from: 'energy',
      to: 'food',
      strength: 0.8,
      type: 'critical',
      description: 'Food storage, processing, and distribution need energy',
    },
    // Water dependencies
    {
      from: 'water',
      to: 'food',
      strength: 0.95,
      type: 'critical',
      description: 'Agriculture and food preparation require water',
    },
    {
      from: 'water',
      to: 'medicine',
      strength: 0.85,
      type: 'critical',
      description: 'Sanitation and medical procedures require clean water',
    },
    // Food dependencies
    {
      from: 'food',
      to: 'medicine',
      strength: 0.6,
      type: 'important',
      description: 'Nutrition affects health outcomes and recovery',
    },
    // Shelter dependencies
    {
      from: 'shelter',
      to: 'medicine',
      strength: 0.5,
      type: 'supportive',
      description: 'Stable housing supports health and medication adherence',
    },
    {
      from: 'shelter',
      to: 'food',
      strength: 0.4,
      type: 'supportive',
      description: 'Housing enables food storage and preparation',
    },
    // Bidirectional water-energy
    {
      from: 'water',
      to: 'energy',
      strength: 0.6,
      type: 'important',
      description: 'Hydroelectric and cooling water for power plants',
    },
  ];

  constructor(config: Partial<DashboardConfig> = {}) {
    this.config = { ...DEFAULT_DASHBOARD_CONFIG, ...config };
  }

  /**
   * Update dashboard with new metrics
   */
  public update(metrics: Partial<DashboardState['metrics']>): DashboardState {
    const now = Date.now();
    const state: DashboardState = {
      timestamp: now,
      scope: 'community',
      metrics: {
        food: metrics.food ?? this.createDefaultMetrics('food') as FoodMetrics,
        water: metrics.water ?? this.createDefaultMetrics('water') as WaterMetrics,
        energy: metrics.energy ?? this.createDefaultMetrics('energy') as EnergyMetrics,
        shelter: metrics.shelter ?? this.createDefaultMetrics('shelter') as ShelterMetrics,
        medicine: metrics.medicine ?? this.createDefaultMetrics('medicine') as MedicineMetrics,
      },
      overallHealth: 0,
      overallStatus: 'unknown',
      alerts: [],
      recommendations: [],
      cascadeRisks: [],
      dependencies: CrossDomainDashboard.STANDARD_DEPENDENCIES,
    };

    // Calculate overall health
    state.overallHealth = this.calculateOverallHealth(state.metrics);
    state.overallStatus = this.healthScoreToStatus(state.overallHealth);

    // Generate alerts
    state.alerts = this.generateAlerts(state);

    // Generate recommendations
    state.recommendations = this.generateRecommendations(state);

    // Calculate cascade risks
    if (this.config.enableCascadeRiskAnalysis) {
      state.cascadeRisks = this.calculateCascadeRisks(state);
    }

    // Store state
    this.currentState = state;
    this.historyBuffer.push(state);

    // Trim history
    const maxHistory = (this.config.historyRetentionDays * 24 * 60 * 60 * 1000) / this.config.updateIntervalMs;
    while (this.historyBuffer.length > maxHistory) {
      this.historyBuffer.shift();
    }

    // Notify alert handlers
    for (const alert of state.alerts) {
      for (const handler of this.alertHandlers) {
        handler(alert);
      }
    }

    return state;
  }

  /**
   * Get current dashboard state
   */
  public getState(): DashboardState | null {
    return this.currentState;
  }

  /**
   * Get dashboard summary
   */
  public getSummary(): DashboardSummary | null {
    if (!this.currentState) return null;

    const state = this.currentState;
    return {
      timestamp: state.timestamp,
      overallHealth: state.overallHealth,
      domainScores: {
        food: state.metrics.food.healthScore,
        water: state.metrics.water.healthScore,
        energy: state.metrics.energy.healthScore,
        shelter: state.metrics.shelter.healthScore,
        medicine: state.metrics.medicine.healthScore,
      },
      alertCounts: {
        info: state.alerts.filter(a => a.severity === 'info').length,
        warning: state.alerts.filter(a => a.severity === 'warning').length,
        critical: state.alerts.filter(a => a.severity === 'critical').length,
        emergency: state.alerts.filter(a => a.severity === 'emergency').length,
      },
      topPriorities: state.recommendations.slice(0, 3).map(r => r.action),
      domainsNeedingAttention: this.getDomainsNeedingAttention(state),
    };
  }

  /**
   * Get historical data for a domain
   */
  public getHistory(domain: ResourceDomain, hoursBack: number = 24): Array<{
    timestamp: number;
    healthScore: number;
    status: DomainHealthStatus;
  }> {
    const cutoff = Date.now() - (hoursBack * 60 * 60 * 1000);
    return this.historyBuffer
      .filter(s => s.timestamp >= cutoff)
      .map(s => ({
        timestamp: s.timestamp,
        healthScore: s.metrics[domain].healthScore,
        status: s.metrics[domain].status,
      }));
  }

  /**
   * Register an alert handler
   */
  public onAlert(handler: (alert: CrossDomainAlert) => void): void {
    this.alertHandlers.push(handler);
  }

  /**
   * Acknowledge an alert
   */
  public acknowledgeAlert(alertId: string): boolean {
    if (!this.currentState) return false;

    const alert = this.currentState.alerts.find(a => a.alertId === alertId);
    if (alert) {
      alert.acknowledged = true;
      return true;
    }
    return false;
  }

  /**
   * Get domain dependencies
   */
  public getDependencies(domain?: ResourceDomain): DomainDependency[] {
    const deps = CrossDomainDashboard.STANDARD_DEPENDENCIES;
    if (domain) {
      return deps.filter(d => d.from === domain || d.to === domain);
    }
    return deps;
  }

  /**
   * Simulate a domain failure to see cascade effects
   */
  public simulateFailure(domain: ResourceDomain): CascadeRisk {
    // BFS to find all affected domains
    const affected: ResourceDomain[] = [];
    const visited = new Set<ResourceDomain>([domain]);
    const queue = [domain];
    let totalProbability = 1.0;

    while (queue.length > 0) {
      const current = queue.shift()!;
      const outgoing = CrossDomainDashboard.STANDARD_DEPENDENCIES.filter(d => d.from === current);

      for (const dep of outgoing) {
        if (!visited.has(dep.to)) {
          visited.add(dep.to);
          affected.push(dep.to);
          totalProbability *= dep.strength;
          if (dep.type === 'critical') {
            queue.push(dep.to); // Critical deps cascade further
          }
        }
      }
    }

    const impactSeverity = affected.length / 4; // Max is 4 other domains

    return {
      originDomain: domain,
      cascadePath: affected,
      probability: Math.min(totalProbability, 0.9),
      impactSeverity,
      riskScore: totalProbability * impactSeverity,
      mitigations: this.generateMitigations(domain, affected),
    };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private createDefaultMetrics(domain: ResourceDomain): DomainMetrics {
    const base: BaseDomainMetrics = {
      domain,
      healthScore: 0.5,
      status: 'unknown',
      trend: 'stable',
      confidence: 0.0,
      lastUpdated: Date.now(),
      activeDataSources: 0,
    };

    switch (domain) {
      case 'food':
        return {
          ...base,
          domain: 'food',
          securityLevel: 'marginal',
          reserveDays: 7,
          nutritionScore: 0.5,
          localProductionPercent: 20,
          communityPrograms: 0,
          seasonalIndex: 0.5,
        };
      case 'water':
        return {
          ...base,
          domain: 'water',
          qualityIndex: 50,
          reserveDays: 3,
          perCapitaConsumption: 150,
          infrastructureHealth: 0.5,
          groundwaterStatus: 'normal',
          recyclingPercent: 10,
        };
      case 'energy':
        return {
          ...base,
          domain: 'energy',
          gridReliability: 0.95,
          renewablePercent: 20,
          peakLoadRatio: 0.7,
          storageHours: 4,
          avgCostPerKwh: 0.15,
          carbonIntensity: 400,
        };
      case 'shelter':
        return {
          ...base,
          domain: 'shelter',
          securityLevel: 'stable',
          affordabilityIndex: 0.3,
          occupancyRate: 0.95,
          emergencyBeds: 100,
          conditionScore: 0.6,
          resourceHubs: 5,
        };
      case 'medicine':
        return {
          ...base,
          domain: 'medicine',
          accessScore: 0.6,
          avgWaitHours: 4,
          preventiveCoverage: 0.5,
          medicineAvailability: 0.7,
          providerDensity: 2.5,
          mentalHealthScore: 0.4,
        };
    }
  }

  private calculateOverallHealth(metrics: DashboardState['metrics']): number {
    // Weighted average with critical domains weighted higher
    const weights = {
      food: 1.0,
      water: 1.2, // Critical
      energy: 1.1, // Critical
      shelter: 0.9,
      medicine: 1.0,
    };

    let weightedSum = 0;
    let totalWeight = 0;

    for (const [domain, weight] of Object.entries(weights)) {
      const score = metrics[domain as ResourceDomain].healthScore;
      weightedSum += score * weight;
      totalWeight += weight;
    }

    return weightedSum / totalWeight;
  }

  private healthScoreToStatus(score: number): DomainHealthStatus {
    if (score >= 0.8) return 'healthy';
    if (score >= 0.6) return 'adequate';
    if (score >= 0.4) return 'stressed';
    if (score > 0) return 'critical';
    return 'unknown';
  }

  private generateAlerts(state: DashboardState): CrossDomainAlert[] {
    const alerts: CrossDomainAlert[] = [];
    let alertCounter = 0;

    // Check each domain against thresholds
    for (const [domain, metrics] of Object.entries(state.metrics)) {
      const threshold = this.config.alertThresholds[domain as ResourceDomain];

      if (metrics.healthScore < threshold) {
        alertCounter++;
        const severity = metrics.healthScore < 0.2 ? 'critical' :
                        metrics.healthScore < 0.3 ? 'warning' : 'info';

        alerts.push({
          alertId: `alert-${state.timestamp}-${alertCounter}`,
          severity,
          affectedDomains: [domain as ResourceDomain],
          title: `${domain.charAt(0).toUpperCase() + domain.slice(1)} Domain Health Low`,
          description: `${domain} health score is ${(metrics.healthScore * 100).toFixed(1)}%, below threshold of ${(threshold * 100).toFixed(1)}%`,
          recommendations: this.getDomainRecommendations(domain as ResourceDomain, metrics),
          triggeredAt: state.timestamp,
          acknowledged: false,
          source: 'automatic',
        });
      }
    }

    // Check for cascade risks
    if (this.config.enablePredictiveAlerts) {
      const highRiskCascades = state.cascadeRisks?.filter(r => r.riskScore > 0.5) ?? [];
      for (const cascade of highRiskCascades) {
        alertCounter++;
        alerts.push({
          alertId: `alert-${state.timestamp}-${alertCounter}`,
          severity: cascade.riskScore > 0.7 ? 'critical' : 'warning',
          affectedDomains: [cascade.originDomain, ...cascade.cascadePath],
          title: `Cascade Risk: ${cascade.originDomain} failure`,
          description: `A failure in ${cascade.originDomain} could cascade to ${cascade.cascadePath.join(', ')}`,
          recommendations: cascade.mitigations,
          triggeredAt: state.timestamp,
          acknowledged: false,
          source: 'predictive',
        });
      }
    }

    return alerts;
  }

  private getDomainRecommendations(domain: ResourceDomain, metrics: DomainMetrics): string[] {
    const recs: string[] = [];

    switch (domain) {
      case 'food':
        const food = metrics as FoodMetrics;
        if (food.reserveDays < 7) recs.push('Increase food reserves to at least 7 days');
        if (food.nutritionScore < 0.5) recs.push('Improve dietary diversity');
        if (food.localProductionPercent < 30) recs.push('Support local food production');
        break;
      case 'water':
        const water = metrics as WaterMetrics;
        if (water.qualityIndex < 70) recs.push('Investigate water quality issues');
        if (water.reserveDays < 3) recs.push('Increase water storage capacity');
        if (water.recyclingPercent < 20) recs.push('Expand water recycling programs');
        break;
      case 'energy':
        const energy = metrics as EnergyMetrics;
        if (energy.gridReliability < 0.99) recs.push('Invest in grid infrastructure');
        if (energy.storageHours < 8) recs.push('Increase energy storage capacity');
        if (energy.renewablePercent < 30) recs.push('Expand renewable energy sources');
        break;
      case 'shelter':
        const shelter = metrics as ShelterMetrics;
        if (shelter.affordabilityIndex < 0.3) recs.push('Address housing affordability');
        if (shelter.emergencyBeds < 50) recs.push('Increase emergency housing capacity');
        if (shelter.conditionScore < 0.5) recs.push('Prioritize housing maintenance');
        break;
      case 'medicine':
        const medicine = metrics as MedicineMetrics;
        if (medicine.accessScore < 0.6) recs.push('Improve healthcare access');
        if (medicine.avgWaitHours > 6) recs.push('Reduce wait times for care');
        if (medicine.mentalHealthScore < 0.5) recs.push('Expand mental health services');
        break;
    }

    return recs;
  }

  private generateRecommendations(state: DashboardState): CrossDomainRecommendation[] {
    const recs: CrossDomainRecommendation[] = [];
    let recCounter = 0;

    // Find domains with declining trends
    const decliningDomains = Object.entries(state.metrics)
      .filter(([_, m]) => m.trend === 'declining')
      .map(([d, _]) => d as ResourceDomain);

    if (decliningDomains.length > 0) {
      recCounter++;
      recs.push({
        recommendationId: `rec-${state.timestamp}-${recCounter}`,
        priority: 1,
        domains: decliningDomains,
        action: `Address declining trends in: ${decliningDomains.join(', ')}`,
        expectedImpact: 'Prevent further deterioration and potential cascade effects',
        resourcesNeeded: 'Assessment and targeted intervention',
        confidence: 0.8,
      });
    }

    // Check for cross-domain opportunities
    const energyMetrics = state.metrics.energy;
    const waterMetrics = state.metrics.water;

    if (energyMetrics.renewablePercent < 30 && waterMetrics.groundwaterStatus === 'normal') {
      recCounter++;
      recs.push({
        recommendationId: `rec-${state.timestamp}-${recCounter}`,
        priority: 2,
        domains: ['energy', 'water'],
        action: 'Consider small-scale hydroelectric generation',
        expectedImpact: 'Increase renewable energy while managing water resources',
        resourcesNeeded: 'Feasibility study and infrastructure investment',
        confidence: 0.6,
      });
    }

    // Food-shelter synergy
    const foodMetrics = state.metrics.food;
    const shelterMetrics = state.metrics.shelter;

    if (foodMetrics.communityPrograms < 3 && shelterMetrics.resourceHubs > 0) {
      recCounter++;
      recs.push({
        recommendationId: `rec-${state.timestamp}-${recCounter}`,
        priority: 2,
        domains: ['food', 'shelter'],
        action: 'Add food programs to existing community resource hubs',
        expectedImpact: 'Leverage existing infrastructure for food security',
        resourcesNeeded: 'Partnerships and program development',
        confidence: 0.75,
      });
    }

    return recs.sort((a, b) => a.priority - b.priority);
  }

  private calculateCascadeRisks(state: DashboardState): CascadeRisk[] {
    const risks: CascadeRisk[] = [];

    // Check each domain for potential cascade origin
    for (const [domain, metrics] of Object.entries(state.metrics)) {
      if (metrics.healthScore < 0.4) {
        // Domain is at risk, calculate cascade
        const risk = this.simulateFailure(domain as ResourceDomain);

        // Adjust probability based on current health
        risk.probability *= (1 - metrics.healthScore);

        if (risk.riskScore > 0.2) {
          risks.push(risk);
        }
      }
    }

    return risks.sort((a, b) => b.riskScore - a.riskScore);
  }

  private generateMitigations(origin: ResourceDomain, affected: ResourceDomain[]): string[] {
    const mitigations: string[] = [];

    mitigations.push(`Strengthen ${origin} domain resilience`);
    mitigations.push(`Establish backup systems for ${affected.join(', ')}`);

    if (affected.includes('medicine')) {
      mitigations.push('Ensure medical facility backup power');
    }
    if (affected.includes('water')) {
      mitigations.push('Maintain emergency water reserves');
    }
    if (affected.includes('food')) {
      mitigations.push('Establish distributed food storage');
    }

    return mitigations;
  }

  private getDomainsNeedingAttention(state: DashboardState): ResourceDomain[] {
    return (Object.entries(state.metrics) as Array<[ResourceDomain, DomainMetrics]>)
      .filter(([_, m]) => m.status === 'stressed' || m.status === 'critical')
      .map(([d, _]) => d);
  }
}

// ============================================================================
// Factory and Singleton
// ============================================================================

let dashboardInstance: CrossDomainDashboard | null = null;

/**
 * Get the singleton dashboard instance
 */
export function getDashboard(config?: Partial<DashboardConfig>): CrossDomainDashboard {
  if (!dashboardInstance) {
    dashboardInstance = new CrossDomainDashboard(config);
  }
  return dashboardInstance;
}

/**
 * Reset the dashboard instance (for testing)
 */
export function resetDashboard(): void {
  dashboardInstance = null;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format health score as percentage string
 */
export function formatHealthScore(score: number): string {
  return `${(score * 100).toFixed(1)}%`;
}

/**
 * Get emoji indicator for status
 */
export function statusEmoji(status: DomainHealthStatus): string {
  switch (status) {
    case 'healthy':
      return '\u2705'; // Green check
    case 'adequate':
      return '\u2705'; // Green check
    case 'stressed':
      return '\u26A0\uFE0F'; // Warning
    case 'critical':
      return '\u274C'; // Red X
    case 'unknown':
      return '\u2753'; // Question mark
  }
}

/**
 * Calculate composite resilience score
 */
export function calculateResilience(state: DashboardState): number {
  // Resilience is based on:
  // 1. Average health scores
  // 2. Diversity of healthy domains
  // 3. Low cascade risk

  const scores = Object.values(state.metrics).map(m => m.healthScore);
  const avgHealth = scores.reduce((a, b) => a + b, 0) / scores.length;

  const healthyCount = scores.filter(s => s > 0.6).length;
  const diversityFactor = healthyCount / 5;

  const maxCascadeRisk = Math.max(...(state.cascadeRisks?.map(r => r.riskScore) ?? [0]));
  const cascadeFactor = 1 - maxCascadeRisk;

  return (avgHealth * 0.4) + (diversityFactor * 0.3) + (cascadeFactor * 0.3);
}
