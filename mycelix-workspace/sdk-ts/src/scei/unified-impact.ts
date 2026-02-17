/**
 * Unified Impact Measurement System - SCEI Integration
 *
 * Provides comprehensive impact measurement across all civilizational domains,
 * integrating Social, Civilizational, Ecological, and Individual (SCEI) metrics.
 *
 * Key features:
 * - Multi-dimensional impact scoring
 * - Cross-domain impact correlation
 * - Temporal impact tracking
 * - Stakeholder-specific views
 * - Impact verification and attestation
 * - Sustainability metrics
 *
 * @packageDocumentation
 * @module scei/unified-impact
 */

import type { ResourceDomain } from '../metabolism/cross-domain-dashboard.js';
import type { AgentPubKey } from '@holochain/client';

// ============================================================================
// SCEI Dimensions
// ============================================================================

/**
 * SCEI Impact Dimensions
 */
export type SCEIDimension = 'social' | 'civilizational' | 'ecological' | 'individual';

/**
 * Impact polarity - positive or negative
 */
export type ImpactPolarity = 'positive' | 'negative' | 'neutral' | 'mixed';

/**
 * Impact timeframe
 */
export type ImpactTimeframe = 'immediate' | 'short_term' | 'medium_term' | 'long_term' | 'generational';

/**
 * Impact magnitude scale
 */
export type ImpactMagnitude = 'negligible' | 'minor' | 'moderate' | 'significant' | 'transformative';

// ============================================================================
// Impact Metrics
// ============================================================================

/**
 * Base impact metric
 */
export interface ImpactMetric {
  /** Metric identifier */
  metricId: string;
  /** Human-readable name */
  name: string;
  /** Description */
  description: string;
  /** SCEI dimension */
  dimension: SCEIDimension;
  /** Related resource domains */
  domains: ResourceDomain[];
  /** Unit of measurement */
  unit: string;
  /** Current value */
  value: number;
  /** Baseline/reference value */
  baseline: number;
  /** Target value */
  target?: number;
  /** Polarity of higher values */
  higherIsBetter: boolean;
  /** Measurement confidence (0-1) */
  confidence: number;
  /** Last measured timestamp */
  measuredAt: number;
}

/**
 * Social dimension metrics
 */
export interface SocialMetrics {
  /** Community cohesion score (0-1) */
  communityCohesion: number;
  /** Trust level within community (0-1) */
  trustLevel: number;
  /** Participation rate in community activities (0-1) */
  participationRate: number;
  /** Social mobility index (0-1) */
  socialMobility: number;
  /** Inclusion score (0-1) */
  inclusionScore: number;
  /** Cultural vitality (0-1) */
  culturalVitality: number;
  /** Social resilience (0-1) */
  socialResilience: number;
}

/**
 * Civilizational dimension metrics
 */
export interface CivilizationalMetrics {
  /** Infrastructure quality (0-1) */
  infrastructureQuality: number;
  /** Economic sustainability (0-1) */
  economicSustainability: number;
  /** Governance effectiveness (0-1) */
  governanceEffectiveness: number;
  /** Innovation capacity (0-1) */
  innovationCapacity: number;
  /** Knowledge preservation (0-1) */
  knowledgePreservation: number;
  /** Institutional trust (0-1) */
  institutionalTrust: number;
  /** Civilizational resilience (0-1) */
  civilizationalResilience: number;
}

/**
 * Ecological dimension metrics
 */
export interface EcologicalMetrics {
  /** Carbon footprint (tons CO2e) */
  carbonFootprint: number;
  /** Biodiversity index (0-1) */
  biodiversityIndex: number;
  /** Resource efficiency (0-1) */
  resourceEfficiency: number;
  /** Waste reduction rate (0-1) */
  wasteReduction: number;
  /** Renewable energy ratio (0-1) */
  renewableRatio: number;
  /** Water quality index (0-1) */
  waterQuality: number;
  /** Land regeneration score (0-1) */
  landRegeneration: number;
  /** Ecological resilience (0-1) */
  ecologicalResilience: number;
}

/**
 * Individual dimension metrics
 */
export interface IndividualMetrics {
  /** Health and wellbeing score (0-1) */
  healthWellbeing: number;
  /** Economic security (0-1) */
  economicSecurity: number;
  /** Learning and growth (0-1) */
  learningGrowth: number;
  /** Autonomy score (0-1) */
  autonomy: number;
  /** Purpose and meaning (0-1) */
  purposeMeaning: number;
  /** Work-life balance (0-1) */
  workLifeBalance: number;
  /** Individual resilience (0-1) */
  individualResilience: number;
}

// ============================================================================
// Impact Assessment
// ============================================================================

/**
 * Complete impact assessment
 */
export interface ImpactAssessment {
  /** Assessment identifier */
  assessmentId: string;
  /** What is being assessed */
  subject: ImpactSubject;
  /** When assessment was created */
  createdAt: number;
  /** Assessment timeframe */
  timeframe: ImpactTimeframe;
  /** Overall impact score (-1 to 1, 0 = neutral) */
  overallScore: number;
  /** Overall magnitude */
  magnitude: ImpactMagnitude;
  /** Overall polarity */
  polarity: ImpactPolarity;
  /** SCEI dimension scores */
  sceiScores: {
    social: number;
    civilizational: number;
    ecological: number;
    individual: number;
  };
  /** Domain-specific impacts */
  domainImpacts: Record<ResourceDomain, DomainImpact>;
  /** Detailed metrics */
  metrics: ImpactMetric[];
  /** Stakeholder impacts */
  stakeholderImpacts: StakeholderImpact[];
  /** Verification status */
  verification: ImpactVerification;
  /** Recommendations */
  recommendations: ImpactRecommendation[];
}

/**
 * Subject of impact assessment
 */
export interface ImpactSubject {
  /** Type of subject */
  type: 'project' | 'organization' | 'community' | 'policy' | 'product' | 'event';
  /** Subject identifier */
  identifier: string;
  /** Subject name */
  name: string;
  /** Subject description */
  description: string;
  /** Geographic scope */
  geographicScope?: string;
  /** Start date */
  startDate?: number;
  /** End date (null if ongoing) */
  endDate?: number | null;
}

/**
 * Domain-specific impact
 */
export interface DomainImpact {
  /** Resource domain */
  domain: ResourceDomain;
  /** Impact score (-1 to 1) */
  score: number;
  /** Magnitude */
  magnitude: ImpactMagnitude;
  /** Polarity */
  polarity: ImpactPolarity;
  /** Specific impacts */
  impacts: Array<{
    description: string;
    value: number;
    unit: string;
    polarity: ImpactPolarity;
  }>;
  /** Confidence in measurement */
  confidence: number;
}

/**
 * Stakeholder-specific impact
 */
export interface StakeholderImpact {
  /** Stakeholder group */
  stakeholder: string;
  /** Impact score (-1 to 1) */
  score: number;
  /** Number of individuals affected */
  affectedCount: number;
  /** Primary benefits */
  benefits: string[];
  /** Primary harms/costs */
  costs: string[];
  /** Distribution equity score (0-1) */
  equityScore: number;
}

/**
 * Impact verification
 */
export interface ImpactVerification {
  /** Verification status */
  status: 'unverified' | 'pending' | 'verified' | 'disputed';
  /** Verification methodology */
  methodology?: string;
  /** Verifiers */
  verifiers: AgentPubKey[];
  /** Attestations */
  attestations: ImpactAttestation[];
  /** Verification timestamp */
  verifiedAt?: number;
}

/**
 * Impact attestation
 */
export interface ImpactAttestation {
  /** Attestation ID */
  attestationId: string;
  /** Attester */
  attester: AgentPubKey;
  /** Attester reputation score */
  attesterReputation: number;
  /** Attestation statement */
  statement: string;
  /** Evidence references */
  evidence: string[];
  /** Timestamp */
  timestamp: number;
  /** Signature */
  signature?: string;
}

/**
 * Impact recommendation
 */
export interface ImpactRecommendation {
  /** Recommendation ID */
  recommendationId: string;
  /** Priority (1 = highest) */
  priority: number;
  /** Category */
  category: 'improve' | 'maintain' | 'mitigate' | 'monitor';
  /** Target SCEI dimension */
  dimension: SCEIDimension;
  /** Action description */
  action: string;
  /** Expected impact */
  expectedImpact: string;
  /** Resources needed */
  resourcesNeeded: string;
}

// ============================================================================
// Impact Tracking
// ============================================================================

/**
 * Impact trend over time
 */
export interface ImpactTrend {
  /** Subject identifier */
  subjectId: string;
  /** Metric being tracked */
  metricId: string;
  /** Data points */
  dataPoints: Array<{
    timestamp: number;
    value: number;
    confidence: number;
  }>;
  /** Trend direction */
  direction: 'improving' | 'stable' | 'declining';
  /** Rate of change */
  changeRate: number;
  /** Projected future value */
  projection?: number;
}

/**
 * Cross-domain correlation
 */
export interface CrossDomainCorrelation {
  /** First domain */
  domainA: ResourceDomain;
  /** Second domain */
  domainB: ResourceDomain;
  /** Correlation coefficient (-1 to 1) */
  correlation: number;
  /** Correlation type */
  type: 'positive' | 'negative' | 'none';
  /** Confidence in correlation */
  confidence: number;
  /** Causal direction if known */
  causalDirection?: 'a_causes_b' | 'b_causes_a' | 'bidirectional' | 'common_cause';
}

// ============================================================================
// Impact Service
// ============================================================================

/**
 * Configuration for impact measurement service
 */
export interface ImpactServiceConfig {
  /** SCEI dimension weights */
  dimensionWeights: Record<SCEIDimension, number>;
  /** Domain weights */
  domainWeights: Record<ResourceDomain, number>;
  /** Minimum confidence threshold */
  minConfidenceThreshold: number;
  /** Verification quorum (number of attesters needed) */
  verificationQuorum: number;
  /** Enable automatic recommendations */
  enableAutoRecommendations: boolean;
  /** Trend analysis window (days) */
  trendWindowDays: number;
}

/**
 * Default configuration
 */
export const DEFAULT_IMPACT_CONFIG: ImpactServiceConfig = {
  dimensionWeights: {
    social: 0.25,
    civilizational: 0.25,
    ecological: 0.25,
    individual: 0.25,
  },
  domainWeights: {
    food: 0.2,
    water: 0.2,
    energy: 0.2,
    shelter: 0.2,
    medicine: 0.2,
  },
  minConfidenceThreshold: 0.6,
  verificationQuorum: 3,
  enableAutoRecommendations: true,
  trendWindowDays: 90,
};

/**
 * Unified Impact Measurement Service
 */
export class UnifiedImpactService {
  private config: ImpactServiceConfig;
  private assessments: Map<string, ImpactAssessment> = new Map();
  private trends: Map<string, ImpactTrend> = new Map();
  private correlations: CrossDomainCorrelation[] = [];

  constructor(config: Partial<ImpactServiceConfig> = {}) {
    this.config = { ...DEFAULT_IMPACT_CONFIG, ...config };
    this.initializeCorrelations();
  }

  // ============================================================================
  // Assessment Operations
  // ============================================================================

  /**
   * Create a new impact assessment
   */
  public createAssessment(
    subject: ImpactSubject,
    timeframe: ImpactTimeframe = 'medium_term'
  ): ImpactAssessment {
    const assessment: ImpactAssessment = {
      assessmentId: `impact-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      subject,
      createdAt: Date.now(),
      timeframe,
      overallScore: 0,
      magnitude: 'negligible',
      polarity: 'neutral',
      sceiScores: {
        social: 0,
        civilizational: 0,
        ecological: 0,
        individual: 0,
      },
      domainImpacts: this.createEmptyDomainImpacts(),
      metrics: [],
      stakeholderImpacts: [],
      verification: {
        status: 'unverified',
        verifiers: [],
        attestations: [],
      },
      recommendations: [],
    };

    this.assessments.set(assessment.assessmentId, assessment);
    return assessment;
  }

  /**
   * Add metrics to an assessment
   */
  public addMetrics(
    assessmentId: string,
    sceiMetrics: {
      social?: Partial<SocialMetrics>;
      civilizational?: Partial<CivilizationalMetrics>;
      ecological?: Partial<EcologicalMetrics>;
      individual?: Partial<IndividualMetrics>;
    }
  ): ImpactAssessment {
    const assessment = this.assessments.get(assessmentId);
    if (!assessment) {
      throw new Error('Assessment not found');
    }

    // Process social metrics
    if (sceiMetrics.social) {
      assessment.sceiScores.social = this.calculateDimensionScore(
        sceiMetrics.social,
        'social'
      );
    }

    // Process civilizational metrics
    if (sceiMetrics.civilizational) {
      assessment.sceiScores.civilizational = this.calculateDimensionScore(
        sceiMetrics.civilizational,
        'civilizational'
      );
    }

    // Process ecological metrics
    if (sceiMetrics.ecological) {
      assessment.sceiScores.ecological = this.calculateDimensionScore(
        sceiMetrics.ecological,
        'ecological'
      );
    }

    // Process individual metrics
    if (sceiMetrics.individual) {
      assessment.sceiScores.individual = this.calculateDimensionScore(
        sceiMetrics.individual,
        'individual'
      );
    }

    // Recalculate overall score
    this.recalculateOverallScore(assessment);

    // Generate recommendations if enabled
    if (this.config.enableAutoRecommendations) {
      assessment.recommendations = this.generateRecommendations(assessment);
    }

    return assessment;
  }

  /**
   * Add domain impact data
   */
  public addDomainImpact(
    assessmentId: string,
    domain: ResourceDomain,
    impacts: Array<{
      description: string;
      value: number;
      unit: string;
      polarity: ImpactPolarity;
    }>
  ): ImpactAssessment {
    const assessment = this.assessments.get(assessmentId);
    if (!assessment) {
      throw new Error('Assessment not found');
    }

    const domainImpact = assessment.domainImpacts[domain];
    domainImpact.impacts = impacts;

    // Calculate domain score
    const positiveImpacts = impacts.filter(i => i.polarity === 'positive').length;
    const negativeImpacts = impacts.filter(i => i.polarity === 'negative').length;
    const totalImpacts = impacts.length;

    domainImpact.score = totalImpacts > 0
      ? (positiveImpacts - negativeImpacts) / totalImpacts
      : 0;
    domainImpact.magnitude = this.scoreToMagnitude(Math.abs(domainImpact.score));
    domainImpact.polarity = this.scoreToPolarity(domainImpact.score);
    domainImpact.confidence = 0.7; // Default confidence

    // Recalculate overall
    this.recalculateOverallScore(assessment);

    return assessment;
  }

  /**
   * Add stakeholder impact
   */
  public addStakeholderImpact(
    assessmentId: string,
    stakeholder: StakeholderImpact
  ): ImpactAssessment {
    const assessment = this.assessments.get(assessmentId);
    if (!assessment) {
      throw new Error('Assessment not found');
    }

    assessment.stakeholderImpacts.push(stakeholder);
    return assessment;
  }

  /**
   * Verify an assessment
   */
  public addVerification(
    assessmentId: string,
    attester: AgentPubKey,
    attestation: Omit<ImpactAttestation, 'attestationId' | 'attester' | 'timestamp'>
  ): ImpactAssessment {
    const assessment = this.assessments.get(assessmentId);
    if (!assessment) {
      throw new Error('Assessment not found');
    }

    const att: ImpactAttestation = {
      attestationId: `att-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      attester,
      timestamp: Date.now(),
      ...attestation,
    };

    assessment.verification.attestations.push(att);
    assessment.verification.verifiers.push(attester);

    // Check if quorum reached
    if (assessment.verification.attestations.length >= this.config.verificationQuorum) {
      assessment.verification.status = 'verified';
      assessment.verification.verifiedAt = Date.now();
    } else {
      assessment.verification.status = 'pending';
    }

    return assessment;
  }

  /**
   * Get assessment by ID
   */
  public getAssessment(assessmentId: string): ImpactAssessment | null {
    return this.assessments.get(assessmentId) ?? null;
  }

  /**
   * Get all assessments for a subject
   */
  public getAssessmentsForSubject(subjectId: string): ImpactAssessment[] {
    return Array.from(this.assessments.values())
      .filter(a => a.subject.identifier === subjectId);
  }

  // ============================================================================
  // Analysis Operations
  // ============================================================================

  /**
   * Calculate aggregated impact across multiple assessments
   */
  public calculateAggregatedImpact(assessmentIds: string[]): {
    overallScore: number;
    sceiScores: Record<SCEIDimension, number>;
    domainScores: Record<ResourceDomain, number>;
    confidence: number;
  } {
    const assessments = assessmentIds
      .map(id => this.assessments.get(id))
      .filter((a): a is ImpactAssessment => a !== undefined);

    if (assessments.length === 0) {
      return {
        overallScore: 0,
        sceiScores: { social: 0, civilizational: 0, ecological: 0, individual: 0 },
        domainScores: { food: 0, water: 0, energy: 0, shelter: 0, medicine: 0 },
        confidence: 0,
      };
    }

    const sceiScores = {
      social: this.average(assessments.map(a => a.sceiScores.social)),
      civilizational: this.average(assessments.map(a => a.sceiScores.civilizational)),
      ecological: this.average(assessments.map(a => a.sceiScores.ecological)),
      individual: this.average(assessments.map(a => a.sceiScores.individual)),
    };

    const domainScores: Record<ResourceDomain, number> = {
      food: this.average(assessments.map(a => a.domainImpacts.food.score)),
      water: this.average(assessments.map(a => a.domainImpacts.water.score)),
      energy: this.average(assessments.map(a => a.domainImpacts.energy.score)),
      shelter: this.average(assessments.map(a => a.domainImpacts.shelter.score)),
      medicine: this.average(assessments.map(a => a.domainImpacts.medicine.score)),
    };

    return {
      overallScore: this.average(assessments.map(a => a.overallScore)),
      sceiScores,
      domainScores,
      confidence: this.average(
        assessments
          .flatMap(a => Object.values(a.domainImpacts).map(d => d.confidence))
      ),
    };
  }

  /**
   * Get cross-domain correlations
   */
  public getCorrelations(): CrossDomainCorrelation[] {
    return this.correlations;
  }

  /**
   * Track impact trend
   */
  public trackTrend(
    subjectId: string,
    metricId: string,
    value: number,
    confidence: number = 0.8
  ): ImpactTrend {
    const key = `${subjectId}:${metricId}`;
    let trend = this.trends.get(key);

    if (!trend) {
      trend = {
        subjectId,
        metricId,
        dataPoints: [],
        direction: 'stable',
        changeRate: 0,
      };
      this.trends.set(key, trend);
    }

    trend.dataPoints.push({
      timestamp: Date.now(),
      value,
      confidence,
    });

    // Keep only data within window
    const windowMs = this.config.trendWindowDays * 24 * 60 * 60 * 1000;
    const cutoff = Date.now() - windowMs;
    trend.dataPoints = trend.dataPoints.filter(dp => dp.timestamp >= cutoff);

    // Calculate trend direction
    if (trend.dataPoints.length >= 3) {
      const first = trend.dataPoints.slice(0, Math.floor(trend.dataPoints.length / 2));
      const second = trend.dataPoints.slice(Math.floor(trend.dataPoints.length / 2));
      const firstAvg = this.average(first.map(dp => dp.value));
      const secondAvg = this.average(second.map(dp => dp.value));

      trend.changeRate = (secondAvg - firstAvg) / firstAvg;

      if (trend.changeRate > 0.05) {
        trend.direction = 'improving';
      } else if (trend.changeRate < -0.05) {
        trend.direction = 'declining';
      } else {
        trend.direction = 'stable';
      }
    }

    return trend;
  }

  /**
   * Generate impact report
   */
  public generateReport(assessmentId: string): ImpactReport {
    const assessment = this.assessments.get(assessmentId);
    if (!assessment) {
      throw new Error('Assessment not found');
    }

    const strengthsWeaknesses = this.analyzeStrengthsWeaknesses(assessment);

    return {
      assessmentId,
      subject: assessment.subject,
      generatedAt: Date.now(),
      executive: {
        overallScore: assessment.overallScore,
        magnitude: assessment.magnitude,
        polarity: assessment.polarity,
        verificationStatus: assessment.verification.status,
        summary: this.generateSummary(assessment),
      },
      sceiBreakdown: assessment.sceiScores,
      domainBreakdown: Object.fromEntries(
        Object.entries(assessment.domainImpacts).map(([k, v]) => [k, v.score])
      ) as Record<ResourceDomain, number>,
      strengths: strengthsWeaknesses.strengths,
      weaknesses: strengthsWeaknesses.weaknesses,
      recommendations: assessment.recommendations,
      stakeholderSummary: this.summarizeStakeholders(assessment),
      trends: this.getRelatedTrends(assessment.subject.identifier),
    };
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private initializeCorrelations(): void {
    // Pre-defined correlations based on known relationships
    // Domains available: food, water, energy, shelter, medicine

    // Strong correlations
    this.correlations.push({
      domainA: 'water',
      domainB: 'food',
      correlation: 0.85,
      type: 'positive',
      confidence: 0.9,
      causalDirection: 'bidirectional',
    });

    this.correlations.push({
      domainA: 'energy',
      domainB: 'water',
      correlation: 0.75,
      type: 'positive',
      confidence: 0.85,
      causalDirection: 'a_causes_b',
    });

    this.correlations.push({
      domainA: 'shelter',
      domainB: 'medicine',
      correlation: 0.6,
      type: 'positive',
      confidence: 0.8,
      causalDirection: 'a_causes_b',
    });

    this.correlations.push({
      domainA: 'food',
      domainB: 'medicine',
      correlation: 0.7,
      type: 'positive',
      confidence: 0.85,
      causalDirection: 'a_causes_b',
    });
  }

  private createEmptyDomainImpacts(): Record<ResourceDomain, DomainImpact> {
    const domains: ResourceDomain[] = ['food', 'water', 'energy', 'shelter', 'medicine'];
    const impacts: Record<ResourceDomain, DomainImpact> = {} as Record<ResourceDomain, DomainImpact>;

    for (const domain of domains) {
      impacts[domain] = {
        domain,
        score: 0,
        magnitude: 'negligible',
        polarity: 'neutral',
        impacts: [],
        confidence: 0,
      };
    }

    return impacts;
  }

  private calculateDimensionScore(
    metrics: Record<string, number>,
    _dimension: SCEIDimension
  ): number {
    const values = Object.values(metrics).filter(v => typeof v === 'number' && !isNaN(v));
    if (values.length === 0) return 0;

    // Convert to -1 to 1 scale where 0.5 input = 0 output
    const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
    return (avg - 0.5) * 2;
  }

  private recalculateOverallScore(assessment: ImpactAssessment): void {
    // Weighted average of SCEI scores
    const sceiScore = (
      assessment.sceiScores.social * this.config.dimensionWeights.social +
      assessment.sceiScores.civilizational * this.config.dimensionWeights.civilizational +
      assessment.sceiScores.ecological * this.config.dimensionWeights.ecological +
      assessment.sceiScores.individual * this.config.dimensionWeights.individual
    );

    // Weighted average of domain scores
    const domainScore = Object.entries(assessment.domainImpacts)
      .reduce((sum, [domain, impact]) => {
        return sum + impact.score * this.config.domainWeights[domain as ResourceDomain];
      }, 0);

    // Combine SCEI and domain scores
    assessment.overallScore = (sceiScore + domainScore) / 2;
    assessment.magnitude = this.scoreToMagnitude(Math.abs(assessment.overallScore));
    assessment.polarity = this.scoreToPolarity(assessment.overallScore);
  }

  private scoreToMagnitude(absScore: number): ImpactMagnitude {
    if (absScore < 0.1) return 'negligible';
    if (absScore < 0.25) return 'minor';
    if (absScore < 0.5) return 'moderate';
    if (absScore < 0.75) return 'significant';
    return 'transformative';
  }

  private scoreToPolarity(score: number): ImpactPolarity {
    if (score > 0.1) return 'positive';
    if (score < -0.1) return 'negative';
    return 'neutral';
  }

  private generateRecommendations(assessment: ImpactAssessment): ImpactRecommendation[] {
    const recommendations: ImpactRecommendation[] = [];
    let priority = 1;

    // Check SCEI dimensions
    const dimensions: Array<[SCEIDimension, number]> = [
      ['social', assessment.sceiScores.social],
      ['civilizational', assessment.sceiScores.civilizational],
      ['ecological', assessment.sceiScores.ecological],
      ['individual', assessment.sceiScores.individual],
    ];

    for (const [dimension, score] of dimensions) {
      if (score < -0.2) {
        recommendations.push({
          recommendationId: `rec-${Date.now()}-${priority}`,
          priority: priority++,
          category: 'mitigate',
          dimension,
          action: `Address negative ${dimension} impacts`,
          expectedImpact: `Improve ${dimension} score from ${(score * 100).toFixed(0)}% to neutral or positive`,
          resourcesNeeded: 'Assessment and targeted intervention',
        });
      } else if (score > 0.3) {
        recommendations.push({
          recommendationId: `rec-${Date.now()}-${priority}`,
          priority: priority++,
          category: 'maintain',
          dimension,
          action: `Maintain strong ${dimension} performance`,
          expectedImpact: `Sustain current ${dimension} benefits`,
          resourcesNeeded: 'Continued monitoring and support',
        });
      }
    }

    // Check domains
    for (const [domain, impact] of Object.entries(assessment.domainImpacts)) {
      if (impact.score < -0.2) {
        recommendations.push({
          recommendationId: `rec-${Date.now()}-${priority}`,
          priority: priority++,
          category: 'mitigate',
          dimension: 'civilizational',
          action: `Address negative ${domain} domain impact`,
          expectedImpact: `Reduce harm to ${domain} resources`,
          resourcesNeeded: 'Domain-specific intervention',
        });
      }
    }

    return recommendations.slice(0, 10); // Max 10 recommendations
  }

  private analyzeStrengthsWeaknesses(assessment: ImpactAssessment): {
    strengths: string[];
    weaknesses: string[];
  } {
    const strengths: string[] = [];
    const weaknesses: string[] = [];

    // SCEI analysis
    if (assessment.sceiScores.social > 0.2) strengths.push('Strong social impact');
    if (assessment.sceiScores.social < -0.2) weaknesses.push('Negative social consequences');

    if (assessment.sceiScores.ecological > 0.2) strengths.push('Positive ecological contribution');
    if (assessment.sceiScores.ecological < -0.2) weaknesses.push('Environmental concerns');

    if (assessment.sceiScores.individual > 0.2) strengths.push('Benefits individual wellbeing');
    if (assessment.sceiScores.individual < -0.2) weaknesses.push('Individual burden');

    // Domain analysis
    for (const [domain, impact] of Object.entries(assessment.domainImpacts)) {
      if (impact.score > 0.3) {
        strengths.push(`Strong ${domain} sector impact`);
      }
      if (impact.score < -0.3) {
        weaknesses.push(`${domain} sector concerns`);
      }
    }

    // Verification
    if (assessment.verification.status === 'verified') {
      strengths.push('Third-party verified impact claims');
    }

    return { strengths, weaknesses };
  }

  private generateSummary(assessment: ImpactAssessment): string {
    const polarity = assessment.polarity === 'positive' ? 'positive' :
                     assessment.polarity === 'negative' ? 'negative' : 'mixed';
    const magnitude = assessment.magnitude;

    return `This ${assessment.subject.type} demonstrates ${magnitude} ${polarity} impact ` +
           `with an overall score of ${(assessment.overallScore * 100).toFixed(1)}%. ` +
           `Verification status: ${assessment.verification.status}.`;
  }

  private summarizeStakeholders(assessment: ImpactAssessment): Array<{
    stakeholder: string;
    netImpact: string;
    affectedCount: number;
  }> {
    return assessment.stakeholderImpacts.map(si => ({
      stakeholder: si.stakeholder,
      netImpact: si.score > 0.1 ? 'positive' : si.score < -0.1 ? 'negative' : 'neutral',
      affectedCount: si.affectedCount,
    }));
  }

  private getRelatedTrends(subjectId: string): ImpactTrend[] {
    return Array.from(this.trends.values())
      .filter(t => t.subjectId === subjectId);
  }

  private average(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
}

// ============================================================================
// Report Types
// ============================================================================

/**
 * Impact report for export/display
 */
export interface ImpactReport {
  assessmentId: string;
  subject: ImpactSubject;
  generatedAt: number;
  executive: {
    overallScore: number;
    magnitude: ImpactMagnitude;
    polarity: ImpactPolarity;
    verificationStatus: ImpactVerification['status'];
    summary: string;
  };
  sceiBreakdown: Record<SCEIDimension, number>;
  domainBreakdown: Record<ResourceDomain, number>;
  strengths: string[];
  weaknesses: string[];
  recommendations: ImpactRecommendation[];
  stakeholderSummary: Array<{
    stakeholder: string;
    netImpact: string;
    affectedCount: number;
  }>;
  trends: ImpactTrend[];
}

// ============================================================================
// Factory and Singleton
// ============================================================================

let impactServiceInstance: UnifiedImpactService | null = null;

/**
 * Get the singleton impact service instance
 */
export function getImpactService(config?: Partial<ImpactServiceConfig>): UnifiedImpactService {
  if (!impactServiceInstance) {
    impactServiceInstance = new UnifiedImpactService(config);
  }
  return impactServiceInstance;
}

/**
 * Reset the impact service instance (for testing)
 */
export function resetImpactService(): void {
  impactServiceInstance = null;
}
