/**
 * Cross-Domain Reputation Bridge
 *
 * Aggregates reputation scores across all Mycelix domain hApps into a unified
 * trust profile. Uses the K-Vector multi-dimensional trust model for
 * comprehensive reputation assessment.
 *
 * Key capabilities:
 * - Aggregate reputation from 12+ domain hApps
 * - K-Vector based multi-dimensional trust
 * - Domain-specific weight configuration
 * - Cross-domain trust correlation analysis
 * - Historical reputation tracking
 * - Byzantine-resistant aggregation
 *
 * @module bridge/cross-domain-reputation
 */

import type { AgentPubKey, ActionHash } from '@holochain/client';

// ============================================================================
// Domain Definitions
// ============================================================================

/**
 * All supported domain hApps in the Mycelix ecosystem
 */
export type ReputationDomain =
  | 'Identity'
  | 'Finance'
  | 'Property'
  | 'Energy'
  | 'Health'
  | 'Governance'
  | 'Education'
  | 'Marketplace'
  | 'SupplyChain'
  | 'Media'
  | 'Justice'
  | 'Knowledge'
  | 'Fabrication'
  | 'Music'
  | 'DeSci'
  | 'Food'
  | 'Shelter';

/**
 * Domain categories for grouped weighting
 */
export type DomainCategory =
  | 'Essential' // Identity, Health, Food, Shelter
  | 'Economic' // Finance, Property, Marketplace, Energy
  | 'Social' // Governance, Education, Media, Music
  | 'Technical' // SupplyChain, Fabrication, Knowledge, DeSci
  | 'Justice'; // Justice (special category)

// ============================================================================
// K-Vector Types (mirrors Rust SDK)
// ============================================================================

/**
 * K-Vector 9-dimensional trust profile
 *
 * Mirrors the Rust KVector struct from sdk/src/matl/kvector.rs
 */
export interface KVector {
  /** Reputation - historical trust accumulated (0.0-1.0) */
  k_r: number;
  /** Activity - recent participation level (0.0-1.0) */
  k_a: number;
  /** Integrity - consistency and honesty (0.0-1.0) */
  k_i: number;
  /** Performance - task completion quality (0.0-1.0) */
  k_p: number;
  /** Mutual - reciprocity in relationships (0.0-1.0) */
  k_m: number;
  /** Stake - economic commitment level (0.0-1.0) */
  k_s: number;
  /** Harmonic - systemic contribution (0.0-1.0) */
  k_h: number;
  /** Topology - network position influence (0.0-1.0) */
  k_topo: number;
  /** Verification - identity/credential validity (0.0-1.0) */
  k_v: number;
}

/**
 * Default K-Vector weights (from MATL spec)
 */
export const DEFAULT_KVECTOR_WEIGHTS = {
  w_r: 0.25,
  w_a: 0.12,
  w_i: 0.20,
  w_p: 0.12,
  w_m: 0.05,
  w_s: 0.07,
  w_h: 0.05,
  w_topo: 0.04,
  w_v: 0.10,
} as const;

// ============================================================================
// Single Domain Reputation
// ============================================================================

/**
 * Reputation data from a single domain hApp
 */
export interface DomainReputationRecord {
  /** The domain this reputation is from */
  domain: ReputationDomain;
  /** Domain category */
  category: DomainCategory;
  /** Agent's K-Vector in this domain */
  kVector: KVector;
  /** Computed trust score for this domain (0-1) */
  trustScore: number;
  /** Number of interactions in this domain */
  interactionCount: number;
  /** Timestamp of first interaction */
  firstInteraction?: number;
  /** Timestamp of most recent interaction */
  lastInteraction?: number;
  /** Domain-specific metadata */
  metadata?: Record<string, unknown>;
  /** Whether this domain data is verified */
  verified: boolean;
  /** Data freshness (how recently fetched) */
  fetchedAt: number;
}

/**
 * Reputation event that updates domain reputation
 */
export interface ReputationEvent {
  eventId: string;
  domain: ReputationDomain;
  agentPubKey: AgentPubKey;
  eventType: ReputationEventType;
  impact: number; // -1.0 to 1.0
  kVectorDelta: Partial<KVector>;
  description: string;
  timestamp: number;
  /** Reference to the on-chain action that triggered this */
  sourceActionHash?: ActionHash;
  /** Witnesses who attested to this event */
  witnesses?: AgentPubKey[];
}

export type ReputationEventType =
  | 'PositiveInteraction'
  | 'NegativeInteraction'
  | 'VerificationCompleted'
  | 'CredentialIssued'
  | 'DisputeResolved'
  | 'ContributionMade'
  | 'ViolationReported'
  | 'StakeIncreased'
  | 'StakeDecreased';

// ============================================================================
// Cross-Domain Aggregated Reputation
// ============================================================================

/**
 * Fully aggregated cross-domain reputation profile
 */
export interface CrossDomainReputation {
  /** Agent this reputation belongs to */
  agentPubKey: AgentPubKey;
  /** DID if identity is verified */
  did?: string;
  /** Aggregated K-Vector across all domains */
  aggregatedKVector: KVector;
  /** Overall trust score (0-1) */
  overallTrustScore: number;
  /** Trust level classification */
  trustLevel: TrustLevel;
  /** Per-domain reputation records */
  domainReputations: Map<ReputationDomain, DomainReputationRecord>;
  /** Category-level aggregations */
  categoryScores: Map<DomainCategory, number>;
  /** Total interactions across all domains */
  totalInteractions: number;
  /** Number of domains with activity */
  activeDomainCount: number;
  /** Cross-domain correlation analysis */
  correlationAnalysis: CorrelationAnalysis;
  /** Historical trend */
  trend: ReputationTrend;
  /** Byzantine risk assessment */
  byzantineRisk: number;
  /** When this aggregation was computed */
  computedAt: number;
}

export type TrustLevel =
  | 'HighlyTrusted' // score >= 0.85
  | 'Trusted' // score >= 0.70
  | 'Conditional' // score >= 0.50
  | 'Suspicious' // score >= 0.30
  | 'Untrusted' // score < 0.30
  | 'Unknown'; // no data

/**
 * Cross-domain correlation insights
 */
export interface CorrelationAnalysis {
  /** Domains that show consistent behavior */
  consistentDomains: ReputationDomain[];
  /** Domains with anomalous behavior relative to others */
  anomalousDomains: ReputationDomain[];
  /** Correlation coefficient between domain pairs */
  domainCorrelations: DomainCorrelation[];
  /** Overall consistency score (0-1) */
  consistencyScore: number;
}

export interface DomainCorrelation {
  domain1: ReputationDomain;
  domain2: ReputationDomain;
  correlation: number; // -1 to 1
}

export type ReputationTrend = 'Improving' | 'Stable' | 'Declining' | 'Volatile' | 'Insufficient';

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration for cross-domain reputation aggregation
 */
export interface CrossDomainReputationConfig {
  /** Custom domain weights (overrides defaults) */
  domainWeights?: Partial<Record<ReputationDomain, number>>;
  /** Category weights */
  categoryWeights?: Partial<Record<DomainCategory, number>>;
  /** K-Vector dimension weights */
  kVectorWeights?: Partial<typeof DEFAULT_KVECTOR_WEIGHTS>;
  /** Minimum interactions to be considered "active" in a domain */
  minInteractionsForActive?: number;
  /** How long domain data is considered fresh (ms) */
  dataFreshnessTtlMs?: number;
  /** Byzantine detection threshold */
  byzantineThreshold?: number;
  /** Enable correlation analysis */
  enableCorrelationAnalysis?: boolean;
  /** Historical lookback period for trend analysis (ms) */
  trendLookbackMs?: number;
}

/**
 * Default domain weights by category importance
 */
export const DEFAULT_DOMAIN_WEIGHTS: Record<ReputationDomain, number> = {
  // Essential (highest weight)
  Identity: 0.12,
  Health: 0.08,
  Food: 0.06,
  Shelter: 0.06,
  // Economic
  Finance: 0.10,
  Property: 0.06,
  Marketplace: 0.06,
  Energy: 0.05,
  // Social
  Governance: 0.07,
  Education: 0.05,
  Media: 0.04,
  Music: 0.03,
  // Technical
  SupplyChain: 0.05,
  Fabrication: 0.04,
  Knowledge: 0.04,
  DeSci: 0.03,
  // Justice
  Justice: 0.06,
};

export const DEFAULT_CATEGORY_WEIGHTS: Record<DomainCategory, number> = {
  Essential: 0.35,
  Economic: 0.25,
  Social: 0.15,
  Technical: 0.15,
  Justice: 0.10,
};

// ============================================================================
// Cross-Domain Reputation Service
// ============================================================================

/**
 * Service for aggregating and querying cross-domain reputation
 */
export class CrossDomainReputationService {
  private config: Required<CrossDomainReputationConfig>;
  private reputationCache: Map<string, CrossDomainReputation> = new Map();
  private eventHistory: Map<string, ReputationEvent[]> = new Map();

  constructor(config: CrossDomainReputationConfig = {}) {
    this.config = {
      domainWeights: { ...DEFAULT_DOMAIN_WEIGHTS, ...config.domainWeights },
      categoryWeights: { ...DEFAULT_CATEGORY_WEIGHTS, ...config.categoryWeights },
      kVectorWeights: { ...DEFAULT_KVECTOR_WEIGHTS, ...config.kVectorWeights },
      minInteractionsForActive: config.minInteractionsForActive ?? 3,
      dataFreshnessTtlMs: config.dataFreshnessTtlMs ?? 5 * 60 * 1000, // 5 minutes
      byzantineThreshold: config.byzantineThreshold ?? 0.34,
      enableCorrelationAnalysis: config.enableCorrelationAnalysis ?? true,
      trendLookbackMs: config.trendLookbackMs ?? 30 * 24 * 60 * 60 * 1000, // 30 days
    };
  }

  // --------------------------------------------------------------------------
  // Core Aggregation
  // --------------------------------------------------------------------------

  /**
   * Aggregate reputation from all domains for an agent
   */
  async aggregateReputation(
    agentPubKey: AgentPubKey,
    domainRecords: DomainReputationRecord[],
    did?: string
  ): Promise<CrossDomainReputation> {
    const now = Date.now();

    // Build domain map
    const domainReputations = new Map<ReputationDomain, DomainReputationRecord>();
    for (const record of domainRecords) {
      domainReputations.set(record.domain, record);
    }

    // Aggregate K-Vector
    const aggregatedKVector = this.aggregateKVectors(domainRecords);

    // Calculate overall trust score
    const overallTrustScore = this.calculateOverallTrustScore(
      aggregatedKVector,
      domainRecords
    );

    // Calculate category scores
    const categoryScores = this.calculateCategoryScores(domainRecords);

    // Count active domains
    const activeDomainCount = domainRecords.filter(
      (r) => r.interactionCount >= this.config.minInteractionsForActive
    ).length;

    // Total interactions
    const totalInteractions = domainRecords.reduce(
      (sum, r) => sum + r.interactionCount,
      0
    );

    // Correlation analysis
    const correlationAnalysis = this.config.enableCorrelationAnalysis
      ? this.analyzeCorrelations(domainRecords)
      : this.emptyCorrelationAnalysis();

    // Byzantine risk
    const byzantineRisk = this.assessByzantineRisk(
      domainRecords,
      correlationAnalysis
    );

    // Trend analysis
    const trend = this.analyzeTrend(agentPubKey);

    // Trust level
    const trustLevel = this.classifyTrustLevel(overallTrustScore, activeDomainCount);

    const reputation: CrossDomainReputation = {
      agentPubKey,
      did,
      aggregatedKVector,
      overallTrustScore,
      trustLevel,
      domainReputations,
      categoryScores,
      totalInteractions,
      activeDomainCount,
      correlationAnalysis,
      trend,
      byzantineRisk,
      computedAt: now,
    };

    // Cache the result
    this.reputationCache.set(agentPubKey.toString(), reputation);

    return reputation;
  }

  /**
   * Get cached or fetch fresh cross-domain reputation
   */
  async getReputation(agentPubKey: AgentPubKey): Promise<CrossDomainReputation | null> {
    const cached = this.reputationCache.get(agentPubKey.toString());
    if (cached && Date.now() - cached.computedAt < this.config.dataFreshnessTtlMs) {
      return cached;
    }
    return null;
  }

  // --------------------------------------------------------------------------
  // K-Vector Aggregation
  // --------------------------------------------------------------------------

  /**
   * Aggregate K-Vectors from multiple domains using weighted averaging
   */
  private aggregateKVectors(records: DomainReputationRecord[]): KVector {
    if (records.length === 0) {
      return this.defaultKVector();
    }

    const weights = this.config.domainWeights as Record<ReputationDomain, number>;
    let totalWeight = 0;
    const sums = {
      k_r: 0, k_a: 0, k_i: 0, k_p: 0, k_m: 0, k_s: 0, k_h: 0, k_topo: 0, k_v: 0,
    };

    for (const record of records) {
      const weight = weights[record.domain] ?? 0.05;
      totalWeight += weight;

      sums.k_r += record.kVector.k_r * weight;
      sums.k_a += record.kVector.k_a * weight;
      sums.k_i += record.kVector.k_i * weight;
      sums.k_p += record.kVector.k_p * weight;
      sums.k_m += record.kVector.k_m * weight;
      sums.k_s += record.kVector.k_s * weight;
      sums.k_h += record.kVector.k_h * weight;
      sums.k_topo += record.kVector.k_topo * weight;
      sums.k_v += record.kVector.k_v * weight;
    }

    if (totalWeight === 0) {
      return this.defaultKVector();
    }

    return {
      k_r: sums.k_r / totalWeight,
      k_a: sums.k_a / totalWeight,
      k_i: sums.k_i / totalWeight,
      k_p: sums.k_p / totalWeight,
      k_m: sums.k_m / totalWeight,
      k_s: sums.k_s / totalWeight,
      k_h: sums.k_h / totalWeight,
      k_topo: sums.k_topo / totalWeight,
      k_v: sums.k_v / totalWeight,
    };
  }

  /**
   * Calculate trust score from K-Vector
   */
  calculateKVectorTrustScore(kv: KVector): number {
    const w = this.config.kVectorWeights;
    return (
      kv.k_r * (w.w_r ?? DEFAULT_KVECTOR_WEIGHTS.w_r) +
      kv.k_a * (w.w_a ?? DEFAULT_KVECTOR_WEIGHTS.w_a) +
      kv.k_i * (w.w_i ?? DEFAULT_KVECTOR_WEIGHTS.w_i) +
      kv.k_p * (w.w_p ?? DEFAULT_KVECTOR_WEIGHTS.w_p) +
      kv.k_m * (w.w_m ?? DEFAULT_KVECTOR_WEIGHTS.w_m) +
      kv.k_s * (w.w_s ?? DEFAULT_KVECTOR_WEIGHTS.w_s) +
      kv.k_h * (w.w_h ?? DEFAULT_KVECTOR_WEIGHTS.w_h) +
      kv.k_topo * (w.w_topo ?? DEFAULT_KVECTOR_WEIGHTS.w_topo) +
      kv.k_v * (w.w_v ?? DEFAULT_KVECTOR_WEIGHTS.w_v)
    );
  }

  // --------------------------------------------------------------------------
  // Category Scoring
  // --------------------------------------------------------------------------

  /**
   * Calculate aggregate scores per domain category
   */
  private calculateCategoryScores(
    records: DomainReputationRecord[]
  ): Map<DomainCategory, number> {
    const categories: DomainCategory[] = [
      'Essential',
      'Economic',
      'Social',
      'Technical',
      'Justice',
    ];

    const categoryScores = new Map<DomainCategory, number>();

    for (const category of categories) {
      const categoryRecords = records.filter((r) => r.category === category);
      if (categoryRecords.length === 0) {
        categoryScores.set(category, 0.5); // Neutral if no data
        continue;
      }

      const avgScore =
        categoryRecords.reduce((sum, r) => sum + r.trustScore, 0) /
        categoryRecords.length;
      categoryScores.set(category, avgScore);
    }

    return categoryScores;
  }

  // --------------------------------------------------------------------------
  // Overall Trust Calculation
  // --------------------------------------------------------------------------

  /**
   * Calculate overall trust score combining K-Vector and domain scores
   */
  private calculateOverallTrustScore(
    aggregatedKVector: KVector,
    records: DomainReputationRecord[]
  ): number {
    if (records.length === 0) {
      return 0.5; // Neutral for unknown
    }

    // K-Vector component (60% weight)
    const kVectorScore = this.calculateKVectorTrustScore(aggregatedKVector);

    // Domain-weighted average (40% weight)
    const weights = this.config.domainWeights as Record<ReputationDomain, number>;
    let domainScoreSum = 0;
    let domainWeightSum = 0;

    for (const record of records) {
      const weight = weights[record.domain] ?? 0.05;
      domainScoreSum += record.trustScore * weight;
      domainWeightSum += weight;
    }

    const domainAvgScore =
      domainWeightSum > 0 ? domainScoreSum / domainWeightSum : 0.5;

    // Combine
    return kVectorScore * 0.6 + domainAvgScore * 0.4;
  }

  // --------------------------------------------------------------------------
  // Correlation Analysis
  // --------------------------------------------------------------------------

  /**
   * Analyze cross-domain correlations to detect anomalies
   */
  private analyzeCorrelations(records: DomainReputationRecord[]): CorrelationAnalysis {
    if (records.length < 2) {
      return this.emptyCorrelationAnalysis();
    }

    const domainCorrelations: DomainCorrelation[] = [];
    const scores = records.map((r) => r.trustScore);
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const stdDev = Math.sqrt(
      scores.reduce((sum, s) => sum + Math.pow(s - avgScore, 2), 0) / scores.length
    );

    const consistentDomains: ReputationDomain[] = [];
    const anomalousDomains: ReputationDomain[] = [];

    // Identify anomalous domains (more than 1.5 std devs from mean)
    for (const record of records) {
      const deviation = Math.abs(record.trustScore - avgScore);
      if (stdDev > 0 && deviation > 1.5 * stdDev) {
        anomalousDomains.push(record.domain);
      } else {
        consistentDomains.push(record.domain);
      }
    }

    // Calculate pairwise correlations for top domains
    for (let i = 0; i < records.length && i < 5; i++) {
      for (let j = i + 1; j < records.length && j < 5; j++) {
        const kv1 = records[i].kVector;
        const kv2 = records[j].kVector;
        const correlation = this.calculateKVectorCorrelation(kv1, kv2);
        domainCorrelations.push({
          domain1: records[i].domain,
          domain2: records[j].domain,
          correlation,
        });
      }
    }

    // Overall consistency score
    const consistencyScore =
      stdDev > 0 ? Math.max(0, 1 - stdDev) : 1.0;

    return {
      consistentDomains,
      anomalousDomains,
      domainCorrelations,
      consistencyScore,
    };
  }

  /**
   * Calculate correlation between two K-Vectors
   */
  private calculateKVectorCorrelation(kv1: KVector, kv2: KVector): number {
    const v1 = [kv1.k_r, kv1.k_a, kv1.k_i, kv1.k_p, kv1.k_m, kv1.k_s, kv1.k_h, kv1.k_topo, kv1.k_v];
    const v2 = [kv2.k_r, kv2.k_a, kv2.k_i, kv2.k_p, kv2.k_m, kv2.k_s, kv2.k_h, kv2.k_topo, kv2.k_v];

    const mean1 = v1.reduce((a, b) => a + b, 0) / v1.length;
    const mean2 = v2.reduce((a, b) => a + b, 0) / v2.length;

    let num = 0;
    let den1 = 0;
    let den2 = 0;

    for (let i = 0; i < v1.length; i++) {
      const d1 = v1[i] - mean1;
      const d2 = v2[i] - mean2;
      num += d1 * d2;
      den1 += d1 * d1;
      den2 += d2 * d2;
    }

    const denom = Math.sqrt(den1 * den2);
    return denom > 0 ? num / denom : 0;
  }

  private emptyCorrelationAnalysis(): CorrelationAnalysis {
    return {
      consistentDomains: [],
      anomalousDomains: [],
      domainCorrelations: [],
      consistencyScore: 0.5,
    };
  }

  // --------------------------------------------------------------------------
  // Byzantine Risk Assessment
  // --------------------------------------------------------------------------

  /**
   * Assess risk of Byzantine behavior based on cross-domain patterns
   */
  private assessByzantineRisk(
    records: DomainReputationRecord[],
    correlation: CorrelationAnalysis
  ): number {
    if (records.length === 0) return 0.5;

    let risk = 0;

    // Factor 1: Anomalous domains (inconsistent behavior)
    const anomalyRatio = correlation.anomalousDomains.length / records.length;
    risk += anomalyRatio * 0.3;

    // Factor 2: Low integrity scores
    const avgIntegrity =
      records.reduce((sum, r) => sum + r.kVector.k_i, 0) / records.length;
    risk += (1 - avgIntegrity) * 0.3;

    // Factor 3: Unverified domains
    const unverifiedRatio =
      records.filter((r) => !r.verified).length / records.length;
    risk += unverifiedRatio * 0.2;

    // Factor 4: Low verification scores
    const avgVerification =
      records.reduce((sum, r) => sum + r.kVector.k_v, 0) / records.length;
    risk += (1 - avgVerification) * 0.2;

    return Math.min(risk, 1.0);
  }

  // --------------------------------------------------------------------------
  // Trend Analysis
  // --------------------------------------------------------------------------

  /**
   * Analyze reputation trend over time
   */
  private analyzeTrend(agentPubKey: AgentPubKey): ReputationTrend {
    const events = this.eventHistory.get(agentPubKey.toString()) ?? [];
    const cutoff = Date.now() - this.config.trendLookbackMs;
    const recentEvents = events.filter((e) => e.timestamp > cutoff);

    if (recentEvents.length < 5) {
      return 'Insufficient';
    }

    // Calculate average impact over time windows
    const midpoint = cutoff + this.config.trendLookbackMs / 2;
    const earlyEvents = recentEvents.filter((e) => e.timestamp < midpoint);
    const lateEvents = recentEvents.filter((e) => e.timestamp >= midpoint);

    if (earlyEvents.length === 0 || lateEvents.length === 0) {
      return 'Stable';
    }

    const earlyAvg =
      earlyEvents.reduce((sum, e) => sum + e.impact, 0) / earlyEvents.length;
    const lateAvg =
      lateEvents.reduce((sum, e) => sum + e.impact, 0) / lateEvents.length;

    const diff = lateAvg - earlyAvg;
    const variance =
      recentEvents.reduce((sum, e) => sum + Math.pow(e.impact - (earlyAvg + lateAvg) / 2, 2), 0) /
      recentEvents.length;

    if (variance > 0.3) {
      return 'Volatile';
    } else if (diff > 0.1) {
      return 'Improving';
    } else if (diff < -0.1) {
      return 'Declining';
    } else {
      return 'Stable';
    }
  }

  // --------------------------------------------------------------------------
  // Trust Classification
  // --------------------------------------------------------------------------

  /**
   * Classify trust level based on score and activity
   */
  private classifyTrustLevel(score: number, activeDomains: number): TrustLevel {
    if (activeDomains === 0) {
      return 'Unknown';
    }

    if (score >= 0.85) {
      return 'HighlyTrusted';
    } else if (score >= 0.70) {
      return 'Trusted';
    } else if (score >= 0.50) {
      return 'Conditional';
    } else if (score >= 0.30) {
      return 'Suspicious';
    } else {
      return 'Untrusted';
    }
  }

  // --------------------------------------------------------------------------
  // Event Recording
  // --------------------------------------------------------------------------

  /**
   * Record a reputation event
   */
  recordEvent(event: ReputationEvent): void {
    const key = event.agentPubKey.toString();
    const events = this.eventHistory.get(key) ?? [];
    events.push(event);

    // Keep only events within lookback period
    const cutoff = Date.now() - this.config.trendLookbackMs;
    const filtered = events.filter((e) => e.timestamp > cutoff);
    this.eventHistory.set(key, filtered);
  }

  /**
   * Get event history for an agent
   */
  getEventHistory(agentPubKey: AgentPubKey): ReputationEvent[] {
    return this.eventHistory.get(agentPubKey.toString()) ?? [];
  }

  // --------------------------------------------------------------------------
  // Utility Methods
  // --------------------------------------------------------------------------

  /**
   * Default K-Vector for unknown agents
   */
  private defaultKVector(): KVector {
    return {
      k_r: 0.5,
      k_a: 0.0,
      k_i: 0.5,
      k_p: 0.5,
      k_m: 0.5,
      k_s: 0.0,
      k_h: 0.5,
      k_topo: 0.0,
      k_v: 0.0,
    };
  }

  /**
   * Get domain category for a domain
   */
  getDomainCategory(domain: ReputationDomain): DomainCategory {
    const categoryMap: Record<ReputationDomain, DomainCategory> = {
      Identity: 'Essential',
      Health: 'Essential',
      Food: 'Essential',
      Shelter: 'Essential',
      Finance: 'Economic',
      Property: 'Economic',
      Marketplace: 'Economic',
      Energy: 'Economic',
      Governance: 'Social',
      Education: 'Social',
      Media: 'Social',
      Music: 'Social',
      SupplyChain: 'Technical',
      Fabrication: 'Technical',
      Knowledge: 'Technical',
      DeSci: 'Technical',
      Justice: 'Justice',
    };
    return categoryMap[domain];
  }

  /**
   * Create a domain reputation record
   */
  createDomainRecord(
    domain: ReputationDomain,
    kVector: KVector,
    interactionCount: number,
    verified: boolean = false,
    metadata?: Record<string, unknown>
  ): DomainReputationRecord {
    return {
      domain,
      category: this.getDomainCategory(domain),
      kVector,
      trustScore: this.calculateKVectorTrustScore(kVector),
      interactionCount,
      verified,
      metadata,
      fetchedAt: Date.now(),
    };
  }

  /**
   * Clear cache for an agent
   */
  clearCache(agentPubKey?: AgentPubKey): void {
    if (agentPubKey) {
      this.reputationCache.delete(agentPubKey.toString());
    } else {
      this.reputationCache.clear();
    }
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Create a new cross-domain reputation service
 */
export function createCrossDomainReputationService(
  config?: CrossDomainReputationConfig
): CrossDomainReputationService {
  return new CrossDomainReputationService(config);
}

/**
 * Check if reputation meets trust threshold for an operation
 */
export function meetsTrustThreshold(
  reputation: CrossDomainReputation,
  requiredLevel: TrustLevel
): boolean {
  const levels: TrustLevel[] = [
    'Unknown',
    'Untrusted',
    'Suspicious',
    'Conditional',
    'Trusted',
    'HighlyTrusted',
  ];

  const currentIndex = levels.indexOf(reputation.trustLevel);
  const requiredIndex = levels.indexOf(requiredLevel);

  return currentIndex >= requiredIndex;
}

/**
 * Get recommended actions based on reputation
 */
export function getReputationRecommendations(
  reputation: CrossDomainReputation
): string[] {
  const recommendations: string[] = [];

  if (reputation.trustLevel === 'Unknown') {
    recommendations.push('Verify identity in Identity domain');
    recommendations.push('Build reputation through interactions');
  }

  if (reputation.byzantineRisk > 0.5) {
    recommendations.push('High Byzantine risk - require additional verification');
  }

  if (reputation.correlationAnalysis.anomalousDomains.length > 0) {
    recommendations.push(
      `Review activity in anomalous domains: ${reputation.correlationAnalysis.anomalousDomains.join(', ')}`
    );
  }

  if (reputation.trend === 'Declining') {
    recommendations.push('Recent negative trend - monitor closely');
  }

  if (reputation.aggregatedKVector.k_v < 0.5) {
    recommendations.push('Low verification score - complete identity verification');
  }

  return recommendations;
}

/**
 * Format reputation for display
 */
export function formatReputationSummary(reputation: CrossDomainReputation): string {
  const lines = [
    `Trust Level: ${reputation.trustLevel}`,
    `Overall Score: ${(reputation.overallTrustScore * 100).toFixed(1)}%`,
    `Active Domains: ${reputation.activeDomainCount}`,
    `Total Interactions: ${reputation.totalInteractions}`,
    `Trend: ${reputation.trend}`,
    `Byzantine Risk: ${(reputation.byzantineRisk * 100).toFixed(1)}%`,
  ];
  return lines.join('\n');
}
