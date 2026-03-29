// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Knowledge Metabolism Engine
 *
 * Manages the lifecycle of claims - from birth through death to decomposition.
 * Key principle: Claims are living things that must be actively maintained.
 *
 * @packageDocumentation
 * @module metabolism/engine
 */

import {
  type LifecyclePhase,
  type DeathReason,
  type ClaimHealth,
  type Tombstone,
  type TombstoneSummary,
  type LifecycleTransition,
  type TransitionRule,
  type MetabolizingClaim,
  type DecompositionResult,
  type MetabolismConfig,
  type MetabolismEvent,
  DEFAULT_METABOLISM_CONFIG,
} from './types.js';
import { EmpiricalLevel } from '../epistemic/index.js';
import {
  getSCEIEventBus,
  createSCEIEvent,
  type SCEIEventBus,
} from '../scei/event-bus.js';
import { getSCEIMetrics, type SCEIMetricsCollector } from '../scei/metrics.js';
import {
  getSCEIPersistence,
  type SCEIPersistenceManager,
} from '../scei/persistence.js';

import type { EpistemicClassification, Evidence } from '../epistemic/index.js';

// ============================================================================
// METABOLISM ENGINE
// ============================================================================

/**
 * Knowledge Metabolism Engine
 *
 * Manages the full lifecycle of claims in the knowledge graph.
 *
 * @example
 * ```typescript
 * const engine = new MetabolismEngine();
 *
 * // Birth a new claim
 * const claim = await engine.birthClaim({
 *   content: "The Earth orbits the Sun",
 *   classification: { empirical: 4, normative: 3, materiality: 3 },
 *   authorId: "agent-123",
 *   domain: "astronomy"
 * });
 *
 * // Add evidence
 * await engine.addEvidence(claim.id, evidence);
 *
 * // Check health
 * const health = engine.assessHealth(claim.id);
 *
 * // When a claim dies, get its tombstone
 * const tombstone = engine.getTombstone(deadClaimId);
 * ```
 */
export class MetabolismEngine {
  private claims: Map<string, MetabolizingClaim> = new Map();
  private tombstones: Map<string, Tombstone> = new Map();
  private config: MetabolismConfig;
  private transitionRules: TransitionRule[];
  private eventListeners: Array<(event: MetabolismEvent) => void> = [];
  private initialized: boolean = false;

  // SCEI Infrastructure
  private sceiEventBus: SCEIEventBus;
  private sceiMetrics: SCEIMetricsCollector;
  private persistence: SCEIPersistenceManager;

  constructor(config: Partial<MetabolismConfig> = {}) {
    this.config = { ...DEFAULT_METABOLISM_CONFIG, ...config };
    this.transitionRules = this.createDefaultRules();
    this.sceiEventBus = getSCEIEventBus();
    this.sceiMetrics = getSCEIMetrics();
    this.persistence = getSCEIPersistence();
  }

  // ==========================================================================
  // PERSISTENCE
  // ==========================================================================

  /**
   * Load claims and tombstones from persistence layer
   *
   * Call this on startup to restore previous session's data.
   *
   * @example
   * ```typescript
   * const engine = new MetabolismEngine({ enablePersistence: true });
   * await engine.loadFromPersistence();
   * ```
   */
  async loadFromPersistence(): Promise<{ claims: number; tombstones: number }> {
    if (!this.config.enablePersistence) {
      return { claims: 0, tombstones: 0 };
    }

    try {
      // Load claims
      const storage = this.persistence.getStorage();
      const claimItems = await storage.query<MetabolizingClaim>('metabolism:claims', {
        sortBy: 'createdAt',
        sortOrder: 'asc',
      });

      for (const item of claimItems) {
        this.claims.set(item.value.id, item.value);
      }

      // Load tombstones
      const tombstoneItems = await storage.query<Tombstone>('metabolism:tombstones', {
        sortBy: 'createdAt',
        sortOrder: 'asc',
      });

      for (const item of tombstoneItems) {
        this.tombstones.set(item.value.claimId, item.value);
      }

      this.initialized = true;

      this.sceiMetrics.incrementCounter('persistence_operations_total', {
        namespace: 'metabolism',
        operation: 'load',
      });
      this.sceiMetrics.setGauge('metabolism_loaded_claims', claimItems.length);
      this.sceiMetrics.setGauge('metabolism_loaded_tombstones', tombstoneItems.length);

      return { claims: claimItems.length, tombstones: tombstoneItems.length };
    } catch (error) {
      console.error('MetabolismEngine: Failed to load from persistence:', error);
      return { claims: 0, tombstones: 0 };
    }
  }

  /**
   * Persist a claim to storage
   */
  private async persistClaim(claim: MetabolizingClaim): Promise<void> {
    if (!this.config.enablePersistence) {
      return;
    }

    try {
      await this.persistence.saveClaim(claim.id, claim, {
        ttl: this.config.claimPersistenceTtlMs || undefined,
        tags: [claim.domain, claim.phase, ...claim.tags],
      });

      this.sceiMetrics.incrementCounter('persistence_operations_total', {
        namespace: 'metabolism:claims',
        operation: 'save',
      });
    } catch (error) {
      console.error('MetabolismEngine: Failed to persist claim:', error);
    }
  }

  /**
   * Persist a tombstone to storage
   */
  private async persistTombstone(tombstone: Tombstone): Promise<void> {
    if (!this.config.enablePersistence) {
      return;
    }

    try {
      await this.persistence.saveTombstone(tombstone.claimId, tombstone);

      this.sceiMetrics.incrementCounter('persistence_operations_total', {
        namespace: 'metabolism:tombstones',
        operation: 'save',
      });
    } catch (error) {
      console.error('MetabolismEngine: Failed to persist tombstone:', error);
    }
  }

  /**
   * Check if persistence is enabled and data has been loaded
   */
  isPersistenceReady(): boolean {
    return this.config.enablePersistence && this.initialized;
  }

  // ==========================================================================
  // CLAIM LIFECYCLE
  // ==========================================================================

  /**
   * Birth a new claim into the system
   */
  async birthClaim(input: {
    content: string;
    classification: EpistemicClassification;
    authorId: string;
    domain: string;
    evidence?: Evidence[];
    tags?: string[];
    confidence?: number;
  }): Promise<MetabolizingClaim> {
    const now = Date.now();
    const id = `claim_${now}_${Math.random().toString(36).slice(2, 11)}`;

    const claim: MetabolizingClaim = {
      id,
      content: input.content,
      classification: input.classification,
      confidence: input.confidence ?? 0.5,
      phase: 'nascent',
      health: this.calculateInitialHealth(input),
      evidence: input.evidence ?? [],
      activityHistory: [
        {
          type: 'evidence_added',
          timestamp: now,
          actorId: input.authorId,
          details: { event: 'birth' },
        },
      ],
      transitionHistory: [],
      dependencies: [],
      dependents: [],
      authorId: input.authorId,
      createdAt: now,
      updatedAt: now,
      lastActivityAt: now,
      domain: input.domain,
      tags: input.tags ?? [],
    };

    this.claims.set(id, claim);

    // Persist to storage
    await this.persistClaim(claim);

    this.emit({ type: 'claim_born', claim });

    // Emit SCEI event
    const sceiEvent = createSCEIEvent('metabolism:claim_born', 'metabolism', {
      claimId: id,
      domain: input.domain,
      initialConfidence: claim.confidence,
      empiricalLevel: input.classification.empirical,
    });
    void this.sceiEventBus.emit(sceiEvent);

    // Record metrics
    this.sceiMetrics.incrementCounter('metabolism_claims_born', {
      domain: input.domain,
    });
    this.sceiMetrics.setGauge('metabolism_active_claims', this.claims.size);

    return claim;
  }

  /**
   * Add evidence to a claim
   */
  async addEvidence(claimId: string, evidence: Evidence): Promise<ClaimHealth> {
    const claim = this.claims.get(claimId);
    if (!claim) {
      throw new Error(`Claim not found: ${claimId}`);
    }

    if (claim.phase === 'dead' || claim.phase === 'decomposing') {
      throw new Error(`Cannot add evidence to ${claim.phase} claim`);
    }

    claim.evidence.push(evidence);
    claim.activityHistory.push({
      type: 'evidence_added',
      timestamp: Date.now(),
      actorId: evidence.source,
    });
    claim.lastActivityAt = Date.now();
    claim.updatedAt = Date.now();

    // Recalculate health
    claim.health = this.calculateHealth(claim);

    // Check for transitions
    if (this.config.autoTransition) {
      await this.checkTransitions(claim);
    }

    // Persist updated claim
    await this.persistClaim(claim);

    this.emit({ type: 'health_updated', claimId, health: claim.health });

    // Emit SCEI event for health update
    const sceiEvent = createSCEIEvent('metabolism:health_updated', 'metabolism', {
      claimId,
      overallHealth: claim.health.overallHealth,
      evidenceStrength: claim.health.evidenceStrength,
      phase: claim.phase,
    });
    void this.sceiEventBus.emit(sceiEvent);

    // Record health metric
    this.sceiMetrics.setGauge('metabolism_claim_health', claim.health.overallHealth, {
      claim_id: claimId,
    });

    return claim.health;
  }

  /**
   * Challenge a claim
   */
  async challengeClaim(
    claimId: string,
    challengerId: string,
    reason: string,
    contradictingEvidence?: Evidence[]
  ): Promise<ClaimHealth> {
    const claim = this.claims.get(claimId);
    if (!claim) {
      throw new Error(`Claim not found: ${claimId}`);
    }

    claim.activityHistory.push({
      type: 'challenged',
      timestamp: Date.now(),
      actorId: challengerId,
      details: { reason, hasEvidence: !!contradictingEvidence },
    });

    if (contradictingEvidence) {
      claim.activityHistory.push({
        type: 'contradicted',
        timestamp: Date.now(),
        actorId: challengerId,
        details: { evidenceCount: contradictingEvidence.length },
      });
    }

    claim.health.challengeCount++;
    claim.lastActivityAt = Date.now();
    claim.health = this.calculateHealth(claim);

    if (this.config.autoTransition) {
      await this.checkTransitions(claim);
    }

    this.emit({ type: 'health_updated', claimId, health: claim.health });
    return claim.health;
  }

  /**
   * Corroborate a claim (independent confirmation)
   */
  async corroborateClaim(
    claimId: string,
    corroboratorId: string,
    independentEvidence: Evidence
  ): Promise<ClaimHealth> {
    const claim = this.claims.get(claimId);
    if (!claim) {
      throw new Error(`Claim not found: ${claimId}`);
    }

    claim.evidence.push(independentEvidence);
    claim.activityHistory.push({
      type: 'corroborated',
      timestamp: Date.now(),
      actorId: corroboratorId,
    });

    claim.health.corroborationScore += 0.1;
    claim.health.corroborationScore = Math.min(1, claim.health.corroborationScore);
    claim.lastActivityAt = Date.now();
    claim.health = this.calculateHealth(claim);

    if (this.config.autoTransition) {
      await this.checkTransitions(claim);
    }

    this.emit({ type: 'health_updated', claimId, health: claim.health });
    return claim.health;
  }

  /**
   * Kill a claim (with reason)
   */
  async killClaim(
    claimId: string,
    reason: DeathReason,
    lessonLearned: string,
    refutingEvidence: Evidence[] = [],
    supersededBy?: string
  ): Promise<Tombstone> {
    const claim = this.claims.get(claimId);
    if (!claim) {
      throw new Error(`Claim not found: ${claimId}`);
    }

    // Transition to dead
    await this.transitionTo(claim, 'dead', `Killed: ${reason}`);

    // Create tombstone
    const tombstone: Tombstone = {
      claimId,
      originalContent: claim.content,
      originalClassification: claim.classification,
      bornAt: claim.createdAt,
      diedAt: Date.now(),
      deathReason: reason,
      lessonLearned,
      refutingEvidence,
      affectedClaims: claim.dependents,
      supersededBy,
      tags: claim.tags,
    };

    this.tombstones.set(claimId, tombstone);

    // Persist tombstone and update claim in storage
    await this.persistTombstone(tombstone);
    await this.persistClaim(claim); // Persist with 'dead' phase

    this.emit({ type: 'claim_died', claimId, tombstone });
    this.emit({ type: 'tombstone_created', tombstone });

    // Emit SCEI event
    const sceiEvent = createSCEIEvent('metabolism:claim_died', 'metabolism', {
      claimId,
      deathReason: reason,
      lifetime: tombstone.diedAt - tombstone.bornAt,
      affectedClaims: claim.dependents.length,
    });
    void this.sceiEventBus.emit(sceiEvent);

    // Record metrics
    this.sceiMetrics.incrementCounter('metabolism_claims_died', {
      reason,
    });
    this.sceiMetrics.recordHistogram(
      'metabolism_claim_lifetime_ms',
      tombstone.diedAt - tombstone.bornAt,
      { reason }
    );
    this.sceiMetrics.setGauge('metabolism_active_claims', this.claims.size);
    this.sceiMetrics.setGauge('metabolism_tombstone_count', this.tombstones.size);

    // Propagate death to dependents
    await this.propagateDeath(claim, tombstone);

    return tombstone;
  }

  /**
   * Decompose a dead claim
   */
  async decomposeClaim(claimId: string): Promise<DecompositionResult> {
    const claim = this.claims.get(claimId);
    const tombstone = this.tombstones.get(claimId);

    if (!claim || claim.phase !== 'dead') {
      throw new Error(`Can only decompose dead claims: ${claimId}`);
    }

    // Transition to decomposing
    await this.transitionTo(claim, 'decomposing', 'Decomposition started');

    const result: DecompositionResult = {
      originalClaimId: claimId,
      extractedClaims: this.extractSubClaims(claim, tombstone),
      reusableEvidence: this.identifyReusableEvidence(claim, tombstone),
      methodologicalInsights: this.extractMethodologicalInsights(claim, tombstone),
      antiPatterns: this.identifyAntiPatterns(claim, tombstone),
      decomposedAt: Date.now(),
    };

    this.emit({ type: 'claim_decomposed', result });

    return result;
  }

  // ==========================================================================
  // HEALTH ASSESSMENT
  // ==========================================================================

  /**
   * Calculate health metrics for a claim
   */
  calculateHealth(claim: MetabolizingClaim): ClaimHealth {
    const now = Date.now();
    const daysSinceActivity = (now - claim.lastActivityAt) / (1000 * 60 * 60 * 24);

    // Evidence strength based on E-level and count
    const evidenceStrength = this.calculateEvidenceStrength(claim);

    // Citation impact
    const citationCount = claim.dependents.length;

    // Challenge ratio
    const challengeCount = claim.activityHistory.filter(
      (a) => a.type === 'challenged'
    ).length;

    // Corroboration from independent sources
    const corroborationScore = claim.activityHistory.filter(
      (a) => a.type === 'corroborated'
    ).length * 0.15;

    // Contradiction score
    const contradictionScore = claim.activityHistory.filter(
      (a) => a.type === 'contradicted'
    ).length * 0.2;

    // Calculate overall health
    const overallHealth = this.computeOverallHealth({
      confidence: claim.confidence,
      evidenceStrength,
      citationCount,
      challengeCount,
      daysSinceActivity,
      corroborationScore: Math.min(1, corroborationScore),
      contradictionScore: Math.min(1, contradictionScore),
      authorReputation: claim.health.authorReputation,
      overallHealth: 0, // Will be calculated
    });

    return {
      confidence: claim.confidence,
      evidenceStrength,
      citationCount,
      challengeCount,
      daysSinceActivity,
      corroborationScore: Math.min(1, corroborationScore),
      contradictionScore: Math.min(1, contradictionScore),
      authorReputation: claim.health.authorReputation,
      overallHealth,
    };
  }

  /**
   * Assess the current health of a claim
   */
  assessHealth(claimId: string): ClaimHealth | null {
    const claim = this.claims.get(claimId);
    if (!claim) return null;

    claim.health = this.calculateHealth(claim);
    return claim.health;
  }

  // ==========================================================================
  // TOMBSTONE MANAGEMENT
  // ==========================================================================

  /**
   * Get a tombstone by claim ID
   */
  getTombstone(claimId: string): Tombstone | null {
    return this.tombstones.get(claimId) ?? null;
  }

  /**
   * Search tombstones by various criteria
   */
  searchTombstones(query: {
    deathReason?: DeathReason;
    domain?: string;
    tag?: string;
    dateRange?: { start: number; end: number };
    contentContains?: string;
  }): TombstoneSummary[] {
    const results: TombstoneSummary[] = [];

    for (const tombstone of this.tombstones.values()) {
      let matches = true;

      if (query.deathReason && tombstone.deathReason !== query.deathReason) {
        matches = false;
      }

      if (query.tag && !tombstone.tags.includes(query.tag)) {
        matches = false;
      }

      if (query.dateRange) {
        if (
          tombstone.diedAt < query.dateRange.start ||
          tombstone.diedAt > query.dateRange.end
        ) {
          matches = false;
        }
      }

      if (
        query.contentContains &&
        !tombstone.originalContent
          .toLowerCase()
          .includes(query.contentContains.toLowerCase())
      ) {
        matches = false;
      }

      if (matches) {
        results.push({
          claimId: tombstone.claimId,
          originalContent: tombstone.originalContent,
          deathReason: tombstone.deathReason,
          lessonLearned: tombstone.lessonLearned,
          diedAt: tombstone.diedAt,
        });
      }
    }

    return results.sort((a, b) => b.diedAt - a.diedAt);
  }

  /**
   * Check if a new claim is similar to a tombstoned claim
   *
   * Prevents repeating past mistakes.
   */
  checkAgainstTombstones(content: string, _domain: string): {
    similar: boolean;
    warnings: Array<{ tombstone: TombstoneSummary; similarity: number }>;
  } {
    const warnings: Array<{ tombstone: TombstoneSummary; similarity: number }> = [];

    for (const tombstone of this.tombstones.values()) {
      const similarity = this.calculateSimilarity(content, tombstone.originalContent);

      if (similarity > 0.7) {
        warnings.push({
          tombstone: {
            claimId: tombstone.claimId,
            originalContent: tombstone.originalContent,
            deathReason: tombstone.deathReason,
            lessonLearned: tombstone.lessonLearned,
            diedAt: tombstone.diedAt,
          },
          similarity,
        });
      }
    }

    return {
      similar: warnings.length > 0,
      warnings: warnings.sort((a, b) => b.similarity - a.similarity),
    };
  }

  // ==========================================================================
  // TRANSITION MANAGEMENT
  // ==========================================================================

  /**
   * Check and apply lifecycle transitions
   */
  async checkTransitions(claim: MetabolizingClaim): Promise<LifecycleTransition | null> {
    // Sort rules by priority
    const sortedRules = [...this.transitionRules].sort(
      (a, b) => b.priority - a.priority
    );

    for (const rule of sortedRules) {
      if (rule.from === claim.phase && rule.condition(claim.health, claim)) {
        return this.transitionTo(claim, rule.to, rule.description);
      }
    }

    return null;
  }

  /**
   * Force a transition to a specific phase
   */
  async transitionTo(
    claim: MetabolizingClaim,
    toPhase: LifecyclePhase,
    reason: string
  ): Promise<LifecycleTransition> {
    const transition: LifecycleTransition = {
      claimId: claim.id,
      fromPhase: claim.phase,
      toPhase,
      timestamp: Date.now(),
      reason,
      healthAtTransition: { ...claim.health },
      triggeredBy: 'system',
    };

    claim.phase = toPhase;
    claim.transitionHistory.push(transition);
    claim.updatedAt = Date.now();

    // Trim history if too long
    if (claim.transitionHistory.length > this.config.maxTransitionHistory) {
      claim.transitionHistory = claim.transitionHistory.slice(
        -this.config.maxTransitionHistory
      );
    }

    this.emit({ type: 'claim_transitioned', transition });
    return transition;
  }

  // ==========================================================================
  // EVENT HANDLING
  // ==========================================================================

  /**
   * Subscribe to metabolism events
   */
  onEvent(listener: (event: MetabolismEvent) => void): () => void {
    this.eventListeners.push(listener);
    return () => {
      const index = this.eventListeners.indexOf(listener);
      if (index >= 0) {
        this.eventListeners.splice(index, 1);
      }
    };
  }

  private emit(event: MetabolismEvent): void {
    for (const listener of this.eventListeners) {
      try {
        listener(event);
      } catch (e) {
        console.error('Metabolism event listener error:', e);
      }
    }
  }

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  /**
   * Get metabolism statistics
   */
  getStats(): MetabolismStats {
    const phaseCount: Record<LifecyclePhase, number> = {
      nascent: 0,
      growing: 0,
      mature: 0,
      stagnant: 0,
      challenged: 0,
      dying: 0,
      dead: 0,
      decomposing: 0,
    };

    let totalHealth = 0;
    let healthCount = 0;

    for (const claim of this.claims.values()) {
      phaseCount[claim.phase]++;
      if (claim.phase !== 'dead' && claim.phase !== 'decomposing') {
        totalHealth += claim.health.overallHealth;
        healthCount++;
      }
    }

    const deathReasonCount: Record<DeathReason, number> = {
      refuted: 0,
      superseded: 0,
      obsolete: 0,
      retracted: 0,
      merged: 0,
      decomposed: 0,
      abandoned: 0,
    };

    for (const tombstone of this.tombstones.values()) {
      deathReasonCount[tombstone.deathReason]++;
    }

    return {
      totalClaims: this.claims.size,
      totalTombstones: this.tombstones.size,
      claimsByPhase: phaseCount,
      tombstonesByReason: deathReasonCount,
      averageHealth: healthCount > 0 ? totalHealth / healthCount : 0,
      oldestClaim: this.getOldestClaim(),
      newestClaim: this.getNewestClaim(),
    };
  }

  // ==========================================================================
  // PRIVATE HELPERS
  // ==========================================================================

  private calculateInitialHealth(input: {
    classification: EpistemicClassification;
    evidence?: Evidence[];
    confidence?: number;
  }): ClaimHealth {
    return {
      confidence: input.confidence ?? 0.5,
      evidenceStrength: (input.evidence?.length ?? 0) * 0.1,
      citationCount: 0,
      challengeCount: 0,
      daysSinceActivity: 0,
      corroborationScore: 0,
      contradictionScore: 0,
      authorReputation: 0.5,
      overallHealth: 0.5,
    };
  }

  private calculateEvidenceStrength(claim: MetabolizingClaim): number {
    if (claim.evidence.length === 0) return 0;

    // Base on E-level ceiling and evidence count
    const eCeiling = [0.4, 0.6, 0.8, 0.95, 0.99][claim.classification.empirical];
    const countFactor = Math.min(1, claim.evidence.length * 0.15);

    return eCeiling * countFactor;
  }

  private computeOverallHealth(health: ClaimHealth): number {
    // Weighted combination of factors
    const weights = {
      confidence: 0.2,
      evidenceStrength: 0.25,
      corroboration: 0.2,
      freshness: 0.15,
      stability: 0.2,
    };

    const freshnessScore = Math.max(0, 1 - health.daysSinceActivity / 365);
    const stabilityScore = Math.max(
      0,
      1 - health.challengeCount * 0.1 - health.contradictionScore
    );

    const overall =
      health.confidence * weights.confidence +
      health.evidenceStrength * weights.evidenceStrength +
      health.corroborationScore * weights.corroboration +
      freshnessScore * weights.freshness +
      stabilityScore * weights.stability;

    return Math.max(0, Math.min(1, overall));
  }

  private createDefaultRules(): TransitionRule[] {
    return [
      // nascent → growing
      {
        from: 'nascent',
        to: 'growing',
        condition: (h) => h.evidenceStrength >= this.config.nascentExitThreshold,
        priority: 10,
        description: 'Sufficient evidence to start growing',
      },
      // growing → mature
      {
        from: 'growing',
        to: 'mature',
        condition: (h) =>
          h.evidenceStrength >= 0.6 &&
          h.corroborationScore >= 0.3 &&
          h.citationCount >= 3,
        priority: 10,
        description: 'Well-evidenced with corroboration and citations',
      },
      // any → stagnant (except dead/decomposing)
      {
        from: 'mature',
        to: 'stagnant',
        condition: (h) => h.daysSinceActivity > this.config.stagnationThresholdDays,
        priority: 5,
        description: 'No activity for extended period',
      },
      {
        from: 'growing',
        to: 'stagnant',
        condition: (h) => h.daysSinceActivity > this.config.stagnationThresholdDays,
        priority: 5,
        description: 'No activity for extended period',
      },
      // any → challenged
      {
        from: 'mature',
        to: 'challenged',
        condition: (h) => h.challengeCount >= 3 || h.contradictionScore >= this.config.challengeThreshold,
        priority: 15,
        description: 'Under significant challenge',
      },
      {
        from: 'growing',
        to: 'challenged',
        condition: (h) => h.challengeCount >= 2 || h.contradictionScore >= this.config.challengeThreshold,
        priority: 15,
        description: 'Under challenge while still growing',
      },
      // challenged → dying
      {
        from: 'challenged',
        to: 'dying',
        condition: (h) => h.contradictionScore >= this.config.dyingThreshold,
        priority: 20,
        description: 'Contradictions exceeding survivable threshold',
      },
      // dying → dead
      {
        from: 'dying',
        to: 'dead',
        condition: (h) => h.overallHealth < this.config.deathThreshold,
        priority: 25,
        description: 'Health too low to survive',
      },
      // challenged → growing (if challenge resolved)
      {
        from: 'challenged',
        to: 'growing',
        condition: (h) =>
          h.corroborationScore > h.contradictionScore + 0.2 &&
          h.evidenceStrength >= 0.4,
        priority: 12,
        description: 'Challenge resolved through corroboration',
      },
      // stagnant → mature (revived)
      {
        from: 'stagnant',
        to: 'mature',
        condition: (h) => h.daysSinceActivity < 7 && h.overallHealth >= 0.6,
        priority: 8,
        description: 'Revived through new activity',
      },
    ];
  }

  private async propagateDeath(
    claim: MetabolizingClaim,
    tombstone: Tombstone
  ): Promise<void> {
    // Notify dependents that a dependency has died
    for (const dependentId of claim.dependents) {
      const dependent = this.claims.get(dependentId);
      if (dependent) {
        dependent.activityHistory.push({
          type: 'contradicted',
          timestamp: Date.now(),
          actorId: 'system',
          details: {
            event: 'dependency_died',
            deadClaimId: claim.id,
            deathReason: tombstone.deathReason,
          },
        });

        // Recalculate health
        dependent.health = this.calculateHealth(dependent);

        if (this.config.autoTransition) {
          await this.checkTransitions(dependent);
        }
      }
    }
  }

  private extractSubClaims(
    _claim: MetabolizingClaim,
    _tombstone: Tombstone | undefined
  ): Array<{ content: string; confidence: number; rationale: string }> {
    // In a real implementation, this would use NLP to break down the claim
    // For now, return empty - this would require human review
    return [];
  }

  private identifyReusableEvidence(
    claim: MetabolizingClaim,
    tombstone: Tombstone | undefined
  ): Evidence[] {
    // Evidence that wasn't part of the refutation might still be valid
    if (!tombstone) return claim.evidence;

    const refutingIds = new Set(
      tombstone.refutingEvidence.map((e) => e.data)
    );
    return claim.evidence.filter((e) => !refutingIds.has(e.data));
  }

  private extractMethodologicalInsights(
    claim: MetabolizingClaim,
    tombstone: Tombstone | undefined
  ): string[] {
    const insights: string[] = [];

    if (tombstone?.deathReason === 'refuted') {
      insights.push(
        'Claim was refuted - review evidence gathering methodology'
      );
    }

    if (claim.health.corroborationScore < 0.1 && claim.confidence > 0.7) {
      insights.push(
        'High confidence without independent corroboration - seek diverse sources'
      );
    }

    return insights;
  }

  private identifyAntiPatterns(
    claim: MetabolizingClaim,
    _tombstone: Tombstone | undefined
  ): string[] {
    const patterns: string[] = [];

    if (
      claim.classification.empirical <= EmpiricalLevel.E1_Testimonial &&
      claim.confidence > 0.8
    ) {
      patterns.push('Over-confidence for testimonial evidence');
    }

    if (claim.evidence.length < 2 && claim.confidence > 0.7) {
      patterns.push('High confidence with minimal evidence');
    }

    return patterns;
  }

  private calculateSimilarity(a: string, b: string): number {
    // Simple Jaccard similarity on words
    const wordsA = new Set(a.toLowerCase().split(/\s+/));
    const wordsB = new Set(b.toLowerCase().split(/\s+/));

    let intersection = 0;
    for (const word of wordsA) {
      if (wordsB.has(word)) intersection++;
    }

    const union = wordsA.size + wordsB.size - intersection;
    return union > 0 ? intersection / union : 0;
  }

  private getOldestClaim(): number | null {
    let oldest: number | null = null;
    for (const claim of this.claims.values()) {
      if (oldest === null || claim.createdAt < oldest) {
        oldest = claim.createdAt;
      }
    }
    return oldest;
  }

  private getNewestClaim(): number | null {
    let newest: number | null = null;
    for (const claim of this.claims.values()) {
      if (newest === null || claim.createdAt > newest) {
        newest = claim.createdAt;
      }
    }
    return newest;
  }
}

/**
 * Statistics for the metabolism system
 */
export interface MetabolismStats {
  totalClaims: number;
  totalTombstones: number;
  claimsByPhase: Record<LifecyclePhase, number>;
  tombstonesByReason: Record<DeathReason, number>;
  averageHealth: number;
  oldestClaim: number | null;
  newestClaim: number | null;
}

// Singleton
let instance: MetabolismEngine | null = null;

export function getMetabolismEngine(
  config?: Partial<MetabolismConfig>
): MetabolismEngine {
  if (!instance) {
    instance = new MetabolismEngine(config);
  }
  return instance;
}

export function resetMetabolismEngine(): void {
  instance = null;
}
