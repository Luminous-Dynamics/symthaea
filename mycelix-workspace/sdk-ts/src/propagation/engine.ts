// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Safe Belief Propagator
 *
 * Manages how belief updates flow through the knowledge graph with
 * circuit breakers to prevent cascade failures.
 *
 * Key safety mechanisms:
 * - Rate limits per claim and globally
 * - Quarantine for suspicious patterns
 * - Human-in-the-loop for high-centrality nodes
 * - Depth limits to prevent infinite propagation
 * - Calibration gating (requires CalibrationEngine approval)
 *
 * @packageDocumentation
 * @module propagation/engine
 */

import {
  type BlockReason,
  type PropagationRequest,
  type PropagationResult,
  type AffectedClaim,
  type PropagationTrace,
  type CircuitBreakerCheck,
  type QuarantinedPropagation,
  type SuspiciousPattern,
  type QuarantineDecision,
  type ClaimRateLimitState,
  type GlobalRateLimitState,
  type HumanReviewRequest,
  type HumanReviewDecision,
  type PropagationEvent,
  type PropagationConfig,
  DEFAULT_PROPAGATION_CONFIG,
  DEFAULT_CIRCUIT_BREAKER_CONFIG,
} from './types.js';
import {
  getSCEIEventBus,
  createSCEIEvent,
  type SCEIEventBus,
} from '../scei/event-bus.js';
import { getSCEIMetrics, type SCEIMetricsCollector } from '../scei/metrics.js';

import type { CalibrationEngine } from '../calibration/engine.js';

// ============================================================================
// SAFE BELIEF PROPAGATOR
// ============================================================================

/**
 * Safe Belief Propagator
 *
 * Propagates belief updates through the knowledge graph with comprehensive
 * safety mechanisms to prevent cascade failures.
 *
 * @example
 * ```typescript
 * const propagator = new SafeBeliefPropagator(calibrationEngine);
 *
 * // Request a propagation
 * const result = await propagator.propagate({
 *   id: 'prop-123',
 *   sourceClaimId: 'claim-456',
 *   updateType: 'confidence_increase',
 *   newConfidence: 0.85,
 *   oldConfidence: 0.7,
 *   initiatedBy: 'evidence-789',
 *   createdAt: Date.now(),
 *   priority: 1,
 * });
 *
 * if (result.state === 'blocked') {
 *   console.log(`Blocked: ${result.blockReason}`);
 * }
 *
 * if (result.pendingHumanReview) {
 *   console.log('Awaiting human review');
 * }
 * ```
 */
export class SafeBeliefPropagator {
  private config: PropagationConfig;
  private calibrationEngine: CalibrationEngine | null;

  // Rate limiting state
  private claimRateLimits: Map<string, ClaimRateLimitState> = new Map();
  private globalRateLimit: GlobalRateLimitState = {
    propagationsThisHour: 0,
    hourStarted: Date.now(),
    quarantinesThisHour: 0,
    blocksThisHour: 0,
  };

  // Statistics tracking
  private stats = {
    totalPropagations: 0,
    successfulPropagations: 0,
    blockedPropagations: 0,
  };

  // Quarantine and review queues
  private quarantine: Map<string, QuarantinedPropagation> = new Map();
  private humanReviewQueue: Map<string, HumanReviewRequest> = new Map();
  private pendingDecisions: Map<string, QuarantineDecision | HumanReviewDecision> = new Map();

  // Event listeners
  private eventListeners: Array<(event: PropagationEvent) => void> = [];

  // Claim graph (simplified - in real implementation this would query the DHT)
  private claimDependencies: Map<string, string[]> = new Map();
  private claimConfidences: Map<string, number> = new Map();
  private claimCentrality: Map<string, number> = new Map();
  private claimDomains: Map<string, string> = new Map();

  // SCEI Infrastructure
  private sceiEventBus: SCEIEventBus;
  private sceiMetrics: SCEIMetricsCollector;

  constructor(
    calibrationEngine: CalibrationEngine | null = null,
    config: Partial<PropagationConfig> = {}
  ) {
    this.calibrationEngine = calibrationEngine;
    this.config = {
      ...DEFAULT_PROPAGATION_CONFIG,
      ...config,
      circuitBreaker: {
        ...DEFAULT_CIRCUIT_BREAKER_CONFIG,
        ...config.circuitBreaker,
      },
    };
    this.sceiEventBus = getSCEIEventBus();
    this.sceiMetrics = getSCEIMetrics();
  }

  /**
   * Subscribe to propagation events
   */
  on(listener: (event: PropagationEvent) => void): void {
    this.eventListeners.push(listener);
  }

  // ==========================================================================
  // MAIN PROPAGATION
  // ==========================================================================

  /**
   * Request a belief propagation
   *
   * This is the main entry point. The propagator will:
   * 1. Check circuit breakers
   * 2. Check rate limits
   * 3. Check calibration (if engine available)
   * 4. Detect suspicious patterns
   * 5. Queue for human review if needed
   * 6. Execute propagation if safe
   */
  async propagate(request: PropagationRequest): Promise<PropagationResult> {
    const startTime = Date.now();
    this.stats.totalPropagations++;
    this.emit({ type: 'propagation_started', request });

    // Initialize trace if enabled
    const trace: PropagationTrace | undefined = this.config.enableTracing
      ? {
          request,
          steps: [],
          circuitBreakerChecks: [],
          totalVisited: 0,
          totalModified: 0,
          peakClaimsInMemory: 0,
        }
      : undefined;

    // Step 1: Pre-flight circuit breaker checks
    const preflightResult = await this.preflightChecks(request, trace);
    if (preflightResult.blocked) {
      this.stats.blockedPropagations++;
      const result = this.createBlockedResult(
        request,
        preflightResult.reason!,
        startTime,
        trace
      );
      this.emit({ type: 'propagation_blocked', request, reason: preflightResult.reason! });
      return result;
    }

    // Step 2: Check rate limits
    const rateLimitResult = this.checkRateLimits(request, trace);
    if (rateLimitResult.blocked) {
      this.stats.blockedPropagations++;
      const result = this.createBlockedResult(
        request,
        rateLimitResult.reason!,
        startTime,
        trace
      );
      this.emit({ type: 'propagation_blocked', request, reason: rateLimitResult.reason! });
      return result;
    }

    // Step 3: Check calibration (only if enabled and engine available)
    if (this.config.enableCalibrationGating && this.calibrationEngine) {
      const domain = this.claimDomains.get(request.sourceClaimId) ?? 'default';
      const calibrationResult = this.calibrationEngine.canPropagate(
        { confidence: request.newConfidence, domain },
        domain
      );
      if (!calibrationResult.allowed) {
        this.stats.blockedPropagations++;
        const result = this.createBlockedResult(
          request,
          'calibration_failed',
          startTime,
          trace
        );
        result.warnings.push(calibrationResult.reason);
        this.emit({ type: 'propagation_blocked', request, reason: 'calibration_failed' });
        return result;
      }
    }

    // Step 4: Detect suspicious patterns
    const suspicionResult = await this.detectSuspiciousPatterns(request);
    if (suspicionResult.shouldQuarantine) {
      return this.quarantinePropagation(request, suspicionResult, startTime, trace);
    }

    // Step 5: Check if human review is required
    const needsReview = this.requiresHumanReview(request);
    if (needsReview) {
      return this.requestHumanReview(request, 'large_impact', startTime, trace);
    }

    // Step 6: Execute propagation
    const result = await this.executePropagation(request, startTime, trace);
    if (result.state === 'completed') {
      this.stats.successfulPropagations++;
    }
    return result;
  }

  /**
   * Execute the actual propagation
   */
  private async executePropagation(
    request: PropagationRequest,
    startTime: number,
    trace?: PropagationTrace
  ): Promise<PropagationResult> {
    const affectedClaims: AffectedClaim[] = [];
    const visited = new Set<string>();
    const queue: Array<{ claimId: string; depth: number }> = [];

    // Start with dependents of the source claim
    const dependents = this.claimDependencies.get(request.sourceClaimId) ?? [];
    for (const dep of dependents) {
      queue.push({ claimId: dep, depth: 1 });
    }

    // BFS through dependency graph
    while (queue.length > 0) {
      const { claimId, depth } = queue.shift()!;

      // Circuit breaker: max depth
      if (depth > this.config.circuitBreaker.maxDepth) {
        trace?.steps.push({
          claimId,
          depth,
          action: 'block',
          reason: 'max_depth_exceeded',
          timestamp: Date.now(),
        });
        continue;
      }

      // Circuit breaker: max affected claims
      if (affectedClaims.length >= this.config.circuitBreaker.maxAffectedClaims) {
        trace?.steps.push({
          claimId,
          depth,
          action: 'block',
          reason: 'max_affected_claims_exceeded',
          timestamp: Date.now(),
        });
        this.emit({
          type: 'circuit_breaker_tripped',
          check: 'maxAffectedClaims',
          value: affectedClaims.length,
          threshold: this.config.circuitBreaker.maxAffectedClaims,
        });
        break;
      }

      // Skip if already visited
      if (visited.has(claimId)) {
        trace?.steps.push({
          claimId,
          depth,
          action: 'skip',
          reason: 'already_visited',
          timestamp: Date.now(),
        });
        continue;
      }
      visited.add(claimId);

      // Check centrality for human review
      const centrality = this.claimCentrality.get(claimId) ?? 0;
      if (centrality >= this.config.circuitBreaker.centralityThresholdForReview) {
        // High-centrality claims require human review
        trace?.steps.push({
          claimId,
          depth,
          action: 'block',
          reason: 'high_centrality_requires_review',
          timestamp: Date.now(),
        });

        affectedClaims.push({
          claimId,
          oldConfidence: this.claimConfidences.get(claimId) ?? 0.5,
          newConfidence: this.claimConfidences.get(claimId) ?? 0.5,
          depth,
          relationship: 'dependent',
          applied: false,
          reason: 'Requires human review due to high centrality',
        });
        continue;
      }

      // Calculate new confidence
      const oldConfidence = this.claimConfidences.get(claimId) ?? 0.5;
      const newConfidence = this.calculatePropagatedConfidence(
        request,
        oldConfidence,
        depth
      );

      // Circuit breaker: max confidence jump (use epsilon for floating point)
      const confidenceJump = Math.abs(newConfidence - oldConfidence);
      const EPSILON = 1e-10;
      if (confidenceJump > this.config.circuitBreaker.maxConfidenceJump + EPSILON) {
        trace?.steps.push({
          claimId,
          depth,
          action: 'block',
          reason: 'confidence_jump_too_large',
          confidenceBefore: oldConfidence,
          confidenceAfter: newConfidence,
          timestamp: Date.now(),
        });

        affectedClaims.push({
          claimId,
          oldConfidence,
          newConfidence,
          depth,
          relationship: 'dependent',
          applied: false,
          reason: `Confidence jump ${(confidenceJump * 100).toFixed(1)}% exceeds limit`,
        });
        continue;
      }

      // Apply update
      this.claimConfidences.set(claimId, newConfidence);

      trace?.steps.push({
        claimId,
        depth,
        action: 'update',
        reason: 'propagation_applied',
        confidenceBefore: oldConfidence,
        confidenceAfter: newConfidence,
        timestamp: Date.now(),
      });

      affectedClaims.push({
        claimId,
        oldConfidence,
        newConfidence,
        depth,
        relationship: 'dependent',
        applied: true,
      });

      this.emit({ type: 'claim_updated', claimId, oldConfidence, newConfidence });

      // Queue dependents for next level
      const nextDependents = this.claimDependencies.get(claimId) ?? [];
      for (const nextDep of nextDependents) {
        queue.push({ claimId: nextDep, depth: depth + 1 });
      }
    }

    // Update rate limits
    this.recordPropagation(request.sourceClaimId);

    // Update trace stats
    if (trace) {
      trace.totalVisited = visited.size;
      trace.totalModified = affectedClaims.filter((c) => c.applied).length;
      trace.peakClaimsInMemory = visited.size;
    }

    const result: PropagationResult = {
      requestId: request.id,
      state: 'completed',
      affectedClaims,
      depthReached: Math.max(0, ...affectedClaims.map((c) => c.depth)),
      durationMs: Date.now() - startTime,
      pendingHumanReview: affectedClaims.some((c) => !c.applied && c.reason?.includes('review')),
      warnings: [],
      trace,
    };

    this.emit({ type: 'propagation_completed', result });

    // Emit SCEI event
    const sceiEvent = createSCEIEvent('propagation:completed', 'propagation', {
      requestId: request.id,
      sourceClaimId: request.sourceClaimId,
      affectedClaimsCount: affectedClaims.length,
      appliedCount: affectedClaims.filter((c) => c.applied).length,
      depthReached: result.depthReached,
      durationMs: result.durationMs,
    });
    void this.sceiEventBus.emit(sceiEvent);

    // Record metrics
    this.sceiMetrics.incrementCounter('propagation_completed', {
      update_type: request.updateType,
    });
    this.sceiMetrics.recordHistogram('propagation_duration_ms', result.durationMs);
    this.sceiMetrics.recordHistogram('propagation_affected_claims', affectedClaims.length);
    this.sceiMetrics.recordHistogram('propagation_depth', result.depthReached);

    return result;
  }

  // ==========================================================================
  // CIRCUIT BREAKERS
  // ==========================================================================

  private async preflightChecks(
    request: PropagationRequest,
    trace?: PropagationTrace
  ): Promise<{ blocked: boolean; reason?: BlockReason }> {
    const checks: CircuitBreakerCheck[] = [];

    // Check if source claim exists and is valid
    if (!this.claimConfidences.has(request.sourceClaimId)) {
      checks.push({
        check: 'source_claim_exists',
        passed: false,
        currentValue: 0,
        threshold: 1,
        actionIfFailed: 'block',
      });
      trace?.circuitBreakerChecks.push(...checks);
      return { blocked: true, reason: 'source_claim_invalid' };
    }

    // Check confidence jump on source
    // Use small epsilon for floating point comparison (0.8 - 0.5 = 0.30000000000000004 in JS)
    const EPSILON = 1e-10;
    const currentConfidence = this.claimConfidences.get(request.sourceClaimId) ?? 0.5;
    const confidenceJump = Math.abs(request.newConfidence - currentConfidence);
    const maxJump = this.config.circuitBreaker.maxConfidenceJump;
    checks.push({
      check: 'source_confidence_jump',
      passed: confidenceJump <= maxJump + EPSILON,
      currentValue: confidenceJump,
      threshold: maxJump,
      actionIfFailed: 'block',
    });

    if (confidenceJump > maxJump + EPSILON) {
      trace?.circuitBreakerChecks.push(...checks);
      return { blocked: true, reason: 'confidence_jump_too_large' };
    }

    // Estimate affected claims
    const estimatedAffected = this.estimateAffectedClaims(request.sourceClaimId);
    checks.push({
      check: 'estimated_affected_claims',
      passed: estimatedAffected <= this.config.circuitBreaker.maxAffectedClaims,
      currentValue: estimatedAffected,
      threshold: this.config.circuitBreaker.maxAffectedClaims,
      actionIfFailed: 'require_review',
    });

    trace?.circuitBreakerChecks.push(...checks);

    // If too many estimated affected, require human review
    if (estimatedAffected > this.config.circuitBreaker.maxAffectedClaims) {
      return { blocked: true, reason: 'human_review_required' };
    }

    return { blocked: false };
  }

  private checkRateLimits(
    request: PropagationRequest,
    _trace?: PropagationTrace
  ): { blocked: boolean; reason?: BlockReason } {
    const now = Date.now();

    // Reset hourly counters if needed
    this.maybeResetHourlyCounters(now);

    // Check global rate limit
    if (this.globalRateLimit.propagationsThisHour >= this.config.circuitBreaker.globalRateLimit) {
      this.emit({ type: 'rate_limit_exceeded', global: true });
      return { blocked: true, reason: 'rate_limit_exceeded' };
    }

    // Check per-claim rate limit
    const claimLimit = this.claimRateLimits.get(request.sourceClaimId);
    if (claimLimit) {
      // Check cooldown
      if (now - claimLimit.lastPropagation < this.config.circuitBreaker.cooldownPeriodMs) {
        return { blocked: true, reason: 'cooldown_active' };
      }

      // Check hourly limit
      if (claimLimit.propagationsThisHour >= this.config.circuitBreaker.perClaimRateLimit) {
        this.emit({ type: 'rate_limit_exceeded', claimId: request.sourceClaimId, global: false });
        return { blocked: true, reason: 'rate_limit_exceeded' };
      }
    }

    return { blocked: false };
  }

  // ==========================================================================
  // SUSPICIOUS PATTERN DETECTION
  // ==========================================================================

  private async detectSuspiciousPatterns(
    request: PropagationRequest
  ): Promise<{ shouldQuarantine: boolean; patterns: SuspiciousPattern[]; score: number }> {
    const patterns: SuspiciousPattern[] = [];
    let suspicionScore = 0;

    // Pattern 1: Rapid updates to same claim
    const claimLimit = this.claimRateLimits.get(request.sourceClaimId);
    if (claimLimit && claimLimit.propagationsThisHour >= 5) {
      const pattern: SuspiciousPattern = {
        type: 'rapid_updates',
        confidence: 0.7,
        description: `${claimLimit.propagationsThisHour} updates to this claim in the last hour`,
        evidence: [`propagations_this_hour: ${claimLimit.propagationsThisHour}`],
      };
      patterns.push(pattern);
      suspicionScore += 0.3;
    }

    // Pattern 2: Confidence oscillation
    // (Would need historical confidence data - simplified here)
    const oldConfidence = this.claimConfidences.get(request.sourceClaimId) ?? 0.5;
    const change = request.newConfidence - oldConfidence;
    const isOscillation = this.detectOscillation(request.sourceClaimId, change);
    if (isOscillation) {
      const pattern: SuspiciousPattern = {
        type: 'confidence_oscillation',
        confidence: 0.6,
        description: 'Confidence is oscillating back and forth',
        evidence: ['Recent changes show alternating direction'],
      };
      patterns.push(pattern);
      suspicionScore += 0.4;
    }

    // Pattern 3: Unusual source
    if (this.isUnusualSource(request.initiatedBy)) {
      const pattern: SuspiciousPattern = {
        type: 'unusual_source',
        confidence: 0.5,
        description: 'Update from an unusual or new source',
        evidence: [`source: ${request.initiatedBy}`],
      };
      patterns.push(pattern);
      suspicionScore += 0.2;
    }

    const shouldQuarantine =
      this.config.circuitBreaker.enableQuarantine &&
      suspicionScore >= this.config.circuitBreaker.quarantineThreshold;

    return { shouldQuarantine, patterns, score: suspicionScore };
  }

  // ==========================================================================
  // QUARANTINE
  // ==========================================================================

  private quarantinePropagation(
    request: PropagationRequest,
    suspicion: { patterns: SuspiciousPattern[]; score: number },
    startTime: number,
    trace?: PropagationTrace
  ): PropagationResult {
    const quarantined: QuarantinedPropagation = {
      request,
      quarantineReason: suspicion.patterns.map((p) => p.description).join('; '),
      suspicionScore: suspicion.score,
      detectedPatterns: suspicion.patterns,
      quarantinedAt: Date.now(),
      reviewDeadline: Date.now() + this.config.quarantineTimeoutMs,
      potentialImpact: {
        estimatedAffectedClaims: this.estimateAffectedClaims(request.sourceClaimId),
        highCentralityClaims: this.getHighCentralityDependents(request.sourceClaimId),
        maxConfidenceChange: Math.abs(
          request.newConfidence - (this.claimConfidences.get(request.sourceClaimId) ?? 0.5)
        ),
      },
    };

    this.quarantine.set(request.id, quarantined);
    this.globalRateLimit.quarantinesThisHour++;

    this.emit({ type: 'quarantine_added', quarantine: quarantined });

    return {
      requestId: request.id,
      state: 'quarantined',
      blockReason: 'quarantined',
      affectedClaims: [],
      depthReached: 0,
      durationMs: Date.now() - startTime,
      pendingHumanReview: true,
      warnings: suspicion.patterns.map((p) => p.description),
      trace,
    };
  }

  /**
   * Decide on a quarantined propagation
   */
  async decideQuarantine(decision: QuarantineDecision): Promise<PropagationResult | null> {
    const quarantined = this.quarantine.get(decision.quarantineId);
    if (!quarantined) {
      return null;
    }

    this.pendingDecisions.set(decision.quarantineId, decision);
    this.quarantine.delete(decision.quarantineId);

    this.emit({ type: 'quarantine_decided', decision });

    if (decision.decision === 'approve') {
      // Re-run propagation
      const modifiedRequest = {
        ...quarantined.request,
        ...decision.modifiedRequest,
      };
      return this.executePropagation(modifiedRequest, Date.now());
    }

    // Rejected
    return {
      requestId: quarantined.request.id,
      state: 'rejected',
      affectedClaims: [],
      depthReached: 0,
      durationMs: 0,
      pendingHumanReview: false,
      warnings: [`Rejected: ${decision.reason}`],
    };
  }

  // ==========================================================================
  // HUMAN REVIEW
  // ==========================================================================

  private requiresHumanReview(request: PropagationRequest): boolean {
    // Check explicit flag first
    if (request.requiresHumanReview === true) {
      return true;
    }

    // Check centrality threshold
    const centrality = this.claimCentrality.get(request.sourceClaimId) ?? 0;
    const threshold = this.config.circuitBreaker.centralityThresholdForReview;
    if (centrality >= threshold) {
      return true;
    }

    // Check estimated affected claims (more than 50% of max)
    const estimatedAffected = this.estimateAffectedClaims(request.sourceClaimId);
    if (estimatedAffected > this.config.circuitBreaker.maxAffectedClaims * 0.5) {
      return true;
    }

    return false;
  }

  private requestHumanReview(
    request: PropagationRequest,
    reason: HumanReviewRequest['reviewReason'],
    startTime: number,
    trace?: PropagationTrace
  ): PropagationResult {
    const reviewRequest: HumanReviewRequest = {
      id: `review_${request.id}`,
      propagationRequest: request,
      reviewReason: reason,
      claimsToReview: this.getHighCentralityDependents(request.sourceClaimId),
      estimatedImpact: {
        affectedClaims: this.estimateAffectedClaims(request.sourceClaimId),
        maxConfidenceChange: Math.abs(
          request.newConfidence - (this.claimConfidences.get(request.sourceClaimId) ?? 0.5)
        ),
      },
      createdAt: Date.now(),
      deadline: Date.now() + this.config.humanReviewTimeoutMs,
      status: 'pending',
    };

    this.humanReviewQueue.set(reviewRequest.id, reviewRequest);
    this.emit({ type: 'human_review_requested', request: reviewRequest });

    return {
      requestId: request.id,
      state: 'pending',
      blockReason: 'human_review_required',
      affectedClaims: [],
      depthReached: 0,
      durationMs: Date.now() - startTime,
      pendingHumanReview: true,
      warnings: [`Requires human review: ${reason}`],
      trace,
    };
  }

  /**
   * Submit a human review decision
   */
  async submitHumanReview(decision: HumanReviewDecision): Promise<PropagationResult | null> {
    const reviewRequest = this.humanReviewQueue.get(decision.requestId);
    if (!reviewRequest) {
      return null;
    }

    reviewRequest.status = decision.decision === 'approve' || decision.decision === 'approve_with_modifications'
      ? 'approved'
      : 'rejected';

    this.pendingDecisions.set(decision.requestId, decision);
    this.humanReviewQueue.delete(decision.requestId);

    this.emit({ type: 'human_review_decided', decision });

    if (decision.decision === 'approve' || decision.decision === 'approve_with_modifications') {
      // Modify config if needed
      const savedConfig = { ...this.config.circuitBreaker };
      if (decision.modifications) {
        if (decision.modifications.maxDepth) {
          this.config.circuitBreaker.maxDepth = decision.modifications.maxDepth;
        }
        if (decision.modifications.maxAffectedClaims) {
          this.config.circuitBreaker.maxAffectedClaims = decision.modifications.maxAffectedClaims;
        }
      }

      // Execute
      const result = await this.executePropagation(
        reviewRequest.propagationRequest,
        Date.now()
      );

      // Restore config
      this.config.circuitBreaker = savedConfig;

      return result;
    }

    return {
      requestId: reviewRequest.propagationRequest.id,
      state: 'rejected',
      affectedClaims: [],
      depthReached: 0,
      durationMs: 0,
      pendingHumanReview: false,
      warnings: [`Rejected by reviewer: ${decision.reason}`],
    };
  }

  // ==========================================================================
  // HELPERS
  // ==========================================================================

  private calculatePropagatedConfidence(
    request: PropagationRequest,
    currentConfidence: number,
    depth: number
  ): number {
    // Confidence change decays with depth
    const changeAtSource = request.newConfidence - request.oldConfidence;
    const decayFactor = Math.pow(0.8, depth); // 80% retained per level
    const change = changeAtSource * decayFactor;

    return Math.max(0, Math.min(1, currentConfidence + change));
  }

  private estimateAffectedClaims(claimId: string, maxDepth: number = 5): number {
    const visited = new Set<string>();
    const queue = [{ id: claimId, depth: 0 }];

    while (queue.length > 0) {
      const { id, depth } = queue.shift()!;
      if (depth > maxDepth || visited.has(id)) continue;
      visited.add(id);

      const deps = this.claimDependencies.get(id) ?? [];
      for (const dep of deps) {
        queue.push({ id: dep, depth: depth + 1 });
      }
    }

    return visited.size - 1; // Exclude source
  }

  private getHighCentralityDependents(claimId: string): string[] {
    const result: string[] = [];
    const deps = this.claimDependencies.get(claimId) ?? [];

    for (const dep of deps) {
      const centrality = this.claimCentrality.get(dep) ?? 0;
      if (centrality >= this.config.circuitBreaker.centralityThresholdForReview) {
        result.push(dep);
      }
    }

    return result;
  }

  private recordPropagation(claimId: string): void {
    const now = Date.now();

    // Update global counter
    this.globalRateLimit.propagationsThisHour++;

    // Update claim counter
    let claimLimit = this.claimRateLimits.get(claimId);
    if (!claimLimit || now - claimLimit.hourStarted > 60 * 60 * 1000) {
      claimLimit = {
        claimId,
        propagationsThisHour: 0,
        lastPropagation: 0,
        hourStarted: now,
      };
    }
    claimLimit.propagationsThisHour++;
    claimLimit.lastPropagation = now;
    this.claimRateLimits.set(claimId, claimLimit);
  }

  private maybeResetHourlyCounters(now: number): void {
    if (now - this.globalRateLimit.hourStarted > 60 * 60 * 1000) {
      this.globalRateLimit = {
        propagationsThisHour: 0,
        hourStarted: now,
        quarantinesThisHour: 0,
        blocksThisHour: 0,
      };
    }
  }

  private detectOscillation(_claimId: string, _currentChange: number): boolean {
    // Simplified - would track historical changes in real implementation
    return false;
  }

  private isUnusualSource(sourceId: string): boolean {
    // Simplified - would check against known sources
    return sourceId.startsWith('unknown_');
  }

  private createBlockedResult(
    request: PropagationRequest,
    reason: BlockReason,
    startTime: number,
    trace?: PropagationTrace
  ): PropagationResult {
    this.globalRateLimit.blocksThisHour++;

    // Record blocked metric
    this.sceiMetrics.incrementCounter('propagation_blocked', {
      reason,
      update_type: request.updateType,
    });

    return {
      requestId: request.id,
      state: 'blocked',
      blockReason: reason,
      affectedClaims: [],
      depthReached: 0,
      durationMs: Date.now() - startTime,
      pendingHumanReview: reason === 'human_review_required',
      warnings: [`Blocked: ${reason}`],
      trace,
    };
  }

  // ==========================================================================
  // GRAPH MANAGEMENT (for testing/simulation)
  // ==========================================================================

  /**
   * Register a claim (for testing/simulation)
   */
  registerClaim(
    claimId: string,
    confidence: number,
    dependents: string[] = [],
    options: { domain?: string; centrality?: number; centralityScore?: number } = {}
  ): void {
    this.claimConfidences.set(claimId, confidence);
    this.claimDependencies.set(claimId, dependents);
    const centralityValue = options.centrality ?? options.centralityScore;
    if (centralityValue !== undefined) {
      this.claimCentrality.set(claimId, centralityValue);
    }
    if (options.domain !== undefined) {
      this.claimDomains.set(claimId, options.domain);
    }
  }

  /**
   * Get current claim confidence
   */
  getClaimConfidence(claimId: string): number | undefined {
    return this.claimConfidences.get(claimId);
  }

  // ==========================================================================
  // EVENTS
  // ==========================================================================

  /**
   * Subscribe to propagation events
   */
  onEvent(listener: (event: PropagationEvent) => void): () => void {
    this.eventListeners.push(listener);
    return () => {
      const index = this.eventListeners.indexOf(listener);
      if (index >= 0) {
        this.eventListeners.splice(index, 1);
      }
    };
  }

  private emit(event: PropagationEvent): void {
    for (const listener of this.eventListeners) {
      try {
        listener(event);
      } catch (e) {
        console.error('Propagation event listener error:', e);
      }
    }
  }

  // ==========================================================================
  // STATISTICS
  // ==========================================================================

  getStats(): PropagationStats {
    const humanReviewSize = this.humanReviewQueue.size;
    return {
      totalPropagations: this.stats.totalPropagations,
      successfulPropagations: this.stats.successfulPropagations,
      blockedPropagations: this.stats.blockedPropagations,
      quarantinedCount: this.quarantine.size,
      pendingHumanReviewCount: humanReviewSize,
      humanReviewQueueSize: humanReviewSize,
      globalRateLimit: { ...this.globalRateLimit },
      trackedClaims: this.claimConfidences.size,
    };
  }
}

/**
 * Statistics for the propagation system
 */
export interface PropagationStats {
  totalPropagations: number;
  successfulPropagations: number;
  blockedPropagations: number;
  quarantinedCount: number;
  pendingHumanReviewCount: number;
  humanReviewQueueSize: number;
  globalRateLimit: GlobalRateLimitState;
  trackedClaims: number;
}

// Singleton
let instance: SafeBeliefPropagator | null = null;

export function getSafeBeliefPropagator(
  calibrationEngine?: CalibrationEngine | null,
  config?: Partial<PropagationConfig>
): SafeBeliefPropagator {
  if (!instance) {
    instance = new SafeBeliefPropagator(calibrationEngine, config);
  }
  return instance;
}

export function resetSafeBeliefPropagator(): void {
  instance = null;
}
