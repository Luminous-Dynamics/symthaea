/**
 * @mycelix/sdk Epistemic Validator Agent System
 *
 * Transforms Symthaea civic agents into epistemic validators that:
 * - Classify every response claim with E/N/M dimensions
 * - Validate responses against the Distributed Knowledge Graph before delivery
 * - Store agent responses as epistemic claims for verification
 * - Enable agents to become trusted sources in the knowledge ecosystem
 *
 * Philosophy: Every agent utterance is a claim. Claims must be accountable.
 * Agents that consistently provide accurate information build epistemic reputation.
 *
 * @packageDocumentation
 * @module innovations/epistemic-agents
 */

import {
  type CalibrationEngine,
  getCalibrationEngine,
} from '../../calibration/engine.js';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  type EpistemicPosition,
  toClassificationCode,
} from '../../integrations/epistemic-markets/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  type ReputationScore,
} from '../../matl/index.js';
import {
  AgentRunner,
  type KnowledgeRetriever,
  DEFAULT_AGENT_CONFIG,
} from '../../symthaea/agent-runner.js';

import type {
  AgentConfig,
  AgentResponse,
  ConversationContext,
  KnowledgeSource,
  CivicAgentDomain,
} from '../../symthaea/types.js';

// ============================================================================
// TYPES
// ============================================================================

/**
 * An epistemic claim extracted from agent response
 */
export interface EpistemicClaim {
  /** Unique claim ID */
  id: string;
  /** The claim text */
  text: string;
  /** Epistemic classification */
  position: EpistemicPosition;
  /** Classification code (e.g., "E3-N2-M2") */
  classificationCode: string;
  /** Confidence in the claim */
  confidence: number;
  /** Calibration-adjusted confidence */
  calibratedConfidence: number;
  /** Sources supporting this claim */
  sources: KnowledgeSource[];
  /** Agent that made the claim */
  agentId: string;
  /** Conversation context */
  conversationId: string;
  /** When the claim was made */
  timestamp: number;
  /** Domain of the claim */
  domain: CivicAgentDomain;
  /** Whether this claim was validated against DKG */
  dkgValidated: boolean;
  /** DKG validation result */
  dkgValidation?: DKGValidationResult;
  /** Whether this claim can be used as a verification source */
  isVerificationSource: boolean;
}

/**
 * Result of validating a claim against the Distributed Knowledge Graph
 */
export interface DKGValidationResult {
  /** Whether the claim is consistent with DKG */
  consistent: boolean;
  /** Confidence in the validation */
  validationConfidence: number;
  /** Supporting claims found in DKG */
  supportingClaims: string[];
  /** Contradicting claims found in DKG */
  contradictingClaims: string[];
  /** Recommendation based on validation */
  recommendation: 'approve' | 'flag' | 'escalate' | 'reject';
  /** Reasoning for the recommendation */
  reasoning: string;
}

/**
 * Extended agent response with epistemic metadata
 */
export interface EpistemicAgentResponse extends AgentResponse {
  /** Claims extracted from this response */
  claims: EpistemicClaim[];
  /** Overall epistemic quality score */
  epistemicQuality: number;
  /** Whether response was modified based on DKG validation */
  dkgModified: boolean;
  /** Epistemic metadata */
  epistemicMetadata: EpistemicResponseMetadata;
}

/**
 * Epistemic-specific response metadata
 */
export interface EpistemicResponseMetadata {
  /** Total claims extracted */
  claimCount: number;
  /** Claims validated against DKG */
  validatedCount: number;
  /** Claims flagged for review */
  flaggedCount: number;
  /** Agent's current epistemic reputation */
  agentReputation: number;
  /** Calibration status for this domain */
  calibrationStatus: 'sufficient' | 'insufficient' | 'degrading';
  /** Time spent on DKG validation */
  validationLatencyMs: number;
}

/**
 * Configuration for epistemic validation
 */
export interface EpistemicConfig {
  /** Enable DKG validation before response */
  enableDKGValidation: boolean;
  /** Minimum confidence to store claim in DKG */
  minConfidenceForStorage: number;
  /** Enable calibration integration */
  enableCalibration: boolean;
  /** Auto-adjust confidence based on calibration */
  autoAdjustConfidence: boolean;
  /** Flag claims that contradict DKG */
  flagContradictions: boolean;
  /** Escalate high-stakes claims for review */
  escalateHighStakes: boolean;
  /** Materiality threshold for high-stakes */
  highStakesThreshold: MaterialityLevel;
  /** Enable agent as verification source */
  enableVerificationSource: boolean;
  /** Minimum reputation to be verification source */
  minVerificationReputation: number;
}

/**
 * Agent epistemic statistics
 */
export interface AgentEpistemicStats {
  /** Total claims made */
  totalClaims: number;
  /** Claims validated as correct */
  validatedCorrect: number;
  /** Claims validated as incorrect */
  validatedIncorrect: number;
  /** Claims still pending validation */
  pendingValidation: number;
  /** Current epistemic reputation */
  reputation: ReputationScore;
  /** Calibration quality (Brier score) */
  brierScore: number | null;
  /** Bias direction (over/under confident) */
  inflationBias: number | null;
  /** Claims by epistemic classification */
  claimsByClassification: Map<string, number>;
  /** Claims by domain */
  claimsByDomain: Map<CivicAgentDomain, number>;
  /** Trend in accuracy */
  accuracyTrend: 'improving' | 'stable' | 'degrading';
}

/**
 * DKG integration interface
 */
export interface DKGClient {
  /** Query DKG for supporting/contradicting claims */
  queryClaims(
    claimText: string,
    domain: string
  ): Promise<{
    supporting: Array<{ id: string; confidence: number; text: string }>;
    contradicting: Array<{ id: string; confidence: number; text: string }>;
  }>;
  /** Store a new claim in DKG */
  storeClaim(claim: EpistemicClaim): Promise<string>;
  /** Update claim validation status */
  updateClaimStatus(
    claimId: string,
    status: 'validated' | 'invalidated' | 'disputed'
  ): Promise<void>;
  /** Get claim by ID */
  getClaim(claimId: string): Promise<EpistemicClaim | null>;
}

// ============================================================================
// DEFAULT CONFIGURATION
// ============================================================================

const DEFAULT_EPISTEMIC_CONFIG: EpistemicConfig = {
  enableDKGValidation: true,
  minConfidenceForStorage: 0.6,
  enableCalibration: true,
  autoAdjustConfidence: true,
  flagContradictions: true,
  escalateHighStakes: true,
  highStakesThreshold: MaterialityLevel.Persistent,
  enableVerificationSource: true,
  minVerificationReputation: 0.7,
};

// ============================================================================
// EPISTEMIC AGENT RUNNER
// ============================================================================

/**
 * Epistemic Validator Agent Runner
 *
 * Extends the base AgentRunner with epistemic validation capabilities:
 * - Every response is decomposed into individual claims
 * - Claims are classified with E/N/M dimensions
 * - Claims are validated against the DKG
 * - Agent builds epistemic reputation over time
 * - Calibration tracks agent accuracy
 *
 * @example
 * ```typescript
 * const config: AgentConfig = {
 *   id: 'benefits-agent-001',
 *   name: 'Benefits Navigator',
 *   domain: 'benefits',
 *   // ... other config
 * };
 *
 * const agent = new EpistemicAgentRunner(config);
 * agent.setDKGClient(dkgClient);
 *
 * const response = await agent.processMessageWithValidation(
 *   conversationId,
 *   'Am I eligible for SNAP benefits?',
 *   'web'
 * );
 *
 * console.log(`Claims made: ${response.claims.length}`);
 * console.log(`Epistemic quality: ${response.epistemicQuality}`);
 * ```
 */
class EpistemicAgentRunner extends AgentRunner {
  private epistemicConfig: EpistemicConfig;
  private calibrationEngine: CalibrationEngine;
  private dkgClient: DKGClient | null = null;
  private agentReputation: ReputationScore;
  private claimHistory: Map<string, EpistemicClaim> = new Map();
  private epistemicStats: AgentEpistemicStats;

  constructor(
    config: AgentConfig,
    epistemicConfig: Partial<EpistemicConfig> = {}
  ) {
    super(config);
    this.epistemicConfig = { ...DEFAULT_EPISTEMIC_CONFIG, ...epistemicConfig };
    this.calibrationEngine = getCalibrationEngine();
    this.agentReputation = createReputation(config.id);
    this.epistemicStats = this.initEpistemicStats();
  }

  /**
   * Set the DKG client for claim validation and storage
   */
  setDKGClient(client: DKGClient): void {
    this.dkgClient = client;
  }

  /**
   * Process a message with full epistemic validation
   *
   * This extends the base processMessage with:
   * 1. Claim extraction from response
   * 2. E/N/M classification
   * 3. DKG validation
   * 4. Calibration adjustment
   * 5. Claim storage
   */
  async processMessageWithValidation(
    conversationId: string,
    message: string,
    channel: ConversationContext['channel'],
    citizenDid?: string
  ): Promise<EpistemicAgentResponse> {
    const validationStartTime = Date.now();

    // Get base response from parent
    const baseResponse = await this.processMessage(
      conversationId,
      message,
      channel,
      citizenDid
    );

    // Extract claims from response
    const rawClaims = await this.extractClaims(
      baseResponse,
      conversationId
    );

    // Classify each claim
    const classifiedClaims = rawClaims.map((claim) =>
      this.classifyClaim(claim)
    );

    // Apply calibration if enabled
    const calibratedClaims = this.epistemicConfig.enableCalibration
      ? await this.applyCalibratedConfidence(classifiedClaims)
      : classifiedClaims;

    // Validate against DKG if enabled
    const validatedClaims = this.epistemicConfig.enableDKGValidation && this.dkgClient
      ? await this.validateAgainstDKG(calibratedClaims)
      : calibratedClaims;

    // Check for contradictions and escalation
    const { claims: processedClaims, modified } = this.processValidationResults(
      validatedClaims,
      baseResponse
    );

    // Calculate epistemic quality
    const epistemicQuality = this.calculateEpistemicQuality(processedClaims);

    // Store claims in DKG if they meet threshold
    await this.storeClaims(processedClaims);

    // Update statistics
    this.updateEpistemicStats(processedClaims);

    const validationLatencyMs = Date.now() - validationStartTime;

    // Build epistemic response
    const epistemicMetadata: EpistemicResponseMetadata = {
      claimCount: processedClaims.length,
      validatedCount: processedClaims.filter((c) => c.dkgValidated).length,
      flaggedCount: processedClaims.filter(
        (c) => c.dkgValidation?.recommendation === 'flag'
      ).length,
      agentReputation: reputationValue(this.agentReputation),
      calibrationStatus: this.getCalibrationStatus(),
      validationLatencyMs,
    };

    return {
      ...baseResponse,
      claims: processedClaims,
      epistemicQuality,
      dkgModified: modified,
      epistemicMetadata,
    };
  }

  /**
   * Record the outcome of a claim for calibration
   */
  async recordClaimOutcome(
    claimId: string,
    outcome: 'correct' | 'incorrect' | 'ambiguous',
    evidence: Array<{ type: string; value: string }> = []
  ): Promise<void> {
    const claim = this.claimHistory.get(claimId);
    if (!claim) {
      throw new Error(`Claim ${claimId} not found`);
    }

    // Record in calibration engine
    await this.calibrationEngine.recordResolution(
      {
        id: claimId,
        confidence: claim.confidence,
        domain: claim.domain,
        sources: claim.sources.map((s) => s.id),
      },
      outcome,
      evidence.map((e) => ({
        type: e.type,
        reference: e.value,
        description: `Evidence of type ${e.type}`,
      }))
    );

    // Update agent reputation
    if (outcome === 'correct') {
      this.agentReputation = recordPositive(this.agentReputation);
      this.epistemicStats.validatedCorrect++;
    } else if (outcome === 'incorrect') {
      this.agentReputation = recordNegative(this.agentReputation);
      this.epistemicStats.validatedIncorrect++;
    }
    this.epistemicStats.pendingValidation--;

    // Update DKG if available
    if (this.dkgClient) {
      await this.dkgClient.updateClaimStatus(
        claimId,
        outcome === 'correct' ? 'validated' : 'invalidated'
      );
    }
  }

  /**
   * Check if this agent can serve as a verification source
   */
  isVerificationSource(): boolean {
    if (!this.epistemicConfig.enableVerificationSource) {
      return false;
    }
    return reputationValue(this.agentReputation) >= this.epistemicConfig.minVerificationReputation;
  }

  /**
   * Get agent's epistemic statistics
   */
  getEpistemicStats(): AgentEpistemicStats {
    return { ...this.epistemicStats };
  }

  /**
   * Get agent's current reputation
   */
  getReputation(): ReputationScore & { positive: number; negative: number } {
    return {
      ...this.agentReputation,
      // Aliases for test compatibility
      positive: this.agentReputation.positiveCount - 1, // Remove prior
      negative: this.agentReputation.negativeCount - 1, // Remove prior
    };
  }

  // ==========================================================================
  // PRIVATE METHODS
  // ==========================================================================

  /**
   * Extract individual claims from agent response
   */
  private async extractClaims(
    response: AgentResponse,
    conversationId: string
  ): Promise<EpistemicClaim[]> {
    const claims: EpistemicClaim[] = [];
    const sentences = this.splitIntoSentences(response.text);

    for (const sentence of sentences) {
      // Skip non-claim sentences
      if (this.isNonClaim(sentence)) {
        continue;
      }

      const claim: EpistemicClaim = {
        id: this.generateClaimId(),
        text: sentence,
        position: {
          empirical: EmpiricalLevel.Testimonial,
          normative: NormativeLevel.Communal,
          materiality: MaterialityLevel.Temporal,
        },
        classificationCode: '',
        confidence: response.confidence,
        calibratedConfidence: response.confidence,
        sources: response.sources,
        agentId: response.metadata.domain,
        conversationId,
        timestamp: Date.now(),
        domain: response.metadata.domain,
        dkgValidated: false,
        isVerificationSource: this.isVerificationSource(),
      };

      claims.push(claim);
    }

    return claims;
  }

  /**
   * Classify a claim with E/N/M dimensions
   */
  private classifyClaim(claim: EpistemicClaim): EpistemicClaim {
    const position = this.inferEpistemicPosition(claim);
    return {
      ...claim,
      position,
      classificationCode: toClassificationCode(position),
    };
  }

  /**
   * Infer epistemic position from claim content and context
   */
  private inferEpistemicPosition(claim: EpistemicClaim): EpistemicPosition {
    const text = claim.text.toLowerCase();
    const sources = claim.sources;

    // Infer empirical level
    let empirical: EmpiricalLevel = EmpiricalLevel.Testimonial;
    if (sources.some((s) => s.type === 'regulation' || s.type === 'policy')) {
      empirical = EmpiricalLevel.Cryptographic; // Official sources
    } else if (sources.some((s) => s.type === 'dkg')) {
      empirical = EmpiricalLevel.PrivateVerify; // DKG-verified
    } else if (this.containsQuantifiableClaim(text)) {
      empirical = EmpiricalLevel.Measurable;
    } else if (text.includes('i feel') || text.includes('in my opinion')) {
      empirical = EmpiricalLevel.Subjective;
    }

    // Infer normative level
    let normative: NormativeLevel = NormativeLevel.Communal;
    if (
      text.includes('federal') ||
      text.includes('law requires') ||
      text.includes('everyone')
    ) {
      normative = NormativeLevel.Universal;
    } else if (text.includes('in this area') || text.includes('local')) {
      normative = NormativeLevel.Network;
    } else if (text.includes('your') || text.includes('you may')) {
      normative = NormativeLevel.Personal;
    }

    // Infer materiality level based on domain and content
    let materiality: MaterialityLevel = MaterialityLevel.Temporal;
    if (
      claim.domain === 'voting' ||
      claim.domain === 'justice' ||
      text.includes('permanent') ||
      text.includes('lifetime')
    ) {
      materiality = MaterialityLevel.Foundational;
    } else if (
      claim.domain === 'benefits' ||
      claim.domain === 'permits' ||
      text.includes('valid for') ||
      text.includes('expires')
    ) {
      materiality = MaterialityLevel.Persistent;
    } else if (
      text.includes('today') ||
      text.includes('currently') ||
      text.includes('right now')
    ) {
      materiality = MaterialityLevel.Ephemeral;
    }

    return { empirical, normative, materiality };
  }

  /**
   * Apply calibration-adjusted confidence
   */
  private async applyCalibratedConfidence(
    claims: EpistemicClaim[]
  ): Promise<EpistemicClaim[]> {
    const adjustedClaims: EpistemicClaim[] = [];

    for (const claim of claims) {
      try {
        const adjusted = await this.calibrationEngine.adjustConfidence(
          claim.confidence,
          claim.domain
        );

        adjustedClaims.push({
          ...claim,
          calibratedConfidence: this.epistemicConfig.autoAdjustConfidence
            ? adjusted.adjustedConfidence
            : claim.confidence,
        });
      } catch {
        // Calibration data insufficient - use original confidence
        adjustedClaims.push(claim);
      }
    }

    return adjustedClaims;
  }

  /**
   * Validate claims against the Distributed Knowledge Graph
   */
  private async validateAgainstDKG(
    claims: EpistemicClaim[]
  ): Promise<EpistemicClaim[]> {
    if (!this.dkgClient) {
      return claims;
    }

    const validatedClaims: EpistemicClaim[] = [];

    for (const claim of claims) {
      const dkgResult = await this.dkgClient.queryClaims(
        claim.text,
        claim.domain
      );

      const validation = this.buildValidationResult(dkgResult, claim);

      validatedClaims.push({
        ...claim,
        dkgValidated: true,
        dkgValidation: validation,
      });
    }

    return validatedClaims;
  }

  /**
   * Build DKG validation result from query response
   */
  private buildValidationResult(
    dkgResult: {
      supporting: Array<{ id: string; confidence: number; text: string }>;
      contradicting: Array<{ id: string; confidence: number; text: string }>;
    },
    claim: EpistemicClaim
  ): DKGValidationResult {
    const supportingWeight = dkgResult.supporting.reduce(
      (sum, c) => sum + c.confidence,
      0
    );
    const contradictingWeight = dkgResult.contradicting.reduce(
      (sum, c) => sum + c.confidence,
      0
    );

    const totalWeight = supportingWeight + contradictingWeight;
    const consistent = totalWeight === 0 || supportingWeight > contradictingWeight;
    const validationConfidence =
      totalWeight > 0
        ? Math.abs(supportingWeight - contradictingWeight) / totalWeight
        : 0;

    // Determine recommendation
    let recommendation: DKGValidationResult['recommendation'] = 'approve';
    let reasoning = 'No contradicting evidence found.';

    if (contradictingWeight > supportingWeight) {
      if (contradictingWeight > 2) {
        recommendation = 'reject';
        reasoning = `Strong contradicting evidence: ${dkgResult.contradicting.length} claims disagree.`;
      } else if (claim.position.materiality === MaterialityLevel.Foundational) {
        recommendation = 'escalate';
        reasoning = 'High-stakes claim with contradicting evidence requires review.';
      } else {
        recommendation = 'flag';
        reasoning = `Potential contradiction with ${dkgResult.contradicting.length} DKG claims.`;
      }
    } else if (dkgResult.supporting.length > 0) {
      recommendation = 'approve';
      reasoning = `Supported by ${dkgResult.supporting.length} DKG claims.`;
    }

    return {
      consistent,
      validationConfidence,
      supportingClaims: dkgResult.supporting.map((c) => c.id),
      contradictingClaims: dkgResult.contradicting.map((c) => c.id),
      recommendation,
      reasoning,
    };
  }

  /**
   * Process validation results and potentially modify response
   */
  private processValidationResults(
    claims: EpistemicClaim[],
    _originalResponse: AgentResponse
  ): { claims: EpistemicClaim[]; modified: boolean } {
    let modified = false;

    const processedClaims = claims.map((claim) => {
      if (!claim.dkgValidation) {
        return claim;
      }

      // Handle contradictions
      if (
        this.epistemicConfig.flagContradictions &&
        !claim.dkgValidation.consistent
      ) {
        modified = true;
        // Reduce confidence for contradicted claims
        return {
          ...claim,
          calibratedConfidence: claim.calibratedConfidence * 0.5,
        };
      }

      // Handle high-stakes escalation
      if (
        this.epistemicConfig.escalateHighStakes &&
        this.isMaterialityAboveThreshold(claim.position.materiality) &&
        claim.dkgValidation.recommendation === 'escalate'
      ) {
        modified = true;
      }

      return claim;
    });

    return { claims: processedClaims, modified };
  }

  /**
   * Store claims in DKG that meet confidence threshold
   */
  private async storeClaims(claims: EpistemicClaim[]): Promise<void> {
    if (!this.dkgClient) {
      // Store locally for calibration tracking
      for (const claim of claims) {
        this.claimHistory.set(claim.id, claim);
      }
      return;
    }

    for (const claim of claims) {
      // Store locally
      this.claimHistory.set(claim.id, claim);

      // Store in DKG if meets threshold
      if (claim.calibratedConfidence >= this.epistemicConfig.minConfidenceForStorage) {
        try {
          await this.dkgClient.storeClaim(claim);
        } catch (error) {
          console.error(`Failed to store claim ${claim.id} in DKG:`, error);
        }
      }
    }
  }

  /**
   * Calculate overall epistemic quality of claims
   */
  private calculateEpistemicQuality(claims: EpistemicClaim[]): number {
    if (claims.length === 0) {
      return 1.0;
    }

    let totalQuality = 0;

    for (const claim of claims) {
      let quality = claim.calibratedConfidence;

      // Boost for DKG validation
      if (claim.dkgValidation?.consistent) {
        quality *= 1.2;
      }

      // Boost for high empirical level
      if (
        claim.position.empirical === EmpiricalLevel.Measurable ||
        claim.position.empirical === EmpiricalLevel.Cryptographic
      ) {
        quality *= 1.1;
      }

      // Penalty for contradictions
      if (claim.dkgValidation && !claim.dkgValidation.consistent) {
        quality *= 0.7;
      }

      totalQuality += Math.min(1.0, quality);
    }

    return totalQuality / claims.length;
  }

  /**
   * Update epistemic statistics
   */
  private updateEpistemicStats(claims: EpistemicClaim[]): void {
    this.epistemicStats.totalClaims += claims.length;
    this.epistemicStats.pendingValidation += claims.length;
    this.epistemicStats.reputation = this.agentReputation;

    for (const claim of claims) {
      // Update by classification
      const count =
        this.epistemicStats.claimsByClassification.get(claim.classificationCode) || 0;
      this.epistemicStats.claimsByClassification.set(
        claim.classificationCode,
        count + 1
      );

      // Update by domain
      const domainCount =
        this.epistemicStats.claimsByDomain.get(claim.domain) || 0;
      this.epistemicStats.claimsByDomain.set(claim.domain, domainCount + 1);
    }
  }

  /**
   * Get calibration status for this agent
   */
  private getCalibrationStatus(): EpistemicResponseMetadata['calibrationStatus'] {
    const stats = this.calibrationEngine.getStats();
    if (stats.totalResolutions < 30) {
      return 'insufficient';
    }
    // Would check trend from calibration report
    return 'sufficient';
  }

  /**
   * Initialize epistemic statistics
   */
  private initEpistemicStats(): AgentEpistemicStats {
    return {
      totalClaims: 0,
      validatedCorrect: 0,
      validatedIncorrect: 0,
      pendingValidation: 0,
      reputation: this.agentReputation,
      brierScore: null,
      inflationBias: null,
      claimsByClassification: new Map(),
      claimsByDomain: new Map(),
      accuracyTrend: 'stable',
    };
  }

  /**
   * Split response text into sentences
   */
  private splitIntoSentences(text: string): string[] {
    return text
      .split(/[.!?]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 10);
  }

  /**
   * Check if a sentence is not a claim (greeting, question, etc.)
   */
  private isNonClaim(sentence: string): boolean {
    const lower = sentence.toLowerCase();
    const nonClaimPatterns = [
      /^(hello|hi|hey|good morning|good afternoon)/,
      /^(how can i|what would you|could you)/,
      /\?$/,
      /^(thank|thanks|you're welcome)/,
      /^(i'm here to|let me|i can|i'd be happy)/,
    ];
    return nonClaimPatterns.some((p) => p.test(lower));
  }

  /**
   * Check if text contains quantifiable claims
   */
  private containsQuantifiableClaim(text: string): boolean {
    const quantPatterns = [
      /\$[\d,]+/,
      /\d+%/,
      /\d+ (days?|weeks?|months?|years?)/,
      /\d+ (people|persons|households)/,
    ];
    return quantPatterns.some((p) => p.test(text));
  }

  /**
   * Check if materiality is above threshold
   */
  private isMaterialityAboveThreshold(materiality: MaterialityLevel): boolean {
    const levels = [
      MaterialityLevel.Ephemeral,
      MaterialityLevel.Temporal,
      MaterialityLevel.Persistent,
      MaterialityLevel.Foundational,
    ];
    const threshold = levels.indexOf(this.epistemicConfig.highStakesThreshold);
    const current = levels.indexOf(materiality);
    return current >= threshold;
  }

  /**
   * Generate unique claim ID
   */
  private generateClaimId(): string {
    return `claim-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }
}

// ============================================================================
// EPISTEMIC KNOWLEDGE RETRIEVER
// ============================================================================

/**
 * Knowledge retriever that integrates with the epistemic system
 *
 * Extends standard knowledge retrieval with:
 * - DKG integration for claim-backed knowledge
 * - Confidence weighting based on claim calibration
 * - Source reputation tracking
 */
class EpistemicKnowledgeRetriever implements KnowledgeRetriever {
  private dkgClient: DKGClient | null = null;
  private sourceReputation: Map<string, ReputationScore> = new Map();

  setDKGClient(client: DKGClient): void {
    this.dkgClient = client;
  }

  async retrieve(
    query: string,
    options: {
      domain: CivicAgentDomain;
      maxResults: number;
      minRelevance: number;
    }
  ): Promise<KnowledgeSource[]> {
    const sources: KnowledgeSource[] = [];

    // If DKG is available, query it for relevant claims
    if (this.dkgClient) {
      const dkgResults = await this.dkgClient.queryClaims(query, options.domain);

      for (const claim of dkgResults.supporting.slice(0, options.maxResults)) {
        sources.push({
          type: 'dkg',
          id: claim.id,
          title: claim.text.substring(0, 50),
          relevance: claim.confidence,
        });
      }
    }

    return sources.filter((s) => s.relevance >= options.minRelevance);
  }

  /**
   * Update source reputation based on claim outcome
   */
  recordSourceOutcome(sourceId: string, correct: boolean): void {
    let rep = this.sourceReputation.get(sourceId);
    if (!rep) {
      rep = createReputation(sourceId);
    }
    rep = correct ? recordPositive(rep) : recordNegative(rep);
    this.sourceReputation.set(sourceId, rep);
  }

  /**
   * Get source reputation
   */
  getSourceReputation(sourceId: string): number {
    const rep = this.sourceReputation.get(sourceId);
    return rep ? reputationValue(rep) : 0.5;
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create an epistemic agent runner with default configuration
 */
function createEpistemicAgent(
  config: AgentConfig,
  epistemicConfig?: Partial<EpistemicConfig>
): EpistemicAgentRunner {
  return new EpistemicAgentRunner(config, epistemicConfig);
}

/**
 * Create an epistemic agent from a domain preset
 */
function createEpistemicAgentFromDomain(
  domain: CivicAgentDomain,
  overrides?: Partial<AgentConfig>
): EpistemicAgentRunner {
  const baseConfig: AgentConfig = {
    id: `epistemic-${domain}-${Date.now()}`,
    name: `Epistemic ${domain.charAt(0).toUpperCase() + domain.slice(1)} Agent`,
    domain,
    ...DEFAULT_AGENT_CONFIG,
    ...overrides,
  };

  return new EpistemicAgentRunner(baseConfig);
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  EpistemicAgentRunner,
  EpistemicKnowledgeRetriever,
  createEpistemicAgent,
  createEpistemicAgentFromDomain,
  DEFAULT_EPISTEMIC_CONFIG,
};
