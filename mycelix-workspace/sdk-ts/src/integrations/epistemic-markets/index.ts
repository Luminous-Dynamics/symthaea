// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Epistemic Markets Integration
 *
 * hApp-specific adapter for Mycelix Epistemic Markets providing:
 * - 3D Epistemic Classification (Empirical x Normative x Materiality)
 * - Multi-dimensional stakes (monetary, reputation, social, commitment, time)
 * - Reasoning trace capture for wisdom extraction
 * - Belief dependency graphs for epistemic lineage
 * - Cross-hApp market integration via Bridge
 * - MATL-weighted resolution mechanisms
 *
 * @packageDocumentation
 * @module integrations/epistemic-markets
 * @see {@link EpistemicMarketsService} - Main service class
 * @see {@link getEpistemicMarketsService} - Singleton accessor
 *
 * @example Creating a prediction market
 * ```typescript
 * import { getEpistemicMarketsService, EmpiricalLevel, NormativeLevel, MaterialityLevel } from '@mycelix/sdk/integrations/epistemic-markets';
 *
 * const service = getEpistemicMarketsService();
 *
 * // Create market with epistemic position
 * const market = await service.createMarket({
 *   question: 'Will solar capacity exceed 1TW globally by 2027?',
 *   description: 'Based on IEA projections and current growth rates',
 *   outcomes: ['Yes', 'No'],
 *   epistemicPosition: {
 *     empirical: EmpiricalLevel.Measurable,
 *     normative: NormativeLevel.Universal,
 *     materiality: MaterialityLevel.Persistent,
 *   },
 *   mechanism: { type: 'LMSR', liquidityParameter: 100 },
 *   closesAt: Date.now() + 30 * 24 * 60 * 60 * 1000,
 * });
 * ```
 */

import { LocalBridge, createReputationQuery, createBroadcastEvent } from '../../bridge/index.js';
import { FLCoordinator, AggregationMethod } from '../../fl/index.js';
import {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  compositeScore,
  createPoGQ,
  type ReputationScore,
} from '../../matl/index.js';

// ============================================================================
// EPISTEMIC CLASSIFICATION (3D Cube from Mycelix Charter)
// ============================================================================

/**
 * Empirical axis: How can this claim be verified?
 *
 * @remarks
 * Maps to the E-axis of the Mycelix Epistemic Charter:
 * - E0 Subjective: Personal experience, cannot be externally verified
 * - E1 Testimonial: Can be verified by witnesses
 * - E2 PrivateVerify: Zero-knowledge proofs, private credentials
 * - E3 Cryptographic: On-chain verification, cryptographic signatures
 * - E4 Measurable: Scientifically reproducible measurements
 */
export enum EmpiricalLevel {
  Subjective = 'Subjective',
  Testimonial = 'Testimonial',
  PrivateVerify = 'PrivateVerify',
  Cryptographic = 'Cryptographic',
  Measurable = 'Measurable',
}

/**
 * Normative axis: Who must agree for this to be "true"?
 *
 * @remarks
 * Maps to the N-axis of the Mycelix Epistemic Charter:
 * - N0 Personal: Only the individual needs to agree
 * - N1 Communal: Community consensus required
 * - N2 Network: Network-wide agreement needed
 * - N3 Universal: Objective truth, universal agreement
 */
export enum NormativeLevel {
  Personal = 'Personal',
  Communal = 'Communal',
  Network = 'Network',
  Universal = 'Universal',
}

/**
 * Materiality axis: How long does this claim matter?
 *
 * @remarks
 * Maps to the M-axis of the Mycelix Epistemic Charter:
 * - M0 Ephemeral: Fleeting, doesn't persist
 * - M1 Temporal: Time-limited relevance
 * - M2 Persistent: Long-term record
 * - M3 Foundational: Core to understanding, permanent importance
 */
export enum MaterialityLevel {
  Ephemeral = 'Ephemeral',
  Temporal = 'Temporal',
  Persistent = 'Persistent',
  Foundational = 'Foundational',
}

/**
 * 3D Epistemic Position - determines market behavior and resolution
 *
 * @example
 * ```typescript
 * // Scientific claim: measurable, universal, persistent
 * const scientificPosition: EpistemicPosition = {
 *   empirical: EmpiricalLevel.Measurable,
 *   normative: NormativeLevel.Universal,
 *   materiality: MaterialityLevel.Persistent,
 * };
 *
 * // Personal prediction: testimonial, personal, temporal
 * const personalPosition: EpistemicPosition = {
 *   empirical: EmpiricalLevel.Testimonial,
 *   normative: NormativeLevel.Personal,
 *   materiality: MaterialityLevel.Temporal,
 * };
 * ```
 */
export interface EpistemicPosition {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
}

/**
 * Get recommended resolution mechanism based on epistemic position
 */
export function getRecommendedResolution(position: EpistemicPosition): ResolutionMechanism {
  switch (position.empirical) {
    case EmpiricalLevel.Measurable:
    case EmpiricalLevel.Cryptographic:
      return ResolutionMechanism.Automated;
    case EmpiricalLevel.PrivateVerify:
      return ResolutionMechanism.ZkVerified;
    case EmpiricalLevel.Testimonial:
      return position.normative === NormativeLevel.Universal ||
        position.normative === NormativeLevel.Network
        ? ResolutionMechanism.OracleConsensus
        : ResolutionMechanism.CommunityVote;
    case EmpiricalLevel.Subjective:
    default:
      return ResolutionMechanism.ReputationStaked;
  }
}

/**
 * Get recommended market duration based on materiality
 */
export function getRecommendedDurationDays(materiality: MaterialityLevel): number {
  switch (materiality) {
    case MaterialityLevel.Ephemeral:
      return 1;
    case MaterialityLevel.Temporal:
      return 7;
    case MaterialityLevel.Persistent:
      return 30;
    case MaterialityLevel.Foundational:
      return 90;
    default:
      return 14;
  }
}

/**
 * Convert to epistemic module classification code (e.g., "E3-N2-M2")
 */
export function toClassificationCode(position: EpistemicPosition): string {
  const eLevel = ['Subjective', 'Testimonial', 'PrivateVerify', 'Cryptographic', 'Measurable'].indexOf(
    position.empirical
  );
  const nLevel = ['Personal', 'Communal', 'Network', 'Universal'].indexOf(position.normative);
  const mLevel = ['Ephemeral', 'Temporal', 'Persistent', 'Foundational'].indexOf(position.materiality);
  return `E${eLevel}-N${nLevel}-M${mLevel}`;
}

// ============================================================================
// MARKET MECHANISMS
// ============================================================================

/**
 * How market resolution is determined
 */
export enum ResolutionMechanism {
  /** Triggered by on-chain event or oracle feed */
  Automated = 'Automated',
  /** Outcome proven via zero-knowledge proof */
  ZkVerified = 'ZkVerified',
  /** MATL-weighted oracle votes */
  OracleConsensus = 'OracleConsensus',
  /** Broader participation, MATL-weighted */
  CommunityVote = 'CommunityVote',
  /** Attestors stake their MATL score */
  ReputationStaked = 'ReputationStaked',
  /** Disputed markets escalate to governance */
  GovernanceEscalation = 'GovernanceEscalation',
}

/**
 * Market pricing mechanism
 */
export type MarketMechanism =
  | { type: 'Binary'; yesShares: number; noShares: number }
  | { type: 'LMSR'; liquidityParameter: number; subsidyPool: number; outcomeQuantities: number[] }
  | { type: 'CDA'; bids: Order[]; asks: Order[] }
  | { type: 'Parimutuel'; pools: Array<{ outcomeId: string; amount: number }> };

/**
 * Order in a continuous double auction
 */
export interface Order {
  agent: string;
  outcome: string;
  price: number;
  quantity: number;
  timestamp: number;
}

/**
 * Market status
 */
export enum MarketStatus {
  Open = 'Open',
  Closed = 'Closed',
  Resolving = 'Resolving',
  Resolved = 'Resolved',
  Disputed = 'Disputed',
  Cancelled = 'Cancelled',
}

// ============================================================================
// MULTI-DIMENSIONAL STAKES
// ============================================================================

/**
 * Stake visibility level
 */
export enum Visibility {
  Private = 'Private',
  Limited = 'Limited',
  Community = 'Community',
  Public = 'Public',
}

/**
 * Commitment types for stake-based accountability
 */
export type Commitment =
  | { type: 'BeliefUpdate'; topic: string }
  | { type: 'Investigation'; hours: number; topic: string }
  | { type: 'Donation'; amount: number; recipient: string }
  | { type: 'Mentorship'; hours: number; domain: string }
  | { type: 'Custom'; description: string };

/**
 * Monetary stake component
 */
export interface MonetaryStake {
  amount: number;
  currency: string;
  escrowId?: string;
}

/**
 * Reputation stake component - MATL score at risk
 */
export interface ReputationStake {
  domains: string[];
  stakePercentage: number;
  confidenceMultiplier: number;
}

/**
 * Social visibility stake
 */
export interface SocialStake {
  visibility: Visibility;
  identityLink?: string;
  witnesses?: string[];
}

/**
 * Commitment-based stake
 */
export interface CommitmentStake {
  ifCorrect: Commitment[];
  ifWrong: Commitment[];
}

/**
 * Time investment stake
 */
export interface TimeStake {
  researchHours: number;
  evidenceSubmitted: string[];
}

/**
 * Multi-dimensional stake combining all components
 *
 * @remarks
 * Revolutionary feature allowing participants to stake beyond money:
 * - Reputation at risk (MATL score)
 * - Social visibility (public predictions)
 * - Future commitments (actions if right/wrong)
 * - Time investment (research hours, evidence)
 */
export interface MultiDimensionalStake {
  monetary?: MonetaryStake;
  reputation?: ReputationStake;
  social?: SocialStake;
  commitment?: CommitmentStake;
  time?: TimeStake;
}

/**
 * Calculate total stake value for ranking/weighting
 */
export function calculateStakeValue(stake: MultiDimensionalStake): number {
  let total = 0;

  if (stake.monetary) {
    total += stake.monetary.amount;
  }

  if (stake.reputation) {
    // Reputation is valuable: 1% of MATL ≈ 100 units
    total += stake.reputation.stakePercentage * 10000 * stake.reputation.confidenceMultiplier;
  }

  if (stake.social) {
    total += stake.social.visibility === Visibility.Public ? 500 : stake.social.visibility === Visibility.Community ? 200 : stake.social.visibility === Visibility.Limited ? 50 : 0;
    if (stake.social.identityLink) {
      total *= 2; // Identity-linked predictions worth more
    }
  }

  if (stake.commitment?.ifWrong) {
    for (const c of stake.commitment.ifWrong) {
      if (c.type === 'Investigation') total += c.hours * 20;
      else if (c.type === 'Donation') total += c.amount;
      else if (c.type === 'Mentorship') total += c.hours * 30;
      else total += 50;
    }
  }

  if (stake.time) {
    total += stake.time.researchHours * 15;
    total += stake.time.evidenceSubmitted.length * 25;
  }

  return total;
}

/**
 * Create a reputation-only stake
 */
export function createReputationOnlyStake(
  domains: string[],
  stakePercentage: number,
  confidence: number
): MultiDimensionalStake {
  return {
    reputation: {
      domains,
      stakePercentage,
      confidenceMultiplier: confidence,
    },
  };
}

/**
 * Create a monetary-only stake
 */
export function createMonetaryOnlyStake(amount: number, currency: string): MultiDimensionalStake {
  return {
    monetary: {
      amount,
      currency,
    },
  };
}

// ============================================================================
// REASONING TRACES (For Wisdom Extraction)
// ============================================================================

/**
 * Evidence type for reasoning steps
 */
export type StepSupport =
  | { type: 'Evidence'; sources: string[]; strength: number }
  | { type: 'Inference'; fromSteps: number[]; rule: string }
  | { type: 'Testimony'; source: string; credibility: number }
  | { type: 'Prior'; basis: string }
  | { type: 'Assumption'; id: string };

/**
 * A single step in the reasoning chain
 */
export interface ReasoningStep {
  stepNumber: number;
  claim: string;
  support: StepSupport;
  confidenceContribution: number;
}

/**
 * Explicit assumption with falsification criteria
 */
export interface Assumption {
  id: string;
  statement: string;
  confidence: number;
  falsificationCriteria: string;
  importance: number;
}

/**
 * Update trigger - what evidence would change my mind
 */
export interface UpdateTrigger {
  description: string;
  evidenceType: string;
  updateMagnitude: number;
  direction: 'Increase' | 'Decrease' | { depends: string };
}

/**
 * Alternative framing considered but rejected
 */
export interface AlternativeFraming {
  framing: string;
  whyRejected: string;
  confidenceIfCorrect: number;
}

/**
 * Information source with credibility assessment
 */
export interface InformationSource {
  sourceType:
    | 'AcademicPaper'
    | 'JournalisticReport'
    | 'ExpertOpinion'
    | 'DataSet'
    | 'PredictionMarket'
    | 'PersonalExperience'
    | { other: string };
  reference: string;
  credibilityAssessment: number;
  keyClaims: string[];
}

/**
 * Confidence breakdown
 */
export interface ConfidenceBreakdown {
  baseRate: number;
  evidenceAdjustment: number;
  modelUncertainty: number;
  knownUnknowns: number;
  unknownUnknownsAllowance: number;
}

/**
 * Full reasoning trace - captures the thinking behind a prediction
 *
 * @remarks
 * The foundation for wisdom extraction. Every prediction includes:
 * - Explicit reasoning steps
 * - Assumptions with falsification criteria
 * - Update triggers (cruxes)
 * - Acknowledged weaknesses
 * - Alternatives considered
 */
export interface ReasoningTrace {
  summary: string;
  steps: ReasoningStep[];
  assumptions: Assumption[];
  updateTriggers: UpdateTrigger[];
  acknowledgedWeaknesses: string[];
  alternativesConsidered: AlternativeFraming[];
  sources: InformationSource[];
  confidenceBreakdown: ConfidenceBreakdown;
}

// ============================================================================
// WISDOM SEEDS (Gifts to the Future)
// ============================================================================

/**
 * Lesson applicability conditions
 */
export interface LessonApplicability {
  domains: string[];
  timeHorizons: string[];
  conditions: string[];
  exceptions: string[];
}

/**
 * A wisdom lesson (if correct or incorrect)
 */
export interface WisdomLesson {
  lesson: string;
  confidence: number;
  applicability: LessonApplicability;
  caveats: string[];
}

/**
 * Activation condition for wisdom seeds
 */
export type ActivationCondition =
  | { type: 'AfterResolution' }
  | { type: 'AtTime'; timestamp: number }
  | { type: 'SimilarPrediction'; similarityThreshold: number }
  | { type: 'DomainActivity'; domain: string; threshold: number }
  | { type: 'OnEvent'; eventType: string };

/**
 * Wisdom seed - lessons for the future regardless of outcome
 */
export interface WisdomSeed {
  ifCorrect: WisdomLesson;
  ifIncorrect: WisdomLesson;
  metaLesson?: string;
  letterToFuture?: string;
  activationConditions: ActivationCondition[];
}

// ============================================================================
// MARKET AND PREDICTION TYPES
// ============================================================================

/**
 * Market outcome
 */
export interface Outcome {
  id: string;
  description: string;
  currentProbability: number;
}

/**
 * Full market structure
 */
export interface Market {
  id: string;
  question: string;
  description: string;
  creator: string;
  epistemicPosition: EpistemicPosition;
  outcomes: Outcome[];
  mechanism: MarketMechanism;
  resolution: ResolutionMechanism;
  minParticipantMatl: number;
  minOracleMatl: number;
  tags: string[];
  sourceHapp?: string;
  resolutionSource?: string;
  createdAt: number;
  closesAt: number;
  resolutionDeadline: number;
  resolvedAt?: number;
  status: MarketStatus;
  resolvedOutcome?: string;
  totalStakeValue: number;
  predictorCount: number;
}

/**
 * Input for creating a market
 */
export interface CreateMarketInput {
  question: string;
  description: string;
  outcomes: string[];
  epistemicPosition: EpistemicPosition;
  mechanism: MarketMechanism;
  closesAt: number;
  resolutionDeadline: number;
  minParticipantMatl?: number;
  minOracleMatl?: number;
  tags: string[];
  sourceHapp?: string;
  resolutionSource?: string;
  initialSubsidy?: number;
}

/**
 * Prediction with full reasoning trace
 */
export interface Prediction {
  id: string;
  marketId: string;
  predictor: string;
  outcome: string;
  confidence: number;
  stake: MultiDimensionalStake;
  reasoning: ReasoningTrace;
  dependsOn: BeliefDependency[];
  temporalCommitment: TemporalCommitment;
  wisdomSeed?: WisdomSeed;
  epistemicLineage: string[];
  createdAt: number;
  updatedAt: number;
  updates: PredictionUpdate[];
  resolved?: PredictionResolution;
}

/**
 * Input for submitting a prediction
 */
export interface SubmitPredictionInput {
  marketId: string;
  outcome: string;
  confidence: number;
  stake: MultiDimensionalStake;
  reasoning: ReasoningTrace;
  dependsOn?: BeliefDependency[];
  temporalCommitment?: TemporalCommitment;
  wisdomSeed?: WisdomSeed;
  epistemicLineage?: string[];
}

/**
 * Belief dependency for building belief graphs
 */
export interface BeliefDependency {
  dependsOn:
    | { type: 'Prediction'; id: string }
    | { type: 'KnowledgeClaim'; id: string }
    | { type: 'ExternalFact'; description: string }
    | { type: 'FutureEvent'; description: string; expectedBy: number };
  dependencyType:
    | 'PositiveEvidence'
    | 'NegativeEvidence'
    | 'Prerequisite'
    | 'Causal'
    | 'Correlated'
    | 'SpecializationOf';
  strength: number;
  conditional: {
    ifTrue: number;
    ifFalse: number;
    dependencyProbability: number;
  };
}

/**
 * Temporal commitment type
 */
export type TemporalCommitment =
  | { type: 'Standard' }
  | { type: 'Locked'; lockUntil: number; earlyExitPenalty: number }
  | { type: 'Vesting'; schedule: Array<{ timestamp: number; percentageUnlocked: number }> }
  | { type: 'Legacy'; successorPolicy: string; knowledgePackage?: string }
  | {
      type: 'Generational';
      generation: number;
      cohortId: string;
      expectedGenerations: number;
    };

/**
 * Prediction update record
 */
export interface PredictionUpdate {
  timestamp: number;
  oldConfidence: number;
  newConfidence: number;
  reason:
    | { type: 'NewEvidence'; description: string; source: string }
    | { type: 'DependencyResolved'; dependencyId: string; outcome: boolean }
    | { type: 'ReasoningCorrection'; error: string; correction: string }
    | { type: 'TimePassage' }
    | { type: 'ExternalEvent'; event: string }
    | { type: 'CruxResolved'; crux: string; outcome: boolean }
    | { type: 'PeerFeedback'; from: string; feedback: string };
  reasoningDelta: string;
}

/**
 * Prediction resolution result
 */
export interface PredictionResolution {
  resolvedAt: number;
  outcome: string;
  wasCorrect: boolean;
  brierContribution: number;
  reflection?: {
    correctReasoning: string[];
    incorrectReasoning: string[];
    lessonsLearned: string[];
    adviceForFuture: string;
    assumptionOutcomes: Record<string, boolean>;
  };
}

// ============================================================================
// EPISTEMIC MARKETS SERVICE
// ============================================================================

/**
 * EpistemicMarketsService - Full integration with Mycelix Epistemic Markets hApp
 *
 * @example
 * ```typescript
 * const service = new EpistemicMarketsService();
 *
 * // Create a market
 * const market = service.createMarket({
 *   question: 'Will AI models pass the Turing test by 2026?',
 *   description: 'Based on public Turing test competitions',
 *   outcomes: ['Yes', 'No'],
 *   epistemicPosition: {
 *     empirical: EmpiricalLevel.Testimonial,
 *     normative: NormativeLevel.Network,
 *     materiality: MaterialityLevel.Persistent,
 *   },
 *   mechanism: { type: 'LMSR', liquidityParameter: 100, subsidyPool: 1000, outcomeQuantities: [500, 500] },
 *   closesAt: Date.now() + 365 * 24 * 60 * 60 * 1000,
 *   resolutionDeadline: Date.now() + 400 * 24 * 60 * 60 * 1000,
 *   tags: ['AI', 'technology', 'prediction'],
 * });
 *
 * // Submit prediction with reasoning
 * const prediction = service.submitPrediction({
 *   marketId: market.id,
 *   outcome: 'Yes',
 *   confidence: 0.65,
 *   stake: createReputationOnlyStake(['technology', 'ai'], 0.05, 1.2),
 *   reasoning: {
 *     summary: 'Based on rapid progress in LLMs and multimodal systems...',
 *     steps: [...],
 *     assumptions: [...],
 *     updateTriggers: [...],
 *     acknowledgedWeaknesses: ['Limited domain expertise'],
 *     alternativesConsidered: [],
 *     sources: [],
 *     confidenceBreakdown: { baseRate: 0.3, evidenceAdjustment: 0.35, modelUncertainty: 0.1, knownUnknowns: 0.1, unknownUnknownsAllowance: 0.15 },
 *   },
 * });
 * ```
 */
export class EpistemicMarketsService {
  private markets: Map<string, Market> = new Map();
  private predictions: Map<string, Prediction> = new Map();
  private predictorReputations: Map<string, ReputationScore> = new Map();
  private bridge: LocalBridge;
  private flCoordinator: FLCoordinator;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('epistemic-markets');

    this.flCoordinator = new FLCoordinator({
      minParticipants: 5,
      aggregationMethod: AggregationMethod.TrustWeighted,
      byzantineTolerance: 0.34,
    });
  }

  // ======== MARKET OPERATIONS ========

  /**
   * Create a new prediction market
   */
  createMarket(input: CreateMarketInput): Market {
    const id = `market-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
    const now = Date.now();

    if (input.closesAt <= now) {
      throw new Error('Market close time must be in the future');
    }

    if (input.outcomes.length < 2) {
      throw new Error('Market must have at least 2 outcomes');
    }

    const initialProb = 1.0 / input.outcomes.length;
    const outcomes: Outcome[] = input.outcomes.map((desc, i) => ({
      id: `outcome_${i}`,
      description: desc,
      currentProbability: initialProb,
    }));

    const market: Market = {
      id,
      question: input.question,
      description: input.description,
      creator: 'local-agent', // Would be actual agent in real implementation
      epistemicPosition: input.epistemicPosition,
      outcomes,
      mechanism: input.mechanism,
      resolution: getRecommendedResolution(input.epistemicPosition),
      minParticipantMatl: input.minParticipantMatl ?? 0,
      minOracleMatl: input.minOracleMatl ?? 0.7,
      tags: input.tags,
      sourceHapp: input.sourceHapp,
      resolutionSource: input.resolutionSource,
      createdAt: now,
      closesAt: input.closesAt,
      resolutionDeadline: input.resolutionDeadline,
      status: MarketStatus.Open,
      totalStakeValue: 0,
      predictorCount: 0,
    };

    this.markets.set(id, market);

    // Broadcast to bridge
    const eventPayload = new TextEncoder().encode(
      JSON.stringify({
        marketId: id,
        question: input.question,
        epistemicPosition: toClassificationCode(input.epistemicPosition),
      })
    );
    this.bridge.send('governance', createBroadcastEvent('epistemic-markets', 'market_created', eventPayload));

    return market;
  }

  /**
   * Get a market by ID
   */
  getMarket(id: string): Market | undefined {
    return this.markets.get(id);
  }

  /**
   * List all open markets
   */
  listOpenMarkets(): Market[] {
    return Array.from(this.markets.values())
      .filter((m) => m.status === MarketStatus.Open)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  /**
   * Search markets by tag
   */
  searchMarketsByTag(tag: string): Market[] {
    return Array.from(this.markets.values()).filter((m) =>
      m.tags.some((t) => t.toLowerCase().includes(tag.toLowerCase()))
    );
  }

  /**
   * Close a market for new predictions
   */
  closeMarket(marketId: string): Market {
    const market = this.markets.get(marketId);
    if (!market) throw new Error('Market not found');
    if (market.status !== MarketStatus.Open) throw new Error('Market is not open');

    market.status = MarketStatus.Closed;
    return market;
  }

  // ======== PREDICTION OPERATIONS ========

  /**
   * Submit a prediction with reasoning trace
   */
  submitPrediction(input: SubmitPredictionInput): Prediction {
    const market = this.markets.get(input.marketId);
    if (!market) throw new Error('Market not found');
    if (market.status !== MarketStatus.Open) throw new Error('Market is not open');

    // Validate reasoning trace
    if (input.reasoning.summary.length < 50) {
      throw new Error('Reasoning summary must be at least 50 characters');
    }
    if (input.reasoning.steps.length === 0) {
      throw new Error('At least one reasoning step required');
    }
    if (input.reasoning.assumptions.length === 0) {
      throw new Error('At least one assumption must be made explicit');
    }

    const id = `pred-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
    const now = Date.now();

    const prediction: Prediction = {
      id,
      marketId: input.marketId,
      predictor: 'local-agent',
      outcome: input.outcome,
      confidence: input.confidence,
      stake: input.stake,
      reasoning: input.reasoning,
      dependsOn: input.dependsOn ?? [],
      temporalCommitment: input.temporalCommitment ?? { type: 'Standard' },
      wisdomSeed: input.wisdomSeed,
      epistemicLineage: input.epistemicLineage ?? [],
      createdAt: now,
      updatedAt: now,
      updates: [],
    };

    this.predictions.set(id, prediction);

    // Update market stats
    market.predictorCount += 1;
    market.totalStakeValue += calculateStakeValue(input.stake);

    // Update predictor reputation
    let rep = this.predictorReputations.get(prediction.predictor) || createReputation(prediction.predictor);
    rep = recordPositive(rep); // Participation is positive
    this.predictorReputations.set(prediction.predictor, rep);
    this.bridge.setReputation('epistemic-markets', prediction.predictor, rep);

    return prediction;
  }

  /**
   * Get prediction by ID
   */
  getPrediction(id: string): Prediction | undefined {
    return this.predictions.get(id);
  }

  /**
   * Get all predictions for a market
   */
  getPredictionsForMarket(marketId: string): Prediction[] {
    return Array.from(this.predictions.values())
      .filter((p) => p.marketId === marketId)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  /**
   * Update a prediction's confidence
   */
  updatePrediction(
    predictionId: string,
    newConfidence: number,
    reason: PredictionUpdate['reason'],
    reasoningDelta: string
  ): Prediction {
    const prediction = this.predictions.get(predictionId);
    if (!prediction) throw new Error('Prediction not found');

    const update: PredictionUpdate = {
      timestamp: Date.now(),
      oldConfidence: prediction.confidence,
      newConfidence,
      reason,
      reasoningDelta,
    };

    prediction.updates.push(update);
    prediction.confidence = newConfidence;
    prediction.updatedAt = Date.now();

    return prediction;
  }

  // ======== REPUTATION OPERATIONS ========

  /**
   * Get predictor reputation
   */
  getPredictorReputation(predictorId: string): ReputationScore {
    return this.predictorReputations.get(predictorId) || createReputation(predictorId);
  }

  /**
   * Check if predictor is trustworthy
   */
  isPredictorTrustworthy(predictorId: string, threshold = 0.7): boolean {
    const rep = this.predictorReputations.get(predictorId);
    if (!rep) return false;
    return isTrustworthy(rep, threshold);
  }

  /**
   * Calculate Brier score for a resolved prediction
   */
  calculateBrierScore(predictionId: string): number {
    const prediction = this.predictions.get(predictionId);
    if (!prediction || !prediction.resolved) throw new Error('Prediction not resolved');

    const market = this.markets.get(prediction.marketId);
    if (!market) throw new Error('Market not found');

    // Brier score: mean squared error
    // Verify outcome exists in market
    const _outcomeIndex = market.outcomes.findIndex((o) => o.id === prediction.outcome);
    if (_outcomeIndex === -1) throw new Error('Outcome not found in market');
    const wasCorrect = prediction.resolved.wasCorrect;

    // If prediction was for the correct outcome
    const probabilityForCorrect = wasCorrect ? prediction.confidence : 1 - prediction.confidence;
    return Math.pow(1 - probabilityForCorrect, 2);
  }

  // ======== WISDOM OPERATIONS ========

  /**
   * Get wisdom seeds for a domain
   */
  getWisdomSeedsForDomain(domain: string): Array<{
    predictionId: string;
    predictor: string;
    marketQuestion: string;
    wisdomSeed: WisdomSeed;
    resolved: boolean;
    wasCorrect?: boolean;
  }> {
    return Array.from(this.predictions.values())
      .filter((p) => p.wisdomSeed && p.wisdomSeed.ifCorrect.applicability.domains.includes(domain))
      .map((p) => ({
        predictionId: p.id,
        predictor: p.predictor,
        marketQuestion: this.markets.get(p.marketId)?.question || '',
        wisdomSeed: p.wisdomSeed!,
        resolved: !!p.resolved,
        wasCorrect: p.resolved?.wasCorrect,
      }));
  }

  /**
   * Get belief dependents (predictions that depend on this one)
   */
  getBeliefDependents(predictionId: string): Prediction[] {
    return Array.from(this.predictions.values()).filter((p) =>
      p.dependsOn.some((d) => d.dependsOn.type === 'Prediction' && d.dependsOn.id === predictionId)
    );
  }

  // ======== CROSS-HAPP OPERATIONS ========

  /**
   * Query predictor reputation from other hApps
   */
  queryExternalReputation(predictorId: string): void {
    const query = createReputationQuery('epistemic-markets', predictorId);
    this.bridge.send('governance', query);
    this.bridge.send('knowledge', query);
    this.bridge.send('praxis', query);
  }

  /**
   * Create market from governance proposal
   */
  createMarketFromGovernance(proposalId: string, question: string, closesAt: number): Market {
    return this.createMarket({
      question,
      description: `Prediction market for governance proposal ${proposalId}`,
      outcomes: ['Proposal Passes', 'Proposal Fails'],
      epistemicPosition: {
        empirical: EmpiricalLevel.Cryptographic,
        normative: NormativeLevel.Network,
        materiality: MaterialityLevel.Persistent,
      },
      mechanism: { type: 'Binary', yesShares: 1000, noShares: 1000 },
      closesAt,
      resolutionDeadline: closesAt + 7 * 24 * 60 * 60 * 1000,
      tags: ['governance', 'proposal'],
      sourceHapp: 'governance',
      resolutionSource: `governance:proposal:${proposalId}`,
    });
  }

  /**
   * Get FL coordinator for collaborative model training
   */
  getFLCoordinator(): FLCoordinator {
    return this.flCoordinator;
  }

  /**
   * Get bridge for cross-hApp communication
   */
  getBridge(): LocalBridge {
    return this.bridge;
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export {
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  compositeScore,
  createPoGQ,
  AggregationMethod,
};

// Default service instance
let defaultService: EpistemicMarketsService | null = null;

/**
 * Get the default epistemic markets service instance
 */
export function getEpistemicMarketsService(): EpistemicMarketsService {
  if (!defaultService) {
    defaultService = new EpistemicMarketsService();
  }
  return defaultService;
}

// ============================================================================
// Holochain Conductor Bridge Client
// ============================================================================

/** Holochain conductor bridge client for Epistemic Markets */
export class EpistemicMarketsBridgeClient {
  constructor(
    private client: {
      callZome(input: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: any;
      }): Promise<any>;
    }
  ) {}

  // ---- Claims ----

  async createClaim(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload,
    });
  }

  async getClaim(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload,
    });
  }

  async updateClaim(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'update_claim',
      payload,
    });
  }

  async listClaims(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'list_claims',
      payload,
    });
  }

  async submitEvidence(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'submit_evidence',
      payload,
    });
  }

  async resolveClaim(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'claims',
      fn_name: 'resolve_claim',
      payload,
    });
  }

  // ---- Markets Integration ----

  async createMarket(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'create_market',
      payload,
    });
  }

  async getMarket(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'get_market',
      payload,
    });
  }

  async listOpenMarkets(payload?: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'list_open_markets',
      payload: payload ?? null,
    });
  }

  async submitPrediction(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'submit_prediction',
      payload,
    });
  }

  async updatePrediction(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'update_prediction',
      payload,
    });
  }

  async resolveMarket(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'resolve_market',
      payload,
    });
  }

  async closeMarket(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'close_market',
      payload,
    });
  }

  async getPredictionsForMarket(payload: any): Promise<any> {
    return this.client.callZome({
      role_name: 'knowledge',
      zome_name: 'markets_integration',
      fn_name: 'get_predictions_for_market',
      payload,
    });
  }
}
