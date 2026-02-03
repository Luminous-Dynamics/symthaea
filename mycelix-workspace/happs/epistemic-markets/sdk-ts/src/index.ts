/**
 * Mycelix Epistemic Markets SDK
 *
 * Revolutionary prediction market infrastructure featuring:
 * - Question Markets: Trade on what's worth knowing
 * - Multi-dimensional Stakes: Money, reputation, social, commitment
 * - MATL-weighted Oracles: 45% Byzantine tolerance
 * - Cross-hApp Integration: Markets about any Mycelix event
 *
 * @packageDocumentation
 */

import {
  AppClient,
  AgentPubKey,
  EntryHash,
  ActionHash,
  Record as HolochainRecord,
} from "@holochain/client";

// ============================================================================
// TYPES - Epistemic Classification
// ============================================================================

/** Empirical axis: How can this claim be verified? */
export type EmpiricalLevel =
  | "Subjective" // E0: No external verification
  | "Testimonial" // E1: Witness testimony
  | "PrivateVerify" // E2: ZK proofs, credentials
  | "Cryptographic" // E3: On-chain verification
  | "Measurable"; // E4: Scientific reproducibility

/** Normative axis: Who must agree? */
export type NormativeLevel =
  | "Personal" // N0: Individual perspective
  | "Communal" // N1: Community consensus
  | "Network" // N2: Network-wide agreement
  | "Universal"; // N3: Objective truth

/** Materiality axis: How long does this matter? */
export type MaterialityLevel =
  | "Ephemeral" // M0: Doesn't persist
  | "Temporal" // M1: Time-limited
  | "Persistent" // M2: Permanent record
  | "Foundational"; // M3: Core understanding

/** 3D epistemic position */
export interface EpistemicPosition {
  empirical: EmpiricalLevel;
  normative: NormativeLevel;
  materiality: MaterialityLevel;
}

// ============================================================================
// TYPES - Stakes
// ============================================================================

export interface MonetaryStake {
  amount: number;
  currency: string;
  escrowId?: EntryHash;
}

export interface ReputationStake {
  domains: string[];
  stakePercentage: number;
  confidenceMultiplier: number;
}

export interface SocialStake {
  visibility: "Private" | "Limited" | "Community" | "Public";
  identityLink?: string;
}

export interface Commitment {
  type: "BeliefUpdate" | "Investigation" | "Donation" | "Mentorship" | "Custom";
  details: Record<string, unknown>;
}

export interface CommitmentStake {
  ifCorrect: Commitment[];
  ifWrong: Commitment[];
}

export interface TimeStake {
  researchHours: number;
  evidenceSubmitted: EntryHash[];
}

export interface MultiDimensionalStake {
  monetary?: MonetaryStake;
  reputation?: ReputationStake;
  social?: SocialStake;
  commitment?: CommitmentStake;
  time?: TimeStake;
}

// ============================================================================
// TYPES - Markets
// ============================================================================

export interface Outcome {
  id: string;
  description: string;
  currentProbability: number;
}

export type MarketStatus =
  | "Open"
  | "Closed"
  | "Resolving"
  | "Resolved"
  | "Disputed"
  | "Cancelled";

export type MarketMechanism =
  | { type: "Binary"; yesShares: number; noShares: number }
  | { type: "LMSR"; liquidityParameter: number; subsidyPool: number }
  | { type: "CDA"; bids: Order[]; asks: Order[] }
  | { type: "Parimutuel"; pools: [string, number][] };

export interface Order {
  agent: AgentPubKey;
  outcome: string;
  price: number;
  quantity: number;
  timestamp: number;
}

export interface Market {
  id: EntryHash;
  question: string;
  description: string;
  creator: AgentPubKey;
  epistemicPosition: EpistemicPosition;
  outcomes: Outcome[];
  mechanism: MarketMechanism;
  minParticipantMatl: number;
  minOracleMatl: number;
  tags: string[];
  sourceHapp?: string;
  createdAt: number;
  closesAt: number;
  resolutionDeadline: number;
  resolvedAt?: number;
  status: MarketStatus;
  resolvedOutcome?: string;
  totalStakeValue: number;
  predictorCount: number;
}

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
  initialSubsidy?: number;
}

// ============================================================================
// TYPES - Question Markets
// ============================================================================

export interface QuestionValue {
  decisionRelevance: number;
  currentUncertainty: number;
  curiositySignals: number;
  actionability: number;
  timeSensitivity: number;
  domainImportance: number;
}

export type QuestionStatus =
  | "Trading"
  | "Spawned"
  | "Expired"
  | { Merged: { into: EntryHash } }
  | { Rejected: { reason: string } };

export interface QuestionMarket {
  id: EntryHash;
  questionText: string;
  context: string;
  proposer: AgentPubKey;
  proposerMatl: number;
  value: QuestionValue;
  currentPrice: number;
  totalShares: number;
  priceHistory: PricePoint[];
  spawnThreshold: number;
  spawnedMarket?: EntryHash;
  domains: string[];
  createdAt: number;
  expiresAt: number;
  status: QuestionStatus;
  curiosityCount: number;
}

export interface PricePoint {
  timestamp: number;
  price: number;
  volume: number;
}

export interface ProposeQuestionInput {
  questionText: string;
  context: string;
  domains: string[];
  spawnThreshold?: number;
  expiresAt?: number;
}

// ============================================================================
// TYPES - Resolution
// ============================================================================

export interface MatlScore {
  quality: number;
  consistency: number;
  reputation: number;
  composite: number;
  domainScores: [string, number][];
}

export interface EvidenceLink {
  source: string;
  url?: string;
  description: string;
  epistemicLevel: number;
}

export interface OracleVote {
  marketId: EntryHash;
  oracle: AgentPubKey;
  outcome: string;
  confidence: number;
  reasoning: string;
  evidence: EvidenceLink[];
  timestamp: number;
  matlWeight: number;
  reputationStake: number;
}

export type ResolutionState =
  | "Pending"
  | { Voting: { startedAt: number; deadline: number } }
  | { Resolved: { outcome: string; confidence: number; resolvedAt: number } }
  | {
      Disputed: {
        disputedAt: number;
        disputer: AgentPubKey;
        reason: string;
        escalationLevel: number;
      };
    }
  | { EscalatedToGovernance: { proposalId: EntryHash } }
  | { Failed: { reason: string } };

export interface ByzantineAnalysis {
  suspiciousOracles: { agent: AgentPubKey; suspicionScore: number; reasons: string[] }[];
  networkHealth: number;
  recommendation: "Proceed" | "IncreaseQuorum" | "ExtendVoting" | "Escalate" | "Halt";
  effectiveTolerance: number;
}

export interface ResolutionResult {
  success: boolean;
  outcome?: string;
  confidence: number;
  byzantineAnalysis: ByzantineAnalysis;
}

// ============================================================================
// CLIENTS
// ============================================================================

/**
 * Client for prediction markets
 */
export class MarketsClient {
  constructor(
    private client: AppClient,
    private roleName: string = "epistemic_markets",
    private zomeName: string = "epistemic_markets"
  ) {}

  async createMarket(input: CreateMarketInput): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_market",
      payload: input,
    });
  }

  async getMarket(marketHash: EntryHash): Promise<Market | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_market",
      payload: marketHash,
    });
  }

  async listMarkets(): Promise<Market[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "list_markets",
      payload: null,
    });
  }

  async listOpenMarkets(): Promise<Market[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "list_open_markets",
      payload: null,
    });
  }

  async searchByTag(tag: string): Promise<Market[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "search_markets_by_tag",
      payload: tag,
    });
  }

  async closeMarket(marketHash: EntryHash): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "close_market",
      payload: marketHash,
    });
  }
}

/**
 * Client for question markets (trade on what's worth knowing)
 */
export class QuestionMarketsClient {
  constructor(
    private client: AppClient,
    private roleName: string = "epistemic_markets",
    private zomeName: string = "question_markets"
  ) {}

  async proposeQuestion(input: ProposeQuestionInput): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "propose_question",
      payload: input,
    });
  }

  async signalCuriosity(
    questionMarket: EntryHash,
    reason?: string
  ): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "signal_curiosity",
      payload: { question_market: questionMarket, reason },
    });
  }

  async buyShares(
    questionMarket: EntryHash,
    shares: number,
    maxPrice?: number,
    reasoning?: string
  ): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "buy_question_shares",
      payload: {
        question_market: questionMarket,
        shares,
        max_price: maxPrice,
        reasoning,
      },
    });
  }

  async getQuestionMarket(hash: EntryHash): Promise<QuestionMarket | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_question_market",
      payload: hash,
    });
  }

  async listQuestionMarkets(): Promise<QuestionMarket[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "list_question_markets",
      payload: null,
    });
  }

  async getTopQuestions(limit: number = 10): Promise<QuestionMarket[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_top_questions",
      payload: limit,
    });
  }
}

/**
 * Client for oracle-based resolution
 */
export class ResolutionClient {
  constructor(
    private client: AppClient,
    private roleName: string = "epistemic_markets",
    private zomeName: string = "resolution"
  ) {}

  async startResolution(marketId: EntryHash): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "start_resolution",
      payload: marketId,
    });
  }

  async submitVote(input: {
    resolutionId: EntryHash;
    outcome: string;
    confidence: number;
    reasoning: string;
    evidence: EvidenceLink[];
    reputationStakePct?: number;
  }): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "submit_oracle_vote",
      payload: {
        resolution_id: input.resolutionId,
        outcome: input.outcome,
        confidence: input.confidence,
        reasoning: input.reasoning,
        evidence: input.evidence,
        reputation_stake_pct: input.reputationStakePct,
      },
    });
  }

  async finalizeResolution(resolutionId: EntryHash): Promise<ResolutionResult> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "finalize_resolution",
      payload: resolutionId,
    });
  }

  async disputeResolution(
    resolutionId: EntryHash,
    reason: string,
    evidence: EvidenceLink[]
  ): Promise<EntryHash> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "dispute_resolution",
      payload: { resolution_id: resolutionId, reason, evidence },
    });
  }
}

// ============================================================================
// UNIFIED CLIENT
// ============================================================================

/**
 * Unified client for all epistemic market operations
 */
export class EpistemicMarketsClient {
  public readonly markets: MarketsClient;
  public readonly questions: QuestionMarketsClient;
  public readonly resolution: ResolutionClient;

  constructor(client: AppClient, roleName: string = "epistemic_markets") {
    this.markets = new MarketsClient(client, roleName, "epistemic_markets");
    this.questions = new QuestionMarketsClient(client, roleName, "question_markets");
    this.resolution = new ResolutionClient(client, roleName, "resolution");
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Calculate composite value of a multi-dimensional stake
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
    const visibilityValues = {
      Private: 0,
      Limited: 50,
      Community: 200,
      Public: 500,
    };
    total += visibilityValues[stake.social.visibility];
    if (stake.social.identityLink) {
      total *= 2;
    }
  }

  if (stake.commitment) {
    for (const c of stake.commitment.ifWrong) {
      if (c.type === "Investigation") {
        total += ((c.details as { hours: number }).hours || 0) * 20;
      } else if (c.type === "Donation") {
        total += (c.details as { amount: number }).amount || 0;
      } else if (c.type === "Mentorship") {
        total += ((c.details as { hours: number }).hours || 0) * 30;
      } else {
        total += 50;
      }
    }
  }

  if (stake.time) {
    total += stake.time.researchHours * 15;
    total += stake.time.evidenceSubmitted.length * 25;
  }

  return total;
}

/**
 * Recommend resolution mechanism from epistemic position
 */
export function recommendResolution(
  position: EpistemicPosition
): "Automated" | "ZkVerified" | "OracleConsensus" | "CommunityVote" | "ReputationStaked" {
  switch (position.empirical) {
    case "Measurable":
    case "Cryptographic":
      return "Automated";
    case "PrivateVerify":
      return "ZkVerified";
    case "Testimonial":
      return position.normative === "Universal" || position.normative === "Network"
        ? "OracleConsensus"
        : "CommunityVote";
    case "Subjective":
      return "ReputationStaked";
  }
}

/**
 * Calculate Brier score for a set of predictions
 */
export function calculateBrierScore(
  predictions: { confidence: number; wasCorrect: boolean }[]
): number {
  if (predictions.length === 0) return 1;

  const sum = predictions.reduce((acc, p) => {
    const outcome = p.wasCorrect ? 1 : 0;
    return acc + Math.pow(p.confidence - outcome, 2);
  }, 0);

  return sum / predictions.length;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default EpistemicMarketsClient;
