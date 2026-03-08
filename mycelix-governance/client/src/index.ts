/**
 * Mycelix Governance SDK
 *
 * Decentralized governance infrastructure featuring:
 * - Φ-Weighted Voting: Consciousness-gated decision making
 * - Collective Mirror: Swarm wisdom sensing (Mirror, not Oracle)
 * - Quadratic Voting: Voice credits for nuanced preference expression
 * - ZK-Verified Voting: Privacy-preserving eligibility proofs
 * - Delegation with Decay: Trust-based vote delegation
 *
 * @packageDocumentation
 */

import {
  AppClient,
  ActionHash,
  Record as HolochainRecord,
} from "@holochain/client";

// ============================================================================
// TYPES - Proposal Tiers
// ============================================================================

/** Proposal tier determines quorum and approval thresholds */
export type ProposalTier = "Basic" | "Major" | "Constitutional";

/** Get quorum requirement for a tier */
export function getQuorumRequirement(tier: ProposalTier): number {
  switch (tier) {
    case "Basic":
      return 0.15; // 15%
    case "Major":
      return 0.25; // 25%
    case "Constitutional":
      return 0.4; // 40%
  }
}

/** Get approval threshold for a tier */
export function getApprovalThreshold(tier: ProposalTier): number {
  switch (tier) {
    case "Basic":
      return 0.5; // Simple majority
    case "Major":
      return 0.6; // 60% supermajority
    case "Constitutional":
      return 0.67; // 2/3 supermajority
  }
}

// ============================================================================
// TYPES - Voting
// ============================================================================

/** Vote choice */
export type VoteChoice = "For" | "Against" | "Abstain";

/** Basic vote */
export interface Vote {
  id: string;
  proposalId: string;
  voter: string;
  choice: VoteChoice;
  weight: number;
  votedAt: number;
  reason?: string;
}

/** Φ weight components for consciousness-gated voting */
export interface PhiWeight {
  /** Φ score from consciousness integration (0-1) */
  phiScore: number;
  /** K-Vector trust score (0-1) */
  kTrust: number;
  /** Stake-based weight (0-1) */
  stakeWeight: number;
  /** Historical participation score (0-1) */
  participationScore: number;
  /** Domain-specific reputation (0-1) */
  domainReputation: number;
}

/** Φ-weighted vote */
export interface PhiWeightedVote {
  id: string;
  proposalId: string;
  voter: string;
  choice: VoteChoice;
  phiWeight: PhiWeight;
  effectiveWeight: number;
  votedAt: number;
  reason?: string;
}

/** Input for casting a Φ-weighted vote */
export interface CastPhiVoteInput {
  proposalId: string;
  voterDid: string;
  tier: ProposalTier;
  choice: VoteChoice;
  reason?: string;
}

// ============================================================================
// TYPES - Delegation
// ============================================================================

/** Delegation decay model */
export type DelegationDecay =
  | { None: null }
  | { Linear: { decayRateDays: number } }
  | { Exponential: { halfLifeDays: number } }
  | { Step: { stepIntervalDays: number; dropPerStep: number } };

/** Vote delegation */
export interface Delegation {
  id: string;
  delegator: string;
  delegate: string;
  domain?: string;
  decay: DelegationDecay;
  createdAt: number;
  expiresAt?: number;
  active: boolean;
}

/** Input for creating a delegation */
export interface CreateDelegationInput {
  delegateDid: string;
  domain?: string;
  decay: DelegationDecay;
  expiresAt?: number;
}

// ============================================================================
// TYPES - Tally Results
// ============================================================================

/** Tally segment by Φ tier */
export interface TallySegment {
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  voterCount: number;
}

/** Breakdown by Φ tier */
export interface PhiTierBreakdown {
  highPhiVotes: TallySegment; // Φ >= 0.6
  mediumPhiVotes: TallySegment; // 0.4 <= Φ < 0.6
  lowPhiVotes: TallySegment; // Φ < 0.4
}

/** Φ-weighted vote tally */
export interface PhiWeightedTally {
  proposalId: string;
  tier: ProposalTier;
  phiVotesFor: number;
  phiVotesAgainst: number;
  phiAbstentions: number;
  rawVotesFor: number;
  rawVotesAgainst: number;
  rawAbstentions: number;
  averagePhi: number;
  totalPhiWeight: number;
  eligibleVoters: number;
  quorumRequirement: number;
  quorumReached: boolean;
  approvalThreshold: number;
  approved: boolean;
  talliedAt: number;
  finalTally: boolean;
  phiTierBreakdown: PhiTierBreakdown;
}

/** Input for tallying Φ-weighted votes */
export interface TallyPhiVotesInput {
  proposalId: string;
  tier: ProposalTier;
  eligibleVoters?: number;
  /** Whether to automatically generate a collective mirror reflection (default: true) */
  generateReflection?: boolean;
}

// ============================================================================
// TYPES - Collective Mirror (Swarm Wisdom Sensing)
// ============================================================================

/**
 * The Collective Mirror reflects the group's state without claiming wisdom.
 * Philosophy: Mirror, not Oracle.
 *
 * It shows:
 * - Topology: How is agreement structured?
 * - Shadow: What values are absent?
 * - Signal Integrity: Is this consensus or echo chamber?
 * - Trajectory: Where is the group heading?
 */

/** Echo chamber risk level */
export type EchoChamberRiskLevel = "Low" | "Moderate" | "High" | "Critical";

/** Topology pattern */
export type TopologyPattern =
  | "Mesh" // Distributed, healthy
  | "HubAndSpoke" // Centralized around few voices
  | "Polarized" // Two opposing camps
  | "Monopole" // Single dominant voice
  | "Unknown"; // Insufficient data

/** Trend direction */
export type TrendDirection = "Rising" | "Stable" | "Falling" | "Unknown";

/**
 * Collective mirror reflection of proposal voting
 *
 * This reflects what IS, not what SHOULD BE.
 * It helps the group see itself clearly.
 */
export interface ProposalReflection {
  id: string;
  proposalId: string;
  timestamp: number;
  voterCount: number;

  // === TOPOLOGY ===
  /** Agreement structure pattern */
  topologyPattern: TopologyPattern;
  /** Agreement centralization (0=distributed, 1=single voice) */
  centralization: number;
  /** Number of distinct clusters */
  clusterCount: number;

  // === SHADOW ===
  /** Harmonies absent from the conversation */
  absentHarmonies: string[];
  /** Percentage of harmonies represented (0-1) */
  harmonyCoverage: number;

  // === SIGNAL INTEGRITY ===
  /** Average epistemic level of voters (0-1) */
  averageEpistemicLevel: number;
  /** Echo chamber risk assessment */
  echoChamberRisk: EchoChamberRiskLevel;
  /** Whether high agreement is backed by verification */
  agreementVerified: boolean;

  // === TRAJECTORY ===
  /** Direction of agreement trend */
  agreementTrend: TrendDirection;
  /** Direction of centralization trend */
  centralizationTrend: TrendDirection;
  /** Warning: group converging too rapidly */
  rapidConvergenceWarning: boolean;
  /** Warning: group fragmenting */
  fragmentationWarning: boolean;

  // === VOTE SUMMARY ===
  votesFor: number;
  votesAgainst: number;
  abstentions: number;
  approvalRatio: number;
  polarization: number;

  // === PROMPTS & INTERVENTIONS ===
  /** Suggested interventions */
  suggestedInterventions: string[];
  /** Reflection prompts for the group */
  reflectionPrompts: string[];

  /** Whether this reflection suggests review before finalizing */
  needsReview: boolean;
  /** Human-readable summary */
  summary: string;
}

/** Input for reflecting on a proposal */
export interface ReflectOnProposalInput {
  proposalId: string;
}

// ============================================================================
// TYPES - Quadratic Voting
// ============================================================================

/** Voice credit allocation */
export interface VoiceCredits {
  id: string;
  voterDid: string;
  totalCredits: number;
  creditsSpent: number;
  creditsRemaining: number;
  epoch: number;
  updatedAt: number;
}

/** Quadratic vote */
export interface QuadraticVote {
  id: string;
  proposalId: string;
  voter: string;
  creditsSpent: number;
  voteStrength: number; // sqrt(creditsSpent)
  choice: VoteChoice;
  votedAt: number;
}

/** Input for casting a quadratic vote */
export interface CastQuadraticVoteInput {
  proposalId: string;
  voterDid: string;
  creditsToSpend: number;
  choice: VoteChoice;
}

/** Quadratic vote tally */
export interface QuadraticTally {
  proposalId: string;
  quadraticFor: number;
  quadraticAgainst: number;
  quadraticAbstain: number;
  rawCreditsFor: number;
  rawCreditsAgainst: number;
  rawCreditsAbstain: number;
  uniqueVoters: number;
  totalCreditsSpent: number;
  approved: boolean;
  talliedAt: number;
}

// ============================================================================
// TYPES - ZK-Verified Voting
// ============================================================================

/** Eligibility proof types */
export type ProofType =
  | "MembershipProof"
  | "AgeProof"
  | "ReputationThreshold"
  | "StakeThreshold"
  | "DomainExpertise"
  | "CompositeProof";

/** ZK eligibility proof */
export interface EligibilityProof {
  id: string;
  voterDid: string;
  proofType: ProofType;
  proofData: string; // Serialized proof
  circuitId: string;
  publicInputsHash: string;
  verified: boolean;
  verifiedAt?: number;
  expiresAt: number;
}

/** Verified vote */
export interface VerifiedVote {
  id: string;
  proposalId: string;
  voter: string;
  eligibilityProofHash: ActionHash;
  voterCommitment: string; // ZK commitment to voter identity
  choice: VoteChoice;
  votedAt: number;
}

// ============================================================================
// CLIENTS
// ============================================================================

/**
 * Client for Φ-weighted voting operations
 */
export class PhiVotingClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "voting"
  ) {}

  /**
   * Cast a Φ-weighted vote
   *
   * The vote weight is calculated from:
   * - Φ score (consciousness integration)
   * - K-Vector trust
   * - Stake weight
   * - Participation history
   * - Domain reputation
   */
  async castPhiVote(input: CastPhiVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "cast_phi_vote",
      payload: {
        proposal_id: input.proposalId,
        voter_did: input.voterDid,
        tier: input.tier,
        choice: input.choice,
        reason: input.reason,
      },
    });
  }

  /**
   * Get all Φ-weighted votes for a proposal
   */
  async getProposalVotes(proposalId: string): Promise<PhiWeightedVote[]> {
    const records = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_phi_votes",
      payload: proposalId,
    });
    return records;
  }

  /**
   * Tally Φ-weighted votes and optionally generate reflection
   */
  async tallyVotes(input: TallyPhiVotesInput): Promise<PhiWeightedTally> {
    const record = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "tally_phi_votes",
      payload: {
        proposal_id: input.proposalId,
        tier: input.tier,
        eligible_voters: input.eligibleVoters,
        generate_reflection: input.generateReflection ?? true,
      },
    });
    return record;
  }

  /**
   * Get the latest tally for a proposal
   */
  async getLatestTally(proposalId: string): Promise<PhiWeightedTally | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_latest_phi_tally",
      payload: proposalId,
    });
  }
}

/**
 * Client for Collective Mirror (swarm wisdom sensing)
 *
 * Philosophy: The Mirror reflects, it doesn't judge.
 * These functions show the group how it looks, without
 * claiming to know what's wise.
 */
export class CollectiveMirrorClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "voting"
  ) {}

  /**
   * Generate a collective mirror reflection for a proposal
   *
   * This analyzes current voting patterns and returns a reflection
   * showing the group's topology, shadow, signal integrity, and trajectory.
   */
  async reflectOnProposal(proposalId: string): Promise<ProposalReflection> {
    const record = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "reflect_on_proposal",
      payload: { proposal_id: proposalId },
    });
    return record;
  }

  /**
   * Get all reflections for a proposal (sorted by time)
   */
  async getProposalReflections(proposalId: string): Promise<ProposalReflection[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_proposal_reflections",
      payload: proposalId,
    });
  }

  /**
   * Get the latest reflection for a proposal
   */
  async getLatestReflection(proposalId: string): Promise<ProposalReflection | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_latest_reflection",
      payload: proposalId,
    });
  }

  /**
   * Check if a proposal needs human review based on collective sensing
   *
   * Returns true if concerning patterns are detected:
   * - Echo chamber risk is High or Critical
   * - Rapid convergence warning
   * - Fragmentation warning
   * - Low harmony coverage (<30%)
   * - High centralization (>80%)
   */
  async proposalNeedsReview(proposalId: string): Promise<boolean> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "proposal_needs_review",
      payload: proposalId,
    });
  }
}

/**
 * Client for delegation operations
 */
export class DelegationClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "voting"
  ) {}

  /**
   * Create a vote delegation
   */
  async createDelegation(input: CreateDelegationInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_delegation",
      payload: {
        delegate_did: input.delegateDid,
        domain: input.domain,
        decay: input.decay,
        expires_at: input.expiresAt,
      },
    });
  }

  /**
   * Revoke a delegation
   */
  async revokeDelegation(delegationHash: ActionHash): Promise<void> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "revoke_delegation",
      payload: delegationHash,
    });
  }

  /**
   * Get all delegations by a delegator
   */
  async getMyDelegations(): Promise<Delegation[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_my_delegations",
      payload: null,
    });
  }

  /**
   * Get all delegations to a delegate
   */
  async getDelegationsToMe(): Promise<Delegation[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_delegations_to_me",
      payload: null,
    });
  }
}

/**
 * Client for quadratic voting operations
 */
export class QuadraticVotingClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "voting"
  ) {}

  /**
   * Cast a quadratic vote
   *
   * Vote strength = sqrt(credits spent)
   * This allows expressing intensity of preference
   */
  async castQuadraticVote(input: CastQuadraticVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "cast_quadratic_vote",
      payload: {
        proposal_id: input.proposalId,
        voter_did: input.voterDid,
        credits_to_spend: input.creditsToSpend,
        choice: input.choice,
      },
    });
  }

  /**
   * Get voice credit balance
   */
  async getVoiceCredits(voterDid: string): Promise<VoiceCredits> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_voice_credits",
      payload: voterDid,
    });
  }

  /**
   * Tally quadratic votes
   */
  async tallyQuadraticVotes(proposalId: string): Promise<QuadraticTally> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "tally_quadratic_votes",
      payload: { proposal_id: proposalId },
    });
  }
}

// ============================================================================
// UNIFIED CLIENT
// ============================================================================

/**
 * Unified client for all governance operations
 *
 * @example
 * ```typescript
 * import { AppWebsocket } from '@holochain/client';
 * import { GovernanceClient } from '@mycelix/governance-sdk';
 *
 * const client = await AppWebsocket.connect('ws://localhost:8888');
 * const governance = new GovernanceClient(client);
 *
 * // Cast a Φ-weighted vote
 * await governance.phi.castPhiVote({
 *   proposalId: "proposal-123",
 *   voterDid: "did:mycelix:alice",
 *   tier: "Major",
 *   choice: "For",
 *   reason: "Aligns with community values"
 * });
 *
 * // Get collective mirror reflection
 * const reflection = await governance.mirror.reflectOnProposal("proposal-123");
 * console.log("Topology:", reflection.topologyPattern);
 * console.log("Echo chamber risk:", reflection.echoChamberRisk);
 *
 * // Check if proposal needs human review
 * if (await governance.mirror.proposalNeedsReview("proposal-123")) {
 *   console.log("Concerning patterns detected - consider extending discussion");
 * }
 *
 * // Tally votes (automatically generates reflection)
 * const tally = await governance.phi.tallyVotes({
 *   proposalId: "proposal-123",
 *   tier: "Major",
 *   eligibleVoters: 100
 * });
 * ```
 */
export class GovernanceClient {
  public readonly phi: PhiVotingClient;
  public readonly mirror: CollectiveMirrorClient;
  public readonly delegation: DelegationClient;
  public readonly quadratic: QuadraticVotingClient;

  constructor(client: AppClient, roleName: string = "governance") {
    this.phi = new PhiVotingClient(client, roleName, "voting");
    this.mirror = new CollectiveMirrorClient(client, roleName, "voting");
    this.delegation = new DelegationClient(client, roleName, "voting");
    this.quadratic = new QuadraticVotingClient(client, roleName, "voting");
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Calculate effective vote weight from Φ components
 *
 * Formula: effectiveWeight = phiScore * (
 *   0.3 * kTrust +
 *   0.2 * stakeWeight +
 *   0.3 * participationScore +
 *   0.2 * domainReputation
 * )
 */
export function calculateEffectiveWeight(phiWeight: PhiWeight): number {
  const componentWeight =
    0.3 * phiWeight.kTrust +
    0.2 * phiWeight.stakeWeight +
    0.3 * phiWeight.participationScore +
    0.2 * phiWeight.domainReputation;

  return phiWeight.phiScore * componentWeight;
}

/**
 * Calculate quadratic vote strength from credits
 */
export function calculateQuadraticStrength(credits: number): number {
  return Math.sqrt(credits);
}

/**
 * Format reflection for human display
 */
export function formatReflectionSummary(reflection: ProposalReflection): string {
  const lines: string[] = [];

  lines.push(`Reflection for ${reflection.proposalId}`);
  lines.push(`Generated: ${new Date(reflection.timestamp / 1000).toISOString()}`);
  lines.push(`Voters: ${reflection.voterCount}`);
  lines.push("");
  lines.push("=== TOPOLOGY ===");
  lines.push(`Pattern: ${reflection.topologyPattern}`);
  lines.push(`Centralization: ${(reflection.centralization * 100).toFixed(1)}%`);
  lines.push(`Clusters: ${reflection.clusterCount}`);
  lines.push("");
  lines.push("=== SHADOW ===");
  lines.push(`Harmony Coverage: ${(reflection.harmonyCoverage * 100).toFixed(1)}%`);
  if (reflection.absentHarmonies.length > 0) {
    lines.push(`Absent: ${reflection.absentHarmonies.join(", ")}`);
  }
  lines.push("");
  lines.push("=== SIGNAL INTEGRITY ===");
  lines.push(`Echo Chamber Risk: ${reflection.echoChamberRisk}`);
  lines.push(`Agreement Verified: ${reflection.agreementVerified ? "Yes" : "No"}`);
  lines.push(`Average Epistemic Level: ${(reflection.averageEpistemicLevel * 100).toFixed(1)}%`);
  lines.push("");
  lines.push("=== TRAJECTORY ===");
  lines.push(`Agreement Trend: ${reflection.agreementTrend}`);
  lines.push(`Centralization Trend: ${reflection.centralizationTrend}`);
  if (reflection.rapidConvergenceWarning) {
    lines.push("⚠️ RAPID CONVERGENCE WARNING");
  }
  if (reflection.fragmentationWarning) {
    lines.push("⚠️ FRAGMENTATION WARNING");
  }
  lines.push("");
  lines.push("=== VOTE SUMMARY ===");
  lines.push(`For: ${reflection.votesFor}`);
  lines.push(`Against: ${reflection.votesAgainst}`);
  lines.push(`Abstain: ${reflection.abstentions}`);
  lines.push(`Approval: ${(reflection.approvalRatio * 100).toFixed(1)}%`);
  lines.push(`Polarization: ${(reflection.polarization * 100).toFixed(1)}%`);

  if (reflection.needsReview) {
    lines.push("");
    lines.push("🔍 THIS PROPOSAL NEEDS REVIEW BEFORE FINALIZING");
  }

  if (reflection.reflectionPrompts.length > 0) {
    lines.push("");
    lines.push("=== REFLECTION PROMPTS ===");
    reflection.reflectionPrompts.forEach((prompt, i) => {
      lines.push(`${i + 1}. ${prompt}`);
    });
  }

  if (reflection.suggestedInterventions.length > 0) {
    lines.push("");
    lines.push("=== SUGGESTED INTERVENTIONS ===");
    reflection.suggestedInterventions.forEach((intervention, i) => {
      lines.push(`${i + 1}. ${intervention}`);
    });
  }

  return lines.join("\n");
}

/**
 * Get concern severity from reflection (0-10)
 */
export function getReflectionConcernSeverity(reflection: ProposalReflection): number {
  let severity = 0;

  switch (reflection.echoChamberRisk) {
    case "Critical":
      severity += 4;
      break;
    case "High":
      severity += 3;
      break;
    case "Moderate":
      severity += 1;
      break;
  }

  if (reflection.rapidConvergenceWarning) severity += 2;
  if (reflection.fragmentationWarning) severity += 2;
  if (reflection.harmonyCoverage < 0.3) severity += 2;
  if (reflection.centralization > 0.8) severity += 1;

  return Math.min(severity, 10);
}

/**
 * Determine if vote result should be considered legitimate
 * based on collective sensing metrics
 */
export function isLegitimateResult(
  tally: PhiWeightedTally,
  reflection: ProposalReflection
): { legitimate: boolean; concerns: string[] } {
  const concerns: string[] = [];

  // Check quorum
  if (!tally.quorumReached) {
    concerns.push("Quorum not reached");
  }

  // Check echo chamber
  if (
    reflection.echoChamberRisk === "High" ||
    reflection.echoChamberRisk === "Critical"
  ) {
    concerns.push(`Echo chamber risk: ${reflection.echoChamberRisk}`);
  }

  // Check rapid convergence
  if (reflection.rapidConvergenceWarning) {
    concerns.push("Rapid convergence detected - possible groupthink");
  }

  // Check harmony coverage
  if (reflection.harmonyCoverage < 0.3) {
    concerns.push(
      `Low value diversity: only ${(reflection.harmonyCoverage * 100).toFixed(0)}% of harmonies represented`
    );
  }

  // Check centralization
  if (reflection.centralization > 0.8) {
    concerns.push(
      `High centralization: ${(reflection.centralization * 100).toFixed(0)}% of agreement from few voices`
    );
  }

  // Check Φ distribution
  if (tally.phiTierBreakdown.highPhiVotes.voterCount === 0) {
    concerns.push("No high-Φ voters participated");
  }

  return {
    legitimate: concerns.length === 0,
    concerns,
  };
}

/**
 * The Eight Harmonies of value alignment
 */
export const SEVEN_HARMONIES = [
  "ResonantCoherence",
  "PanSentientFlourishing",
  "IntegralWisdom",
  "InfinitePlay",
  "UniversalInterconnectedness",
  "SacredReciprocity",
  "EvolutionaryProgression",
] as const;

export type Harmony = (typeof SEVEN_HARMONIES)[number];

/**
 * Get description of a harmony
 */
export function getHarmonyDescription(harmony: Harmony): string {
  const descriptions: Record<Harmony, string> = {
    ResonantCoherence: "Internal consistency and alignment of beliefs and actions",
    PanSentientFlourishing: "Care for the wellbeing of all conscious beings",
    IntegralWisdom: "Integration of multiple ways of knowing",
    InfinitePlay: "Openness, creativity, and evolutionary exploration",
    UniversalInterconnectedness: "Recognition of deep connections between all things",
    SacredReciprocity: "Balanced giving and receiving in relationships",
    EvolutionaryProgression: "Commitment to growth and positive development",
  };
  return descriptions[harmony];
}

// ============================================================================
// TYPES - Discussion System
// ============================================================================

/** Stance on a proposal */
export type Stance = "Support" | "Oppose" | "Neutral" | "Amend";

/** Discussion contribution with harmony tagging */
export interface DiscussionContribution {
  id: string;
  proposalId: string;
  contributor: string;
  content: string;
  harmonyTags: string[];
  stance?: Stance;
  parentId?: string;
  createdAt: number;
  edited: boolean;
}

/** Input for adding a discussion contribution */
export interface AddContributionInput {
  proposalId: string;
  contributorDid: string;
  content: string;
  harmonyTags: string[];
  stance?: Stance;
  parentId?: string;
}

/** Harmony presence in discussion */
export interface HarmonyPresence {
  harmony: string;
  presence: number;
  exampleContributionId?: string;
}

/** Discussion reflection - collective sensing of deliberation phase */
export interface DiscussionReflection {
  id: string;
  proposalId: string;
  timestamp: number;

  // Participation metrics
  contributorCount: number;
  contributionCount: number;
  avgContributionsPerParticipant: number;
  maxThreadDepth: number;

  // Harmony analysis
  harmonyCoverage: HarmonyPresence[];
  harmonyDiversity: number;
  absentHarmonies: string[];

  // Stance distribution
  supportCount: number;
  opposeCount: number;
  neutralCount: number;
  amendCount: number;
  preliminarySentiment: number;

  // Discussion health
  voiceConcentration: number;
  crossCampEngagement: number;
  substantivenessScore: number;

  // Readiness signals
  discussionSaturated: boolean;
  unaddressedConcerns: string[];
  readyForVote: boolean;
  readinessReasoning: string;

  summary: string;
}

/** Discussion readiness check result */
export interface DiscussionReadiness {
  isReady: boolean;
  reasons: string[];
  reflection?: DiscussionReflection;
}

// ============================================================================
// TYPES - Council System (Holonic Governance)
// ============================================================================

/** Council type determines powers and processes */
export type CouncilType =
  | { Root: null }
  | { Domain: { domain: string } }
  | { Regional: { region: string } }
  | { WorkingGroup: { focus: string; expires?: number } }
  | { Advisory: null }
  | { Emergency: { expires: number } };

/** Council status */
export type CouncilStatus = "Active" | "Dormant" | "Dissolved" | "Suspended";

/** Council member role */
export type MemberRole =
  | { Member: null }
  | { Facilitator: null }
  | { Steward: null }
  | { Observer: null }
  | { Delegate: { fromCouncil: string } };

/** Membership status */
export type MembershipStatus = "Active" | "OnLeave" | "Suspended" | "Removed";

/** A governance council */
export interface Council {
  id: string;
  name: string;
  purpose: string;
  councilType: CouncilType;
  parentCouncilId?: string;
  phiThreshold: number;
  quorum: number;
  supermajority: number;
  canSpawnChildren: boolean;
  maxDelegationDepth: number;
  status: CouncilStatus;
  createdAt: number;
  lastActivity: number;
}

/** Council membership */
export interface CouncilMembership {
  id: string;
  councilId: string;
  memberDid: string;
  role: MemberRole;
  phiScore: number;
  votingWeight: number;
  canDelegate: boolean;
  status: MembershipStatus;
  joinedAt: number;
  lastParticipation: number;
}

/** Input for creating a council */
export interface CreateCouncilInput {
  name: string;
  purpose: string;
  councilType: CouncilType;
  parentCouncilId?: string;
  phiThreshold: number;
  quorum: number;
  supermajority: number;
  canSpawnChildren: boolean;
  maxDelegationDepth: number;
}

/** Input for joining a council */
export interface JoinCouncilInput {
  councilId: string;
  memberDid: string;
  role: MemberRole;
  phiScore: number;
}

/** Phi distribution statistics */
export interface PhiDistribution {
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
}

/** Aggregate health of child councils */
export interface AggregateChildHealth {
  healthyCount: number;
  strugglingCount: number;
  dormantCount: number;
  averageHealth: number;
  childrenNeedingAttention: string[];
}

/** Risk category for council health */
export type RiskCategory =
  | "Participation"
  | "Consciousness"
  | "DecisionQuality"
  | "HarmonyGap"
  | "ChildHealth"
  | "ParentDisconnect"
  | "PowerConcentration"
  | "Stagnation";

/** Risk factor for council health */
export interface RiskFactor {
  category: RiskCategory;
  severity: number;
  description: string;
}

/** Vitality trend for councils */
export type VitalityTrend = "Thriving" | "Stable" | "Declining" | "Critical";

/** Holonic reflection of a council's collective state */
export interface HolonicReflection {
  id: string;
  councilId: string;
  timestamp: number;
  holonDepth: number;

  // Member sensing
  memberCount: number;
  phiQualifiedCount: number;
  averagePhi: number;
  phiDistribution: PhiDistribution;
  participationRate: number;

  // Decision health
  decisionsLast30Days: number;
  averageConsensus: number;
  contentionRatio: number;
  implementationRate: number;

  // Holonic health
  childCount: number;
  childHealth?: AggregateChildHealth;
  parentCoherence?: number;
  collaborationScore: number;

  // Harmony analysis
  harmonyPresence: HarmonyPresence[];
  harmonyCoverage: number;
  absentHarmonies: string[];

  // Vitality signals
  vitalityTrend: VitalityTrend;
  riskFactors: RiskFactor[];
  opportunities: string[];
  healthScore: number;

  summary: string;
}

/** Holonic tree node for hierarchical visualization */
export interface HolonicTreeNode {
  councilId: string;
  councilName: string;
  depth: number;
  healthScore: number;
  memberCount: number;
  children: HolonicTreeNode[];
  reflectionSummary?: string;
}

// ============================================================================
// DISCUSSION CLIENT
// ============================================================================

/**
 * Client for discussion operations with collective sensing
 */
export class DiscussionClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "proposals"
  ) {}

  /**
   * Add a contribution to a proposal discussion
   */
  async addContribution(input: AddContributionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "add_contribution",
      payload: {
        proposal_id: input.proposalId,
        contributor_did: input.contributorDid,
        content: input.content,
        harmony_tags: input.harmonyTags,
        stance: input.stance,
        parent_id: input.parentId,
      },
    });
  }

  /**
   * Get all contributions for a proposal discussion
   */
  async getDiscussion(proposalId: string): Promise<DiscussionContribution[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_discussion",
      payload: proposalId,
    });
  }

  /**
   * Get replies to a contribution
   */
  async getReplies(contributionId: string): Promise<DiscussionContribution[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_replies",
      payload: contributionId,
    });
  }

  /**
   * Generate a collective sensing reflection of the discussion phase
   */
  async reflectOnDiscussion(proposalId: string): Promise<DiscussionReflection> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "reflect_on_discussion",
      payload: proposalId,
    });
  }

  /**
   * Get all discussion reflections for a proposal
   */
  async getDiscussionReflections(proposalId: string): Promise<DiscussionReflection[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_discussion_reflections",
      payload: proposalId,
    });
  }

  /**
   * Check if discussion is mature enough for voting
   */
  async isDiscussionReady(proposalId: string): Promise<DiscussionReadiness> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "is_discussion_ready",
      payload: proposalId,
    });
  }
}

// ============================================================================
// COUNCILS CLIENT (HOLONIC GOVERNANCE)
// ============================================================================

/**
 * Client for council operations with holonic mirror
 */
export class CouncilsClient {
  constructor(
    private client: AppClient,
    private roleName: string = "governance",
    private zomeName: string = "councils"
  ) {}

  /**
   * Create a new council
   */
  async createCouncil(input: CreateCouncilInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "create_council",
      payload: {
        name: input.name,
        purpose: input.purpose,
        council_type: input.councilType,
        parent_council_id: input.parentCouncilId,
        phi_threshold: input.phiThreshold,
        quorum: input.quorum,
        supermajority: input.supermajority,
        can_spawn_children: input.canSpawnChildren,
        max_delegation_depth: input.maxDelegationDepth,
      },
    });
  }

  /**
   * Get a council by ID
   */
  async getCouncil(councilId: string): Promise<Council | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_council_by_id",
      payload: councilId,
    });
  }

  /**
   * Get all councils
   */
  async getAllCouncils(): Promise<Council[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_all_councils",
      payload: null,
    });
  }

  /**
   * Get child councils
   */
  async getChildCouncils(councilId: string): Promise<Council[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_child_councils",
      payload: councilId,
    });
  }

  /**
   * Join a council
   */
  async joinCouncil(input: JoinCouncilInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "join_council",
      payload: {
        council_id: input.councilId,
        member_did: input.memberDid,
        role: input.role,
        phi_score: input.phiScore,
      },
    });
  }

  /**
   * Get council members
   */
  async getCouncilMembers(councilId: string): Promise<CouncilMembership[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_council_members",
      payload: councilId,
    });
  }

  /**
   * Generate a holonic reflection for a council
   */
  async reflectOnCouncil(councilId: string): Promise<HolonicReflection> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "reflect_on_council",
      payload: councilId,
    });
  }

  /**
   * Get council reflections
   */
  async getCouncilReflections(councilId: string): Promise<HolonicReflection[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_council_reflections",
      payload: councilId,
    });
  }

  /**
   * Get the latest reflection for a council
   */
  async getLatestCouncilReflection(councilId: string): Promise<HolonicReflection | null> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_latest_council_reflection",
      payload: councilId,
    });
  }

  /**
   * Get holonic health tree (recursive reflection of entire hierarchy)
   */
  async getHolonicTree(rootCouncilId: string): Promise<HolonicTreeNode> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: "get_holonic_tree",
      payload: rootCouncilId,
    });
  }
}

// ============================================================================
// EXTENDED GOVERNANCE CLIENT
// ============================================================================

/**
 * Extended unified client for all governance operations
 *
 * @example
 * ```typescript
 * import { AppWebsocket } from '@holochain/client';
 * import { GovernanceClient } from '@mycelix/governance-sdk';
 *
 * const client = await AppWebsocket.connect('ws://localhost:8888');
 * const governance = new GovernanceClient(client);
 *
 * // Create a council
 * await governance.councils.createCouncil({
 *   name: "Treasury Council",
 *   purpose: "Manage community treasury",
 *   councilType: { Domain: { domain: "treasury" } },
 *   phiThreshold: 0.4,
 *   quorum: 0.5,
 *   supermajority: 0.67,
 *   canSpawnChildren: true,
 *   maxDelegationDepth: 3
 * });
 *
 * // Add discussion contribution with harmony tags
 * await governance.discussion.addContribution({
 *   proposalId: "MIP-001",
 *   contributorDid: "did:mycelix:alice",
 *   content: "This proposal aligns with our values of interconnectedness...",
 *   harmonyTags: ["UniversalInterconnectedness", "SacredReciprocity"],
 *   stance: "Support"
 * });
 *
 * // Check if discussion is ready for voting
 * const readiness = await governance.discussion.isDiscussionReady("MIP-001");
 * if (!readiness.isReady) {
 *   console.log("Discussion not ready:", readiness.reasons.join(", "));
 * }
 *
 * // Get holonic health tree
 * const tree = await governance.councils.getHolonicTree("root-council-id");
 * console.log("Governance health:", tree.healthScore);
 * ```
 */
export class ExtendedGovernanceClient extends GovernanceClient {
  public readonly discussion: DiscussionClient;
  public readonly councils: CouncilsClient;

  constructor(client: AppClient, roleName: string = "governance") {
    super(client, roleName);
    this.discussion = new DiscussionClient(client, roleName, "proposals");
    this.councils = new CouncilsClient(client, roleName, "councils");
  }
}

// ============================================================================
// UTILITY FUNCTIONS - DISCUSSION
// ============================================================================

/**
 * Format discussion reflection for human display
 */
export function formatDiscussionReflection(reflection: DiscussionReflection): string {
  const lines: string[] = [];

  lines.push(`Discussion Reflection for ${reflection.proposalId}`);
  lines.push(`Generated: ${new Date(reflection.timestamp / 1000).toISOString()}`);
  lines.push("");
  lines.push("=== PARTICIPATION ===");
  lines.push(`Contributors: ${reflection.contributorCount}`);
  lines.push(`Total Contributions: ${reflection.contributionCount}`);
  lines.push(`Avg per Participant: ${reflection.avgContributionsPerParticipant.toFixed(1)}`);
  lines.push(`Max Thread Depth: ${reflection.maxThreadDepth}`);
  lines.push("");
  lines.push("=== HARMONY ANALYSIS ===");
  lines.push(`Diversity Score: ${(reflection.harmonyDiversity * 100).toFixed(1)}%`);
  if (reflection.absentHarmonies.length > 0) {
    lines.push(`Absent Harmonies: ${reflection.absentHarmonies.join(", ")}`);
  }
  lines.push("");
  lines.push("=== STANCE DISTRIBUTION ===");
  lines.push(`Support: ${reflection.supportCount}`);
  lines.push(`Oppose: ${reflection.opposeCount}`);
  lines.push(`Neutral: ${reflection.neutralCount}`);
  lines.push(`Amend: ${reflection.amendCount}`);
  lines.push(`Preliminary Sentiment: ${(reflection.preliminarySentiment * 100).toFixed(1)}%`);
  lines.push("");
  lines.push("=== DISCUSSION HEALTH ===");
  lines.push(`Voice Concentration: ${(reflection.voiceConcentration * 100).toFixed(1)}%`);
  lines.push(`Cross-Camp Engagement: ${(reflection.crossCampEngagement * 100).toFixed(1)}%`);
  lines.push(`Substantiveness: ${(reflection.substantivenessScore * 100).toFixed(1)}%`);
  lines.push("");
  lines.push("=== READINESS ===");
  lines.push(`Ready for Vote: ${reflection.readyForVote ? "Yes" : "No"}`);
  lines.push(`Reasoning: ${reflection.readinessReasoning}`);
  if (reflection.unaddressedConcerns.length > 0) {
    lines.push(`Unaddressed Concerns: ${reflection.unaddressedConcerns.join("; ")}`);
  }

  return lines.join("\n");
}

/**
 * Format holonic reflection for human display
 */
export function formatHolonicReflection(reflection: HolonicReflection): string {
  const lines: string[] = [];

  lines.push(`Holonic Reflection for Council ${reflection.councilId}`);
  lines.push(`Generated: ${new Date(reflection.timestamp / 1000).toISOString()}`);
  lines.push(`Holon Depth: ${reflection.holonDepth}`);
  lines.push("");
  lines.push("=== MEMBERSHIP ===");
  lines.push(`Total Members: ${reflection.memberCount}`);
  lines.push(`Phi-Qualified: ${reflection.phiQualifiedCount}`);
  lines.push(`Average Phi: ${reflection.averagePhi.toFixed(3)}`);
  lines.push(`Participation Rate: ${(reflection.participationRate * 100).toFixed(1)}%`);
  lines.push("");
  lines.push("=== HOLONIC HEALTH ===");
  lines.push(`Child Councils: ${reflection.childCount}`);
  if (reflection.childHealth) {
    lines.push(`Healthy Children: ${reflection.childHealth.healthyCount}`);
    lines.push(`Struggling Children: ${reflection.childHealth.strugglingCount}`);
    lines.push(`Dormant Children: ${reflection.childHealth.dormantCount}`);
  }
  lines.push(`Collaboration Score: ${(reflection.collaborationScore * 100).toFixed(1)}%`);
  lines.push("");
  lines.push("=== VITALITY ===");
  lines.push(`Trend: ${reflection.vitalityTrend}`);
  lines.push(`Health Score: ${(reflection.healthScore * 100).toFixed(1)}%`);
  if (reflection.riskFactors.length > 0) {
    lines.push(`Risk Factors: ${reflection.riskFactors.map((r) => r.description).join("; ")}`);
  }
  if (reflection.opportunities.length > 0) {
    lines.push(`Opportunities: ${reflection.opportunities.join("; ")}`);
  }

  return lines.join("\n");
}

// ============================================================================
// REAL-TIME SIGNALS
// ============================================================================

/**
 * Governance signal types emitted by voting coordinator
 */
export type GovernanceSignal =
  | {
      type: "VoteCast";
      payload: {
        proposal_id: string;
        voter_did: string;
        choice: VoteChoice;
        weight: number;
        is_phi_weighted: boolean;
      };
    }
  | {
      type: "TallyCompleted";
      payload: {
        proposal_id: string;
        approved: boolean;
        phi_votes_for: number;
        phi_votes_against: number;
        quorum_reached: boolean;
        voter_count: number;
      };
    }
  | {
      type: "ReflectionGenerated";
      payload: {
        proposal_id: string;
        reflection_id: string;
        needs_review: boolean;
        echo_chamber_risk: string;
        health_score: number;
      };
    }
  | {
      type: "DelegationChanged";
      payload: {
        delegator_did: string;
        delegate_did: string;
        domain: string;
        action: "created" | "revoked";
      };
    }
  | {
      type: "QuadraticVoteCast";
      payload: {
        proposal_id: string;
        voter_did: string;
        credits_spent: number;
        weight: number;
      };
    }
  | {
      type: "ProposalStatusChanged";
      payload: {
        proposal_id: string;
        old_status: string;
        new_status: string;
      };
    };

/**
 * Council signal types emitted by councils coordinator
 */
export type CouncilSignal =
  | {
      type: "CouncilCreated";
      payload: {
        council_id: string;
        council_name: string;
        council_type: string;
        parent_id: string | null;
      };
    }
  | {
      type: "MemberJoined";
      payload: {
        council_id: string;
        member_did: string;
        role: string;
        phi_score: number;
      };
    }
  | {
      type: "HolonicReflectionGenerated";
      payload: {
        council_id: string;
        reflection_id: string;
        health_score: number;
        vitality_trend: string;
        risk_count: number;
      };
    }
  | {
      type: "DecisionRecorded";
      payload: {
        council_id: string;
        decision_id: string;
        title: string;
        passed: boolean;
        phi_weighted_result: number;
      };
    }
  | {
      type: "CouncilStatusChanged";
      payload: {
        council_id: string;
        old_status: string;
        new_status: string;
      };
    };

/** Combined signal type for all governance events */
export type AnyGovernanceSignal = GovernanceSignal | CouncilSignal;

/** Signal handler callback type */
export type SignalHandler<T> = (signal: T) => void;

/**
 * Signal subscription manager for real-time governance updates
 */
export class GovernanceSignalSubscriber {
  private client: AppClient;
  private handlers: Map<string, Set<SignalHandler<AnyGovernanceSignal>>> =
    new Map();
  private unsubscribe: (() => void) | null = null;

  constructor(client: AppClient) {
    this.client = client;
  }

  /**
   * Subscribe to a specific signal type
   */
  on<T extends AnyGovernanceSignal["type"]>(
    signalType: T,
    handler: SignalHandler<Extract<AnyGovernanceSignal, { type: T }>>
  ): () => void {
    if (!this.handlers.has(signalType)) {
      this.handlers.set(signalType, new Set());
    }
    this.handlers.get(signalType)!.add(handler as SignalHandler<AnyGovernanceSignal>);

    // Start listening if not already
    this.ensureListening();

    // Return unsubscribe function
    return () => {
      this.handlers.get(signalType)?.delete(handler as SignalHandler<AnyGovernanceSignal>);
    };
  }

  /**
   * Subscribe to all signals
   */
  onAll(handler: SignalHandler<AnyGovernanceSignal>): () => void {
    if (!this.handlers.has("*")) {
      this.handlers.set("*", new Set());
    }
    this.handlers.get("*")!.add(handler);

    this.ensureListening();

    return () => {
      this.handlers.get("*")?.delete(handler);
    };
  }

  /**
   * Subscribe to vote-related signals
   */
  onVote(handler: SignalHandler<Extract<GovernanceSignal, { type: "VoteCast" | "TallyCompleted" | "QuadraticVoteCast" }>>): () => void {
    const unsubVote = this.on("VoteCast", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "VoteCast" }>>);
    const unsubTally = this.on("TallyCompleted", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "TallyCompleted" }>>);
    const unsubQuad = this.on("QuadraticVoteCast", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "QuadraticVoteCast" }>>);
    return () => {
      unsubVote();
      unsubTally();
      unsubQuad();
    };
  }

  /**
   * Subscribe to council-related signals
   */
  onCouncil(handler: SignalHandler<CouncilSignal>): () => void {
    const unsubCreated = this.on("CouncilCreated", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "CouncilCreated" }>>);
    const unsubJoined = this.on("MemberJoined", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "MemberJoined" }>>);
    const unsubReflection = this.on("HolonicReflectionGenerated", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "HolonicReflectionGenerated" }>>);
    const unsubDecision = this.on("DecisionRecorded", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "DecisionRecorded" }>>);
    const unsubStatus = this.on("CouncilStatusChanged", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "CouncilStatusChanged" }>>);
    return () => {
      unsubCreated();
      unsubJoined();
      unsubReflection();
      unsubDecision();
      unsubStatus();
    };
  }

  /**
   * Subscribe to reflection signals (collective sensing)
   */
  onReflection(handler: SignalHandler<Extract<AnyGovernanceSignal, { type: "ReflectionGenerated" | "HolonicReflectionGenerated" }>>): () => void {
    const unsubProposal = this.on("ReflectionGenerated", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "ReflectionGenerated" }>>);
    const unsubHolonic = this.on("HolonicReflectionGenerated", handler as SignalHandler<Extract<AnyGovernanceSignal, { type: "HolonicReflectionGenerated" }>>);
    return () => {
      unsubProposal();
      unsubHolonic();
    };
  }

  private ensureListening(): void {
    if (this.unsubscribe) return;

    // Subscribe to app signals
    this.client.on("signal", (signal) => {
      try {
        const parsed = signal.payload as AnyGovernanceSignal;
        if (!parsed || typeof parsed !== "object" || !("type" in parsed)) {
          return;
        }

        // Notify specific handlers
        const typeHandlers = this.handlers.get(parsed.type);
        if (typeHandlers) {
          typeHandlers.forEach((handler) => handler(parsed));
        }

        // Notify wildcard handlers
        const wildcardHandlers = this.handlers.get("*");
        if (wildcardHandlers) {
          wildcardHandlers.forEach((handler) => handler(parsed));
        }
      } catch {
        // Ignore non-governance signals
      }
    });

    this.unsubscribe = () => {
      // Note: @holochain/client doesn't expose off/removeListener
      // In production, you'd track the listener and remove it
    };
  }

  /**
   * Stop listening to all signals
   */
  disconnect(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = null;
    }
    this.handlers.clear();
  }
}

/**
 * Create a signal subscriber for real-time governance updates
 *
 * @example
 * ```typescript
 * const subscriber = createSignalSubscriber(client);
 *
 * // Subscribe to specific signals
 * subscriber.on("VoteCast", (signal) => {
 *   console.log(`Vote cast on ${signal.payload.proposal_id}`);
 * });
 *
 * // Subscribe to all council events
 * subscriber.onCouncil((signal) => {
 *   console.log(`Council event: ${signal.type}`);
 * });
 *
 * // Subscribe to all reflection events
 * subscriber.onReflection((signal) => {
 *   console.log(`New reflection: ${signal.payload.health_score}`);
 * });
 *
 * // Cleanup
 * subscriber.disconnect();
 * ```
 */
export function createSignalSubscriber(
  client: AppClient
): GovernanceSignalSubscriber {
  return new GovernanceSignalSubscriber(client);
}

// ============================================================================
// EXPORTS
// ============================================================================

export { ExtendedGovernanceClient as FullGovernanceClient };
export default GovernanceClient;
