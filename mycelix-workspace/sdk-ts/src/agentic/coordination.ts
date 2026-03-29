// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Multi-Agent Coordination
 *
 * Enables AI agents to collaborate with trust-weighted influence.
 * Supports consensus mechanisms for agent groups.
 *
 * @example
 * ```typescript
 * import { AgentGroup, CoordinationConfig, Proposal, VoteType } from '@mycelix/sdk';
 *
 * // Create a group of trading agents
 * const group = new AgentGroup();
 *
 * // Add agents with their trust scores
 * group.addMember('agent-1', 0.8);
 * group.addMember('agent-2', 0.6);
 * group.addMember('agent-3', 0.7);
 *
 * // Create a proposal
 * const proposal = new Proposal('increase-position', 'Increase BTC position by 10%');
 * const proposalId = group.submitProposal(proposal);
 *
 * // Submit votes
 * group.vote('agent-1', proposalId, VoteType.Approve);
 * group.vote('agent-2', proposalId, VoteType.Approve);
 * group.vote('agent-3', proposalId, VoteType.Reject);
 *
 * // Check consensus (trust-weighted)
 * const result = group.checkConsensus(proposalId);
 * if (result) {
 *   console.log(`Decision: ${result.decision}, Weight: ${result.approvalWeight}`);
 * }
 * ```
 *
 * @packageDocumentation
 * @module coordination
 */

// =============================================================================
// Types
// =============================================================================

/**
 * Coordination configuration
 */
export interface CoordinationConfig {
  /** Minimum trust required to participate in group */
  minTrustThreshold: number;
  /** Approval threshold for consensus (weighted) */
  approvalThreshold: number;
  /** Minimum participation rate required */
  minParticipation: number;
  /** Maximum time for voting (ms) */
  votingTimeoutMs: number;
  /** Whether to use quadratic voting (sqrt of trust) */
  quadraticVoting: boolean;
  /** Maximum group size */
  maxGroupSize: number;
}

/**
 * Default coordination configuration
 */
export const DEFAULT_COORDINATION_CONFIG: CoordinationConfig = {
  minTrustThreshold: 0.3,
  approvalThreshold: 0.67, // 2/3 majority
  minParticipation: 0.5, // 50% must vote
  votingTimeoutMs: 60_000,
  quadraticVoting: false,
  maxGroupSize: 100,
};

/**
 * Vote type
 */
export enum VoteType {
  /** Approve the proposal */
  Approve = 'approve',
  /** Reject the proposal */
  Reject = 'reject',
  /** Abstain from voting */
  Abstain = 'abstain',
}

/**
 * Consensus decision outcome
 */
export enum ConsensusDecision {
  /** Proposal approved */
  Approved = 'approved',
  /** Proposal rejected */
  Rejected = 'rejected',
  /** No consensus reached (quorum not met) */
  NoQuorum = 'no_quorum',
  /** Voting still in progress */
  Pending = 'pending',
  /** Proposal expired */
  Expired = 'expired',
}

/**
 * A proposal for group decision
 */
export class Proposal {
  /** Unique proposal ID */
  public readonly id: string;
  /** Proposal type/action */
  public readonly action: string;
  /** Description */
  public readonly description: string;
  /** Creator agent ID */
  public creator: string = '';
  /** Creation timestamp (ms) */
  public readonly createdAt: number;
  /** Expiry timestamp (ms) */
  public expiresAt: number;
  /** Required quorum (weighted trust) */
  public requiredQuorum: number = 0.5;
  /** Metadata */
  public metadata: Map<string, string> = new Map();

  constructor(action: string, description: string) {
    this.id = `prop-${Date.now()}`;
    this.action = action;
    this.description = description;
    this.createdAt = Date.now();
    this.expiresAt = this.createdAt + 3600_000; // 1 hour default
  }

  /** Set the creator */
  withCreator(creator: string): this {
    this.creator = creator;
    return this;
  }

  /** Set expiry duration */
  withDurationMs(durationMs: number): this {
    this.expiresAt = this.createdAt + durationMs;
    return this;
  }

  /** Set required quorum */
  withQuorum(quorum: number): this {
    this.requiredQuorum = Math.max(0, Math.min(1, quorum));
    return this;
  }

  /** Add metadata */
  withMetadata(key: string, value: string): this {
    this.metadata.set(key, value);
    return this;
  }

  /** Check if expired */
  isExpired(): boolean {
    return Date.now() > this.expiresAt;
  }
}

/**
 * A vote on a proposal
 */
export interface Vote {
  /** Agent casting the vote */
  agentId: string;
  /** Vote type */
  voteType: VoteType;
  /** Agent's trust at time of vote */
  trustScore: number;
  /** Calculated weight (may differ from trust if quadratic) */
  weight: number;
  /** Timestamp */
  timestamp: number;
}

/**
 * Consensus result
 */
export interface ConsensusResult {
  /** Proposal ID */
  proposalId: string;
  /** Final decision */
  decision: ConsensusDecision;
  /** Total approval weight */
  approvalWeight: number;
  /** Total rejection weight */
  rejectionWeight: number;
  /** Total abstention weight */
  abstentionWeight: number;
  /** Participation rate */
  participationRate: number;
  /** Whether quorum was reached */
  quorumReached: boolean;
  /** Number of voters */
  voterCount: number;
  /** Total group members */
  totalMembers: number;
}

/**
 * Group member (internal)
 */
interface GroupMember {
  agentId: string;
  trustScore: number;
  joinedAt: number;
  participationCount: number;
}

/**
 * Proposal state (internal)
 */
interface ProposalState {
  proposal: Proposal;
  votes: Map<string, Vote>;
  finalized: boolean;
  result?: ConsensusResult;
}

// =============================================================================
// Errors
// =============================================================================

/**
 * Coordination error types
 */
export enum CoordinationErrorType {
  NotMember = 'NOT_MEMBER',
  ProposalNotFound = 'PROPOSAL_NOT_FOUND',
  ProposalFinalized = 'PROPOSAL_FINALIZED',
  ProposalExpired = 'PROPOSAL_EXPIRED',
  GroupFull = 'GROUP_FULL',
  TrustTooLow = 'TRUST_TOO_LOW',
  AlreadyMember = 'ALREADY_MEMBER',
}

/**
 * Coordination error
 */
export class CoordinationError extends Error {
  constructor(
    public readonly type: CoordinationErrorType,
    message: string,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'CoordinationError';
  }
}

// =============================================================================
// AgentGroup
// =============================================================================

/**
 * A coordinated group of agents
 *
 * Manages membership, proposals, voting, and consensus.
 */
export class AgentGroup {
  /** Group ID */
  public readonly id: string;
  /** Configuration */
  private config: CoordinationConfig;
  /** Group members */
  private members: Map<string, GroupMember> = new Map();
  /** Active proposals */
  private proposals: Map<string, ProposalState> = new Map();
  /** Total trust in group */
  private totalTrust: number = 0;

  constructor(config: Partial<CoordinationConfig> = {}) {
    this.id = `group-${Date.now()}`;
    this.config = { ...DEFAULT_COORDINATION_CONFIG, ...config };
  }

  // ---------------------------------------------------------------------------
  // Membership Management
  // ---------------------------------------------------------------------------

  /**
   * Add a member to the group
   * @throws {CoordinationError} If group is full, trust too low, or already member
   */
  addMember(agentId: string, trustScore: number): void {
    // Check group size
    if (this.members.size >= this.config.maxGroupSize) {
      throw new CoordinationError(
        CoordinationErrorType.GroupFull,
        `Group has reached maximum size of ${this.config.maxGroupSize}`,
        { maxSize: this.config.maxGroupSize }
      );
    }

    // Check trust threshold
    if (trustScore < this.config.minTrustThreshold) {
      throw new CoordinationError(
        CoordinationErrorType.TrustTooLow,
        `Trust score ${trustScore} below threshold ${this.config.minTrustThreshold}`,
        { agentId, trust: trustScore, required: this.config.minTrustThreshold }
      );
    }

    // Check if already member
    if (this.members.has(agentId)) {
      throw new CoordinationError(
        CoordinationErrorType.AlreadyMember,
        `Agent ${agentId} is already a member`,
        { agentId }
      );
    }

    // Add member
    this.members.set(agentId, {
      agentId,
      trustScore,
      joinedAt: Date.now(),
      participationCount: 0,
    });

    this.totalTrust += trustScore;
  }

  /**
   * Remove a member from the group
   */
  removeMember(agentId: string): boolean {
    const member = this.members.get(agentId);
    if (!member) {
      return false;
    }

    this.totalTrust -= member.trustScore;
    this.members.delete(agentId);
    return true;
  }

  /**
   * Update a member's trust score
   */
  updateMemberTrust(agentId: string, newTrust: number): boolean {
    const member = this.members.get(agentId);
    if (!member) {
      return false;
    }

    this.totalTrust -= member.trustScore;
    member.trustScore = newTrust;
    this.totalTrust += newTrust;
    return true;
  }

  /**
   * Check if an agent is a member
   */
  isMember(agentId: string): boolean {
    return this.members.has(agentId);
  }

  /**
   * Get member count
   */
  memberCount(): number {
    return this.members.size;
  }

  /**
   * Get total trust in group
   */
  getTotalTrust(): number {
    return this.totalTrust;
  }

  /**
   * Get all member IDs
   */
  getMemberIds(): string[] {
    return Array.from(this.members.keys());
  }

  /**
   * Get member trust score
   */
  getMemberTrust(agentId: string): number | undefined {
    return this.members.get(agentId)?.trustScore;
  }

  // ---------------------------------------------------------------------------
  // Proposal Management
  // ---------------------------------------------------------------------------

  /**
   * Submit a proposal for voting
   * @returns The proposal ID
   */
  submitProposal(proposal: Proposal): string {
    const state: ProposalState = {
      proposal,
      votes: new Map(),
      finalized: false,
    };

    this.proposals.set(proposal.id, state);
    return proposal.id;
  }

  /**
   * Get a proposal by ID
   */
  getProposal(proposalId: string): Proposal | undefined {
    return this.proposals.get(proposalId)?.proposal;
  }

  /**
   * Get all active (non-finalized) proposal IDs
   */
  getActiveProposals(): string[] {
    return Array.from(this.proposals.entries())
      .filter(([_, state]) => !state.finalized)
      .map(([id]) => id);
  }

  // ---------------------------------------------------------------------------
  // Voting
  // ---------------------------------------------------------------------------

  /**
   * Cast a vote on a proposal
   * @throws {CoordinationError} If not member, proposal not found, finalized, or expired
   */
  vote(agentId: string, proposalId: string, voteType: VoteType): void {
    // Check membership
    const member = this.members.get(agentId);
    if (!member) {
      throw new CoordinationError(
        CoordinationErrorType.NotMember,
        `Agent ${agentId} is not a member of this group`,
        { agentId }
      );
    }

    // Get proposal
    const state = this.proposals.get(proposalId);
    if (!state) {
      throw new CoordinationError(
        CoordinationErrorType.ProposalNotFound,
        `Proposal ${proposalId} not found`,
        { proposalId }
      );
    }

    // Check if finalized
    if (state.finalized) {
      throw new CoordinationError(
        CoordinationErrorType.ProposalFinalized,
        `Proposal ${proposalId} has already been finalized`,
        { proposalId }
      );
    }

    // Check if expired
    if (state.proposal.isExpired()) {
      throw new CoordinationError(
        CoordinationErrorType.ProposalExpired,
        `Proposal ${proposalId} has expired`,
        { proposalId }
      );
    }

    // Calculate weight
    const weight = this.config.quadraticVoting
      ? Math.sqrt(member.trustScore)
      : member.trustScore;

    const vote: Vote = {
      agentId,
      voteType,
      trustScore: member.trustScore,
      weight,
      timestamp: Date.now(),
    };

    state.votes.set(agentId, vote);
  }

  /**
   * Get votes for a proposal
   */
  getVotes(proposalId: string): Vote[] {
    const state = this.proposals.get(proposalId);
    return state ? Array.from(state.votes.values()) : [];
  }

  // ---------------------------------------------------------------------------
  // Consensus
  // ---------------------------------------------------------------------------

  /**
   * Check consensus on a proposal
   * @returns ConsensusResult or undefined if proposal not found
   */
  checkConsensus(proposalId: string): ConsensusResult | undefined {
    const state = this.proposals.get(proposalId);
    if (!state) {
      return undefined;
    }

    // If already finalized, return cached result
    if (state.finalized && state.result) {
      return state.result;
    }

    // Calculate totals
    let approvalWeight = 0;
    let rejectionWeight = 0;
    let abstentionWeight = 0;

    for (const vote of state.votes.values()) {
      switch (vote.voteType) {
        case VoteType.Approve:
          approvalWeight += vote.weight;
          break;
        case VoteType.Reject:
          rejectionWeight += vote.weight;
          break;
        case VoteType.Abstain:
          abstentionWeight += vote.weight;
          break;
      }
    }

    const totalVoteWeight = approvalWeight + rejectionWeight + abstentionWeight;

    // Calculate total possible weight
    const totalPossible = this.config.quadraticVoting
      ? Array.from(this.members.values()).reduce((sum, m) => sum + Math.sqrt(m.trustScore), 0)
      : this.totalTrust;

    const participationRate = totalPossible > 0 ? totalVoteWeight / totalPossible : 0;
    const quorumReached = participationRate >= this.config.minParticipation;

    // Determine decision
    let decision: ConsensusDecision;

    if (state.proposal.isExpired()) {
      decision = ConsensusDecision.Expired;
    } else if (!quorumReached) {
      // Check if it's still possible to reach quorum
      const remainingWeight = totalPossible - totalVoteWeight;
      if (totalVoteWeight + remainingWeight < this.config.minParticipation * totalPossible) {
        decision = ConsensusDecision.NoQuorum;
      } else {
        decision = ConsensusDecision.Pending;
      }
    } else {
      const activeVoteWeight = approvalWeight + rejectionWeight;
      if (activeVoteWeight > 0) {
        const approvalRatio = approvalWeight / activeVoteWeight;
        decision = approvalRatio >= this.config.approvalThreshold
          ? ConsensusDecision.Approved
          : ConsensusDecision.Rejected;
      } else {
        decision = ConsensusDecision.Pending;
      }
    }

    const result: ConsensusResult = {
      proposalId,
      decision,
      approvalWeight,
      rejectionWeight,
      abstentionWeight,
      participationRate,
      quorumReached,
      voterCount: state.votes.size,
      totalMembers: this.members.size,
    };

    // Finalize if decision is final
    if (
      decision === ConsensusDecision.Approved ||
      decision === ConsensusDecision.Rejected ||
      decision === ConsensusDecision.NoQuorum ||
      decision === ConsensusDecision.Expired
    ) {
      state.finalized = true;
      state.result = result;

      // Update participation counts
      for (const vote of state.votes.values()) {
        const member = this.members.get(vote.agentId);
        if (member) {
          member.participationCount++;
        }
      }
    }

    return result;
  }

  // ---------------------------------------------------------------------------
  // Serialization
  // ---------------------------------------------------------------------------

  /**
   * Export group state for persistence
   */
  toJSON(): object {
    return {
      id: this.id,
      config: this.config,
      members: Array.from(this.members.entries()),
      proposals: Array.from(this.proposals.entries()).map(([id, state]) => [
        id,
        {
          proposal: {
            id: state.proposal.id,
            action: state.proposal.action,
            description: state.proposal.description,
            creator: state.proposal.creator,
            createdAt: state.proposal.createdAt,
            expiresAt: state.proposal.expiresAt,
            requiredQuorum: state.proposal.requiredQuorum,
            metadata: Array.from(state.proposal.metadata.entries()),
          },
          votes: Array.from(state.votes.entries()),
          finalized: state.finalized,
          result: state.result,
        },
      ]),
      totalTrust: this.totalTrust,
    };
  }

  /**
   * Import group state from persistence
   */
  static fromJSON(data: ReturnType<AgentGroup['toJSON']>): AgentGroup {
    const parsed = data as {
      id: string;
      config: CoordinationConfig;
      members: [string, GroupMember][];
      proposals: [string, {
        proposal: {
          id: string;
          action: string;
          description: string;
          creator: string;
          createdAt: number;
          expiresAt: number;
          requiredQuorum: number;
          metadata: [string, string][];
        };
        votes: [string, Vote][];
        finalized: boolean;
        result?: ConsensusResult;
      }][];
      totalTrust: number;
    };

    const group = new AgentGroup(parsed.config);
    (group as { id: string }).id = parsed.id;
    group.members = new Map(parsed.members);
    group.totalTrust = parsed.totalTrust;

    // Restore proposals
    for (const [id, stateData] of parsed.proposals) {
      const proposal = new Proposal(stateData.proposal.action, stateData.proposal.description);
      (proposal as { id: string }).id = stateData.proposal.id;
      proposal.creator = stateData.proposal.creator;
      (proposal as { createdAt: number }).createdAt = stateData.proposal.createdAt;
      proposal.expiresAt = stateData.proposal.expiresAt;
      proposal.requiredQuorum = stateData.proposal.requiredQuorum;
      proposal.metadata = new Map(stateData.proposal.metadata);

      group.proposals.set(id, {
        proposal,
        votes: new Map(stateData.votes),
        finalized: stateData.finalized,
        result: stateData.result,
      });
    }

    return group;
  }
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate effective voting weight for trust score
 */
export function calculateVotingWeight(trustScore: number, quadratic: boolean): number {
  return quadratic ? Math.sqrt(trustScore) : trustScore;
}

/**
 * Check if a proposal would pass with current votes
 */
export function wouldProposalPass(
  approvalWeight: number,
  rejectionWeight: number,
  threshold: number
): boolean {
  const activeWeight = approvalWeight + rejectionWeight;
  if (activeWeight === 0) return false;
  return approvalWeight / activeWeight >= threshold;
}

/**
 * Calculate minimum additional approval weight needed to pass
 */
export function weightNeededToPass(
  currentApproval: number,
  currentRejection: number,
  threshold: number
): number {
  // We need: approval / (approval + rejection) >= threshold
  // approval >= threshold * (approval + rejection)
  // approval - threshold * approval >= threshold * rejection
  // approval * (1 - threshold) >= threshold * rejection
  // approval >= threshold * rejection / (1 - threshold)

  const required = (threshold * currentRejection) / (1 - threshold);
  return Math.max(0, required - currentApproval);
}
