/**
 * Unified Governance Voting System
 *
 * Enables cross-domain voting and proposal management across the Mycelix ecosystem.
 * Citizens can participate in governance across multiple domains while respecting
 * domain-specific voting rules, weights, and reputation requirements.
 *
 * Key features:
 * - Cross-domain proposal creation and voting
 * - Domain-specific voting power calculation
 * - Quadratic voting with anti-Sybil protections
 * - Delegation across domain boundaries
 * - Multi-domain quorum requirements
 * - Reputation-weighted voting
 *
 * @module governance/unified-voting
 */

import type { AgentPubKey, ActionHash } from '@holochain/client';

// ============================================================================
// Domain Types
// ============================================================================

/**
 * Governance domains in the Mycelix ecosystem
 */
export type GovernanceDomain =
  | 'Core' // Core protocol governance
  | 'Finance' // Financial system governance
  | 'Property' // Property registry governance
  | 'Energy' // Energy grid governance
  | 'Health' // Health system governance
  | 'Education' // Education governance
  | 'Environment' // Environmental governance
  | 'Justice' // Justice system governance
  | 'Media' // Media governance
  | 'Research' // Research/DeSci governance
  | 'Community'; // Local community governance

/**
 * Proposal scope - determines which domains are affected
 */
export type ProposalScope =
  | 'SingleDomain' // Affects only one domain
  | 'CrossDomain' // Affects multiple specified domains
  | 'Ecosystem' // Affects entire ecosystem (constitutional)
  | 'Emergency'; // Emergency protocol-wide decision

// ============================================================================
// Voting Configuration
// ============================================================================

/**
 * Voting method options
 */
export type VotingMethod =
  | 'SimpleWeighted' // 1 person = weight votes
  | 'Quadratic' // sqrt(tokens) = votes
  | 'Conviction' // votes grow over time
  | 'RankedChoice' // preference ordering
  | 'Approval'; // vote for multiple options

/**
 * Vote choice options
 */
export type VoteChoice = 'For' | 'Against' | 'Abstain';

/**
 * Voting power sources
 */
export interface VotingPowerSource {
  source: VotingPowerType;
  amount: number;
  domain: GovernanceDomain;
  verified: boolean;
}

export type VotingPowerType =
  | 'Reputation' // K-Vector derived
  | 'Stake' // KREDIT staked
  | 'Delegation' // Power delegated from others
  | 'Credential' // Domain-specific credential
  | 'Tenure'; // Time-based (domain participation)

// ============================================================================
// Unified Proposal Types
// ============================================================================

/**
 * Unified proposal that can span multiple domains
 */
export interface UnifiedProposal {
  /** Unique proposal identifier */
  proposalId: string;
  /** Hash on-chain */
  actionHash?: ActionHash;
  /** Title of the proposal */
  title: string;
  /** Detailed description */
  description: string;
  /** Proposer's agent key */
  proposer: AgentPubKey;
  /** Proposer's DID if verified */
  proposerDid?: string;
  /** Proposal scope */
  scope: ProposalScope;
  /** Primary domain */
  primaryDomain: GovernanceDomain;
  /** Additional affected domains */
  affectedDomains: GovernanceDomain[];
  /** Proposal status */
  status: UnifiedProposalStatus;
  /** Voting method */
  votingMethod: VotingMethod;
  /** Domain-specific quorum requirements */
  quorumRequirements: DomainQuorum[];
  /** Voting period start */
  votingStart: number;
  /** Voting period end */
  votingEnd: number;
  /** Minimum reputation to vote */
  minReputationToVote: number;
  /** Minimum reputation to propose */
  minReputationToPropose: number;
  /** Execution payload (if passes) */
  executionPayload?: ExecutionPayload;
  /** Vote tally by domain */
  votesByDomain: Map<GovernanceDomain, DomainVoteTally>;
  /** Overall vote tally */
  overallTally: VoteTally;
  /** Created timestamp */
  createdAt: number;
  /** Last updated */
  updatedAt: number;
  /** Executed timestamp (if executed) */
  executedAt?: number;
  /** Tags for categorization */
  tags: string[];
  /** Discussion thread hash */
  discussionHash?: ActionHash;
}

export type UnifiedProposalStatus =
  | 'Draft' // Being prepared
  | 'Pending' // Waiting for voting to start
  | 'Active' // Voting in progress
  | 'Passed' // Achieved quorum and majority
  | 'Rejected' // Did not pass
  | 'Expired' // Voting period ended without quorum
  | 'Vetoed' // Vetoed by emergency process
  | 'Executing' // Execution in progress
  | 'Executed' // Successfully executed
  | 'Failed'; // Execution failed

/**
 * Quorum requirement for a specific domain
 */
export interface DomainQuorum {
  domain: GovernanceDomain;
  /** Required participation percentage (0-1) */
  participationThreshold: number;
  /** Required approval percentage (0-1) */
  approvalThreshold: number;
  /** Minimum eligible voters */
  minEligibleVoters: number;
  /** Whether this domain's quorum is mandatory */
  mandatory: boolean;
}

/**
 * Vote tally for a domain or overall
 */
export interface VoteTally {
  forWeight: number;
  againstWeight: number;
  abstainWeight: number;
  totalVoters: number;
  eligibleVoters: number;
}

/**
 * Domain-specific vote tally with quorum status
 */
export interface DomainVoteTally extends VoteTally {
  domain: GovernanceDomain;
  quorumMet: boolean;
  approvalMet: boolean;
}

/**
 * Execution payload for passed proposals
 */
export interface ExecutionPayload {
  /** Target domains for execution */
  targetDomains: GovernanceDomain[];
  /** Actions to execute */
  actions: ExecutionAction[];
  /** Execution order (parallel or sequential) */
  executionOrder: 'Parallel' | 'Sequential';
  /** Rollback on failure */
  rollbackOnFailure: boolean;
}

export interface ExecutionAction {
  domain: GovernanceDomain;
  actionType: string;
  parameters: Record<string, unknown>;
  requiredApprovals?: number;
}

// ============================================================================
// Unified Vote Types
// ============================================================================

/**
 * A vote cast on a unified proposal
 */
export interface UnifiedVote {
  /** Vote identifier */
  voteId: string;
  /** Hash on-chain */
  actionHash?: ActionHash;
  /** Proposal being voted on */
  proposalId: string;
  /** Voter's agent key */
  voter: AgentPubKey;
  /** Voter's DID */
  voterDid?: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Domain this vote counts for */
  votingDomain: GovernanceDomain;
  /** Voting power breakdown */
  powerSources: VotingPowerSource[];
  /** Total voting power used */
  totalPower: number;
  /** Effective power after method adjustment */
  effectivePower: number;
  /** Optional reasoning */
  reason?: string;
  /** Delegation chain if delegated */
  delegationChain?: AgentPubKey[];
  /** Vote timestamp */
  timestamp: number;
  /** Signature for verification */
  signature?: Uint8Array;
}

// ============================================================================
// Delegation Types
// ============================================================================

/**
 * Cross-domain delegation
 */
export interface UnifiedDelegation {
  /** Delegation identifier */
  delegationId: string;
  /** Who is delegating */
  delegator: AgentPubKey;
  /** Who receives the delegation */
  delegate: AgentPubKey;
  /** Domains this delegation applies to */
  domains: GovernanceDomain[];
  /** What percentage of power is delegated (0-1) */
  powerPercentage: number;
  /** Maximum power that can be used per proposal */
  maxPowerPerProposal?: number;
  /** Delegation constraints */
  constraints: DelegationConstraints;
  /** When delegation becomes active */
  activeFrom: number;
  /** When delegation expires */
  expiresAt?: number;
  /** Whether delegation is currently active */
  isActive: boolean;
  /** Created timestamp */
  createdAt: number;
  /** Revoked timestamp */
  revokedAt?: number;
}

export interface DelegationConstraints {
  /** Only allow voting on specific proposal types */
  allowedScopes?: ProposalScope[];
  /** Don't allow voting on specific topics */
  excludedTags?: string[];
  /** Require delegate to vote same as delegator's preference if expressed */
  followPreference?: boolean;
  /** Auto-revoke if delegate votes against delegator's stated preference */
  autoRevokeOnConflict?: boolean;
}

// ============================================================================
// Voter Profile
// ============================================================================

/**
 * Unified voter profile across domains
 */
export interface UnifiedVoterProfile {
  /** Agent public key */
  agentPubKey: AgentPubKey;
  /** Verified DID */
  did?: string;
  /** Voting power by domain */
  domainPower: Map<GovernanceDomain, number>;
  /** Total voting power */
  totalPower: number;
  /** Delegation given to others */
  delegationsGiven: UnifiedDelegation[];
  /** Delegations received from others */
  delegationsReceived: UnifiedDelegation[];
  /** Vote history summary */
  voteHistory: VoteHistorySummary;
  /** Reputation scores by domain */
  domainReputation: Map<GovernanceDomain, number>;
  /** Overall governance reputation */
  governanceReputation: number;
  /** Active proposals created */
  activeProposalsCreated: number;
  /** Last vote timestamp */
  lastVoteAt?: number;
  /** Profile updated timestamp */
  updatedAt: number;
}

export interface VoteHistorySummary {
  totalVotes: number;
  proposalsVotedOn: number;
  proposalsCreated: number;
  successfulProposals: number;
  participationRate: number;
  averageTurnout: number;
}

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration for unified voting system
 */
export interface UnifiedVotingConfig {
  /** Default voting method */
  defaultVotingMethod?: VotingMethod;
  /** Default quorum threshold */
  defaultQuorumThreshold?: number;
  /** Default approval threshold */
  defaultApprovalThreshold?: number;
  /** Minimum proposal duration (ms) */
  minProposalDuration?: number;
  /** Maximum proposal duration (ms) */
  maxProposalDuration?: number;
  /** Enable quadratic voting */
  enableQuadraticVoting?: boolean;
  /** Enable conviction voting */
  enableConvictionVoting?: boolean;
  /** Maximum delegation depth */
  maxDelegationDepth?: number;
  /** Sybil resistance threshold */
  sybilResistanceThreshold?: number;
  /** Domain weights for cross-domain proposals */
  domainWeights?: Partial<Record<GovernanceDomain, number>>;
}

const DEFAULT_CONFIG: Required<UnifiedVotingConfig> = {
  defaultVotingMethod: 'Quadratic',
  defaultQuorumThreshold: 0.2,
  defaultApprovalThreshold: 0.5,
  minProposalDuration: 24 * 60 * 60 * 1000, // 24 hours
  maxProposalDuration: 30 * 24 * 60 * 60 * 1000, // 30 days
  enableQuadraticVoting: true,
  enableConvictionVoting: true,
  maxDelegationDepth: 3,
  sybilResistanceThreshold: 0.3,
  domainWeights: {
    Core: 1.5,
    Finance: 1.2,
    Property: 1.0,
    Energy: 1.0,
    Health: 1.2,
    Education: 1.0,
    Environment: 1.1,
    Justice: 1.3,
    Media: 0.9,
    Research: 1.0,
    Community: 0.8,
  },
};

const DEFAULT_DOMAIN_QUORUM: Record<GovernanceDomain, Omit<DomainQuorum, 'domain'>> = {
  Core: { participationThreshold: 0.3, approvalThreshold: 0.66, minEligibleVoters: 100, mandatory: true },
  Finance: { participationThreshold: 0.25, approvalThreshold: 0.6, minEligibleVoters: 50, mandatory: false },
  Property: { participationThreshold: 0.2, approvalThreshold: 0.5, minEligibleVoters: 30, mandatory: false },
  Energy: { participationThreshold: 0.2, approvalThreshold: 0.5, minEligibleVoters: 30, mandatory: false },
  Health: { participationThreshold: 0.25, approvalThreshold: 0.6, minEligibleVoters: 50, mandatory: false },
  Education: { participationThreshold: 0.2, approvalThreshold: 0.5, minEligibleVoters: 20, mandatory: false },
  Environment: { participationThreshold: 0.2, approvalThreshold: 0.55, minEligibleVoters: 40, mandatory: false },
  Justice: { participationThreshold: 0.3, approvalThreshold: 0.66, minEligibleVoters: 50, mandatory: false },
  Media: { participationThreshold: 0.15, approvalThreshold: 0.5, minEligibleVoters: 20, mandatory: false },
  Research: { participationThreshold: 0.2, approvalThreshold: 0.5, minEligibleVoters: 20, mandatory: false },
  Community: { participationThreshold: 0.15, approvalThreshold: 0.5, minEligibleVoters: 10, mandatory: false },
};

// ============================================================================
// Unified Voting Service
// ============================================================================

/**
 * Unified voting service for cross-domain governance
 */
export class UnifiedVotingService {
  private config: Required<UnifiedVotingConfig>;
  private proposals: Map<string, UnifiedProposal> = new Map();
  private votes: Map<string, UnifiedVote[]> = new Map();
  private delegations: Map<string, UnifiedDelegation[]> = new Map();
  private voterProfiles: Map<string, UnifiedVoterProfile> = new Map();

  constructor(config: UnifiedVotingConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // --------------------------------------------------------------------------
  // Proposal Management
  // --------------------------------------------------------------------------

  /**
   * Create a new unified proposal
   */
  createProposal(input: CreateProposalInput): UnifiedProposal {
    const now = Date.now();
    const proposalId = this.generateId('prop');

    // Validate proposal duration
    const duration = input.votingEnd - input.votingStart;
    if (duration < this.config.minProposalDuration) {
      throw new Error(`Proposal duration must be at least ${this.config.minProposalDuration}ms`);
    }
    if (duration > this.config.maxProposalDuration) {
      throw new Error(`Proposal duration cannot exceed ${this.config.maxProposalDuration}ms`);
    }

    // Build quorum requirements
    const allDomains = [input.primaryDomain, ...input.affectedDomains];
    const quorumRequirements = allDomains.map((domain) => ({
      domain,
      ...DEFAULT_DOMAIN_QUORUM[domain],
    }));

    const proposal: UnifiedProposal = {
      proposalId,
      title: input.title,
      description: input.description,
      proposer: input.proposer,
      proposerDid: input.proposerDid,
      scope: this.determineScope(input.primaryDomain, input.affectedDomains),
      primaryDomain: input.primaryDomain,
      affectedDomains: input.affectedDomains,
      status: 'Draft',
      votingMethod: input.votingMethod ?? this.config.defaultVotingMethod,
      quorumRequirements,
      votingStart: input.votingStart,
      votingEnd: input.votingEnd,
      minReputationToVote: input.minReputationToVote ?? 0.1,
      minReputationToPropose: input.minReputationToPropose ?? 0.3,
      executionPayload: input.executionPayload,
      votesByDomain: new Map(allDomains.map((d) => [d, this.emptyDomainTally(d)])),
      overallTally: this.emptyTally(),
      createdAt: now,
      updatedAt: now,
      tags: input.tags ?? [],
    };

    this.proposals.set(proposalId, proposal);
    this.votes.set(proposalId, []);

    return proposal;
  }

  /**
   * Activate a proposal (start voting)
   */
  activateProposal(proposalId: string): UnifiedProposal {
    const proposal = this.getProposal(proposalId);
    if (!proposal) {
      throw new Error(`Proposal not found: ${proposalId}`);
    }

    if (proposal.status !== 'Draft' && proposal.status !== 'Pending') {
      throw new Error(`Cannot activate proposal in status: ${proposal.status}`);
    }

    const now = Date.now();
    if (now < proposal.votingStart) {
      proposal.status = 'Pending';
    } else if (now < proposal.votingEnd) {
      proposal.status = 'Active';
    } else {
      throw new Error('Voting period has already ended');
    }

    proposal.updatedAt = now;
    return proposal;
  }

  /**
   * Get a proposal by ID
   */
  getProposal(proposalId: string): UnifiedProposal | undefined {
    return this.proposals.get(proposalId);
  }

  /**
   * List proposals with filtering
   */
  listProposals(filter?: ProposalFilter): UnifiedProposal[] {
    let proposals = Array.from(this.proposals.values());

    if (filter?.status) {
      proposals = proposals.filter((p) => p.status === filter.status);
    }
    if (filter?.domain) {
      proposals = proposals.filter(
        (p) => p.primaryDomain === filter.domain || p.affectedDomains.includes(filter.domain!)
      );
    }
    if (filter?.scope) {
      proposals = proposals.filter((p) => p.scope === filter.scope);
    }
    if (filter?.proposer) {
      proposals = proposals.filter((p) => p.proposer.toString() === filter.proposer!.toString());
    }
    if (filter?.activeSince) {
      proposals = proposals.filter((p) => p.votingStart >= filter.activeSince!);
    }

    return proposals;
  }

  // --------------------------------------------------------------------------
  // Voting
  // --------------------------------------------------------------------------

  /**
   * Cast a vote on a proposal
   */
  castVote(input: CastVoteInput): UnifiedVote {
    const proposal = this.getProposal(input.proposalId);
    if (!proposal) {
      throw new Error(`Proposal not found: ${input.proposalId}`);
    }

    // Check voting period
    const now = Date.now();
    if (now < proposal.votingStart) {
      throw new Error('Voting has not started');
    }
    if (now > proposal.votingEnd) {
      throw new Error('Voting has ended');
    }

    if (proposal.status !== 'Active') {
      throw new Error(`Cannot vote on proposal in status: ${proposal.status}`);
    }

    // Check domain eligibility
    const votingDomain = input.votingDomain ?? proposal.primaryDomain;
    if (
      votingDomain !== proposal.primaryDomain &&
      !proposal.affectedDomains.includes(votingDomain)
    ) {
      throw new Error(`Domain ${votingDomain} is not part of this proposal`);
    }

    // Check for existing vote
    const existingVotes = this.votes.get(input.proposalId) ?? [];
    const existingVote = existingVotes.find(
      (v) => v.voter.toString() === input.voter.toString() && v.votingDomain === votingDomain
    );
    if (existingVote) {
      throw new Error('Already voted in this domain for this proposal');
    }

    // Calculate voting power
    const powerSources = input.powerSources ?? this.calculateVotingPower(input.voter, votingDomain);
    const totalPower = powerSources.reduce((sum, s) => sum + s.amount, 0);
    const effectivePower = this.applyVotingMethod(totalPower, proposal.votingMethod);

    const vote: UnifiedVote = {
      voteId: this.generateId('vote'),
      proposalId: input.proposalId,
      voter: input.voter,
      voterDid: input.voterDid,
      choice: input.choice,
      votingDomain,
      powerSources,
      totalPower,
      effectivePower,
      reason: input.reason,
      delegationChain: input.delegationChain,
      timestamp: now,
    };

    // Record the vote
    existingVotes.push(vote);
    this.votes.set(input.proposalId, existingVotes);

    // Update tallies
    this.updateTallies(proposal, vote);

    return vote;
  }

  /**
   * Get votes for a proposal
   */
  getVotes(proposalId: string): UnifiedVote[] {
    return this.votes.get(proposalId) ?? [];
  }

  /**
   * Check if an agent has voted
   */
  hasVoted(proposalId: string, voter: AgentPubKey, domain?: GovernanceDomain): boolean {
    const votes = this.votes.get(proposalId) ?? [];
    return votes.some(
      (v) =>
        v.voter.toString() === voter.toString() && (domain === undefined || v.votingDomain === domain)
    );
  }

  // --------------------------------------------------------------------------
  // Delegation
  // --------------------------------------------------------------------------

  /**
   * Create a delegation
   */
  createDelegation(input: CreateDelegationInput): UnifiedDelegation {
    const now = Date.now();

    // Check for circular delegation
    if (this.wouldCreateCircularDelegation(input.delegator, input.delegate, input.domains)) {
      throw new Error('This delegation would create a circular dependency');
    }

    // Check delegation depth
    const depth = this.getDelegationDepth(input.delegate, input.domains[0]);
    if (depth >= this.config.maxDelegationDepth) {
      throw new Error(`Maximum delegation depth (${this.config.maxDelegationDepth}) exceeded`);
    }

    const delegation: UnifiedDelegation = {
      delegationId: this.generateId('del'),
      delegator: input.delegator,
      delegate: input.delegate,
      domains: input.domains,
      powerPercentage: input.powerPercentage ?? 1.0,
      maxPowerPerProposal: input.maxPowerPerProposal,
      constraints: input.constraints ?? {},
      activeFrom: input.activeFrom ?? now,
      expiresAt: input.expiresAt,
      isActive: true,
      createdAt: now,
    };

    // Store delegation
    const key = input.delegator.toString();
    const existing = this.delegations.get(key) ?? [];
    existing.push(delegation);
    this.delegations.set(key, existing);

    return delegation;
  }

  /**
   * Revoke a delegation
   */
  revokeDelegation(delegator: AgentPubKey, delegationId: string): boolean {
    const key = delegator.toString();
    const delegations = this.delegations.get(key);
    if (!delegations) return false;

    const delegation = delegations.find((d) => d.delegationId === delegationId);
    if (!delegation) return false;

    delegation.isActive = false;
    delegation.revokedAt = Date.now();
    return true;
  }

  /**
   * Get delegations given by an agent
   */
  getDelegationsGiven(delegator: AgentPubKey): UnifiedDelegation[] {
    return (this.delegations.get(delegator.toString()) ?? []).filter((d) => d.isActive);
  }

  /**
   * Get delegations received by an agent
   */
  getDelegationsReceived(delegate: AgentPubKey): UnifiedDelegation[] {
    const received: UnifiedDelegation[] = [];
    for (const delegations of this.delegations.values()) {
      for (const d of delegations) {
        if (d.delegate.toString() === delegate.toString() && d.isActive) {
          received.push(d);
        }
      }
    }
    return received;
  }

  // --------------------------------------------------------------------------
  // Proposal Finalization
  // --------------------------------------------------------------------------

  /**
   * Finalize a proposal (check quorum, determine outcome)
   */
  finalizeProposal(proposalId: string): ProposalResult {
    const proposal = this.getProposal(proposalId);
    if (!proposal) {
      throw new Error(`Proposal not found: ${proposalId}`);
    }

    const now = Date.now();
    if (now < proposal.votingEnd && proposal.status === 'Active') {
      throw new Error('Voting period has not ended');
    }

    // Check quorum for each domain
    let allMandatoryQuorumsMet = true;
    let anyQuorumMet = false;

    for (const [domain, tally] of proposal.votesByDomain) {
      const quorum = proposal.quorumRequirements.find((q) => q.domain === domain);
      if (!quorum) continue;

      const participation =
        tally.eligibleVoters > 0 ? tally.totalVoters / tally.eligibleVoters : 0;
      const approval =
        tally.forWeight + tally.againstWeight > 0
          ? tally.forWeight / (tally.forWeight + tally.againstWeight)
          : 0;

      tally.quorumMet = participation >= quorum.participationThreshold;
      tally.approvalMet = approval >= quorum.approvalThreshold;

      if (tally.quorumMet) anyQuorumMet = true;
      if (quorum.mandatory && !tally.quorumMet) {
        allMandatoryQuorumsMet = false;
      }
    }

    // Determine overall outcome
    const overallParticipation =
      proposal.overallTally.eligibleVoters > 0
        ? proposal.overallTally.totalVoters / proposal.overallTally.eligibleVoters
        : 0;
    const overallApproval =
      proposal.overallTally.forWeight + proposal.overallTally.againstWeight > 0
        ? proposal.overallTally.forWeight /
          (proposal.overallTally.forWeight + proposal.overallTally.againstWeight)
        : 0;

    const passed = allMandatoryQuorumsMet && anyQuorumMet && overallApproval >= 0.5;

    // Update status
    if (!anyQuorumMet) {
      proposal.status = 'Expired';
    } else if (passed) {
      proposal.status = 'Passed';
    } else {
      proposal.status = 'Rejected';
    }
    proposal.updatedAt = now;

    return {
      proposalId,
      passed,
      quorumMet: anyQuorumMet && allMandatoryQuorumsMet,
      totalVoters: proposal.overallTally.totalVoters,
      eligibleVoters: proposal.overallTally.eligibleVoters,
      forPercentage: overallApproval,
      participationRate: overallParticipation,
      domainResults: Array.from(proposal.votesByDomain.entries()).map(([domain, tally]) => ({
        domain,
        quorumMet: tally.quorumMet,
        approvalMet: tally.approvalMet,
        forWeight: tally.forWeight,
        againstWeight: tally.againstWeight,
        abstainWeight: tally.abstainWeight,
      })),
    };
  }

  // --------------------------------------------------------------------------
  // Voter Profile
  // --------------------------------------------------------------------------

  /**
   * Get or create voter profile
   */
  getVoterProfile(agentPubKey: AgentPubKey): UnifiedVoterProfile {
    const key = agentPubKey.toString();
    let profile = this.voterProfiles.get(key);

    if (!profile) {
      profile = {
        agentPubKey,
        domainPower: new Map(),
        totalPower: 0,
        delegationsGiven: [],
        delegationsReceived: [],
        voteHistory: {
          totalVotes: 0,
          proposalsVotedOn: 0,
          proposalsCreated: 0,
          successfulProposals: 0,
          participationRate: 0,
          averageTurnout: 0,
        },
        domainReputation: new Map(),
        governanceReputation: 0.5,
        activeProposalsCreated: 0,
        updatedAt: Date.now(),
      };
      this.voterProfiles.set(key, profile);
    }

    // Update delegations
    profile.delegationsGiven = this.getDelegationsGiven(agentPubKey);
    profile.delegationsReceived = this.getDelegationsReceived(agentPubKey);

    return profile;
  }

  // --------------------------------------------------------------------------
  // Helper Methods
  // --------------------------------------------------------------------------

  private generateId(prefix: string): string {
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  }

  private determineScope(primary: GovernanceDomain, affected: GovernanceDomain[]): ProposalScope {
    if (affected.length === 0) return 'SingleDomain';
    if (primary === 'Core' && affected.length > 3) return 'Ecosystem';
    return 'CrossDomain';
  }

  private emptyTally(): VoteTally {
    return {
      forWeight: 0,
      againstWeight: 0,
      abstainWeight: 0,
      totalVoters: 0,
      eligibleVoters: 0,
    };
  }

  private emptyDomainTally(domain: GovernanceDomain): DomainVoteTally {
    return {
      domain,
      forWeight: 0,
      againstWeight: 0,
      abstainWeight: 0,
      totalVoters: 0,
      eligibleVoters: 0,
      quorumMet: false,
      approvalMet: false,
    };
  }

  private calculateVotingPower(_voter: AgentPubKey, domain: GovernanceDomain): VotingPowerSource[] {
    // In production, this would query actual reputation/stake data from cross-domain reputation
    // and K-Vector scores for the given voter
    return [
      {
        source: 'Reputation',
        amount: 1.0,
        domain,
        verified: true,
      },
    ];
  }

  private applyVotingMethod(power: number, method: VotingMethod): number {
    switch (method) {
      case 'Quadratic':
        return Math.sqrt(power);
      case 'SimpleWeighted':
        return power;
      case 'Conviction':
        // Would need time factor in production
        return power;
      case 'RankedChoice':
      case 'Approval':
        return 1; // 1 vote per person
      default:
        return power;
    }
  }

  private updateTallies(proposal: UnifiedProposal, vote: UnifiedVote): void {
    // Update domain tally
    const domainTally = proposal.votesByDomain.get(vote.votingDomain);
    if (domainTally) {
      domainTally.totalVoters++;
      switch (vote.choice) {
        case 'For':
          domainTally.forWeight += vote.effectivePower;
          break;
        case 'Against':
          domainTally.againstWeight += vote.effectivePower;
          break;
        case 'Abstain':
          domainTally.abstainWeight += vote.effectivePower;
          break;
      }
    }

    // Update overall tally
    proposal.overallTally.totalVoters++;
    switch (vote.choice) {
      case 'For':
        proposal.overallTally.forWeight += vote.effectivePower;
        break;
      case 'Against':
        proposal.overallTally.againstWeight += vote.effectivePower;
        break;
      case 'Abstain':
        proposal.overallTally.abstainWeight += vote.effectivePower;
        break;
    }

    proposal.updatedAt = Date.now();
  }

  private wouldCreateCircularDelegation(
    delegator: AgentPubKey,
    delegate: AgentPubKey,
    domains: GovernanceDomain[]
  ): boolean {
    const visited = new Set<string>();
    const queue = [delegate.toString()];

    while (queue.length > 0) {
      const current = queue.shift()!;
      if (current === delegator.toString()) return true;
      if (visited.has(current)) continue;
      visited.add(current);

      const delegations = this.delegations.get(current) ?? [];
      for (const d of delegations) {
        if (d.isActive && d.domains.some((dom) => domains.includes(dom))) {
          queue.push(d.delegate.toString());
        }
      }
    }

    return false;
  }

  private getDelegationDepth(agent: AgentPubKey, domain: GovernanceDomain): number {
    let maxDepth = 0;
    const delegations = this.getDelegationsReceived(agent);

    for (const d of delegations) {
      if (d.domains.includes(domain)) {
        const depth = 1 + this.getDelegationDepth(d.delegator, domain);
        maxDepth = Math.max(maxDepth, depth);
      }
    }

    return maxDepth;
  }
}

// ============================================================================
// Input Types
// ============================================================================

export interface CreateProposalInput {
  title: string;
  description: string;
  proposer: AgentPubKey;
  proposerDid?: string;
  primaryDomain: GovernanceDomain;
  affectedDomains: GovernanceDomain[];
  votingStart: number;
  votingEnd: number;
  votingMethod?: VotingMethod;
  minReputationToVote?: number;
  minReputationToPropose?: number;
  executionPayload?: ExecutionPayload;
  tags?: string[];
}

export interface CastVoteInput {
  proposalId: string;
  voter: AgentPubKey;
  voterDid?: string;
  choice: VoteChoice;
  votingDomain?: GovernanceDomain;
  powerSources?: VotingPowerSource[];
  reason?: string;
  delegationChain?: AgentPubKey[];
}

export interface CreateDelegationInput {
  delegator: AgentPubKey;
  delegate: AgentPubKey;
  domains: GovernanceDomain[];
  powerPercentage?: number;
  maxPowerPerProposal?: number;
  constraints?: DelegationConstraints;
  activeFrom?: number;
  expiresAt?: number;
}

export interface ProposalFilter {
  status?: UnifiedProposalStatus;
  domain?: GovernanceDomain;
  scope?: ProposalScope;
  proposer?: AgentPubKey;
  activeSince?: number;
}

export interface ProposalResult {
  proposalId: string;
  passed: boolean;
  quorumMet: boolean;
  totalVoters: number;
  eligibleVoters: number;
  forPercentage: number;
  participationRate: number;
  domainResults: DomainResult[];
}

export interface DomainResult {
  domain: GovernanceDomain;
  quorumMet: boolean;
  approvalMet: boolean;
  forWeight: number;
  againstWeight: number;
  abstainWeight: number;
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a new unified voting service
 */
export function createUnifiedVotingService(
  config?: UnifiedVotingConfig
): UnifiedVotingService {
  return new UnifiedVotingService(config);
}
