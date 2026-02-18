/**
 * Governance hApp Client for the Master SDK (Phase 3)
 *
 * Complete TypeScript client for the Mycelix Governance hApp providing access
 * to all 10 governance zomes:
 *
 * - **dao** - DAO creation, membership, and management
 * - **proposals** - Proposal lifecycle (create, activate, vote, finalize)
 * - **voting** - Vote casting, MATL-weighted power, and statistics
 * - **delegation** - Vote power delegation and chain management
 * - **treasury** - DAO treasury management and allocations
 * - **execution** - Proposal execution and scheduling
 * - **bridge** - Cross-hApp governance coordination
 *
 * @module @mycelix/sdk/clients/governance
 *
 * @example
 * ```typescript
 * import { GovernanceClient } from '@mycelix/sdk/clients/governance';
 *
 * // Connect to the Governance hApp
 * const governance = await GovernanceClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Create a DAO
 * const dao = await governance.dao.createDAO({
 *   name: 'Community Governance',
 *   description: 'Decentralized decision-making',
 *   charter: 'All major decisions require 2/3 majority...',
 * });
 *
 * // Create a proposal
 * const proposal = await governance.proposals.createProposal({
 *   daoId: dao.id,
 *   title: 'Fund Community Garden',
 *   description: 'Allocate 5000 MYC to...',
 *   proposalType: 'Standard',
 * });
 *
 * // Activate voting
 * await governance.proposals.activateProposal(proposal.id);
 *
 * // Cast a vote
 * await governance.voting.castVoteAuto(proposal.id, 'Approve', 'Great initiative!');
 *
 * // Check status
 * const stats = await governance.voting.getVotingStats(proposal.id);
 * console.log(`Approval: ${stats.approvePercentage}%`);
 * ```
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

// Import all zome clients
import { BridgeClient } from './bridge';
import { ConstitutionClient } from './constitution';
import { CouncilsClient } from './councils';
import { DAOClient } from './dao';
import { DelegationClient } from './delegation';
import { ExecutionClient } from './execution';
import { ProposalsClient } from './proposals';
import { ThresholdSigningClient } from './threshold-signing';
import { TreasuryClient } from './treasury';
import { GovernanceError } from './types';
import { VotingClient } from './voting';

// Note: Client config types are exported via re-exports below
import type { ActionHash } from '../../generated/common';

// ============================================================================
// Client Configuration
// ============================================================================

/**
 * Configuration for the Governance client
 */
export interface GovernanceClientConfig {
  /** Role name for governance DNA (default: 'governance') */
  roleName?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Timeout for zome calls in milliseconds */
  timeout?: number;
  /** Source hApp identifier for bridge events */
  sourceHapp?: string;
}

/**
 * Connection options for creating a new client
 */
export interface GovernanceConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Optional configuration */
  config?: GovernanceClientConfig;
}

const DEFAULT_CONFIG: Required<GovernanceClientConfig> = {
  roleName: 'governance',
  debug: false,
  timeout: 30000,
  sourceHapp: 'governance',
};

// ============================================================================
// Aggregate Governance Client
// ============================================================================

/**
 * Unified Governance hApp Client
 *
 * Provides access to all 7 governance zomes through a single interface.
 * This is the recommended entry point for governance operations.
 *
 * @example
 * ```typescript
 * // Connect directly
 * const governance = await GovernanceClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Or use with existing AppClient
 * const governance = GovernanceClient.fromClient(existingClient);
 *
 * // Access zome clients
 * await governance.dao.createDAO({...});
 * await governance.proposals.createProposal({...});
 * await governance.voting.castVote({...});
 * await governance.delegation.createDelegation({...});
 * await governance.treasury.proposeAllocation({...});
 * await governance.execution.requestExecution({...});
 * await governance.bridge.broadcastEvent({...});
 * ```
 */
export class GovernanceClient {
  // Zome clients
  /** DAO management operations */
  public readonly dao: DAOClient;

  /** Proposal lifecycle operations */
  public readonly proposals: ProposalsClient;

  /** Voting operations */
  public readonly voting: VotingClient;

  /** Delegation operations */
  public readonly delegation: DelegationClient;

  /** Treasury management operations */
  public readonly treasury: TreasuryClient;

  /** Execution operations */
  public readonly execution: ExecutionClient;

  /** Cross-hApp bridge operations */
  public readonly bridge: BridgeClient;

  /** Holonic council management */
  public readonly councils: CouncilsClient;

  /** Constitutional charter and amendments */
  public readonly constitution: ConstitutionClient;

  /** Threshold signing and DKG ceremonies */
  public readonly thresholdSigning: ThresholdSigningClient;

  // Internal state
  private readonly _client: AppClient;
  private readonly _config: Required<GovernanceClientConfig>;

  /**
   * Private constructor - use static factory methods
   */
  private constructor(client: AppClient, config: GovernanceClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    // Initialize all zome clients
    this.dao = new DAOClient(client, baseConfig);
    this.proposals = new ProposalsClient(client, baseConfig);
    this.voting = new VotingClient(client, baseConfig);
    this.delegation = new DelegationClient(client, baseConfig);
    this.treasury = new TreasuryClient(client, baseConfig);
    this.execution = new ExecutionClient(client, baseConfig);
    this.bridge = new BridgeClient(client, {
      ...baseConfig,
      sourceHapp: this._config.sourceHapp,
    });
    this.councils = new CouncilsClient(client, baseConfig);
    this.constitution = new ConstitutionClient(client, baseConfig);
    this.thresholdSigning = new ThresholdSigningClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[governance-sdk] Client initialized with 10 zome clients');
    }
  }

  /**
   * Connect to Holochain and create a GovernanceClient
   *
   * @param options - Connection options
   * @returns Connected GovernanceClient
   *
   * @example
   * ```typescript
   * const governance = await GovernanceClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   *   config: { debug: true },
   * });
   * ```
   */
  static async connect(options: GovernanceConnectionOptions): Promise<GovernanceClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      // Verify connection
      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      if (options.config?.debug) {
        console.log(`[governance-sdk] Connected to ${options.url}`);
      }

      return new GovernanceClient(client, options.config);
    } catch (error) {
      throw new GovernanceError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Create a GovernanceClient from an existing AppClient
   *
   * Use this when you already have a Holochain connection.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration
   * @returns GovernanceClient
   *
   * @example
   * ```typescript
   * const existingClient = await AppWebsocket.connect({...});
   * const governance = GovernanceClient.fromClient(existingClient);
   * ```
   */
  static fromClient(
    client: AppClient,
    config: GovernanceClientConfig = {}
  ): GovernanceClient {
    return new GovernanceClient(client, config);
  }

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Get the underlying Holochain AppClient
   */
  getClient(): AppClient {
    return this._client;
  }

  /**
   * Get the current agent's public key
   */
  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  /**
   * Check if connected to Holochain
   */
  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }

  // ============================================================================
  // Convenience Methods (Cross-Zome Operations)
  // ============================================================================

  /**
   * Get comprehensive DAO status
   *
   * Returns DAO info, active proposals, member count, and treasury balance.
   *
   * @param daoId - DAO identifier
   * @returns DAO status summary
   */
  async getDAOStatus(daoId: ActionHash): Promise<{
    dao: Awaited<ReturnType<DAOClient['getDAO']>>;
    activeProposals: Awaited<ReturnType<ProposalsClient['getActiveProposals']>>;
    memberCount: number;
    stats: Awaited<ReturnType<DAOClient['getDAOStats']>>;
  }> {
    const [dao, activeProposals, stats] = await Promise.all([
      this.dao.getDAO(daoId),
      this.proposals.getActiveProposals(daoId),
      this.dao.getDAOStats(daoId),
    ]);

    return {
      dao,
      activeProposals,
      memberCount: stats.memberCount,
      stats,
    };
  }

  /**
   * Get comprehensive proposal status
   *
   * Returns proposal info, voting stats, and quorum status.
   *
   * @param proposalId - Proposal identifier
   * @returns Proposal status summary
   */
  async getProposalStatus(proposalId: ActionHash): Promise<{
    proposal: Awaited<ReturnType<ProposalsClient['getProposal']>>;
    stats: Awaited<ReturnType<VotingClient['getVotingStats']>>;
    quorum: Awaited<ReturnType<VotingClient['checkQuorum']>>;
    votes: Awaited<ReturnType<VotingClient['getVotesForProposal']>>;
  }> {
    const [proposal, stats, quorum, votes] = await Promise.all([
      this.proposals.getProposal(proposalId),
      this.voting.getVotingStats(proposalId),
      this.voting.checkQuorum(proposalId),
      this.voting.getVotesForProposal(proposalId),
    ]);

    return {
      proposal,
      stats,
      quorum,
      votes,
    };
  }

  /**
   * Get member governance profile
   *
   * Returns membership info, voting power, delegations, and participation.
   *
   * @param daoId - DAO identifier
   * @param memberDid - Member's DID
   * @returns Member governance profile
   */
  async getMemberProfile(daoId: ActionHash, memberDid: string): Promise<{
    membership: Awaited<ReturnType<DAOClient['getMembership']>>;
    votingPower: Awaited<ReturnType<VotingClient['getVotingPowerBreakdown']>>;
    delegationsFrom: Awaited<ReturnType<DelegationClient['getDelegationsFrom']>>;
    delegationsTo: Awaited<ReturnType<DelegationClient['getDelegationsTo']>>;
    activity: Awaited<ReturnType<DAOClient['getMemberActivity']>>;
    participationScore: Awaited<ReturnType<BridgeClient['getParticipationScore']>>;
  }> {
    const [membership, votingPower, delegationsFrom, delegationsTo, activity, participationScore] =
      await Promise.all([
        this.dao.getMembership(daoId, memberDid),
        this.voting.getVotingPowerBreakdown({ daoId, voterDid: memberDid, includeDelegated: true }),
        this.delegation.getDelegationsFrom(daoId, memberDid),
        this.delegation.getDelegationsTo(daoId, memberDid),
        this.dao.getMemberActivity(daoId, memberDid),
        this.bridge.getParticipationScore(memberDid),
      ]);

    return {
      membership,
      votingPower,
      delegationsFrom,
      delegationsTo,
      activity,
      participationScore,
    };
  }

  /**
   * Execute full proposal workflow
   *
   * Creates a proposal, activates it, and optionally schedules execution.
   *
   * @param input - Proposal creation input
   * @param activateImmediately - Whether to activate immediately (default: true)
   * @returns Created and optionally activated proposal
   */
  async createAndActivateProposal(
    input: Parameters<ProposalsClient['createProposal']>[0],
    activateImmediately: boolean = true
  ): Promise<Awaited<ReturnType<ProposalsClient['createProposal']>>> {
    const proposal = await this.proposals.createProposal(input);

    if (activateImmediately) {
      return this.proposals.activateProposal(proposal.id);
    }

    return proposal;
  }

  /**
   * Get list of zome capabilities
   *
   * Useful for debugging and introspection.
   */
  getCapabilities(): Record<string, string[]> {
    return {
      dao: [
        'createDAO',
        'getDAO',
        'updateDAO',
        'listDAOs',
        'archiveDAO',
        'joinDAO',
        'leaveDAO',
        'getMembers',
        'getMembership',
        'updateMemberRole',
        'removeMember',
        'getDAOStats',
      ],
      proposals: [
        'createProposal',
        'getProposal',
        'updateProposal',
        'listProposals',
        'getActiveProposals',
        'activateProposal',
        'cancelProposal',
        'vetoProposal',
        'finalizeProposal',
        'markExecuted',
        'getProposalHistory',
      ],
      voting: [
        'castVote',
        'castVoteAuto',
        'castDelegatedVote',
        'changeVote',
        'getVotesForProposal',
        'getVote',
        'hasVoted',
        'getVotingPower',
        'getVotingPowerBreakdown',
        'calculateMATLVotingPower',
        'checkQuorum',
        'getVotingStats',
      ],
      delegation: [
        'createDelegation',
        'revokeDelegation',
        'updateDelegation',
        'getDelegation',
        'listDelegations',
        'getDelegationsFrom',
        'getDelegationsTo',
        'hasDelegated',
        'getDelegatedPowerReceived',
        'getDelegationChain',
        'wouldCreateCycle',
      ],
      treasury: [
        'createTreasury',
        'getTreasury',
        'updateTreasury',
        'addSigner',
        'removeSigner',
        'getBalance',
        'deposit',
        'proposeAllocation',
        'getAllocation',
        'approveAllocation',
        'executeAllocation',
        'cancelAllocation',
      ],
      execution: [
        'requestExecution',
        'executeNow',
        'getExecution',
        'listExecutions',
        'acknowledgeExecution',
        'retryExecution',
        'cancelExecution',
        'revertExecution',
        'scheduleExecution',
        'cancelSchedule',
        'getScheduledExecutions',
      ],
      bridge: [
        'broadcastEvent',
        'getRecentEvents',
        'getEventsByType',
        'getEventsForDAO',
        'getEventsForProposal',
        'queryGovernance',
        'registerCrossHappProposal',
        'getCrossHappProposals',
        'getParticipationScore',
        'reportParticipationScore',
        'registerHapp',
        'getRegisteredHapps',
      ],
    };
  }
}

// ============================================================================
// Re-exports
// ============================================================================

// Zome clients
export { DAOClient, type DAOClientConfig } from './dao';
export { ProposalsClient, type ProposalsClientConfig, VOTING_PERIODS } from './proposals';
export { VotingClient, type VotingClientConfig } from './voting';
export { DelegationClient, type DelegationClientConfig } from './delegation';
export { TreasuryClient, type TreasuryClientConfig } from './treasury';
export { ExecutionClient, type ExecutionClientConfig } from './execution';
export { BridgeClient, type BridgeClientConfig } from './bridge';
export { CouncilsClient, type CouncilsClientConfig } from './councils';
export { ConstitutionClient, type ConstitutionClientConfig } from './constitution';
export { ThresholdSigningClient, type ThresholdSigningClientConfig } from './threshold-signing';

// Types
export * from './types';

// Default export
export default GovernanceClient;
