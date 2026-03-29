// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Governance Client
 *
 * Unified client for the Governance hApp SDK, providing access to
 * DAOs, proposals, voting, and delegation functionality.
 *
 * @module @mycelix/sdk/integrations/governance
 */

import {
  type AppClient,
  AppWebsocket,
} from '@holochain/client';

import { GovernanceSdkError } from './types';
import { DaoClient } from './zomes/dao';
import { DelegationClient } from './zomes/delegation';
import { ProposalClient } from './zomes/proposal';
import { VotingClient } from './zomes/voting';

import type {
  ParticipationScore,
  GovernanceEvent,
} from './types';

/**
 * Configuration for the Governance client
 */
export interface GovernanceClientConfig {
  /** Role ID for the governance DNA */
  roleId: string;
  /** Zome name */
  zomeName: string;
}

const DEFAULT_CONFIG: GovernanceClientConfig = {
  roleId: 'governance',
  zomeName: 'governance',
};

/**
 * Connection options for creating a new client
 */
export interface GovernanceConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Optional app client constructor args */
  appClientArgs?: Record<string, unknown>;
}

/**
 * Unified Mycelix Governance Client
 *
 * Provides access to all governance functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixGovernanceClient } from '@mycelix/sdk/integrations/governance';
 *
 * // Connect to Holochain conductor
 * const governance = await MycelixGovernanceClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // Create a DAO
 * const dao = await governance.dao.createDao({
 *   id: 'community-dao',
 *   name: 'Community Governance',
 *   description: 'Decentralized community decision-making',
 *   constitution: 'All major decisions require 2/3 majority...',
 *   founder_did: 'did:mycelix:abc123',
 * });
 *
 * // Create a proposal
 * const proposal = await governance.proposals.createProposal({
 *   id: 'prop-001',
 *   dao_id: 'community-dao',
 *   title: 'Fund community garden project',
 *   description: 'Allocate 5000 tokens to...',
 *   proposer_did: 'did:mycelix:abc123',
 *   proposal_type: 'Standard',
 *   quorum_percentage: 33,
 *   approval_threshold: 51,
 * });
 *
 * // Activate voting
 * await governance.proposals.activateProposal('prop-001');
 *
 * // Cast a vote
 * await governance.voting.castVote({
 *   id: 'vote-001',
 *   proposal_id: 'prop-001',
 *   voter_did: 'did:mycelix:def456',
 *   choice: 'Yes',
 *   weight: 100,
 *   reason: 'Great initiative for the community',
 * });
 *
 * // Check voting stats
 * const stats = await governance.voting.getVotingStats('prop-001');
 * console.log(`Approval: ${stats.yes_percentage}%`);
 * ```
 */
export class MycelixGovernanceClient {
  /** DAO management operations */
  public readonly dao: DaoClient;

  /** Proposal lifecycle operations */
  public readonly proposals: ProposalClient;

  /** Voting operations */
  public readonly voting: VotingClient;

  /** Delegation operations */
  public readonly delegation: DelegationClient;

  private readonly config: GovernanceClientConfig;

  /**
   * Create a governance client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<GovernanceClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize sub-clients
    this.dao = new DaoClient(client, this.config);
    this.proposals = new ProposalClient(client, this.config);
    this.voting = new VotingClient(client, this.config);
    this.delegation = new DelegationClient(client, this.config);
  }

  /**
   * Connect to Holochain and create a governance client
   *
   * @param options - Connection options
   * @returns Connected governance client
   *
   * @example
   * ```typescript
   * const governance = await MycelixGovernanceClient.connect({
   *   url: 'ws://localhost:8888',
   *   timeout: 30000,
   * });
   * ```
   */
  static async connect(
    options: GovernanceConnectionOptions
  ): Promise<MycelixGovernanceClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
        ...(options.appClientArgs ?? {}),
      } as Parameters<typeof AppWebsocket.connect>[0]);

      return new MycelixGovernanceClient(client);
    } catch (error) {
      throw new GovernanceSdkError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Create a governance client from an existing AppClient
   *
   * Use this when you already have a Holochain connection and want
   * to add governance functionality.
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Governance client
   *
   * @example
   * ```typescript
   * const existingClient = await AppWebsocket.connect({ url: '...' });
   * const governance = MycelixGovernanceClient.fromClient(existingClient);
   * ```
   */
  static fromClient(
    client: AppClient,
    config: Partial<GovernanceClientConfig> = {}
  ): MycelixGovernanceClient {
    return new MycelixGovernanceClient(client, config);
  }

  // ============================================================================
  // Cross-hApp Integration
  // ============================================================================

  /**
   * Query governance participation for cross-hApp reputation
   *
   * Returns participation metrics that can be used by other hApps
   * to assess governance engagement.
   *
   * @param did - DID to query
   * @returns Participation score
   */
  async getParticipationScore(did: string): Promise<ParticipationScore> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: 'get_participation_score',
        payload: did,
      });
      return result as ParticipationScore;
    } catch (error) {
      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to get participation score: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Query governance data for Bridge integration
   *
   * Returns a summary of governance activity for a DID.
   *
   * @param did - DID to query
   * @returns Governance summary
   */
  async queryGovernance(did: string): Promise<Record<string, unknown>> {
    try {
      const result = await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: 'query_governance',
        payload: did,
      });
      return result as Record<string, unknown>;
    } catch (error) {
      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to query governance: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  /**
   * Emit a governance event for cross-hApp notification
   *
   * Events are broadcast via the Bridge for other hApps to consume.
   *
   * @param event - Governance event
   */
  async emitGovernanceEvent(event: GovernanceEvent): Promise<void> {
    try {
      await this.client.callZome({
        role_name: this.config.roleId,
        zome_name: this.config.zomeName,
        fn_name: 'emit_governance_event',
        payload: event,
      });
    } catch (error) {
      throw new GovernanceSdkError(
        'ZOME_ERROR',
        `Failed to emit governance event: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get comprehensive governance status for a DAO
   *
   * Returns DAO info, active proposals, and member count.
   *
   * @param daoId - DAO identifier
   * @returns DAO status summary
   */
  async getDaoStatus(daoId: string) {
    const [dao, activeProposals, members] = await Promise.all([
      this.dao.getDao(daoId),
      this.proposals.getActiveProposals(daoId),
      this.dao.getMembers(daoId),
    ]);

    return {
      dao,
      activeProposals,
      memberCount: members.length,
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
  async getProposalStatus(proposalId: string) {
    const [proposal, stats, quorum] = await Promise.all([
      this.proposals.getProposal(proposalId),
      this.voting.getVotingStats(proposalId),
      this.voting.checkQuorum(proposalId),
    ]);

    return {
      proposal,
      stats,
      quorum,
    };
  }

  /**
   * Get the underlying Holochain client
   *
   * Useful for advanced operations or debugging.
   */
  getClient(): AppClient {
    return this.client;
  }
}
