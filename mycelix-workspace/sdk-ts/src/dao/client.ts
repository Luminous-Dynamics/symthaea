// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix-DAO Client
 *
 * Unified client for all DAO operations.
 */

import { type AppClient, type AgentPubKey, type CellId } from '@holochain/client';

import { ConvictionClient } from './conviction';
import { DelegationClient } from './delegation';
import { EpistemicClient } from './epistemic';
import { GovernanceClient } from './governance';
import { IdentityClient } from './identity';
import { VotingClient } from './voting';

import type {
  Proposal,
  MemberProfile,
  EpistemicClassification,
  CrossHappReputation,
  TreasurySummary,
} from './types';

/**
 * Configuration options for DAOClient
 */
export interface DAOClientConfig {
  /** AppClient connected to Holochain */
  client: AppClient;
  /** Role name for the DAO (default: 'dao') */
  roleName?: string;
  /** Zome names override */
  zomeNames?: {
    governance?: string;
    epistemic?: string;
    identity?: string;
    oversight?: string;
    treasury?: string;
    bridge?: string;
  };
}

/**
 * Main client for interacting with Mycelix-DAO
 *
 * @example
 * ```typescript
 * import { DAOClient } from '@mycelix/sdk/dao';
 *
 * const client = new DAOClient({
 *   client: appClient,
 *   roleName: 'dao',
 * });
 *
 * // Create a proposal
 * const proposal = await client.governance.createProposal({
 *   proposal_type: 'TechnicalMIP',
 *   title: 'Add ZK Voting',
 *   description: 'Implement zero-knowledge voting...',
 *   classification: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
 *   tags: ['privacy', 'voting'],
 * });
 *
 * // Cast a vote
 * await client.voting.castVote({
 *   proposal: proposal.actionHash,
 *   choice: 'Yes',
 *   justification: 'This improves privacy significantly.',
 * });
 *
 * // Get vote tally
 * const tally = await client.voting.tallyVotes(proposal.actionHash);
 * console.log(`Approval: ${(tally.yes_votes / tally.total_weight * 100).toFixed(1)}%`);
 * ```
 */
export class DAOClient {
  private _client: AppClient;
  private _cellClient: AppClient | null = null;
  private _roleName: string;
  private _zomeNames: Required<NonNullable<DAOClientConfig['zomeNames']>>;

  /** Governance operations (proposals, configuration) */
  public readonly governance: GovernanceClient;

  /** Epistemic operations (classification, DKG, disputes) */
  public readonly epistemic: EpistemicClient;

  /** Identity operations (registration, verification) */
  public readonly identity: IdentityClient;

  /** Voting operations (all mechanisms) */
  public readonly voting: VotingClient;

  /** Delegation operations (liquid democracy) */
  public readonly delegation: DelegationClient;

  /** Conviction voting operations */
  public readonly conviction: ConvictionClient;

  constructor(config: DAOClientConfig) {
    this._client = config.client;
    this._roleName = config.roleName || 'dao';
    this._zomeNames = {
      governance: config.zomeNames?.governance || 'governance',
      epistemic: config.zomeNames?.epistemic || 'epistemic',
      identity: config.zomeNames?.identity || 'identity',
      oversight: config.zomeNames?.oversight || 'oversight',
      treasury: config.zomeNames?.treasury || 'treasury',
      bridge: config.zomeNames?.bridge || 'bridge',
    };

    // Initialize sub-clients
    this.governance = new GovernanceClient(this);
    this.epistemic = new EpistemicClient(this);
    this.identity = new IdentityClient(this);
    this.voting = new VotingClient(this);
    this.delegation = new DelegationClient(this);
    this.conviction = new ConvictionClient(this);
  }

  /**
   * Get the cell client for zome calls
   */
  async getCellClient(): Promise<AppClient> {
    if (this._cellClient) {
      return this._cellClient;
    }

    const appInfo = await this._client.appInfo();
    if (!appInfo) {
      throw new Error('Could not get app info');
    }

    const cell = appInfo.cell_info[this._roleName]?.[0];
    if (!cell) {
      throw new Error(`Could not find cell for role: ${this._roleName}`);
    }

    // Handle provisioned cell - cast to expected structure
    const cellInfo = cell as { provisioned?: { cell_id: unknown } };
    const cellId = cellInfo.provisioned?.cell_id ?? null;
    if (!cellId) {
      throw new Error('Cell is not provisioned');
    }

    this._cellClient = this._client;
    return this._cellClient;
  }

  /**
   * Make a zome call
   */
  async callZome<T>(zomeName: string, fnName: string, payload: unknown): Promise<T> {
    const appInfo = await this._client.appInfo();
    if (!appInfo) {
      throw new Error('Could not get app info');
    }

    const cell = appInfo.cell_info[this._roleName]?.[0];
    if (!cell || !('provisioned' in cell)) {
      throw new Error(`Could not find provisioned cell for role: ${this._roleName}`);
    }

    // Cast to expected structure for type safety
    const cellInfo = cell as { provisioned: { cell_id: CellId } };

    return this._client.callZome({
      cell_id: cellInfo.provisioned.cell_id,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    }) as Promise<T>;
  }

  /**
   * Get the zome name for a subsystem
   */
  getZomeName(subsystem: keyof typeof this._zomeNames): string {
    return this._zomeNames[subsystem];
  }

  /**
   * Get current agent's public key
   */
  async getMyPubKey(): Promise<AgentPubKey> {
    const appInfo = await this._client.appInfo();
    if (!appInfo) {
      throw new Error('Could not get app info');
    }
    return appInfo.agent_pub_key;
  }

  // =========================================================================
  // CONVENIENCE METHODS
  // =========================================================================

  /**
   * Get a comprehensive dashboard for the current user
   */
  async getMyDashboard(): Promise<{
    profile: MemberProfile | null;
    myProposals: Proposal[];
    activeProposals: Proposal[];
    myDelegations: any[];
    myConvictions: any[];
    crossHappReputation: CrossHappReputation;
  }> {
    const myPubKey = await this.getMyPubKey();

    const [profile, myProposals, activeProposals, myDelegations, crossHappReputation] = await Promise.all([
      this.identity.getMyProfile(),
      this.governance.getProposalsByAuthor(myPubKey),
      this.governance.getActiveProposals(),
      this.delegation.getMyDelegations(),
      this.callZome<CrossHappReputation>(
        this._zomeNames.bridge,
        'get_cross_happ_reputation',
        myPubKey
      ),
    ]);

    return {
      profile,
      myProposals,
      activeProposals,
      myDelegations,
      myConvictions: await this.conviction.getMyConvictions(),
      crossHappReputation,
    };
  }

  /**
   * Get treasury summary
   */
  async getTreasurySummary(): Promise<TreasurySummary> {
    return this.callZome<TreasurySummary>(
      this._zomeNames.treasury,
      'get_treasury_summary',
      null
    );
  }

  /**
   * Calculate vote weight for an agent
   */
  async calculateVoteWeight(agent?: AgentPubKey): Promise<number> {
    const pubKey = agent || await this.getMyPubKey();
    return this.callZome<number>(
      this._zomeNames.governance,
      'calculate_vote_weight',
      pubKey
    );
  }

  /**
   * Get suggested classification for a proposal type
   */
  async suggestClassification(proposalType: string): Promise<EpistemicClassification> {
    return this.epistemic.suggestClassification(proposalType);
  }
}
