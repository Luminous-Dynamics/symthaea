/**
 * Governance Client
 *
 * Proposal creation and management.
 */

import { type ActionHash, type AgentPubKey, type Record as HolochainRecord } from '@holochain/client';

import type { DAOClient } from './client';
import type {
  Proposal,
  ProposalType,
  ProposalStatus,
  VotingMechanism,
  CreateProposalInput,
} from './types';

/**
 * Client for governance operations
 */
export class GovernanceClient {
  constructor(private client: DAOClient) {}

  private get zome() {
    return this.client.getZomeName('governance');
  }

  /**
   * Create a new proposal
   *
   * @example
   * ```typescript
   * const proposal = await governance.createProposal({
   *   proposal_type: 'TechnicalMIP',
   *   title: 'Implement Feature X',
   *   description: 'Detailed description of Feature X...',
   *   classification: { e: 'E2', n: 'N2', m: 'M3', override_from_default: false },
   *   tags: ['feature', 'core'],
   * });
   * ```
   */
  async createProposal(input: CreateProposalInput): Promise<{
    record: HolochainRecord;
    actionHash: ActionHash;
    proposal: Proposal;
  }> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'create_proposal',
      input
    );

    const actionHash = (record as any).signed_action.hashed.hash;
    const proposal = this.decodeProposal(record);

    return { record, actionHash, proposal };
  }

  /**
   * Get a proposal by its action hash
   */
  async getProposal(actionHash: ActionHash): Promise<Proposal | null> {
    const record = await this.client.callZome<HolochainRecord | null>(
      this.zome,
      'get_proposal',
      actionHash
    );

    if (!record) return null;
    return this.decodeProposal(record);
  }

  /**
   * Get all proposals
   */
  async getAllProposals(): Promise<Proposal[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_all_proposals',
      null
    );

    return records.map(r => this.decodeProposal(r));
  }

  /**
   * Get active proposals (Review or Voting status)
   */
  async getActiveProposals(): Promise<Proposal[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_active_proposals',
      null
    );

    return records.map(r => this.decodeProposal(r));
  }

  /**
   * Get proposals by type
   */
  async getProposalsByType(proposalType: ProposalType): Promise<Proposal[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_proposals_by_type',
      proposalType
    );

    return records.map(r => this.decodeProposal(r));
  }

  /**
   * Get proposals by author
   */
  async getProposalsByAuthor(author: AgentPubKey): Promise<Proposal[]> {
    const records = await this.client.callZome<HolochainRecord[]>(
      this.zome,
      'get_proposals_by_author',
      author
    );

    return records.map(r => this.decodeProposal(r));
  }

  /**
   * Update proposal status
   */
  async updateProposalStatus(
    proposalHash: ActionHash,
    newStatus: ProposalStatus
  ): Promise<Proposal> {
    const record = await this.client.callZome<HolochainRecord>(
      this.zome,
      'update_proposal_status',
      { proposal_hash: proposalHash, new_status: newStatus }
    );

    return this.decodeProposal(record);
  }

  /**
   * Get governance configuration
   */
  async getConfig(): Promise<any> {
    return this.client.callZome(this.zome, 'get_governance_config', null);
  }

  /**
   * Get suggested voting mechanism for a proposal type
   */
  async getSuggestedMechanism(proposalType: ProposalType): Promise<VotingMechanism> {
    return this.client.callZome<VotingMechanism>(
      this.zome,
      'get_suggested_mechanism',
      proposalType
    );
  }

  /**
   * Decode a proposal from a record
   */
  private decodeProposal(record: HolochainRecord): Proposal {
    // In real implementation, decode from msgpack
    return (record as any).entry.Present.entry as Proposal;
  }
}
