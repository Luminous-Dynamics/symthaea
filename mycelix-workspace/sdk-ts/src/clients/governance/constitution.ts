/**
 * Constitution Zome Client
 *
 * Handles charter management, constitutional amendments, and governance
 * parameters for the Governance hApp.
 *
 * @module @mycelix/sdk/clients/governance/constitution
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { AppClient, Record as HolochainRecord } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

export type AmendmentType =
  | 'AddArticle'
  | 'ModifyArticle'
  | 'RemoveArticle'
  | 'AddRight'
  | 'ModifyRight'
  | 'RemoveRight'
  | 'ModifyPreamble'
  | 'ModifyProcess';

export type AmendmentStatus =
  | 'Draft'
  | 'Deliberation'
  | 'Voting'
  | 'Ratified'
  | 'Rejected'
  | 'Withdrawn';

export type ParameterType =
  | 'Integer'
  | 'Float'
  | 'Percentage'
  | 'Duration'
  | 'Boolean'
  | 'String';

export interface Charter {
  id: string;
  version: number;
  preamble: string;
  articles: string;
  rights: string[];
  amendmentProcess: string;
  adopted: number;
  lastAmended?: number;
}

export interface Amendment {
  id: string;
  charterVersion: number;
  amendmentType: AmendmentType;
  article?: string;
  originalText?: string;
  newText: string;
  rationale: string;
  proposer: string;
  proposalId: string;
  status: AmendmentStatus;
  created: number;
  ratified?: number;
}

export interface GovernanceParameter {
  name: string;
  value: string;
  valueType: ParameterType;
  description: string;
  minValue?: string;
  maxValue?: string;
  updated: number;
  changedByProposal?: string;
}

export interface ProposeAmendmentInput {
  amendmentType: AmendmentType;
  article?: string;
  originalText?: string;
  newText: string;
  rationale: string;
  proposerDid: string;
  proposalId: string;
}

export interface SetParameterInput {
  name: string;
  value: string;
  valueType: ParameterType;
  description: string;
  minValue?: string;
  maxValue?: string;
  proposalId?: string;
}

export interface UpdateParameterInput {
  parameter: string;
  value: string;
  proposalId?: string;
}

// ============================================================================
// Configuration
// ============================================================================

export interface ConstitutionClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: ConstitutionClientConfig = {
  roleName: 'governance',
};

// ============================================================================
// Client
// ============================================================================

/**
 * Client for Constitution operations
 *
 * Manages the constitutional charter, amendments, and governance parameters.
 *
 * @example
 * ```typescript
 * const constitution = new ConstitutionClient(appClient);
 *
 * // Get current charter
 * const charter = await constitution.getCurrentCharter();
 *
 * // Propose an amendment
 * const amendment = await constitution.proposeAmendment({
 *   amendmentType: 'AddRight',
 *   newText: 'Right to data sovereignty',
 *   rationale: 'Members should control their own data',
 *   proposerDid: 'did:mycelix:uhCAk...',
 *   proposalId: 'proposal:123',
 * });
 *
 * // Set a governance parameter
 * await constitution.setParameter({
 *   name: 'quorum_threshold',
 *   value: '0.5',
 *   valueType: 'Float',
 *   description: 'Minimum quorum for standard proposals',
 * });
 * ```
 */
export class ConstitutionClient extends ZomeClient {
  protected readonly zomeName = 'constitution';

  constructor(client: AppClient, config: ConstitutionClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Charter Operations
  // ============================================================================

  async createCharter(charter: Charter): Promise<Charter> {
    const record = await this.callZomeOnce<HolochainRecord>('create_charter', {
      id: charter.id,
      version: charter.version,
      preamble: charter.preamble,
      articles: charter.articles,
      rights: charter.rights,
      amendment_process: charter.amendmentProcess,
      adopted: charter.adopted,
      last_amended: charter.lastAmended,
    });
    return this.mapCharter(record);
  }

  async getCurrentCharter(): Promise<Charter | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_current_charter', null);
    if (!record) return null;
    return this.mapCharter(record);
  }

  // ============================================================================
  // Amendment Operations
  // ============================================================================

  async proposeAmendment(input: ProposeAmendmentInput): Promise<Amendment> {
    const record = await this.callZomeOnce<HolochainRecord>('propose_amendment', {
      amendment_type: input.amendmentType,
      article: input.article,
      original_text: input.originalText,
      new_text: input.newText,
      rationale: input.rationale,
      proposer_did: input.proposerDid,
      proposal_id: input.proposalId,
    });
    return this.mapAmendment(record);
  }

  async ratifyAmendment(amendmentId: string): Promise<Amendment> {
    const record = await this.callZomeOnce<HolochainRecord>('ratify_amendment', amendmentId);
    return this.mapAmendment(record);
  }

  // ============================================================================
  // Parameter Operations
  // ============================================================================

  async setParameter(input: SetParameterInput): Promise<GovernanceParameter> {
    const record = await this.callZomeOnce<HolochainRecord>('set_parameter', {
      name: input.name,
      value: input.value,
      value_type: input.valueType,
      description: input.description,
      min_value: input.minValue,
      max_value: input.maxValue,
      proposal_id: input.proposalId,
    });
    return this.mapParameter(record);
  }

  async updateParameter(input: UpdateParameterInput): Promise<GovernanceParameter> {
    const record = await this.callZomeOnce<HolochainRecord>('update_parameter', {
      parameter: input.parameter,
      value: input.value,
      proposal_id: input.proposalId,
    });
    return this.mapParameter(record);
  }

  async getParameter(name: string): Promise<GovernanceParameter | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_parameter', name);
    if (!record) return null;
    return this.mapParameter(record);
  }

  async listParameters(): Promise<GovernanceParameter[]> {
    const records = await this.callZome<HolochainRecord[]>('list_parameters', null);
    return records.map((r) => this.mapParameter(r));
  }

  // ============================================================================
  // Mappers
  // ============================================================================

  private mapCharter(record: HolochainRecord): Charter {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      version: entry.version,
      preamble: entry.preamble,
      articles: entry.articles,
      rights: entry.rights,
      amendmentProcess: entry.amendment_process,
      adopted: entry.adopted,
      lastAmended: entry.last_amended,
    };
  }

  private mapAmendment(record: HolochainRecord): Amendment {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      id: entry.id,
      charterVersion: entry.charter_version,
      amendmentType: entry.amendment_type,
      article: entry.article,
      originalText: entry.original_text,
      newText: entry.new_text,
      rationale: entry.rationale,
      proposer: entry.proposer,
      proposalId: entry.proposal_id,
      status: entry.status,
      created: entry.created,
      ratified: entry.ratified,
    };
  }

  private mapParameter(record: HolochainRecord): GovernanceParameter {
    const entry = (record as any).entry?.Present?.entry ?? (record as any).entry ?? {};
    return {
      name: entry.name,
      value: entry.value,
      valueType: entry.value_type,
      description: entry.description,
      minValue: entry.min_value,
      maxValue: entry.max_value,
      updated: entry.updated,
      changedByProposal: entry.changed_by_proposal,
    };
  }
}
