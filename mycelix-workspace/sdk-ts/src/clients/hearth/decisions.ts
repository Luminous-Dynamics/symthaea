/**
 * Decisions Zome Client
 *
 * Family decisions, voting, tallying, and finalization for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/decisions
 */

import type {
  CreateDecisionInput,
  CastVoteInput,
  AmendVoteInput,
  FinalizeDecisionInput,
  CloseDecisionInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface DecisionsClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class DecisionsClient {
  private readonly zomeName = 'hearth_decisions';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<DecisionsClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Decision Management
  // ============================================================================

  async createDecision(input: CreateDecisionInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_decision', payload: input });
  }

  async getDecision(decisionHash: ActionHash): Promise<Record | null> {
    return this.client.callZome<Record | null>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_decision', payload: decisionHash });
  }

  async getHearthDecisions(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_decisions', payload: hearthHash });
  }

  async finalizeDecision(input: FinalizeDecisionInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'finalize_decision', payload: input });
  }

  async closeDecision(input: CloseDecisionInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'close_decision', payload: input });
  }

  async getDecisionOutcome(decisionHash: ActionHash): Promise<Record | null> {
    return this.client.callZome<Record | null>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_decision_outcome', payload: decisionHash });
  }

  // ============================================================================
  // Voting
  // ============================================================================

  async castVote(input: CastVoteInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'cast_vote', payload: input });
  }

  async amendVote(input: AmendVoteInput) {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'amend_vote', payload: input });
  }

  async tallyVotes(decisionHash: ActionHash): Promise<Array<[number, number]>> {
    return this.client.callZome<Array<[number, number]>>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'tally_votes', payload: decisionHash });
  }

  async getDecisionVotes(decisionHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_decision_votes', payload: decisionHash });
  }

  async getVoteHistory(decisionHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_vote_history', payload: decisionHash });
  }

  async getMyPendingVotes(hearthHash: ActionHash) {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_pending_votes', payload: hearthHash });
  }
}
