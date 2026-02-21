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
  Decision,
  Vote,
  DecisionOutcome,
} from './types';
import type { ActionHash } from '../../generated/common';

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
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_decision', payload: input });
  }

  async getHearthDecisions(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_decisions', payload: hearthHash });
  }

  async finalizeDecision(decisionHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'finalize_decision', payload: decisionHash });
  }

  // ============================================================================
  // Voting
  // ============================================================================

  async castVote(input: CastVoteInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'cast_vote', payload: input });
  }

  async tallyVotes(decisionHash: ActionHash): Promise<DecisionOutcome> {
    return this.client.callZome<DecisionOutcome>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'tally_votes', payload: decisionHash });
  }

  async getMyPendingVotes(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_pending_votes', payload: hearthHash });
  }
}
