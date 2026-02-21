/**
 * Hearth Decisions SDK client.
 * Wraps zome calls to the hearth-decisions coordinator.
 */

import type { AppClient, Record as HolochainRecord, ActionHash } from '@holochain/client';
import type {
  CreateDecisionInput,
  CastVoteInput,
  FinalizeDecisionInput,
  CloseDecisionInput,
  AmendVoteInput,
} from './types';

const ROLE_NAME = 'hearth';
const ZOME_NAME = 'hearth_decisions';

export class DecisionsClient {
  constructor(private readonly client: AppClient, private readonly roleName = ROLE_NAME) {}

  async createDecision(input: CreateDecisionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'create_decision',
      payload: input,
    });
  }

  async castVote(input: CastVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'cast_vote',
      payload: input,
    });
  }

  async amendVote(input: AmendVoteInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'amend_vote',
      payload: input,
    });
  }

  async tallyVotes(decisionHash: ActionHash): Promise<Array<[number, number]>> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'tally_votes',
      payload: decisionHash,
    });
  }

  async finalizeDecision(input: FinalizeDecisionInput): Promise<HolochainRecord> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'finalize_decision',
      payload: input,
    });
  }

  async closeDecision(decisionHash: ActionHash): Promise<HolochainRecord> {
    const input: CloseDecisionInput = { decision_hash: decisionHash };
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'close_decision',
      payload: input,
    });
  }

  async getHearthDecisions(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_hearth_decisions',
      payload: hearthHash,
    });
  }

  async getDecisionVotes(decisionHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_decision_votes',
      payload: decisionHash,
    });
  }

  async getMyPendingVotes(hearthHash: ActionHash): Promise<HolochainRecord[]> {
    return this.client.callZome({
      role_name: this.roleName,
      zome_name: ZOME_NAME,
      fn_name: 'get_my_pending_votes',
      payload: hearthHash,
    });
  }
}
