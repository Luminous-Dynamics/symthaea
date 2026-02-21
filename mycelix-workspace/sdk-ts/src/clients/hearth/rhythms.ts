/**
 * Rhythms Zome Client
 *
 * Daily rhythms, occurrence logging, and presence status tracking
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/rhythms
 */

import type {
  CreateRhythmInput,
  LogOccurrenceInput,
  SetPresenceInput,
} from './types';
import type { ActionHash } from '../../generated/common';

export interface RhythmsClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class RhythmsClient {
  private readonly zomeName = 'hearth_rhythms';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<RhythmsClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Rhythm Management
  // ============================================================================

  async createRhythm(input: CreateRhythmInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_rhythm', payload: input });
  }

  async getTodayRhythms(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_today_rhythms', payload: hearthHash });
  }

  async getRhythmConsistency(rhythmHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_rhythm_consistency', payload: rhythmHash });
  }

  // ============================================================================
  // Occurrences
  // ============================================================================

  async logOccurrence(input: LogOccurrenceInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'log_occurrence', payload: input });
  }

  async skipOccurrence(rhythmHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'skip_occurrence', payload: rhythmHash });
  }

  // ============================================================================
  // Presence
  // ============================================================================

  async setPresence(input: SetPresenceInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'set_presence', payload: input });
  }

  async getHearthPresence(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_presence', payload: hearthHash });
  }
}
