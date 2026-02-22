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
  DigestEpochInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

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

  async createRhythm(input: CreateRhythmInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_rhythm', payload: input });
  }

  async getHearthRhythms(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_rhythms', payload: hearthHash });
  }

  // ============================================================================
  // Occurrences
  // ============================================================================

  async logOccurrence(input: LogOccurrenceInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'log_occurrence', payload: input });
  }

  async getRhythmOccurrences(rhythmHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_rhythm_occurrences', payload: rhythmHash });
  }

  // ============================================================================
  // Presence
  // ============================================================================

  async setPresence(input: SetPresenceInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'set_presence', payload: input });
  }

  async getHearthPresence(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_presence', payload: hearthHash });
  }

  // ============================================================================
  // Digests
  // ============================================================================

  async createRhythmDigest(input: DigestEpochInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_rhythm_digest', payload: input });
  }
}
