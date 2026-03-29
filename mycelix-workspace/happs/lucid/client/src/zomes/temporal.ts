// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Temporal Zome Client
 *
 * Belief versioning and time-travel queries.
 */

import type { AppClient, Record as HolochainRecord, Timestamp } from '@holochain/client';
import type {
  BeliefVersion,
  RecordVersionInput,
  GraphSnapshot,
  CreateSnapshotInput,
  BeliefAtTimeInput,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class TemporalZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'temporal'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ============================================================================
  // VERSION OPERATIONS
  // ============================================================================

  /** Record a new version of a thought */
  async recordVersion(input: RecordVersionInput): Promise<BeliefVersion> {
    const record = await this.callZome<HolochainRecord>('record_version', input);
    return decodeRecord<BeliefVersion>(record);
  }

  /** Get all versions of a thought (history) */
  async getThoughtHistory(thoughtId: string): Promise<BeliefVersion[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_history', thoughtId);
    return decodeRecords<BeliefVersion>(records);
  }

  /** Get belief at a specific timestamp */
  async getBeliefAtTime(thoughtId: string, timestamp: Timestamp): Promise<BeliefVersion | null> {
    const input: BeliefAtTimeInput = { thought_id: thoughtId, timestamp };
    const record = await this.callZome<HolochainRecord | null>('get_belief_at_time', input);
    return record ? decodeRecord<BeliefVersion>(record) : null;
  }

  // ============================================================================
  // SNAPSHOT OPERATIONS
  // ============================================================================

  /** Create a snapshot of the current knowledge graph */
  async createSnapshot(input: CreateSnapshotInput): Promise<GraphSnapshot> {
    const record = await this.callZome<HolochainRecord>('create_snapshot', input);
    return decodeRecord<GraphSnapshot>(record);
  }

  /** Get all snapshots */
  async getMySnapshots(): Promise<GraphSnapshot[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_snapshots', null);
    return decodeRecords<GraphSnapshot>(records);
  }
}
