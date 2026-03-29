// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Temporal Zome Client
 *
 * Version tracking, belief history, and knowledge graph snapshots.
 *
 * @module @mycelix/sdk/clients/lucid/temporal
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  RecordVersionInput,
  BeliefAtTimeInput,
  CreateSnapshotInput,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface TemporalClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: TemporalClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Temporal zome
 *
 * Tracks thought versions over time and provides
 * point-in-time belief queries and snapshots.
 */
export class TemporalClient extends ZomeClient {
  protected readonly zomeName = 'lucid_temporal';

  constructor(client: AppClient, config: TemporalClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Versioning
  // ============================================================================

  /** Record a new version of a thought */
  async recordVersion(input: RecordVersionInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_version', input);
  }

  /** Get all versions of a thought */
  async getThoughtHistory(thoughtId: string): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_thought_history', thoughtId);
  }

  /** Get belief state at a specific timestamp */
  async getBeliefAtTime(input: BeliefAtTimeInput): Promise<HolochainRecord | null> {
    return this.callZomeOrNull<HolochainRecord>('get_belief_at_time', input);
  }

  // ============================================================================
  // Snapshots
  // ============================================================================

  /** Create a snapshot of the current knowledge graph */
  async createSnapshot(input: CreateSnapshotInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('create_snapshot', input);
  }

  /** Get all my knowledge graph snapshots */
  async getMySnapshots(): Promise<HolochainRecord[]> {
    return this.callZome<HolochainRecord[]>('get_my_snapshots', null);
  }
}
