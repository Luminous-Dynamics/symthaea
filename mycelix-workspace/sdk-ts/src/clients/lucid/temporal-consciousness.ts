// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Temporal Consciousness Zome Client
 *
 * Belief trajectory analysis and consciousness evolution tracking.
 *
 * @module @mycelix/sdk/clients/lucid/temporal-consciousness
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  RecordSnapshotInput,
  BeliefTrajectory,
  TrajectoryAnalysis,
  RecordEvolutionInput,
  ConsciousnessEvolution,
} from './types';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface TemporalConsciousnessClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: TemporalConsciousnessClientConfig = {
  roleName: 'lucid',
};

/**
 * Client for the LUCID Temporal Consciousness zome
 *
 * Tracks belief trajectories (stability, direction, phase) and
 * records consciousness evolution summaries over time.
 */
export class TemporalConsciousnessClient extends ZomeClient {
  protected readonly zomeName = 'lucid_temporal_consciousness';

  constructor(client: AppClient, config: TemporalConsciousnessClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Belief Snapshots & Trajectories
  // ============================================================================

  /** Record a snapshot of a belief's current state */
  async recordBeliefSnapshot(input: RecordSnapshotInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_belief_snapshot', input);
  }

  /** Get the trajectory for a thought (all snapshots over time) */
  async getBeliefTrajectory(thoughtId: string): Promise<BeliefTrajectory | null> {
    return this.callZomeOrNull<BeliefTrajectory>('get_belief_trajectory', thoughtId);
  }

  /** Analyze a belief's trajectory (stability, direction, phase) */
  async analyzeBeliefTrajectory(thoughtId: string): Promise<TrajectoryAnalysis> {
    return this.callZome<TrajectoryAnalysis>('analyze_belief_trajectory', thoughtId);
  }

  // ============================================================================
  // Consciousness Evolution
  // ============================================================================

  /** Record a consciousness evolution summary for a time period */
  async recordConsciousnessEvolution(input: RecordEvolutionInput): Promise<HolochainRecord> {
    return this.callZomeOnce<HolochainRecord>('record_consciousness_evolution', input);
  }

  /** Get consciousness evolution history */
  async getMyEvolutionHistory(): Promise<ConsciousnessEvolution[]> {
    return this.callZome<ConsciousnessEvolution[]>('get_my_evolution_history', null);
  }
}
