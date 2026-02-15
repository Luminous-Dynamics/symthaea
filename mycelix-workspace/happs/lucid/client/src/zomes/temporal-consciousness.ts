/**
 * Temporal Consciousness Zome Client
 *
 * Belief trajectory tracking and consciousness evolution analysis.
 * Persists consciousness snapshots to Holochain DHT for long-term tracking.
 */

import type { AppClient, Record as HolochainRecord } from '@holochain/client';
import type {
  BeliefTrajectoryRecord,
  TrajectoryAnalysis,
  ConsciousnessEvolution,
  RecordSnapshotInput,
  RecordEvolutionInput,
} from '../types';
import { decodeRecord } from '../utils';

export class TemporalConsciousnessZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'temporal_consciousness'
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
  // TRAJECTORY OPERATIONS
  // ============================================================================

  /** Record a snapshot of a belief's epistemic state */
  async recordBeliefSnapshot(input: RecordSnapshotInput): Promise<BeliefTrajectoryRecord> {
    const record = await this.callZome<HolochainRecord>('record_belief_snapshot', input);
    return decodeRecord<BeliefTrajectoryRecord>(record);
  }

  /** Get the full trajectory for a belief */
  async getBeliefTrajectory(thoughtId: string): Promise<BeliefTrajectoryRecord | null> {
    return await this.callZome<BeliefTrajectoryRecord | null>('get_belief_trajectory', thoughtId);
  }

  /** Analyze a belief's trajectory and get recommendations */
  async analyzeBeliefTrajectory(thoughtId: string): Promise<TrajectoryAnalysis> {
    return await this.callZome<TrajectoryAnalysis>('analyze_belief_trajectory', thoughtId);
  }

  // ============================================================================
  // EVOLUTION OPERATIONS
  // ============================================================================

  /** Record a consciousness evolution summary for a time period */
  async recordConsciousnessEvolution(input: RecordEvolutionInput): Promise<ConsciousnessEvolution> {
    const record = await this.callZome<HolochainRecord>('record_consciousness_evolution', input);
    return decodeRecord<ConsciousnessEvolution>(record);
  }

  /** Get all consciousness evolution records for the current agent */
  async getMyEvolutionHistory(): Promise<ConsciousnessEvolution[]> {
    return await this.callZome<ConsciousnessEvolution[]>('get_my_evolution_history', null);
  }
}
