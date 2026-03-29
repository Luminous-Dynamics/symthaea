// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Bridge Zome Client
 *
 * Cross-hApp integration with Identity, Knowledge DKG, and Symthaea.
 */

import type { AppClient, Record as HolochainRecord, AgentPubKey } from '@holochain/client';
import type {
  FederationRecord,
  ExternalReputation,
  CoherenceResult,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class BridgeZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'bridge'
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
  // FEDERATION OPERATIONS
  // ============================================================================

  /** Record federation of a thought to external hApp */
  async recordFederation(
    thoughtId: string,
    targetHapp: string,
    externalId?: string
  ): Promise<FederationRecord> {
    const record = await this.callZome<HolochainRecord>('record_federation', {
      thought_id: thoughtId,
      target_happ: targetHapp,
      external_id: externalId,
    });
    return decodeRecord<FederationRecord>(record);
  }

  /** Get federation records for a thought */
  async getThoughtFederations(thoughtId: string): Promise<FederationRecord[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_federations', thoughtId);
    return decodeRecords<FederationRecord>(records);
  }

  // ============================================================================
  // REPUTATION OPERATIONS
  // ============================================================================

  /** Cache external reputation score */
  async cacheReputation(
    agent: AgentPubKey,
    kVector: [number, number, number, number, number, number, number, number],
    trustScore: number,
    sourceHapp: string
  ): Promise<ExternalReputation> {
    const record = await this.callZome<HolochainRecord>('cache_reputation', {
      agent,
      k_vector: kVector,
      trust_score: trustScore,
      source_happ: sourceHapp,
    });
    return decodeRecord<ExternalReputation>(record);
  }

  /** Get cached reputation for an agent */
  async getCachedReputation(agent: AgentPubKey): Promise<ExternalReputation | null> {
    const record = await this.callZome<HolochainRecord | null>('get_cached_reputation', agent);
    return record ? decodeRecord<ExternalReputation>(record) : null;
  }

  // ============================================================================
  // SYMTHAEA INTEGRATION
  // ============================================================================

  /** Check coherence with Symthaea consciousness engine */
  async checkCoherenceWithSymthaea(thoughtIds: string[]): Promise<CoherenceResult> {
    return this.callZome<CoherenceResult>('check_coherence_with_symthaea', thoughtIds);
  }
}
