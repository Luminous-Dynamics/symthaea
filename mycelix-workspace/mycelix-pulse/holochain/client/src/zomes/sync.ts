// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Sync Zome Client (CRDT)
 * Offline-first synchronization with conflict resolution
 */

import type { AppClient, AgentPubKey } from '@holochain/client';
import type {
  SyncState,
  VectorClock,
  SyncResult,
  SyncConflict,
  ConflictResolution,
  SyncOpType,
} from '../types';

export class SyncZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'mycelix_mail',
    private zomeName: string = 'mail_sync'
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

  // ==================== SYNC STATE ====================

  /**
   * Initialize sync state for current agent
   */
  async initSyncState(): Promise<void> {
    await this.callZome('init_sync_state', null);
  }

  /**
   * Get current sync state
   */
  async getSyncState(): Promise<SyncState | null> {
    return this.callZome('get_sync_state', null);
  }

  /**
   * Set online/offline status
   */
  async setOnlineStatus(isOnline: boolean): Promise<void> {
    await this.callZome('set_online_status', isOnline);
  }

  /**
   * Get current vector clock
   */
  async getVectorClock(): Promise<VectorClock> {
    const state = await this.getSyncState();
    return state?.vector_clock ?? { clocks: [] };
  }

  // ==================== OPERATIONS ====================

  /**
   * Record a sync operation (called after local changes)
   */
  async recordOperation(
    opType: SyncOpType,
    target?: Uint8Array,
    payload?: unknown,
    dependencies?: string[]
  ): Promise<void> {
    await this.callZome('record_operation', {
      op_type: opType,
      target: target ?? null,
      payload: payload ?? null,
      dependencies: dependencies ?? [],
    });
  }

  /**
   * Get operations since a vector clock
   */
  async getOperationsSince(
    sinceClock?: VectorClock,
    limit?: number
  ): Promise<Array<{
    op_id: string;
    op_type: SyncOpType;
    timestamp: number;
    applied: boolean;
  }>> {
    return this.callZome('get_operations_since', {
      since_clock: sinceClock ?? null,
      limit: limit ?? 50,
    });
  }

  // ==================== PEER SYNC ====================

  /**
   * Sync with a peer
   */
  async syncWithPeer(peer: AgentPubKey, sinceCheckpoint?: string): Promise<SyncResult> {
    return this.callZome('sync_with_peer', {
      peer,
      since_checkpoint: sinceCheckpoint ?? null,
    });
  }

  /**
   * Request sync from peer (via remote signal)
   */
  async requestSyncFromPeer(peer: AgentPubKey): Promise<void> {
    await this.callZome('request_sync_from_peer', peer);
  }

  /**
   * Apply remote operations from peer
   */
  async applyRemoteOperations(
    operations: unknown[],
    fromPeer: AgentPubKey
  ): Promise<{
    applied: number;
    conflicts: number;
    new_clock: VectorClock;
  }> {
    return this.callZome('apply_remote_operations', {
      operations,
      from_peer: fromPeer,
    });
  }

  /**
   * Register a sync peer
   */
  async registerSyncPeer(peer: AgentPubKey): Promise<void> {
    await this.callZome('register_sync_peer', peer);
  }

  /**
   * Get all sync peers
   */
  async getSyncPeers(): Promise<Array<{
    agent: AgentPubKey;
    last_sync: number | null;
    reliability: number;
    is_reachable: boolean;
  }>> {
    return this.callZome('get_sync_peers', null);
  }

  // ==================== CHECKPOINTS ====================

  /**
   * Create a sync checkpoint
   */
  async createCheckpoint(): Promise<string> {
    return this.callZome('create_checkpoint', null);
  }

  /**
   * Get latest checkpoint
   */
  async getLatestCheckpoint(): Promise<{
    checkpoint_id: string;
    vector_clock: VectorClock;
    timestamp: number;
  } | null> {
    return this.callZome('get_latest_checkpoint', null);
  }

  // ==================== OFFLINE QUEUE ====================

  /**
   * Process offline queue (call when coming online)
   */
  async processOfflineQueue(): Promise<number> {
    return this.callZome('process_offline_queue', null);
  }

  /**
   * Get offline queue size
   */
  async getOfflineQueueSize(): Promise<number> {
    const state = await this.getSyncState();
    return state?.pending_ops ?? 0;
  }

  // ==================== CONFLICTS ====================

  /**
   * Get pending conflicts
   */
  async getPendingConflicts(): Promise<SyncConflict[]> {
    return this.callZome('get_pending_conflicts', null);
  }

  /**
   * Resolve conflict manually
   */
  async resolveConflict(
    conflictId: string,
    resolution: 'keep_local' | 'keep_remote' | 'merge',
    mergedValue?: unknown
  ): Promise<void> {
    await this.callZome('resolve_conflict', {
      conflict_id: conflictId,
      resolution,
      merged_value: mergedValue ?? null,
    });
  }

  /**
   * Get conflict history
   */
  async getConflictHistory(limit?: number): Promise<Array<{
    conflict_id: string;
    conflict_type: string;
    resolution: ConflictResolution;
    resolved_at: number;
  }>> {
    return this.callZome('get_conflict_history', { limit: limit ?? 50 });
  }

  // ==================== CRDT MERGE ====================

  /**
   * Merge email states (used internally for CRDT resolution)
   */
  async mergeEmailStates(local: unknown, remote: unknown): Promise<unknown> {
    return this.callZome('merge_email_states', { local, remote });
  }
}
