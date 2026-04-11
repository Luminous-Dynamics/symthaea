// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Mail Holochain Client
 * Unified TypeScript client for all Holochain zomes
 */

import type { AppClient, AgentPubKey, ActionHash } from '@holochain/client';

// Zome clients
import { MessagesZomeClient } from './zomes/messages';
import { TrustZomeClient } from './zomes/trust';
import { SyncZomeClient } from './zomes/sync';
import { FederationZomeClient } from './zomes/federation';

// Signal handling
import { SignalHub, createSignalEmitter } from './signals/SignalHub';

// Types
export * from './types';
export { MessagesZomeClient } from './zomes/messages';
export { TrustZomeClient } from './zomes/trust';
export { SyncZomeClient } from './zomes/sync';
export { FederationZomeClient } from './zomes/federation';
export { SignalHub, createSignalEmitter } from './signals/SignalHub';

/**
 * Unified Mycelix Mail Holochain Client
 */
export class MycelixMailClient {
  public readonly messages: MessagesZomeClient;
  public readonly trust: TrustZomeClient;
  public readonly sync: SyncZomeClient;
  public readonly federation: FederationZomeClient;
  public readonly signals: SignalHub;

  private _myAgentPubKey: AgentPubKey | null = null;

  constructor(
    private client: AppClient,
    private roleName: string = 'mycelix_mail'
  ) {
    this.messages = new MessagesZomeClient(client, roleName);
    this.trust = new TrustZomeClient(client, roleName);
    this.sync = new SyncZomeClient(client, roleName);
    this.federation = new FederationZomeClient(client, roleName);
    this.signals = new SignalHub(client);
  }

  /**
   * Get the current agent's public key
   */
  get myAgentPubKey(): AgentPubKey | null {
    return this._myAgentPubKey;
  }

  /**
   * Initialize the client (call after connecting)
   */
  async initialize(): Promise<void> {
    // Get agent info
    const appInfo = await this.client.appInfo();
    if (appInfo?.agent_pub_key) {
      this._myAgentPubKey = appInfo.agent_pub_key;
    }

    // Initialize sync state
    await this.sync.initSyncState();

    // Process any offline queue
    const processed = await this.sync.processOfflineQueue();
    if (processed > 0) {
      console.log(`Processed ${processed} offline operations`);
    }
  }

  /**
   * Create event emitter for UI frameworks (React, Vue, Svelte)
   */
  createEventEmitter() {
    return createSignalEmitter(this.signals);
  }

  // ==================== CONVENIENCE METHODS ====================

  /**
   * Send email with automatic trust checking
   */
  async sendEmailWithTrustCheck(
    recipient: AgentPubKey,
    subject: string,
    body: string,
    options?: {
      requireTrustLevel?: number;
      attachments?: unknown[];
      priority?: 'Low' | 'Normal' | 'High' | 'Urgent';
    }
  ): Promise<{ sent: boolean; email_hash?: ActionHash; trust_score?: number; reason?: string }> {
    // Check trust score
    const trustScore = await this.trust.getTrustScore({
      subject: recipient,
      category: 'Communication',
    });

    const requiredTrust = options?.requireTrustLevel ?? 0.3;

    if (trustScore.combined_score < requiredTrust) {
      return {
        sent: false,
        trust_score: trustScore.combined_score,
        reason: `Recipient trust score (${trustScore.combined_score.toFixed(2)}) below threshold (${requiredTrust})`,
      };
    }

    // Send the email
    const result = await this.messages.sendEmail({
      recipient,
      subject,
      body,
      attachments: options?.attachments as any,
      priority: options?.priority,
    });

    return {
      sent: true,
      email_hash: result.email_hash,
      trust_score: trustScore.combined_score,
    };
  }

  /**
   * Get inbox with trust scores for each sender
   */
  async getInboxWithTrust(limit?: number): Promise<Array<{
    email: Awaited<ReturnType<MessagesZomeClient['getInbox']>>[0];
    sender_trust: number;
    is_trusted: boolean;
  }>> {
    const emails = await this.messages.getInbox(limit);

    // Get unique senders
    const senders = [...new Set(emails.map(e => e.sender.toString()))];

    // Batch get trust scores
    const trustScores = await this.trust.getBatchTrustScores(
      senders.map(s => s as unknown as AgentPubKey),
      'Communication'
    );

    return emails.map(email => {
      const trustScore = trustScores.get(email.sender.toString());
      return {
        email,
        sender_trust: trustScore?.combined_score ?? 0,
        is_trusted: (trustScore?.combined_score ?? 0) >= 0.3,
      };
    });
  }

  /**
   * Sync with all known peers
   */
  async syncWithAllPeers(): Promise<{
    total_sent: number;
    total_received: number;
    total_conflicts: number;
    peers_synced: number;
  }> {
    const peers = await this.sync.getSyncPeers();
    let totalSent = 0;
    let totalReceived = 0;
    let totalConflicts = 0;
    let peersSynced = 0;

    for (const peer of peers) {
      if (!peer.is_reachable) continue;

      try {
        const result = await this.sync.syncWithPeer(peer.agent);
        totalSent += result.operations_sent;
        totalReceived += result.operations_received;
        totalConflicts += result.conflicts_detected;
        peersSynced++;
      } catch (error) {
        console.warn(`Failed to sync with peer ${peer.agent}:`, error);
      }
    }

    return {
      total_sent: totalSent,
      total_received: totalReceived,
      total_conflicts: totalConflicts,
      peers_synced: peersSynced,
    };
  }

  /**
   * Get comprehensive mailbox status
   */
  async getMailboxStatus(): Promise<{
    inbox_count: number;
    unread_count: number;
    sent_count: number;
    draft_count: number;
    sync_status: {
      is_online: boolean;
      pending_ops: number;
      last_sync: number | null;
    };
    trust_status: {
      attestations_made: number;
      total_stake: number;
    };
    federation_status: {
      known_networks: number;
      pending_envelopes: number;
    };
  }> {
    const [stats, syncState, attestations, stake, networks, pending] = await Promise.all([
      this.messages.getStats(),
      this.sync.getSyncState(),
      this.trust.getMyAttestations(),
      this.trust.getMyStake(),
      this.federation.getAllNetworks(),
      this.federation.getPendingOutbound(),
    ]);

    return {
      inbox_count: stats.total_emails,
      unread_count: stats.unread_count,
      sent_count: stats.sent_count,
      draft_count: stats.draft_count,
      sync_status: {
        is_online: syncState?.is_online ?? false,
        pending_ops: syncState?.pending_ops ?? 0,
        last_sync: syncState?.last_sync ? Number(syncState.last_sync) : null,
      },
      trust_status: {
        attestations_made: attestations.length,
        total_stake: stake.total,
      },
      federation_status: {
        known_networks: networks.length,
        pending_envelopes: pending.length,
      },
    };
  }

  /**
   * Handle going offline
   */
  async goOffline(): Promise<void> {
    await this.sync.setOnlineStatus(false);
  }

  /**
   * Handle coming online
   */
  async goOnline(): Promise<void> {
    await this.sync.setOnlineStatus(true);
    await this.sync.processOfflineQueue();
  }

  // ==================== REACT HOOKS HELPERS ====================

  /**
   * Create a subscription that cleans up automatically
   * Usage: useEffect(() => client.createSubscription('EmailReceived', handler), [])
   */
  createSubscription<T extends import('./types').MycelixSignal['type']>(
    type: T,
    handler: (signal: Extract<import('./types').MycelixSignal, { type: T }>) => void
  ): () => void {
    return this.signals.on(type, handler);
  }
}

/**
 * Create a new Mycelix Mail client
 */
export function createMycelixMailClient(
  client: AppClient,
  roleName?: string
): MycelixMailClient {
  return new MycelixMailClient(client, roleName);
}

/**
 * Default export
 */
export default MycelixMailClient;

// ==================== ADDITIONAL MODULES ====================

// Bootstrap & Enhanced Client Initialization
export * from './bootstrap';

// React Components
export * from './components';

// Vue.js Composables
export * from './vue';

// PWA & Service Worker
export * from './pwa';

// Accessibility
export * from './a11y';

// WebRTC Signaling
export * from './webrtc';

// Theme System
export * from './themes';

// Plugin SDK
export * from './plugins';

// Email Templates
export * from './templates';

// Trust Network Visualization
export * from './visualization';

// Mobile Optimizations
export * from './mobile';

// Version
export const CLIENT_VERSION = '1.0.0';
