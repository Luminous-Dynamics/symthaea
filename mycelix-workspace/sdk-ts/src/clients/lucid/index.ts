// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID hApp Client
 *
 * Complete TypeScript client for the LUCID (Living Unified Consciousness for
 * Integrated Deliberation) hApp, providing access to all 8 zomes:
 *
 * - **thoughts** - Core knowledge graph: thoughts, tags, domains, relationships, semantic search
 * - **bridge** - Symthaea-DHT bridge: Phi streaming, coherence analysis, federation
 * - **collective** - Collective intelligence: belief sharing, Phi-weighted consensus, trust
 * - **reasoning** - Contradiction detection, coherence reports, logical inference
 * - **temporal** - Version tracking, belief history, knowledge graph snapshots
 * - **temporalConsciousness** - Belief trajectories, consciousness evolution
 * - **sources** - Information sources and citation tracking
 * - **privacy** - Sharing policies, access control, ZK proof attestations
 *
 * @module @mycelix/sdk/clients/lucid
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { ThoughtsClient } from './thoughts';
import { LucidBridgeClient } from './bridge';
import { CollectiveClient } from './collective';
import { ReasoningClient } from './reasoning';
import { TemporalClient } from './temporal';
import { TemporalConsciousnessClient } from './temporal-consciousness';
import { SourcesClient } from './sources';
import { PrivacyClient } from './privacy';
import { LucidError } from './types';

// ============================================================================
// Client Configuration
// ============================================================================

export interface LucidClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface LucidConnectionOptions {
  url: string;
  timeout?: number;
  config?: LucidClientConfig;
}

const DEFAULT_CONFIG: Required<LucidClientConfig> = {
  roleName: 'lucid',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate LUCID Client
// ============================================================================

/**
 * Unified LUCID hApp Client
 *
 * @example
 * ```typescript
 * const lucid = await LucidClient.connect({ url: 'ws://localhost:8888' });
 *
 * // Create a thought
 * const thought = await lucid.thoughts.createThought({
 *   content: 'Consciousness emerges from integrated information',
 *   thought_type: 'Belief',
 *   confidence: 'High',
 *   domain: 'consciousness',
 *   tags: ['IIT', 'phi'],
 * });
 *
 * // Stream Phi score from Symthaea
 * await lucid.bridge.streamPhiScore({
 *   phi: 0.73,
 *   coherence: 0.85,
 *   active_thought_count: 42,
 *   engine_version: 'v0.5.0',
 * });
 *
 * // Get Phi-weighted consensus
 * const phiScores = await lucid.bridge.getAllLatestPhiScores();
 * const consensus = await lucid.collective.calculatePhiWeightedConsensus({
 *   belief_hash: beliefHash,
 *   phi_scores: phiScores,
 * });
 * ```
 */
export class LucidClient {
  public readonly thoughts: ThoughtsClient;
  public readonly bridge: LucidBridgeClient;
  public readonly collective: CollectiveClient;
  public readonly reasoning: ReasoningClient;
  public readonly temporal: TemporalClient;
  public readonly temporalConsciousness: TemporalConsciousnessClient;
  public readonly sources: SourcesClient;
  public readonly privacy: PrivacyClient;

  private readonly _client: AppClient;
  private readonly _config: Required<LucidClientConfig>;

  private constructor(client: AppClient, config: LucidClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.thoughts = new ThoughtsClient(client, baseConfig);
    this.bridge = new LucidBridgeClient(client, baseConfig);
    this.collective = new CollectiveClient(client, baseConfig);
    this.reasoning = new ReasoningClient(client, baseConfig);
    this.temporal = new TemporalClient(client, baseConfig);
    this.temporalConsciousness = new TemporalConsciousnessClient(client, baseConfig);
    this.sources = new SourcesClient(client, baseConfig);
    this.privacy = new PrivacyClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[lucid-sdk] Client initialized with 8 zome clients');
    }
  }

  /** Connect to a running Holochain conductor and create a LUCID client */
  static async connect(options: LucidConnectionOptions): Promise<LucidClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new LucidClient(client, options.config);
    } catch (error) {
      throw new LucidError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  /** Create a LUCID client from an existing AppClient */
  static fromClient(client: AppClient, config: LucidClientConfig = {}): LucidClient {
    return new LucidClient(client, config);
  }

  /** Get the underlying Holochain app client */
  getClient(): AppClient {
    return this._client;
  }

  /** Get the current agent's public key */
  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  /** Check if the connection is still alive */
  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/** Create a LucidClient from an existing AppClient */
export function createLucidClient(client: AppClient, config?: LucidClientConfig): LucidClient {
  return LucidClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { ThoughtsClient, type ThoughtsClientConfig } from './thoughts';
export { LucidBridgeClient, type BridgeClientConfig as LucidBridgeClientConfig } from './bridge';
export { CollectiveClient, type CollectiveClientConfig } from './collective';
export { ReasoningClient, type ReasoningClientConfig } from './reasoning';
export { TemporalClient, type TemporalClientConfig } from './temporal';
export { TemporalConsciousnessClient, type TemporalConsciousnessClientConfig } from './temporal-consciousness';
export { SourcesClient, type SourcesClientConfig } from './sources';
export { PrivacyClient, type PrivacyClientConfig } from './privacy';
export * from './types';
export default LucidClient;
