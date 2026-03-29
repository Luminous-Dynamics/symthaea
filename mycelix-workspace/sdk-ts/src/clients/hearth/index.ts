// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Hearth hApp Client
 *
 * Complete TypeScript client for the Mycelix Hearth hApp providing access
 * to all 11 hearth zomes (10 domain + 1 bridge):
 *
 * - **kinship** - Membership, relationships, bond management
 * - **gratitude** - Appreciation expressions and circles
 * - **stories** - Family stories, traditions, collections
 * - **care** - Care schedules, swaps, meal plans
 * - **autonomy** - Graduated autonomy profiles and requests
 * - **emergency** - Emergency plans, alerts, safety check-ins
 * - **decisions** - Family decisions and voting
 * - **resources** - Shared resources, loans, budgets
 * - **milestones** - Life milestones and transitions
 * - **rhythms** - Daily rhythms and presence status
 * - **bridge** - Cross-zome dispatch, cross-cluster communication, governance gating
 *
 * @module @mycelix/sdk/clients/hearth
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { KinshipClient } from './kinship';
import { GratitudeClient } from './gratitude';
import { StoriesClient } from './stories';
import { HearthCareClient } from './care';
import { AutonomyClient } from './autonomy';
import { EmergencyClient } from './emergency';
import { DecisionsClient } from './decisions';
import { ResourcesClient } from './resources';
import { MilestonesClient } from './milestones';
import { RhythmsClient } from './rhythms';
import { BridgeClient } from './bridge';
import {
  HearthError,
  getSignalType,
  type HearthSignal,
  type HearthSignalType,
  type BridgeEventSignal,
} from './types';

// ============================================================================
// Client Configuration
// ============================================================================

export interface HearthClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface HearthConnectionOptions {
  url: string;
  timeout?: number;
  config?: HearthClientConfig;
}

const DEFAULT_CONFIG: Required<HearthClientConfig> = {
  roleName: 'hearth',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Hearth Client
// ============================================================================

/**
 * Unified Hearth hApp Client
 *
 * Provides access to all 11 hearth zomes through a single interface.
 *
 * @example
 * ```typescript
 * const hearth = await HearthClient.connect({ url: 'ws://localhost:8888' });
 *
 * // Create a hearth
 * const h = await hearth.kinship.createHearth({
 *   name: 'Stoltz Family',
 *   description: 'Our family hearth',
 *   hearth_type: 'Nuclear',
 * });
 *
 * // Express gratitude
 * await hearth.gratitude.expressGratitude({
 *   to_agent: memberKey,
 *   message: 'Thank you for cooking dinner!',
 *   gratitude_type: 'Appreciation',
 *   visibility: 'AllMembers',
 *   hearth_hash: hearthHash,
 * });
 *
 * // Raise an emergency alert
 * await hearth.emergency.raiseAlert({
 *   hearth_hash: hearthHash,
 *   alert_type: 'Medical',
 *   severity: 'High',
 *   message: 'Need immediate help',
 * });
 * ```
 */
export class HearthClient {
  public readonly kinship: KinshipClient;
  public readonly gratitude: GratitudeClient;
  public readonly stories: StoriesClient;
  public readonly care: HearthCareClient;
  public readonly autonomy: AutonomyClient;
  public readonly emergency: EmergencyClient;
  public readonly decisions: DecisionsClient;
  public readonly resources: ResourcesClient;
  public readonly milestones: MilestonesClient;
  public readonly rhythms: RhythmsClient;
  public readonly bridge: BridgeClient;

  private readonly _client: AppClient;
  private readonly _config: Required<HearthClientConfig>;

  private constructor(client: AppClient, config: HearthClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.kinship = new KinshipClient(client, baseConfig);
    this.gratitude = new GratitudeClient(client, baseConfig);
    this.stories = new StoriesClient(client, baseConfig);
    this.care = new HearthCareClient(client, baseConfig);
    this.autonomy = new AutonomyClient(client, baseConfig);
    this.emergency = new EmergencyClient(client, baseConfig);
    this.decisions = new DecisionsClient(client, baseConfig);
    this.resources = new ResourcesClient(client, baseConfig);
    this.milestones = new MilestonesClient(client, baseConfig);
    this.rhythms = new RhythmsClient(client, baseConfig);
    this.bridge = new BridgeClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[hearth-sdk] Client initialized with 11 zome clients');
    }
  }

  static async connect(options: HearthConnectionOptions): Promise<HearthClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new HearthClient(client, options.config);
    } catch (error) {
      throw new HearthError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  static fromClient(client: AppClient, config: HearthClientConfig = {}): HearthClient {
    return new HearthClient(client, config);
  }

  getClient(): AppClient {
    return this._client;
  }

  getAgentPubKey(): Uint8Array {
    return this._client.myPubKey;
  }

  async isConnected(): Promise<boolean> {
    try {
      const appInfo = await this._client.appInfo();
      return appInfo !== null;
    } catch {
      return false;
    }
  }

  // ============================================================================
  // Signal Handling
  // ============================================================================

  /**
   * Subscribe to HearthSignal events from all hearth zomes.
   *
   * @param callback - Invoked for each matching signal
   * @param filter - Optional set of signal types to listen for.
   *                 If omitted, all 27 HearthSignal types are received.
   * @returns Unsubscribe function
   *
   * @example
   * ```typescript
   * // Listen to all signals
   * const unsub = hearth.onSignal((signal) => {
   *   console.log('Signal:', getSignalType(signal), signal);
   * });
   *
   * // Listen to specific signal types only
   * const unsub2 = hearth.onSignal(
   *   (signal) => console.log('Emergency!', signal),
   *   ['EmergencyAlert', 'MemberDeparted'],
   * );
   *
   * // Later: unsubscribe
   * unsub();
   * ```
   */
  onSignal(
    callback: (signal: HearthSignal) => void,
    filter?: HearthSignalType[],
  ): () => void {
    const filterSet = filter ? new Set<string>(filter) : null;
    const debug = this._config.debug;

    const wsClient = this._client as AppWebsocket;
    return wsClient.on('signal', (appSignal: unknown) => {
      const sig = appSignal as { zome_name?: string; payload?: unknown };
      if (!sig.payload || typeof sig.payload !== 'object') return;

      // Only process signals from hearth zomes
      const zomeName = sig.zome_name ?? '';
      if (!zomeName.startsWith('hearth_')) return;

      const payload = sig.payload as HearthSignal;
      const signalType = getSignalType(payload);

      if (filterSet && !filterSet.has(signalType)) return;

      if (debug) {
        console.log(`[hearth-sdk] Signal: ${signalType} from ${zomeName}`);
      }

      callback(payload);
    });
  }

  /**
   * Subscribe to BridgeEventSignal events from the bridge zome.
   *
   * Bridge signals use a separate struct (not part of HearthSignal enum)
   * for cross-domain event broadcasting.
   *
   * @param callback - Invoked for each bridge event signal
   * @returns Unsubscribe function
   */
  onBridgeSignal(callback: (signal: BridgeEventSignal) => void): () => void {
    const debug = this._config.debug;

    const wsClient = this._client as AppWebsocket;
    return wsClient.on('signal', (appSignal: unknown) => {
      const sig = appSignal as { zome_name?: string; payload?: unknown };
      if (!sig.payload || typeof sig.payload !== 'object') return;

      if (sig.zome_name !== 'hearth_bridge') return;

      const payload = sig.payload as Record<string, unknown>;
      if (!('signal_type' in payload)) return;

      if (debug) {
        console.log(`[hearth-sdk] Bridge signal: ${payload.signal_type}`);
      }

      callback(payload as unknown as BridgeEventSignal);
    });
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a HearthClient from an existing AppClient
 */
export function createHearthClient(client: AppClient, config?: HearthClientConfig): HearthClient {
  return HearthClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { KinshipClient, type KinshipClientConfig } from './kinship';
export { GratitudeClient, type GratitudeClientConfig } from './gratitude';
export { StoriesClient, type StoriesClientConfig } from './stories';
export { HearthCareClient, type HearthCareClientConfig } from './care';
export { AutonomyClient, type AutonomyClientConfig } from './autonomy';
export { EmergencyClient, type EmergencyClientConfig } from './emergency';
export { DecisionsClient, type DecisionsClientConfig } from './decisions';
export { ResourcesClient, type ResourcesClientConfig } from './resources';
export { MilestonesClient, type MilestonesClientConfig } from './milestones';
export { RhythmsClient, type RhythmsClientConfig } from './rhythms';
export { BridgeClient, type BridgeClientConfig } from './bridge';
export * from './types';
export default HearthClient;
