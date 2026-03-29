// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Emergency hApp Client (Phase 3)
 *
 * Complete TypeScript client for the Mycelix Emergency hApp providing access
 * to all 3 emergency zomes:
 *
 * - **incidents** - Disaster declaration, status tracking, lifecycle
 * - **coordination** - Team formation, sitreps, check-ins
 * - **shelters** - Shelter registration, occupancy, search
 *
 * @module @mycelix/sdk/clients/emergency
 */

import { type AppClient, AppWebsocket } from '@holochain/client';

import { CoordinationClient } from './coordination';
import { IncidentsClient } from './incidents';
import { SheltersClient } from './shelters';
import { EmergencyError } from './types';

import type { ActionHash } from '../../generated/common';

// ============================================================================
// Client Configuration
// ============================================================================

export interface EmergencyClientConfig {
  roleName?: string;
  debug?: boolean;
  timeout?: number;
}

export interface EmergencyConnectionOptions {
  url: string;
  timeout?: number;
  config?: EmergencyClientConfig;
}

const DEFAULT_CONFIG: Required<EmergencyClientConfig> = {
  roleName: 'civic',
  debug: false,
  timeout: 30000,
};

// ============================================================================
// Aggregate Emergency Client
// ============================================================================

/**
 * Unified Emergency hApp Client
 *
 * @example
 * ```typescript
 * const emergency = await EmergencyClient.connect({ url: 'ws://localhost:8888' });
 *
 * const disaster = await emergency.incidents.declareDisaster({
 *   title: 'Flooding - River District',
 *   description: 'Major flooding affecting downtown area',
 *   disasterType: 'Flood',
 *   severity: 'Warning',
 *   latitude: 32.9483,
 *   longitude: -96.7299,
 *   radiusKm: 5,
 * });
 *
 * const team = await emergency.coordination.formTeam({
 *   disasterId: disaster.id,
 *   name: 'Search & Rescue Alpha',
 *   specialization: 'Water Rescue',
 * });
 * ```
 */
export class EmergencyClient {
  public readonly incidents: IncidentsClient;
  public readonly coordination: CoordinationClient;
  public readonly shelters: SheltersClient;

  private readonly _client: AppClient;
  private readonly _config: Required<EmergencyClientConfig>;

  private constructor(client: AppClient, config: EmergencyClientConfig = {}) {
    this._client = client;
    this._config = { ...DEFAULT_CONFIG, ...config };

    const baseConfig = {
      roleName: this._config.roleName,
      timeout: this._config.timeout,
    };

    this.incidents = new IncidentsClient(client, baseConfig);
    this.coordination = new CoordinationClient(client, baseConfig);
    this.shelters = new SheltersClient(client, baseConfig);

    if (this._config.debug) {
      console.log('[emergency-sdk] Client initialized with 3 zome clients');
    }
  }

  static async connect(options: EmergencyConnectionOptions): Promise<EmergencyClient> {
    try {
      const client = await AppWebsocket.connect({
        url: new URL(options.url),
      });

      const appInfo = await client.appInfo();
      if (!appInfo) {
        throw new Error('Failed to get app info from conductor');
      }

      return new EmergencyClient(client, options.config);
    } catch (error) {
      throw new EmergencyError(
        'CONNECTION_ERROR',
        `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
        error
      );
    }
  }

  static fromClient(client: AppClient, config: EmergencyClientConfig = {}): EmergencyClient {
    return new EmergencyClient(client, config);
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
  // Convenience Methods
  // ============================================================================

  /**
   * Get full disaster situation overview
   */
  async getDisasterOverview(disasterId: ActionHash): Promise<{
    disaster: Awaited<ReturnType<IncidentsClient['getDisaster']>>;
    teams: Awaited<ReturnType<CoordinationClient['getTeamsForDisaster']>>;
    shelters: Awaited<ReturnType<SheltersClient['getSheltersForDisaster']>>;
    sitreps: Awaited<ReturnType<CoordinationClient['getSitreps']>>;
  }> {
    const [disaster, teams, shelters, sitreps] = await Promise.all([
      this.incidents.getDisaster(disasterId),
      this.coordination.getTeamsForDisaster(disasterId),
      this.shelters.getSheltersForDisaster(disasterId),
      this.coordination.getSitreps(disasterId),
    ]);

    return { disaster, teams, shelters, sitreps };
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createEmergencyClient(client: AppClient, config?: EmergencyClientConfig): EmergencyClient {
  return EmergencyClient.fromClient(client, config);
}

// ============================================================================
// Re-exports
// ============================================================================

export { IncidentsClient, type IncidentsClientConfig } from './incidents';
export { CoordinationClient, type CoordinationClientConfig } from './coordination';
export { SheltersClient, type SheltersClientConfig } from './shelters';
export * from './types';
export default EmergencyClient;
