// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Gratitude Zome Client
 *
 * Appreciation expressions, gratitude circles, and balance tracking
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/gratitude
 */

import type {
  ExpressGratitudeInput,
  StartCircleInput,
  GratitudeAnchor,
  DigestEpochInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface GratitudeClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class GratitudeClient {
  private readonly zomeName = 'hearth_gratitude';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<GratitudeClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Expressions
  // ============================================================================

  async expressGratitude(input: ExpressGratitudeInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'express_gratitude', payload: input });
  }

  async getGratitudeStream(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_gratitude_stream', payload: hearthHash });
  }

  async getGratitudeBalance(hearthHash: ActionHash): Promise<GratitudeAnchor> {
    return this.client.callZome<GratitudeAnchor>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_gratitude_balance', payload: hearthHash });
  }

  // ============================================================================
  // Appreciation Circles
  // ============================================================================

  async startCircle(input: StartCircleInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'start_appreciation_circle', payload: input });
  }

  async joinCircle(circleHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'join_circle', payload: circleHash });
  }

  async completeCircle(circleHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_circle', payload: circleHash });
  }

  async getHearthCircles(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_circles', payload: hearthHash });
  }

  // ============================================================================
  // Digests
  // ============================================================================

  async createGratitudeDigest(input: DigestEpochInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_gratitude_digest', payload: input });
  }
}
