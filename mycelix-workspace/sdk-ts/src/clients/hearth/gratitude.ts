/**
 * Gratitude Zome Client
 *
 * Appreciation expressions, gratitude circles, and streaks for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/gratitude
 */

import type {
  ExpressGratitudeInput,
  StartCircleInput,
  GratitudeAnchor,
} from './types';
import type { ActionHash } from '../../generated/common';

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

  async expressGratitude(input: ExpressGratitudeInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'express_gratitude', payload: input });
  }

  async getGratitudeStream(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_gratitude_stream', payload: hearthHash });
  }

  async getGratitudeBalance(hearthHash: ActionHash): Promise<GratitudeAnchor> {
    return this.client.callZome<GratitudeAnchor>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_gratitude_balance', payload: hearthHash });
  }

  async getGratitudeMilestones(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_gratitude_milestones', payload: hearthHash });
  }

  // ============================================================================
  // Appreciation Circles
  // ============================================================================

  async startCircle(input: StartCircleInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'start_circle', payload: input });
  }

  async joinCircle(circleHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'join_circle', payload: circleHash });
  }

  async completeCircle(circleHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_circle', payload: circleHash });
  }

  // ============================================================================
  // Digests
  // ============================================================================

  async createGratitudeDigest(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_gratitude_digest', payload: hearthHash });
  }
}
