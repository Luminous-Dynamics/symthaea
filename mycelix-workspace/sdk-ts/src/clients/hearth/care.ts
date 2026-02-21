/**
 * Hearth Care Zome Client
 *
 * Care schedules, swaps, meal plans, and care load balancing for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/care
 */

import type {
  CreateCareScheduleInput,
  ProposeSwapInput,
  CreateMealPlanInput,
} from './types';
import type { ActionHash } from '../../generated/common';

export interface HearthCareClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class HearthCareClient {
  private readonly zomeName = 'hearth_care';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<HearthCareClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Care Schedules
  // ============================================================================

  async createCareSchedule(input: CreateCareScheduleInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_care_schedule', payload: input });
  }

  async completeTask(scheduleHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_task', payload: scheduleHash });
  }

  async getMyCareduties(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_careduties', payload: hearthHash });
  }

  async getHearthSchedule(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_schedule', payload: hearthHash });
  }

  async getCareLoadBalance(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_care_load_balance', payload: hearthHash });
  }

  // ============================================================================
  // Care Swaps
  // ============================================================================

  async proposeSwap(input: ProposeSwapInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'propose_swap', payload: input });
  }

  async acceptSwap(swapHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'accept_swap', payload: swapHash });
  }

  // ============================================================================
  // Meal Plans
  // ============================================================================

  async createMealPlan(input: CreateMealPlanInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_meal_plan', payload: input });
  }

  // ============================================================================
  // Digests
  // ============================================================================

  async createCareDigest(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_care_digest', payload: hearthHash });
  }
}
