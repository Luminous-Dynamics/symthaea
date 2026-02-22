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
  DigestEpochInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

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

  async createCareSchedule(input: CreateCareScheduleInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_care_schedule', payload: input });
  }

  async completeTask(scheduleHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'complete_task', payload: scheduleHash });
  }

  async getMyCareDuties(): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_my_care_duties', payload: null });
  }

  async getHearthSchedule(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_schedule', payload: hearthHash });
  }

  // ============================================================================
  // Care Swaps
  // ============================================================================

  async proposeSwap(input: ProposeSwapInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'propose_swap', payload: input });
  }

  async acceptSwap(swapHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'accept_swap', payload: swapHash });
  }

  async declineSwap(swapHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'decline_swap', payload: swapHash });
  }

  // ============================================================================
  // Meal Plans
  // ============================================================================

  async createMealPlan(input: CreateMealPlanInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_meal_plan', payload: input });
  }

  async getHearthMealPlans(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_hearth_meal_plans', payload: hearthHash });
  }

  // ============================================================================
  // Digests
  // ============================================================================

  async createCareDigest(input: DigestEpochInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_care_digest', payload: input });
  }
}
