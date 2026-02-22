/**
 * Emergency Zome Client
 *
 * Emergency plans, alerts, safety check-ins, and missing member tracking
 * for Hearth clusters.
 *
 * @module @mycelix/sdk/clients/hearth/emergency
 */

import type {
  CreateEmergencyPlanInput,
  UpdatePlanInput,
  RaiseAlertInput,
  CheckInInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { Record } from '@holochain/client';

export interface EmergencyClientConfig {
  roleName?: string;
  timeout?: number;
}

interface ZomeCallable {
  callZome<T>(params: { role_name: string; zome_name: string; fn_name: string; payload: unknown }): Promise<T>;
}

export class EmergencyClient {
  private readonly zomeName = 'hearth_emergency';

  constructor(
    private readonly client: ZomeCallable,
    private readonly config: Required<Pick<EmergencyClientConfig, 'roleName' | 'timeout'>>,
  ) {}

  // ============================================================================
  // Emergency Plans
  // ============================================================================

  async createEmergencyPlan(input: CreateEmergencyPlanInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_emergency_plan', payload: input });
  }

  async updateEmergencyPlan(input: UpdatePlanInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'update_emergency_plan', payload: input });
  }

  async getEmergencyPlan(hearthHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_emergency_plan', payload: hearthHash });
  }

  // ============================================================================
  // Alerts
  // ============================================================================

  async raiseAlert(input: RaiseAlertInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'raise_alert', payload: input });
  }

  async resolveAlert(alertHash: ActionHash): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'resolve_alert', payload: alertHash });
  }

  async getActiveAlerts(hearthHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_alerts', payload: hearthHash });
  }

  // ============================================================================
  // Safety Check-Ins
  // ============================================================================

  async checkIn(input: CheckInInput): Promise<Record> {
    return this.client.callZome<Record>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'check_in', payload: input });
  }

  async getAlertCheckins(alertHash: ActionHash): Promise<Record[]> {
    return this.client.callZome<Record[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_alert_checkins', payload: alertHash });
  }
}
