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
  RaiseAlertInput,
  CheckInInput,
  EmergencyPlan,
  EmergencyAlert,
  SafetyCheckIn,
} from './types';
import type { ActionHash } from '../../generated/common';

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

  async createPlan(input: CreateEmergencyPlanInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'create_plan', payload: input });
  }

  // ============================================================================
  // Alerts
  // ============================================================================

  async raiseAlert(input: RaiseAlertInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'raise_alert', payload: input });
  }

  async resolveAlert(alertHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'resolve_alert', payload: alertHash });
  }

  async getActiveAlerts(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_active_alerts', payload: hearthHash });
  }

  // ============================================================================
  // Safety Check-Ins
  // ============================================================================

  async checkIn(input: CheckInInput) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'check_in', payload: input });
  }

  async markSafe(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'mark_safe', payload: hearthHash });
  }

  async requestHelp(hearthHash: ActionHash) {
    return this.client.callZome<unknown>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'request_help', payload: hearthHash });
  }

  async getMissingCheckins(hearthHash: ActionHash) {
    return this.client.callZome<unknown[]>({ role_name: this.config.roleName, zome_name: this.zomeName, fn_name: 'get_missing_checkins', payload: hearthHash });
  }
}
