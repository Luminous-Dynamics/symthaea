// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Privacy Zome Client
 *
 * Sharing policies and access control.
 */

import type { AppClient, Record as HolochainRecord, ActionHash, AgentPubKey } from '@holochain/client';
import type {
  SharingPolicy,
  SetPolicyInput,
  AccessGrant,
  GrantAccessInput,
  CheckAccessInput,
} from '../types';
import { decodeRecord, decodeRecords } from '../utils';

export class PrivacyZomeClient {
  constructor(
    private client: AppClient,
    private roleName: string = 'lucid',
    private zomeName: string = 'privacy'
  ) {}

  private async callZome<T>(fnName: string, payload?: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload: payload ?? null,
    });
    return result as T;
  }

  // ============================================================================
  // POLICY OPERATIONS
  // ============================================================================

  /** Set sharing policy for a thought */
  async setSharingPolicy(input: SetPolicyInput): Promise<SharingPolicy> {
    const record = await this.callZome<HolochainRecord>('set_sharing_policy', input);
    return decodeRecord<SharingPolicy>(record);
  }

  /** Get sharing policy for a thought */
  async getSharingPolicy(thoughtId: string): Promise<SharingPolicy | null> {
    const record = await this.callZome<HolochainRecord | null>('get_sharing_policy', thoughtId);
    return record ? decodeRecord<SharingPolicy>(record) : null;
  }

  // ============================================================================
  // GRANT OPERATIONS
  // ============================================================================

  /** Grant access to a specific agent */
  async grantAccess(input: GrantAccessInput): Promise<AccessGrant> {
    const record = await this.callZome<HolochainRecord>('grant_access', input);
    return decodeRecord<AccessGrant>(record);
  }

  /** Get grants for a thought */
  async getThoughtGrants(thoughtId: string): Promise<AccessGrant[]> {
    const records = await this.callZome<HolochainRecord[]>('get_thought_grants', thoughtId);
    return decodeRecords<AccessGrant>(records);
  }

  /** Check if an agent has access to a thought */
  async checkAccess(thoughtId: string, agent: AgentPubKey): Promise<boolean> {
    const input: CheckAccessInput = { thought_id: thoughtId, agent };
    return this.callZome<boolean>('check_access', input);
  }

  /** Log an access event */
  async logAccess(thoughtId: string, accessType: string): Promise<ActionHash> {
    return this.callZome<ActionHash>('log_access', {
      thought_id: thoughtId,
      access_type: accessType,
    });
  }
}
