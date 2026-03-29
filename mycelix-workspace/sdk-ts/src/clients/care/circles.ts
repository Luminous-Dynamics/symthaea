// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Circles Zome Client
 *
 * Handles care circle creation, membership, and management.
 *
 * @module @mycelix/sdk/clients/care/circles
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type { CareCircle, CreateCareCircleInput, CircleMember } from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface CirclesClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: CirclesClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Care Circle operations
 */
export class CirclesClient extends ZomeClient {
  protected readonly zomeName = 'care_circles';

  constructor(client: AppClient, config: CirclesClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async createCircle(input: CreateCareCircleInput): Promise<CareCircle> {
    const record = await this.callZomeOnce<HolochainRecord>('create_circle', {
      name: input.name,
      description: input.description,
      max_members: input.maxMembers ?? 20,
      area: input.area,
      categories: input.categories ?? [],
      open: input.open ?? true,
    });
    return this.mapCircle(record);
  }

  async getCircle(circleId: ActionHash): Promise<CareCircle | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_circle', circleId);
    if (!record) return null;
    return this.mapCircle(record);
  }

  async joinCircle(circleId: ActionHash): Promise<CircleMember> {
    const record = await this.callZomeOnce<HolochainRecord>('join_circle', circleId);
    return this.mapCircleMember(record);
  }

  async leaveCircle(circleId: ActionHash): Promise<void> {
    await this.callZomeOnce<void>('leave_circle', circleId);
  }

  async getMyCircles(): Promise<CareCircle[]> {
    const records = await this.callZome<HolochainRecord[]>('get_my_circles', null);
    return records.map(r => this.mapCircle(r));
  }

  async getCircleMembers(circleId: ActionHash): Promise<CircleMember[]> {
    const records = await this.callZome<HolochainRecord[]>('get_circle_members', circleId);
    return records.map(r => this.mapCircleMember(r));
  }

  async listOpenCircles(): Promise<CareCircle[]> {
    const records = await this.callZome<HolochainRecord[]>('list_open_circles', null);
    return records.map(r => this.mapCircle(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapCircle(record: HolochainRecord): CareCircle {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      name: entry.name,
      description: entry.description,
      founderDid: entry.founder_did,
      maxMembers: entry.max_members,
      memberCount: entry.member_count,
      area: entry.area,
      categories: entry.categories,
      open: entry.open,
      active: entry.active,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapCircleMember(record: HolochainRecord): CircleMember {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      circleId: entry.circle_id,
      memberDid: entry.member_did,
      memberPubKey: entry.member_pub_key,
      role: entry.role,
      joinedAt: entry.joined_at,
      active: entry.active,
    };
  }
}
