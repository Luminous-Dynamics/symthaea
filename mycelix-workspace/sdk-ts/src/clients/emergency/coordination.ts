/**
 * Coordination Zome Client
 *
 * Handles team formation, zone assignment, situation reports, and check-ins.
 *
 * @module @mycelix/sdk/clients/emergency/coordination
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Team,
  FormTeamInput,
  SitRep,
  SubmitSitRepInput,
  CheckIn,
  CheckInInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface CoordinationClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: CoordinationClientConfig = {
  roleName: 'civic',
};

/**
 * Client for Emergency Coordination operations
 */
export class CoordinationClient extends ZomeClient {
  protected readonly zomeName = 'emergency_coordination';

  constructor(client: AppClient, config: CoordinationClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  async formTeam(input: FormTeamInput): Promise<Team> {
    const record = await this.callZomeOnce<HolochainRecord>('form_team', {
      disaster_id: input.disasterId,
      name: input.name,
      specialization: input.specialization,
      initial_members: input.initialMembers ?? [],
    });
    return this.mapTeam(record);
  }

  async getTeam(teamId: ActionHash): Promise<Team | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_team', teamId);
    if (!record) return null;
    return this.mapTeam(record);
  }

  async getTeamsForDisaster(disasterId: ActionHash): Promise<Team[]> {
    const records = await this.callZome<HolochainRecord[]>('get_teams_for_disaster', disasterId);
    return records.map(r => this.mapTeam(r));
  }

  async assignToZone(teamId: ActionHash, zone: string): Promise<Team> {
    const record = await this.callZomeOnce<HolochainRecord>('assign_to_zone', {
      team_id: teamId,
      zone,
    });
    return this.mapTeam(record);
  }

  async submitSitrep(input: SubmitSitRepInput): Promise<SitRep> {
    const record = await this.callZomeOnce<HolochainRecord>('submit_sitrep', {
      disaster_id: input.disasterId,
      team_id: input.teamId,
      summary: input.summary,
      casualties: input.casualties,
      rescued: input.rescued,
      resources_needed: input.resourcesNeeded ?? [],
    });
    return this.mapSitRep(record);
  }

  async getSitreps(disasterId: ActionHash): Promise<SitRep[]> {
    const records = await this.callZome<HolochainRecord[]>('get_sitreps', disasterId);
    return records.map(r => this.mapSitRep(r));
  }

  async checkin(input: CheckInInput): Promise<CheckIn> {
    const record = await this.callZomeOnce<HolochainRecord>('checkin', {
      disaster_id: input.disasterId,
      team_id: input.teamId,
      latitude: input.latitude,
      longitude: input.longitude,
      status: input.status,
      message: input.message,
    });
    return this.mapCheckIn(record);
  }

  async getCheckins(disasterId: ActionHash): Promise<CheckIn[]> {
    const records = await this.callZome<HolochainRecord[]>('get_checkins', disasterId);
    return records.map(r => this.mapCheckIn(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapTeam(record: HolochainRecord): Team {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      disasterId: entry.disaster_id,
      name: entry.name,
      leaderDid: entry.leader_did,
      members: entry.members,
      assignedZone: entry.assigned_zone,
      specialization: entry.specialization,
      status: entry.status,
      createdAt: entry.created_at,
      updatedAt: entry.updated_at,
    };
  }

  private mapSitRep(record: HolochainRecord): SitRep {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      disasterId: entry.disaster_id,
      teamId: entry.team_id,
      authorDid: entry.author_did,
      summary: entry.summary,
      casualties: entry.casualties,
      rescued: entry.rescued,
      resourcesNeeded: entry.resources_needed,
      createdAt: entry.created_at,
    };
  }

  private mapCheckIn(record: HolochainRecord): CheckIn {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      disasterId: entry.disaster_id,
      agentDid: entry.agent_did,
      teamId: entry.team_id,
      latitude: entry.latitude,
      longitude: entry.longitude,
      status: entry.status,
      message: entry.message,
      checkedInAt: entry.checked_in_at,
    };
  }
}
