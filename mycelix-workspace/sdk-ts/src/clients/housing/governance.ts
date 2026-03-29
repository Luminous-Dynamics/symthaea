// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Housing Governance Zome Client
 *
 * Handles meetings, resolutions, voting, and elections for housing cooperatives.
 *
 * @module @mycelix/sdk/clients/housing/governance
 */

import { ZomeClient, type ZomeClientConfig } from '../../core/zome-client';

import type {
  Meeting,
  ScheduleMeetingInput,
  Resolution,
  ProposeResolutionInput,
  VoteOnResolutionInput,
  Election,
  CreateElectionInput,
  CastBallotInput,
} from './types';
import type { ActionHash } from '../../generated/common';
import type { AppClient, Record as HolochainRecord } from '@holochain/client';

export interface HousingGovernanceClientConfig extends Partial<ZomeClientConfig> {
  roleName?: string;
}

const DEFAULT_CONFIG: HousingGovernanceClientConfig = {
  roleName: 'commons',
};

/**
 * Client for Housing Governance operations
 */
export class HousingGovernanceClient extends ZomeClient {
  protected readonly zomeName = 'housing_governance';

  constructor(client: AppClient, config: HousingGovernanceClientConfig = {}) {
    const mergedConfig = { ...DEFAULT_CONFIG, ...config };
    super(client, { roleName: mergedConfig.roleName! });
  }

  // ============================================================================
  // Meetings
  // ============================================================================

  async scheduleMeeting(input: ScheduleMeetingInput): Promise<Meeting> {
    const record = await this.callZomeOnce<HolochainRecord>('schedule_meeting', {
      building_id: input.buildingId,
      title: input.title,
      description: input.description,
      meeting_type: input.meetingType,
      scheduled_at: input.scheduledAt,
      location: input.location,
      agenda: input.agenda ?? [],
      quorum_required: input.quorumRequired ?? 0.5,
    });
    return this.mapMeeting(record);
  }

  async getMeeting(meetingId: ActionHash): Promise<Meeting | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_meeting', meetingId);
    if (!record) return null;
    return this.mapMeeting(record);
  }

  async getUpcomingMeetings(buildingId: ActionHash): Promise<Meeting[]> {
    const records = await this.callZome<HolochainRecord[]>('get_upcoming_meetings', buildingId);
    return records.map(r => this.mapMeeting(r));
  }

  // ============================================================================
  // Resolutions
  // ============================================================================

  async proposeResolution(input: ProposeResolutionInput): Promise<Resolution> {
    const record = await this.callZomeOnce<HolochainRecord>('propose_resolution', {
      building_id: input.buildingId,
      meeting_id: input.meetingId,
      title: input.title,
      description: input.description,
      quorum_required: input.quorumRequired ?? 0.5,
      threshold: input.threshold ?? 0.5,
      voting_period_hours: input.votingPeriodHours ?? 168,
    });
    return this.mapResolution(record);
  }

  async getResolution(resolutionId: ActionHash): Promise<Resolution | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_resolution', resolutionId);
    if (!record) return null;
    return this.mapResolution(record);
  }

  async voteOnResolution(input: VoteOnResolutionInput): Promise<void> {
    await this.callZomeOnce<void>('vote_on_resolution', {
      resolution_id: input.resolutionId,
      choice: input.choice,
      reason: input.reason,
    });
  }

  async getResolutionsForBuilding(buildingId: ActionHash): Promise<Resolution[]> {
    const records = await this.callZome<HolochainRecord[]>('get_resolutions_for_building', buildingId);
    return records.map(r => this.mapResolution(r));
  }

  // ============================================================================
  // Elections
  // ============================================================================

  async createElection(input: CreateElectionInput): Promise<Election> {
    const record = await this.callZomeOnce<HolochainRecord>('create_election', {
      building_id: input.buildingId,
      title: input.title,
      positions: input.positions,
      voting_starts_at: input.votingStartsAt,
      voting_ends_at: input.votingEndsAt,
    });
    return this.mapElection(record);
  }

  async getElection(electionId: ActionHash): Promise<Election | null> {
    const record = await this.callZomeOrNull<HolochainRecord>('get_election', electionId);
    if (!record) return null;
    return this.mapElection(record);
  }

  async castBallot(input: CastBallotInput): Promise<void> {
    await this.callZomeOnce<void>('cast_ballot', {
      election_id: input.electionId,
      selections: input.selections,
    });
  }

  async getElectionsForBuilding(buildingId: ActionHash): Promise<Election[]> {
    const records = await this.callZome<HolochainRecord[]>('get_elections_for_building', buildingId);
    return records.map(r => this.mapElection(r));
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private mapMeeting(record: HolochainRecord): Meeting {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      title: entry.title,
      description: entry.description,
      meetingType: entry.meeting_type,
      scheduledAt: entry.scheduled_at,
      location: entry.location,
      agenda: entry.agenda,
      minutesTakerDid: entry.minutes_taker_did,
      quorumRequired: entry.quorum_required,
      status: entry.status,
      createdAt: entry.created_at,
    };
  }

  private mapResolution(record: HolochainRecord): Resolution {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      meetingId: entry.meeting_id,
      title: entry.title,
      description: entry.description,
      proposerDid: entry.proposer_did,
      approveVotes: entry.approve_votes,
      rejectVotes: entry.reject_votes,
      abstainVotes: entry.abstain_votes,
      quorumRequired: entry.quorum_required,
      threshold: entry.threshold,
      status: entry.status,
      votingEndsAt: entry.voting_ends_at,
      createdAt: entry.created_at,
    };
  }

  private mapElection(record: HolochainRecord): Election {
    const entry = this.extractEntry<any>(record);
    return {
      id: record.signed_action.hashed.hash as unknown as string,
      buildingId: entry.building_id,
      title: entry.title,
      positions: entry.positions,
      candidates: entry.candidates,
      status: entry.status,
      votingStartsAt: entry.voting_starts_at,
      votingEndsAt: entry.voting_ends_at,
      createdAt: entry.created_at,
    };
  }
}
