/**
 * Disputes Zome Functions
 *
 * Type-safe wrappers for all dispute resolution (MRC) zome calls
 */

import type { AppClient } from '@holochain/client';
import { callZome } from './client';
import type { Dispute, CreateDisputeInput, CastVoteInput, ArbitratorProfile } from '$types';

/**
 * Get a single dispute by claim ID
 *
 * @param client - Holochain client
 * @param claimId - Dispute claim ID
 * @returns Dispute details
 */
export async function getDispute(client: AppClient, claimId: string): Promise<Dispute> {
  return callZome<Dispute>(client, 'disputes', 'get_dispute', {
    claim_id: claimId,
  });
}

/**
 * Get my arbitration cases (if I'm an arbitrator)
 *
 * @param client - Holochain client
 * @returns Array of disputes I'm assigned to
 */
export async function getMyArbitrationCases(client: AppClient): Promise<Dispute[]> {
  return callZome<Dispute[]>(client, 'disputes', 'get_my_arbitration_cases', {});
}

/**
 * Get disputes I filed (as buyer)
 *
 * @param client - Holochain client
 * @returns Array of disputes I filed
 */
export async function getMyDisputes(client: AppClient): Promise<Dispute[]> {
  return callZome<Dispute[]>(client, 'disputes', 'get_my_disputes', {});
}

/**
 * File a new dispute
 *
 * @param client - Holochain client
 * @param input - Dispute data
 * @returns Created dispute
 */
export async function createDispute(
  client: AppClient,
  input: CreateDisputeInput
): Promise<Dispute> {
  return callZome<Dispute>(client, 'disputes', 'file_dispute', input);
}

/**
 * Cast arbitrator vote
 *
 * @param client - Holochain client
 * @param input - Vote data
 * @returns Updated dispute
 */
export async function castArbitratorVote(
  client: AppClient,
  input: CastVoteInput
): Promise<Dispute> {
  return callZome<Dispute>(client, 'disputes', 'cast_arbitrator_vote', input);
}

/**
 * Get arbitrator profile
 *
 * @param client - Holochain client
 * @param agentId - Arbitrator agent ID (optional, defaults to current user)
 * @returns Arbitrator profile
 */
export async function getArbitratorProfile(
  client: AppClient,
  agentId?: string
): Promise<ArbitratorProfile> {
  return callZome<ArbitratorProfile>(client, 'disputes', 'get_arbitrator_profile', {
    agent_id: agentId,
  });
}

/**
 * Check if user is an arbitrator
 *
 * @param client - Holochain client
 * @returns Boolean indicating arbitrator status
 */
export async function isArbitrator(client: AppClient): Promise<boolean> {
  return callZome<boolean>(client, 'disputes', 'is_arbitrator', {});
}

/**
 * Get all disputes (admin only)
 *
 * @param client - Holochain client
 * @returns Array of all disputes
 */
export async function getAllDisputes(client: AppClient): Promise<Dispute[]> {
  return callZome<Dispute[]>(client, 'disputes', 'get_all_disputes', {});
}

/**
 * Get disputes by status
 *
 * @param client - Holochain client
 * @param status - Dispute status ('pending', 'active', 'resolved', 'rejected')
 * @returns Array of disputes with matching status
 */
export async function getDisputesByStatus(
  client: AppClient,
  status: string
): Promise<Dispute[]> {
  return callZome<Dispute[]>(client, 'disputes', 'get_disputes_by_status', {
    status,
  });
}
