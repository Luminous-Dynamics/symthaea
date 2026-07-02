/**
 * Reputation + trust graph zome wrappers
 */

import type { AppClient } from '@holochain/client';
import { callZome } from './client';
import type { ProofTrailItem, ReputationClaim, TrustGraphSnapshot } from '$types';

/**
 * Fetch the trust graph snapshot for a subject agent.
 */
export async function getTrustGraph(
  client: AppClient,
  agentId: string
): Promise<TrustGraphSnapshot> {
  return callZome<TrustGraphSnapshot>(client, 'reputation', 'get_trust_graph', {
    agent_id: agentId,
  });
}

/**
 * Fetch all reputation claims for the subject.
 */
export async function getReputationClaims(
  client: AppClient,
  agentId: string
): Promise<ReputationClaim[]> {
  return callZome<ReputationClaim[]>(client, 'reputation', 'get_reputation_claims', {
    agent_id: agentId,
  });
}

/**
 * Fetch proof trail entries (hashes/CIDs) supporting the subject's reputation.
 */
export async function getProofTrail(
  client: AppClient,
  agentId: string
): Promise<ProofTrailItem[]> {
  return callZome<ProofTrailItem[]>(client, 'reputation', 'get_proof_trail', {
    agent_id: agentId,
  });
}

/**
 * Request a fresh proof/attestation from the subject.
 */
export async function requestProof(
  client: AppClient,
  agentId: string,
  listingHash?: string
): Promise<{ status: 'requested'; message?: string }> {
  return callZome(client, 'reputation', 'request_proof', {
    agent_id: agentId,
    listing_hash: listingHash,
  });
}
