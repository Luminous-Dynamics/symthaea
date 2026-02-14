/**
 * @mycelix/sdk Vue 3 Composables for Governance Module
 *
 * Provides Vue 3 Composition API composables for the Governance hApp integration.
 *
 * @packageDocumentation
 * @module vue/governance
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface DAO {
  id: string;
  name: string;
  description: string;
  constitution: string;
  founder_did: string;
  treasury_wallet_id: string;
  created_at: number;
  member_count: number;
}

export interface Proposal {
  id: string;
  dao_id: string;
  title: string;
  description: string;
  proposer_did: string;
  proposal_type: 'standard' | 'emergency' | 'constitutional';
  status: 'draft' | 'active' | 'passed' | 'failed' | 'executed' | 'vetoed';
  voting_starts: number;
  voting_ends: number;
  quorum_percentage: number;
  approval_threshold: number;
  yes_votes: number;
  no_votes: number;
  abstain_votes: number;
  execution_payload?: string;
  created_at: number;
}

export interface Vote {
  id: string;
  proposal_id: string;
  voter_did: string;
  choice: 'yes' | 'no' | 'abstain';
  weight: number;
  reason?: string;
  delegate_from?: string;
  created_at: number;
}

export interface Delegation {
  id: string;
  delegator_did: string;
  delegate_did: string;
  dao_id: string;
  scope: 'all' | 'category';
  category?: string;
  expires_at?: number;
  created_at: number;
}

// ============================================================================
// DAO Composables
// ============================================================================

/**
 * Composable to create a DAO
 */
export function useCreateDAO(): UseMutationReturn<
  HolochainRecord<DAO>,
  {
    name: string;
    description: string;
    constitution: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a DAO
 */
export function useDAO(
  _daoId: string,
  _options?: UseQueryOptions
): UseQueryReturn<HolochainRecord<DAO> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get DAOs by founder
 */
export function useDAOsByFounder(_founderDid: string): UseQueryReturn<HolochainRecord<DAO>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get DAOs user is member of
 */
export function useMyDAOs(_memberDid: string): UseQueryReturn<HolochainRecord<DAO>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to join a DAO
 */
export function useJoinDAO(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to leave a DAO
 */
export function useLeaveDAO(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Proposal Composables
// ============================================================================

export interface UseProposalOptions extends UseQueryOptions {
  /** Include vote counts */
  includeVotes?: boolean;
  /** Include execution details */
  includeExecution?: boolean;
}

/**
 * Composable to create a proposal
 */
export function useCreateProposal(): UseMutationReturn<
  HolochainRecord<Proposal>,
  {
    dao_id: string;
    title: string;
    description: string;
    proposal_type?: 'standard' | 'emergency' | 'constitutional';
    voting_period_hours: number;
    quorum_percentage: number;
    approval_threshold?: number;
    execution_payload?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a proposal
 */
export function useProposal(
  _proposalId: string,
  _options?: UseProposalOptions
): UseQueryReturn<HolochainRecord<Proposal> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get proposals by DAO
 */
export function useProposalsByDAO(
  _daoId: string,
  _status?: Proposal['status'],
  _options?: UseProposalOptions
): UseQueryReturn<HolochainRecord<Proposal>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get proposals by proposer
 */
export function useProposalsByProposer(
  _proposerDid: string,
  _options?: UseProposalOptions
): UseQueryReturn<HolochainRecord<Proposal>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get active proposals
 */
export function useActiveProposals(_daoId: string): UseQueryReturn<HolochainRecord<Proposal>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to cancel a proposal (proposer only)
 */
export function useCancelProposal(): UseMutationReturn<HolochainRecord<Proposal>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to execute a passed proposal
 */
export function useExecuteProposal(): UseMutationReturn<HolochainRecord<Proposal>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Vote Composables
// ============================================================================

/**
 * Composable to cast a vote
 */
export function useCastVote(): UseMutationReturn<
  HolochainRecord<Vote>,
  {
    proposal_id: string;
    choice: 'yes' | 'no' | 'abstain';
    reason?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get votes for a proposal
 */
export function useVotesForProposal(_proposalId: string): UseQueryReturn<HolochainRecord<Vote>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get user's vote on a proposal
 */
export function useMyVote(
  _proposalId: string,
  _voterDid: string
): UseQueryReturn<HolochainRecord<Vote> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get voting power
 */
export function useVotingPower(_daoId: string, _voterDid: string): UseQueryReturn<number> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to change vote (if allowed)
 */
export function useChangeVote(): UseMutationReturn<
  HolochainRecord<Vote>,
  {
    vote_id: string;
    new_choice: 'yes' | 'no' | 'abstain';
    reason?: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Delegation Composables
// ============================================================================

/**
 * Composable to delegate voting power
 */
export function useDelegate(): UseMutationReturn<
  HolochainRecord<Delegation>,
  {
    delegate_did: string;
    dao_id: string;
    scope?: 'all' | 'category';
    category?: string;
    expires_at?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to revoke delegation
 */
export function useRevokeDelegation(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get delegations from user
 */
export function useMyDelegations(
  _delegatorDid: string
): UseQueryReturn<HolochainRecord<Delegation>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get delegations to user
 */
export function useDelegationsToMe(
  _delegateDid: string
): UseQueryReturn<HolochainRecord<Delegation>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Constitution Composables
// ============================================================================

/**
 * Composable to propose constitution amendment
 */
export function useProposeAmendment(): UseMutationReturn<
  HolochainRecord<Proposal>,
  {
    dao_id: string;
    title: string;
    description: string;
    new_constitution_text: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get constitution history
 */
export function useConstitutionHistory(_daoId: string): UseQueryReturn<
  Array<{
    version: number;
    text: string;
    changed_at: number;
    proposal_id: string;
  }>
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
