// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React Hooks for Mycelix Governance Module
 *
 * Provides React hooks for DAO management, proposals, and voting.
 *
 * @module @mycelix/sdk/react/governance
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Proposal,
  CreateProposalInput,
  Vote,
  CastVoteInput,
  DAO,
  CreateDAOInput,
  DAOMember,
  MemberRole,
  Delegation,
  CreateDelegationInput,
  Execution,
  HolochainRecord,
} from '../governance/index.js';

// ============================================================================
// DAO Hooks
// ============================================================================

/**
 * Hook to fetch a DAO by ID
 */
export function useDAO(_daoId: string): QueryState<HolochainRecord<DAO> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to list all DAOs
 */
export function useDAOList(): QueryState<HolochainRecord<DAO>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch DAO members
 */
export function useDAOMembers(_daoId: string): QueryState<HolochainRecord<DAOMember>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch a specific member
 */
export function useDAOMember(
  _daoId: string,
  _did: string
): QueryState<HolochainRecord<DAOMember> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to create a DAO
 */
export function useCreateDAO(): MutationState<HolochainRecord<DAO>, CreateDAOInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to join a DAO
 */
export function useJoinDAO(): MutationState<
  HolochainRecord<DAOMember>,
  { daoId: string; votingPower?: number }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to update member role
 */
export function useUpdateMemberRole(): MutationState<
  HolochainRecord<DAOMember>,
  { daoId: string; did: string; role: MemberRole }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Proposal Hooks
// ============================================================================

/**
 * Hook to fetch a proposal by ID
 */
export function useProposalById(_proposalId: string): QueryState<HolochainRecord<Proposal> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch proposals for a DAO
 */
export function useProposalsByDAO(_daoId: string): QueryState<HolochainRecord<Proposal>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch active proposals for a DAO
 */
export function useActiveProposals(_daoId: string): QueryState<HolochainRecord<Proposal>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch proposals by proposer
 */
export function useProposalsByProposer(
  _proposerDid: string
): QueryState<HolochainRecord<Proposal>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to create a proposal
 */
export function useCreateProposalMutation(): MutationState<
  HolochainRecord<Proposal>,
  CreateProposalInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to cancel a proposal
 */
export function useCancelProposal(): MutationState<HolochainRecord<Proposal>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to finalize a proposal
 */
export function useFinalizeProposal(): MutationState<HolochainRecord<Proposal>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Voting Hooks
// ============================================================================

/**
 * Hook to fetch votes for a proposal
 */
export function useVotesForProposal(_proposalId: string): QueryState<HolochainRecord<Vote>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch votes by voter
 */
export function useVotesByVoter(_voterDid: string): QueryState<HolochainRecord<Vote>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to check if user has voted
 */
export function useHasVoted(_proposalId: string, _voterDid: string): QueryState<boolean> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get vote tally
 */
export function useVoteTally(_proposalId: string): QueryState<{
  approve: number;
  reject: number;
  abstain: number;
  total: number;
}> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to cast a vote
 */
export function useCastVoteMutation(): MutationState<HolochainRecord<Vote>, CastVoteInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Delegation Hooks
// ============================================================================

/**
 * Hook to fetch delegations from a delegator
 */
export function useDelegationsFrom(
  _delegatorDid: string
): QueryState<HolochainRecord<Delegation>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch delegations to a delegate
 */
export function useDelegationsTo(_delegateDid: string): QueryState<HolochainRecord<Delegation>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to get effective voting power
 */
export function useEffectiveVotingPower(_did: string, _daoId: string): QueryState<number> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to create a delegation
 */
export function useDelegate(): MutationState<HolochainRecord<Delegation>, CreateDelegationInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to revoke a delegation
 */
export function useRevokeDelegation(): MutationState<void, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Execution Hooks
// ============================================================================

/**
 * Hook to fetch pending executions
 */
export function usePendingExecutions(
  _targetHapp?: string
): QueryState<HolochainRecord<Execution>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch executions for a proposal
 */
export function useExecutionsForProposal(
  _proposalId: string
): QueryState<HolochainRecord<Execution>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to request execution
 */
export function useRequestExecution(): MutationState<
  HolochainRecord<Execution>,
  { proposalId: string; targetHapp: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to acknowledge execution
 */
export function useAcknowledgeExecution(): MutationState<
  HolochainRecord<Execution>,
  { executionId: string; success: boolean; error?: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}
