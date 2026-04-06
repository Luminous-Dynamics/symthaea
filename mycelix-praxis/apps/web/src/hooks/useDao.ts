// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * useDao Hook
 *
 * Custom hook for governance proposals and voting from the DAO zome.
 */

import { useState, useEffect, useCallback } from 'react';
import { useHolochain } from '../contexts/HolochainContext';
import type {
  Proposal,
  Vote,
  VoteChoice,
  ProposalCategory,
  ProposalType,
  ActionHash,
  Timestamp,
} from '../types/zomes';

// =============================================================================
// Types
// =============================================================================

export interface UseProposalsResult {
  /** List of proposals */
  proposals: Proposal[];
  /** Whether data is loading */
  loading: boolean;
  /** Error message if fetch failed */
  error: string | null;
  /** Whether using real Holochain data */
  isReal: boolean;
  /** Refresh proposals from source */
  refresh: () => Promise<void>;
  /** Create a new proposal */
  createProposal: (input: CreateProposalInput) => Promise<ActionHash | null>;
  /** Cast a vote on a proposal */
  castVote: (proposalId: string, choice: VoteChoice, votingPower?: number, justification?: string) => Promise<ActionHash | null>;
}

export interface CreateProposalInput {
  title: string;
  description: string;
  proposalType: ProposalType;
  category: ProposalCategory;
  actionsJson?: string;
  votingDeadlineDays?: number;
}

// =============================================================================
// Mock Data
// =============================================================================

const now = Math.floor(Date.now() / 1000);

const MOCK_PROPOSALS: Proposal[] = [
  {
    proposal_id: 'prop-001',
    title: 'Add Rust Programming Course',
    description: 'Proposal to add a comprehensive Rust systems programming course to the curriculum.',
    proposer: 'did:holo:mock-agent-1',
    proposal_type: 'Normal',
    category: 'Curriculum',
    created_at: now - 86400 * 3,
    voting_deadline: now + 86400 * 4,
    actions_json: '{"action": "add_course", "course_id": "rust-101"}',
    votes_for: 15,
    votes_against: 3,
    votes_abstain: 2,
    status: 'Active',
  },
  {
    proposal_id: 'prop-002',
    title: 'Update Credential Verification Protocol',
    description: 'Upgrade the credential verification to use the latest W3C VC 2.0 spec.',
    proposer: 'did:holo:mock-agent-2',
    proposal_type: 'Slow',
    category: 'Protocol',
    created_at: now - 86400 * 7,
    voting_deadline: now + 86400 * 14,
    actions_json: '{"action": "update_protocol", "version": "2.0"}',
    votes_for: 22,
    votes_against: 1,
    votes_abstain: 5,
    status: 'Active',
  },
];

// =============================================================================
// Hook
// =============================================================================

export function useProposals(): UseProposalsResult {
  const { client, status, isReal } = useHolochain();
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchProposals = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      if (isReal && client && status === 'connected') {
        console.log('[useProposals] Fetching from Holochain...');
        const result = await client.dao.list_proposals();
        if (result.length === 0) {
          console.log('[useProposals] No proposals, using mock data');
          setProposals(MOCK_PROPOSALS);
        } else {
          setProposals(result);
        }
      } else {
        console.log('[useProposals] Using mock data');
        await new Promise(resolve => setTimeout(resolve, 400));
        setProposals(MOCK_PROPOSALS);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch proposals';
      console.error('[useProposals] Error:', errorMessage);
      setError(errorMessage);
      setProposals(MOCK_PROPOSALS);
    } finally {
      setLoading(false);
    }
  }, [client, status, isReal]);

  useEffect(() => {
    if (status === 'connected' || status === 'disconnected') {
      fetchProposals();
    }
  }, [status, fetchProposals]);

  const createProposal = useCallback(async (input: CreateProposalInput): Promise<ActionHash | null> => {
    if (!client || status !== 'connected') {
      console.warn('[useProposals] Cannot create proposal - not connected');
      return null;
    }

    const deadlineDays = input.votingDeadlineDays ?? 7;
    const proposal: Proposal = {
      proposal_id: `prop-${Date.now()}`,
      title: input.title,
      description: input.description,
      proposer: '', // Set by zome
      proposal_type: input.proposalType,
      category: input.category,
      created_at: now,
      voting_deadline: now + 86400 * deadlineDays,
      actions_json: input.actionsJson ?? '{}',
      votes_for: 0,
      votes_against: 0,
      votes_abstain: 0,
      status: 'Active',
    };

    try {
      if (isReal) {
        const hash = await client.dao.create_proposal(proposal);
        console.log('[useProposals] Proposal created:', hash);
        await fetchProposals();
        return hash;
      }
      // Mock creation
      setProposals(prev => [proposal, ...prev]);
      return new Uint8Array(32);
    } catch (err) {
      console.error('[useProposals] Failed to create proposal:', err);
      return null;
    }
  }, [client, status, isReal, fetchProposals]);

  const castVote = useCallback(async (
    proposalId: string,
    choice: VoteChoice,
    votingPower: number = 1,
    justification?: string,
  ): Promise<ActionHash | null> => {
    if (!client || status !== 'connected') {
      console.warn('[useProposals] Cannot cast vote - not connected');
      return null;
    }

    const vote: Vote = {
      proposal_id: proposalId,
      voter: '', // Set by zome
      choice,
      voting_power: votingPower,
      timestamp: Math.floor(Date.now() / 1000),
      justification,
    };

    try {
      if (isReal) {
        const hash = await client.dao.cast_vote(vote);
        console.log('[useProposals] Vote cast:', hash);
        await fetchProposals();
        return hash;
      }
      // Mock: update vote counts locally
      setProposals(prev => prev.map(p => {
        if (p.proposal_id !== proposalId) return p;
        const key = choice === 'For' ? 'votes_for' : choice === 'Against' ? 'votes_against' : 'votes_abstain';
        return { ...p, [key]: p[key] + votingPower };
      }));
      return new Uint8Array(32);
    } catch (err) {
      console.error('[useProposals] Failed to cast vote:', err);
      return null;
    }
  }, [client, status, isReal, fetchProposals]);

  return { proposals, loading, error, isReal, refresh: fetchProposals, createProposal, castVote };
}

// =============================================================================
// Export
// =============================================================================

export default useProposals;
