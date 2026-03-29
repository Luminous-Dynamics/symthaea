// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Governance Hooks for Mycelix DAO
 *
 * React hooks for interacting with governance contracts.
 */

'use client';

import { useCallback, useMemo, useState } from 'react';
import {
  useAccount,
  useChainId,
  useReadContract,
  useWriteContract,
  useWaitForTransactionReceipt,
  usePublicClient,
} from 'wagmi';
import { parseEther, formatEther, encodeFunctionData } from 'viem';
import { getContractAddress } from './config';

// ============================================================================
// Types
// ============================================================================

export interface Proposal {
  id: bigint;
  proposer: `0x${string}`;
  targets: `0x${string}`[];
  values: bigint[];
  calldatas: `0x${string}`[];
  description: string;
  startBlock: bigint;
  endBlock: bigint;
  forVotes: bigint;
  againstVotes: bigint;
  abstainVotes: bigint;
  canceled: boolean;
  executed: boolean;
  state: ProposalState;
  category: ProposalCategory;
}

export enum ProposalState {
  Pending = 0,
  Active = 1,
  Canceled = 2,
  Defeated = 3,
  Succeeded = 4,
  Queued = 5,
  Expired = 6,
  Executed = 7,
}

export enum ProposalCategory {
  Standard = 0,
  Treasury = 1,
  Protocol = 2,
  Constitutional = 3,
}

export interface ResonanceBreakdown {
  listening: bigint;
  patronage: bigint;
  community: bigint;
  presence: bigint;
  collaboration: bigint;
  creation: bigint;
  total: bigint;
}

export interface Grant {
  id: number;
  recipient: `0x${string}`;
  amount: bigint;
  description: string;
  createdAt: number;
  claimedAt: number;
  active: boolean;
}

// ============================================================================
// ABIs (simplified)
// ============================================================================

const GOVERNOR_ABI = [
  {
    inputs: [],
    name: 'proposalCount',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'proposalId', type: 'uint256' }],
    name: 'state',
    outputs: [{ type: 'uint8' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'proposalId', type: 'uint256' }],
    name: 'proposalVotes',
    outputs: [
      { name: 'againstVotes', type: 'uint256' },
      { name: 'forVotes', type: 'uint256' },
      { name: 'abstainVotes', type: 'uint256' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [
      { name: 'targets', type: 'address[]' },
      { name: 'values', type: 'uint256[]' },
      { name: 'calldatas', type: 'bytes[]' },
      { name: 'description', type: 'string' },
    ],
    name: 'propose',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      { name: 'proposalId', type: 'uint256' },
      { name: 'support', type: 'uint8' },
    ],
    name: 'castVote',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [
      { name: 'proposalId', type: 'uint256' },
      { name: 'support', type: 'uint8' },
      { name: 'reason', type: 'string' },
    ],
    name: 'castVoteWithReason',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [{ name: 'account', type: 'address' }],
    name: 'getVotes',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
] as const;

const RESONANCE_TOKEN_ABI = [
  {
    inputs: [{ name: 'account', type: 'address' }],
    name: 'balanceOf',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'account', type: 'address' }],
    name: 'getVotes',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'delegatee', type: 'address' }],
    name: 'delegate',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
  {
    inputs: [{ name: 'account', type: 'address' }],
    name: 'delegates',
    outputs: [{ type: 'address' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'account', type: 'address' }],
    name: 'getResonanceBreakdown',
    outputs: [
      { name: 'listening', type: 'uint256' },
      { name: 'patronage', type: 'uint256' },
      { name: 'community', type: 'uint256' },
      { name: 'presence', type: 'uint256' },
      { name: 'collaboration', type: 'uint256' },
      { name: 'creation', type: 'uint256' },
    ],
    stateMutability: 'view',
    type: 'function',
  },
] as const;

const TREASURY_ABI = [
  {
    inputs: [],
    name: 'getBalance',
    outputs: [{ type: 'uint256' }],
    stateMutability: 'view',
    type: 'function',
  },
  {
    inputs: [{ name: 'grantId', type: 'uint256' }],
    name: 'claimGrant',
    outputs: [],
    stateMutability: 'nonpayable',
    type: 'function',
  },
] as const;

// ============================================================================
// Hooks
// ============================================================================

/**
 * Hook for resonance token (voting power)
 */
export function useResonance() {
  const chainId = useChainId();
  const { address } = useAccount();
  const publicClient = usePublicClient();

  // Contract address would come from config
  const tokenAddress = '0x0000000000000000000000000000000000000000' as `0x${string}`;

  // Read: Balance
  const { data: balance, isLoading: isLoadingBalance } = useReadContract({
    address: tokenAddress,
    abi: RESONANCE_TOKEN_ABI,
    functionName: 'balanceOf',
    args: address ? [address] : undefined,
    query: { enabled: !!address },
  });

  // Read: Voting power
  const { data: votingPower, isLoading: isLoadingVotes } = useReadContract({
    address: tokenAddress,
    abi: RESONANCE_TOKEN_ABI,
    functionName: 'getVotes',
    args: address ? [address] : undefined,
    query: { enabled: !!address },
  });

  // Read: Delegate
  const { data: delegate } = useReadContract({
    address: tokenAddress,
    abi: RESONANCE_TOKEN_ABI,
    functionName: 'delegates',
    args: address ? [address] : undefined,
    query: { enabled: !!address },
  });

  // Read: Breakdown
  const { data: breakdown } = useReadContract({
    address: tokenAddress,
    abi: RESONANCE_TOKEN_ABI,
    functionName: 'getResonanceBreakdown',
    args: address ? [address] : undefined,
    query: { enabled: !!address },
  });

  // Write: Delegate
  const { writeContract: delegateWrite, isPending: isDelegating } = useWriteContract();

  const delegateTo = useCallback(
    async (delegatee: `0x${string}`) => {
      delegateWrite({
        address: tokenAddress,
        abi: RESONANCE_TOKEN_ABI,
        functionName: 'delegate',
        args: [delegatee],
      });
    },
    [delegateWrite, tokenAddress]
  );

  const selfDelegate = useCallback(async () => {
    if (!address) return;
    await delegateTo(address);
  }, [address, delegateTo]);

  const resonanceBreakdown = useMemo((): ResonanceBreakdown | undefined => {
    if (!breakdown) return undefined;
    const [listening, patronage, community, presence, collaboration, creation] = breakdown;
    return {
      listening,
      patronage,
      community,
      presence,
      collaboration,
      creation,
      total: listening + patronage + community + presence + collaboration + creation,
    };
  }, [breakdown]);

  return {
    balance: balance ?? 0n,
    votingPower: votingPower ?? 0n,
    delegate: delegate as `0x${string}` | undefined,
    breakdown: resonanceBreakdown,
    isLoading: isLoadingBalance || isLoadingVotes,
    isDelegating,
    delegateTo,
    selfDelegate,
    hasDelegated: delegate && delegate !== address,
  };
}

/**
 * Hook for governance proposals
 */
export function useGovernance() {
  const chainId = useChainId();
  const { address } = useAccount();
  const publicClient = usePublicClient();

  const governorAddress = '0x0000000000000000000000000000000000000000' as `0x${string}`;

  // Read: User's voting power
  const { data: votingPower } = useReadContract({
    address: governorAddress,
    abi: GOVERNOR_ABI,
    functionName: 'getVotes',
    args: address ? [address] : undefined,
    query: { enabled: !!address },
  });

  // Write: Create proposal
  const { writeContract: proposeWrite, data: proposeHash, isPending: isProposing } = useWriteContract();

  const { isLoading: isConfirmingProposal, isSuccess: proposalConfirmed } = useWaitForTransactionReceipt({
    hash: proposeHash,
  });

  const createProposal = useCallback(
    async (params: {
      targets: `0x${string}`[];
      values: bigint[];
      calldatas: `0x${string}`[];
      description: string;
    }) => {
      proposeWrite({
        address: governorAddress,
        abi: GOVERNOR_ABI,
        functionName: 'propose',
        args: [params.targets, params.values, params.calldatas, params.description],
      });
    },
    [proposeWrite, governorAddress]
  );

  // Write: Cast vote
  const { writeContract: voteWrite, data: voteHash, isPending: isVoting } = useWriteContract();

  const { isLoading: isConfirmingVote, isSuccess: voteConfirmed } = useWaitForTransactionReceipt({
    hash: voteHash,
  });

  const castVote = useCallback(
    async (proposalId: bigint, support: 0 | 1 | 2, reason?: string) => {
      if (reason) {
        voteWrite({
          address: governorAddress,
          abi: GOVERNOR_ABI,
          functionName: 'castVoteWithReason',
          args: [proposalId, support, reason],
        });
      } else {
        voteWrite({
          address: governorAddress,
          abi: GOVERNOR_ABI,
          functionName: 'castVote',
          args: [proposalId, support],
        });
      }
    },
    [voteWrite, governorAddress]
  );

  const voteFor = useCallback(
    async (proposalId: bigint, reason?: string) => castVote(proposalId, 1, reason),
    [castVote]
  );

  const voteAgainst = useCallback(
    async (proposalId: bigint, reason?: string) => castVote(proposalId, 0, reason),
    [castVote]
  );

  const abstain = useCallback(
    async (proposalId: bigint, reason?: string) => castVote(proposalId, 2, reason),
    [castVote]
  );

  // Fetch proposal details
  const getProposalState = useCallback(
    async (proposalId: bigint): Promise<ProposalState | null> => {
      if (!publicClient) return null;
      try {
        const state = await publicClient.readContract({
          address: governorAddress,
          abi: GOVERNOR_ABI,
          functionName: 'state',
          args: [proposalId],
        });
        return state as ProposalState;
      } catch {
        return null;
      }
    },
    [publicClient, governorAddress]
  );

  const getProposalVotes = useCallback(
    async (proposalId: bigint) => {
      if (!publicClient) return null;
      try {
        const [againstVotes, forVotes, abstainVotes] = await publicClient.readContract({
          address: governorAddress,
          abi: GOVERNOR_ABI,
          functionName: 'proposalVotes',
          args: [proposalId],
        });
        return { forVotes, againstVotes, abstainVotes };
      } catch {
        return null;
      }
    },
    [publicClient, governorAddress]
  );

  return {
    votingPower: votingPower ?? 0n,
    canPropose: (votingPower ?? 0n) >= parseEther('100'), // Example threshold

    // Proposal creation
    createProposal,
    isProposing: isProposing || isConfirmingProposal,
    proposalConfirmed,

    // Voting
    voteFor,
    voteAgainst,
    abstain,
    isVoting: isVoting || isConfirmingVote,
    voteConfirmed,

    // Queries
    getProposalState,
    getProposalVotes,
  };
}

/**
 * Hook for DAO treasury
 */
export function useTreasury() {
  const chainId = useChainId();
  const { address } = useAccount();

  const treasuryAddress = '0x0000000000000000000000000000000000000000' as `0x${string}`;

  // Read: Treasury balance
  const { data: balance, isLoading } = useReadContract({
    address: treasuryAddress,
    abi: TREASURY_ABI,
    functionName: 'getBalance',
  });

  // Write: Claim grant
  const { writeContract: claimWrite, isPending: isClaiming } = useWriteContract();

  const claimGrant = useCallback(
    async (grantId: number) => {
      claimWrite({
        address: treasuryAddress,
        abi: TREASURY_ABI,
        functionName: 'claimGrant',
        args: [BigInt(grantId)],
      });
    },
    [claimWrite, treasuryAddress]
  );

  return {
    balance: balance ?? 0n,
    balanceFormatted: formatEther(balance ?? 0n),
    isLoading,
    claimGrant,
    isClaiming,
  };
}

/**
 * Combined governance hook
 */
export function useMycelixGovernance() {
  const resonance = useResonance();
  const governance = useGovernance();
  const treasury = useTreasury();

  return {
    // Resonance
    resonance: resonance.balance,
    votingPower: resonance.votingPower,
    resonanceBreakdown: resonance.breakdown,
    delegateTo: resonance.delegateTo,
    selfDelegate: resonance.selfDelegate,

    // Governance
    canPropose: governance.canPropose,
    createProposal: governance.createProposal,
    voteFor: governance.voteFor,
    voteAgainst: governance.voteAgainst,
    abstain: governance.abstain,
    isVoting: governance.isVoting,
    isProposing: governance.isProposing,

    // Treasury
    treasuryBalance: treasury.balance,
    claimGrant: treasury.claimGrant,

    // Loading states
    isLoading: resonance.isLoading || treasury.isLoading,
  };
}
