// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Governance types (Proposal, Vote, DAO, etc.)
 *
 * Auto-generated TypeScript types from Rust SDK.
 * DO NOT EDIT MANUALLY - regenerate with: pnpm generate:types
 *
 * @module @mycelix/sdk/generated/governance
 * @generated
 */

/**
 * A DAO governance proposal
 * @generated
 */
export interface Proposal {
  /** Proposal ID */
  id: string;
  /** DAO this proposal belongs to */
  daoId: string;
  /** Proposal title */
  title: string;
  /** Detailed description */
  description: string;
  /** Proposer DID */
  proposer: string;
  /** Type of proposal */
  proposalType: ProposalType;
  /** Current status */
  status: ProposalStatus;
  /** Voting period in hours */
  votingPeriodHours: number;
  /** Required quorum (0.0-1.0) */
  quorum: number;
  /** Approval threshold (0.0-1.0) */
  threshold: number;
  /** Total approve vote weight */
  approveWeight: number;
  /** Total reject vote weight */
  rejectWeight: number;
  /** Total abstain vote weight */
  abstainWeight: number;
  /** Voting end timestamp */
  votingEnds: number;
  /** Creation timestamp */
  created: number;
}

/**
 * Type of governance proposal
 * @generated
 */
export type ProposalType = 'Standard' | 'Emergency' | 'Constitutional' | 'Parameter' | 'Funding';

/**
 * Status of a governance proposal
 * @generated
 */
export type ProposalStatus = 'Draft' | 'Active' | 'Ended' | 'Approved' | 'Signed' | 'Rejected' | 'Executed' | 'Cancelled' | 'Failed';

/**
 * A vote on a governance proposal
 * @generated
 */
export interface GovernanceVote {
  /** Vote ID */
  id: string;
  /** Proposal being voted on */
  proposalId: string;
  /** Voter DID */
  voter: string;
  /** Vote choice */
  choice: VoteChoice;
  /** Vote weight */
  weight: number;
  /** Reason for vote */
  reason?: string;
  /** Vote timestamp */
  votedAt: number;
}

/**
 * Vote choice options
 * @generated
 */
export type VoteChoice = 'Approve' | 'Reject' | 'Abstain';

/**
 * Decentralized Autonomous Organization
 * @generated
 */
export interface DAO {
  /** DAO ID */
  id: string;
  /** DAO name */
  name: string;
  /** DAO description */
  description: string;
  /** Creator DID */
  creator: string;
  /** Charter document hash */
  charterHash?: string;
  /** Default voting period in hours */
  defaultVotingPeriod: number;
  /** Default quorum */
  defaultQuorum: number;
  /** Default threshold */
  defaultThreshold: number;
  /** Number of members */
  memberCount: number;
  /** Total voting power */
  totalVotingPower: number;
  /** Creation timestamp */
  created: number;
}

/**
 * Voting power delegation
 * @generated
 */
export interface Delegation {
  /** Delegation ID */
  id: string;
  /** Delegator DID */
  delegator: string;
  /** Delegate DID */
  delegate: string;
  /** DAO ID */
  daoId: string;
  /** Delegated voting power */
  power: number;
  /** Expiration timestamp */
  expiresAt?: number;
  /** Creation timestamp */
  created: number;
}
