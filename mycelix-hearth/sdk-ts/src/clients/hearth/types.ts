/**
 * TypeScript types for Hearth Decisions zome.
 * Mirrors the Rust entry types and input structs.
 */

import type { ActionHash, AgentPubKey, Timestamp } from '@holochain/client';

// ============================================================================
// Enums (represented as string unions)
// ============================================================================

export type DecisionType =
  | 'Consensus'
  | 'MajorityVote'
  | 'ElderDecision'
  | 'GuardianDecision';

export type DecisionStatus = 'Open' | 'Closed' | 'Finalized';

export type MemberRole =
  | 'Founder'
  | 'Elder'
  | 'Adult'
  | 'Youth'
  | 'Child'
  | 'Guest'
  | 'Ancestor';

// ============================================================================
// Entry Types
// ============================================================================

export interface Decision {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
  quorum_bp?: number;
  status: DecisionStatus;
  created_by: AgentPubKey;
  created_at: Timestamp;
}

export interface Vote {
  decision_hash: ActionHash;
  voter: AgentPubKey;
  choice: number;
  weight_bp: number;
  reasoning?: string;
  created_at: Timestamp;
}

export interface DecisionOutcome {
  decision_hash: ActionHash;
  chosen_option: number;
  participation_rate_bp: number;
  resolved_at: Timestamp;
}

// ============================================================================
// Input Types
// ============================================================================

export interface CreateDecisionInput {
  hearth_hash: ActionHash;
  title: string;
  description: string;
  decision_type: DecisionType;
  eligible_roles: MemberRole[];
  options: string[];
  deadline: Timestamp;
  quorum_bp?: number;
}

export interface CastVoteInput {
  decision_hash: ActionHash;
  choice: number;
  reasoning?: string;
}

export interface FinalizeDecisionInput {
  decision_hash: ActionHash;
}

export interface CloseDecisionInput {
  decision_hash: ActionHash;
}

export interface AmendVoteInput {
  decision_hash: ActionHash;
  choice: number;
  reasoning?: string;
}
