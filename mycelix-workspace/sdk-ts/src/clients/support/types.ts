// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Support hApp Client Types
 *
 * Type definitions for the Support hApp SDK client,
 * providing coverage of knowledge, tickets, and diagnostics zomes.
 *
 * @module @mycelix/sdk/clients/support/types
 */

// ============================================================================
// Common Types
// ============================================================================

export type Timestamp = number;

// ============================================================================
// Enums (matching Rust enum variants)
// ============================================================================

export type SupportCategory = 'Network' | 'Hardware' | 'Software' | 'Holochain' | 'Mycelix' | 'Security' | 'General';
export type TicketPriority = 'Low' | 'Medium' | 'High' | 'Critical';
export type TicketStatus = 'Open' | 'InProgress' | 'AwaitingUser' | 'Resolved' | 'Closed';
export type AutonomyLevel = 'Advisory' | 'SemiAutonomous' | 'FullAutonomous';
export type ActionType = 'RestartService' | 'ClearCache' | 'UpdateConfig' | 'RunDiagnostic' | { Custom: string };
export type SharingTier = 'LocalOnly' | 'Anonymized' | 'Full';
export type DiagnosticType = 'NetworkCheck' | 'DiskSpace' | 'ServiceStatus' | 'HolochainHealth' | 'MemoryUsage' | { Custom: string };
export type DiagnosticSeverity = 'Healthy' | 'Warning' | 'Error' | 'Critical';
export type DifficultyLevel = 'Beginner' | 'Intermediate' | 'Advanced';
export type ArticleSource = 'Community' | 'PreSeeded' | 'SymthaeaGenerated';
export type FlagReason = 'Harmful' | 'Incorrect' | 'Outdated' | 'Spam';
export type ReputationEvent = 'ResolutionVerified' | 'ArticleUpvoted' | 'ArticleFlagged' | 'HelpProvided';
export type EpistemicStatus = 'Certain' | 'Probable' | 'Uncertain' | 'Unknown' | 'OutOfDomain';

// ============================================================================
// Knowledge Types
// ============================================================================

/** Input for creating a knowledge article */
export interface KnowledgeArticleInput {
  title: string;
  content: string;
  category: SupportCategory;
  tags: string[];
  author: Uint8Array;
  source: ArticleSource;
  difficultyLevel: DifficultyLevel;
  upvotes: number;
  verified: boolean;
  deprecated: boolean;
  deprecationReason: string | null;
  version: number;
}

/** A knowledge article with its action hash */
export interface KnowledgeArticle extends KnowledgeArticleInput {
  actionHash: Uint8Array;
}

/** Input for recording a resolution */
export interface ResolutionInput {
  ticketHash: Uint8Array;
  steps: string[];
  rootCause: string | null;
  timeToResolveMins: number | null;
  effectivenessRating: number | null;
  helper: Uint8Array;
  requester: Uint8Array;
  anonymized: boolean;
  helperSignature: number[];
  requesterSignature: number[];
}

/** Input for flagging an article */
export interface ArticleFlagInput {
  articleHash: Uint8Array;
  flagger: Uint8Array;
  reason: FlagReason;
  description: string;
  createdAt: Timestamp;
}

/** Input for updating an article */
export interface UpdateArticleInput {
  originalHash: Uint8Array;
  updated: KnowledgeArticleInput;
}

/** Input for deprecating an article */
export interface DeprecateInput {
  articleHash: Uint8Array;
  reason: string;
}

// ============================================================================
// Ticket Types
// ============================================================================

/** Input for creating a support ticket */
export interface SupportTicketInput {
  title: string;
  description: string;
  category: SupportCategory;
  priority: TicketPriority;
  status: TicketStatus;
  requester: Uint8Array;
  assignee: Uint8Array | null;
  autonomyLevel: AutonomyLevel;
  systemInfo: string | null;
  isPreemptive: boolean;
  predictionConfidence: number | null;
  createdAt: Timestamp;
  updatedAt: Timestamp;
}

/** Input for a ticket comment */
export interface TicketCommentInput {
  ticketHash: Uint8Array;
  author: Uint8Array;
  content: string;
  isSymthaeaResponse: boolean;
  confidence: number | null;
  epistemicStatus: EpistemicStatus | null;
}

/** Input for an autonomous action */
export interface AutonomousActionInput {
  ticketHash: Uint8Array;
  actionType: ActionType;
  description: string;
  approved: boolean;
  executed: boolean;
  result: string | null;
  success: boolean | null;
  rollbackSteps: string[] | null;
  rollbackState: string | null;
  rolledBack: boolean;
  createdAt: Timestamp;
}

/** Input for undoing an autonomous action */
export interface UndoActionInput {
  originalActionHash: Uint8Array;
  reason: string;
  rollbackResult: string;
  createdAt: Timestamp;
}

/** Input for a preemptive alert */
export interface PreemptiveAlertInput {
  predictedFailure: string;
  expectedTimeToFailure: string | null;
  freeEnergy: number;
  recommendedAction: string;
  autoGeneratedTicket: Uint8Array | null;
  createdAt: Timestamp;
}

/** Input for updating a ticket */
export interface UpdateTicketInput {
  originalHash: Uint8Array;
  updated: SupportTicketInput;
}

/** Input for promoting an alert to a ticket */
export interface PromoteAlertInput {
  alertHash: Uint8Array;
  ticket: SupportTicketInput;
}

// ============================================================================
// Diagnostic Types
// ============================================================================

/** Input for a diagnostic result */
export interface DiagnosticResultInput {
  ticketHash: Uint8Array | null;
  diagnosticType: DiagnosticType;
  findings: string;
  severity: DiagnosticSeverity;
  recommendations: string[];
  agent: Uint8Array;
  scrubbed: boolean;
  createdAt: Timestamp;
}

/** Input for privacy preferences */
export interface PrivacyPreferenceInput {
  agent: Uint8Array;
  sharingTier: SharingTier;
  allowedCategories: SupportCategory[];
  shareSystemInfo: boolean;
  shareResolutionPatterns: boolean;
  shareCognitiveUpdates: boolean;
  updatedAt: Timestamp;
}

/** Input for a cognitive update */
export interface CognitiveUpdateInput {
  category: SupportCategory;
  encoding: number[];
  phi: number;
  resolutionPattern: string;
  sourceAgent: Uint8Array;
  createdAt: Timestamp;
}

// ============================================================================
// Error Types
// ============================================================================

export const SupportErrorCode = {
  NOT_FOUND: 'SUPPORT_NOT_FOUND',
  VALIDATION_FAILED: 'SUPPORT_VALIDATION_FAILED',
  PERMISSION_DENIED: 'SUPPORT_PERMISSION_DENIED',
} as const;

export class SupportError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = 'SupportError';
  }
}
