// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Agentic Framework Types
 *
 * TypeScript types for AI agent management, matching the Rust SDK's agentic module.
 * These types enable epistemic-aware AI agency with verifiable trust profiles.
 *
 * @packageDocumentation
 * @module agentic/types
 */

import type { MoralUncertainty, MoralActionGuidance } from '../epistemic/gis.js';

// Re-export GIS types used in agentic context
export type { MoralUncertainty, MoralActionGuidance };
export { MoralUncertaintyType, MoralActionGuidance as MoralActionGuidanceEnum } from '../epistemic/gis.js';

// =============================================================================
// Agent Classification
// =============================================================================

/**
 * Agent classification determining operational capabilities
 */
export enum AgentClass {
  /** Fully automated, no human-in-loop */
  Autonomous = 'Autonomous',
  /** Human approval required for high-value actions */
  Supervised = 'Supervised',
  /** Human initiates, agent executes */
  Assistive = 'Assistive',
  /** Read-only participation */
  Observer = 'Observer',
}

/**
 * Agent operational status
 */
export enum AgentStatus {
  /** Normal operation */
  Active = 'Active',
  /** Rate-limited due to high activity */
  Throttled = 'Throttled',
  /** Temporarily suspended */
  Suspended = 'Suspended',
  /** Permanently revoked */
  Revoked = 'Revoked',
}

// =============================================================================
// K-Vector Types
// =============================================================================

/**
 * K-Vector dimension values (10 dimensions)
 */
export interface KVectorValues {
  /** Reputation dimension (k_r): Success rate, peer feedback */
  k_r: number;
  /** Activity dimension (k_a): Engagement level */
  k_a: number;
  /** Integrity dimension (k_i): Constraint compliance */
  k_i: number;
  /** Performance dimension (k_p): Output quality */
  k_p: number;
  /** Membership dimension (k_m): Time in network */
  k_m: number;
  /** Stake dimension (k_s): KREDIT efficiency */
  k_s: number;
  /** Historical dimension (k_h): Behavioral consistency */
  k_h: number;
  /** Topological dimension (k_topo): Network connections */
  k_topo: number;
  /** Verification dimension (k_v): Identity verification level */
  k_v?: number;
  /** Coherence dimension (k_coherence): Output consistency metric */
  k_coherence?: number;
}

/**
 * K-Vector history entry
 */
export interface KVectorHistoryEntry {
  /** Timestamp of this snapshot */
  timestamp: number;
  /** K-Vector values at this point */
  k_vector: KVectorValues;
  /** Trust score calculated from the K-Vector */
  trust_score: number;
}

// =============================================================================
// Uncertainty Calibration
// =============================================================================

/**
 * Uncertainty calibration summary
 *
 * Tracks whether an agent is appropriately uncertain about its decisions.
 */
export interface CalibrationSummary {
  /** Agent ID */
  agent_id: string;
  /** Total calibration events recorded */
  total_events: number;
  /** Overall calibration score (0.0-1.0, higher is better) */
  calibration_score: number;
  /** Count of times agent was appropriately uncertain */
  appropriate_uncertainty: number;
  /** Count of times agent was appropriately confident */
  appropriate_confidence: number;
  /** Count of times agent was overconfident (certain when wrong) */
  overconfident: number;
  /** Count of times agent was overcautious (uncertain when could proceed) */
  overcautious: number;
  /** Whether agent tends to be overconfident */
  is_overconfident: boolean;
  /** Whether agent tends to be overcautious */
  is_overcautious: boolean;
}

/**
 * Default calibration state
 */
export function createDefaultCalibration(agentId: string): CalibrationSummary {
  return {
    agent_id: agentId,
    total_events: 0,
    calibration_score: 0.5, // Neutral starting point
    appropriate_uncertainty: 0,
    appropriate_confidence: 0,
    overconfident: 0,
    overcautious: 0,
    is_overconfident: false,
    is_overcautious: false,
  };
}

// =============================================================================
// Escalation Types
// =============================================================================

/**
 * Summary of an escalation request
 *
 * When an agent encounters high moral uncertainty, actions are escalated
 * to the human sponsor for approval.
 */
export interface EscalationSummary {
  /** Agent ID */
  agent_id: string;
  /** Action that was blocked due to uncertainty */
  blocked_action: string;
  /** Total moral uncertainty (RMS of three dimensions) */
  uncertainty_total: number;
  /** Maximum uncertainty in any single dimension */
  uncertainty_max: number;
  /** Guidance level (from MoralActionGuidance) */
  guidance: string;
  /** Recommendations for resolution */
  recommendations: string[];
  /** When the escalation was created */
  timestamp: number;
}

/**
 * Response after resolving an escalation
 */
export interface EscalationResolutionResponse {
  /** Agent ID */
  agent_id: string;
  /** Action that was resolved */
  action: string;
  /** Whether sponsor approved the action */
  approved: boolean;
  /** Whether calibration was updated */
  calibration_updated: boolean;
  /** Remaining pending escalations */
  remaining_escalations: number;
}

// =============================================================================
// Agent API Types
// =============================================================================

/**
 * API error response
 */
export interface ApiError {
  /** Error code (e.g., "NOT_FOUND", "VALIDATION_ERROR") */
  code: string;
  /** Human-readable error message */
  message: string;
  /** Additional details */
  details?: string;
}

/**
 * Request to create a new agent
 */
export interface CreateAgentRequest {
  /** Agent ID (optional - will be generated if not provided) */
  agent_id?: string;
  /** Sponsor DID (must start with "did:") */
  sponsor_did: string;
  /** Agent class */
  agent_class: AgentClass;
  /** Initial KREDIT balance (default: 5000) */
  initial_kredit?: number;
  /** KREDIT cap (default: 10000, range: 100 - 1,000,000,000) */
  kredit_cap?: number;
}

/**
 * Response after creating an agent
 */
export interface CreateAgentResponse {
  /** The created agent summary */
  agent: AgentSummary;
  /** Number of events generated */
  events_generated: number;
}

/**
 * Request to update an agent
 */
export interface UpdateAgentRequest {
  /** New status (optional) */
  status?: AgentStatus;
  /** New KREDIT balance (optional) */
  kredit_balance?: number;
  /** Reason for update */
  reason?: string;
}

/**
 * Summary view of an agent
 */
export interface AgentSummary {
  /** Agent ID */
  agent_id: string;
  /** Sponsor DID */
  sponsor_did: string;
  /** Agent class */
  agent_class: AgentClass;
  /** Current status */
  status: AgentStatus;
  /** KREDIT balance */
  kredit_balance: number;
  /** KREDIT cap */
  kredit_cap: number;
  /** Trust score (from K-Vector, 0.0-1.0) */
  trust_score: number;
  /** Total actions taken */
  total_actions: number;
  /** Success rate (0.0-1.0) */
  success_rate: number;
  /** Created timestamp (Unix epoch seconds) */
  created_at: number;
  /** Last activity timestamp (Unix epoch seconds) */
  last_activity: number;
}

/**
 * List agents response
 */
export interface ListAgentsResponse {
  /** List of agents */
  agents: AgentSummary[];
  /** Total count */
  total: number;
  /** Pagination offset */
  offset: number;
  /** Pagination limit */
  limit: number;
}

/**
 * K-Vector history response
 */
export interface KVectorHistoryResponse {
  /** Agent ID */
  agent_id: string;
  /** History entries */
  history: KVectorHistoryEntry[];
  /** Total entries */
  total: number;
}

/**
 * Summary of an event
 */
export interface EventSummary {
  /** Event ID */
  event_id: number;
  /** Sequence number */
  sequence: number;
  /** Event type (e.g., "Created", "StatusChanged") */
  event_type: string;
  /** Agent ID */
  agent_id: string;
  /** Timestamp */
  timestamp: number;
}

/**
 * Events response
 */
export interface EventsResponse {
  /** Events */
  events: EventSummary[];
  /** Last sequence number */
  last_sequence: number;
  /** Has more events */
  has_more: boolean;
}

// =============================================================================
// Gaming Detection Types
// =============================================================================

/**
 * Types of gaming attacks detected
 */
export enum GamingAttackType {
  /** Artificially perfect success rates */
  SuccessInflation = 'SuccessInflation',
  /** Bot-like constant intervals */
  TimingManipulation = 'TimingManipulation',
  /** Spike-then-quiet patterns */
  ActivityBursting = 'ActivityBursting',
  /** Mutual boosting between agents */
  CircularReferral = 'CircularReferral',
  /** Fake activity generation */
  SyntheticTransactions = 'SyntheticTransactions',
  /** Classification manipulation */
  EpistemicGaming = 'EpistemicGaming',
}

/**
 * Response to detected gaming
 */
export enum GamingResponse {
  /** No action needed */
  None = 'None',
  /** Increase monitoring */
  IncreasedMonitoring = 'IncreasedMonitoring',
  /** Quarantine the agent */
  Quarantine = 'Quarantine',
  /** Revoke the agent */
  Revoke = 'Revoke',
}

/**
 * Result of gaming detection analysis
 */
export interface GamingDetectionResult {
  /** Agent ID */
  agent_id: string;
  /** Whether gaming was detected */
  gaming_detected: boolean;
  /** Confidence in detection (0.0-1.0) */
  confidence: number;
  /** Detected attack types */
  detected_attacks: GamingAttackType[];
  /** Recommended response */
  recommended_action: GamingResponse;
  /** Explanation of findings */
  explanation: string;
}

// =============================================================================
// Coherence (Phi) Types
// =============================================================================

/**
 * Coherence state for an agent
 */
export interface CoherenceState {
  /** Current Phi value */
  phi: number;
  /** Whether coherence is sufficient for high-stakes actions */
  is_coherent: boolean;
  /** Threshold for coherence (default: 0.7) */
  threshold: number;
  /** Last measurement timestamp */
  last_measured: number;
}

/**
 * Result of coherence check for an action
 */
export enum CoherenceCheckResult {
  /** Action allowed */
  Allowed = 'Allowed',
  /** Action requires approval */
  RequiresApproval = 'RequiresApproval',
  /** Action blocked */
  Blocked = 'Blocked',
}

// =============================================================================
// Class Limits
// =============================================================================

/**
 * Class-specific operational limits
 */
export interface ClassLimits {
  /** Maximum KREDIT per 30-day epoch */
  max_kredit_per_epoch: number;
  /** Maximum transactions per hour */
  max_tx_per_hour: number;
  /** Maximum transaction size in SAP */
  max_tx_size_sap: number;
  /** SAP threshold requiring sponsor approval (null = never required) */
  requires_approval_above: number | null;
  /** API calls per minute */
  api_rate_per_minute: number;
}

/**
 * Get class-specific limits
 */
export function getClassLimits(agentClass: AgentClass): ClassLimits {
  switch (agentClass) {
    case AgentClass.Autonomous:
      return {
        max_kredit_per_epoch: 5_000,
        max_tx_per_hour: 100,
        max_tx_size_sap: 1_000,
        requires_approval_above: null,
        api_rate_per_minute: 60,
      };
    case AgentClass.Supervised:
      return {
        max_kredit_per_epoch: 10_000,
        max_tx_per_hour: 500,
        max_tx_size_sap: 5_000,
        requires_approval_above: 1_000,
        api_rate_per_minute: 300,
      };
    case AgentClass.Assistive:
      return {
        max_kredit_per_epoch: 15_000,
        max_tx_per_hour: 1_000,
        max_tx_size_sap: 10_000,
        requires_approval_above: null,
        api_rate_per_minute: 600,
      };
    case AgentClass.Observer:
      return {
        max_kredit_per_epoch: 500,
        max_tx_per_hour: 10,
        max_tx_size_sap: 0,
        requires_approval_above: 0,
        api_rate_per_minute: 60,
      };
  }
}

// =============================================================================
// Constraint Types
// =============================================================================

/**
 * Agent operational constraints
 */
export interface AgentConstraints {
  /** Can vote on governance */
  can_vote_governance: boolean;
  /** Can become validator */
  can_become_validator: boolean;
  /** Can govern HEARTH pools */
  can_govern_hearth: boolean;
  /** Can sponsor other agents */
  can_sponsor_agents: boolean;
  /** Can accumulate CIV */
  can_hold_civ: boolean;
  /** Can hold SAP directly */
  can_hold_sap: boolean;
  /** Can receive CGC */
  can_receive_cgc: boolean;
  /** Can send CGC */
  can_send_cgc: boolean;
  /** Custom action whitelist (null = all allowed) */
  action_whitelist: string[] | null;
  /** Action blacklist */
  action_blacklist: string[];
}

/**
 * Default agent constraints (constitutional constraints enforced)
 */
export function createDefaultConstraints(): AgentConstraints {
  return {
    // Constitutional constraints - always false for AI agents
    can_vote_governance: false,
    can_become_validator: false,
    can_govern_hearth: false,
    can_sponsor_agents: false,
    can_hold_civ: false,
    can_hold_sap: false,
    // CGC permissions
    can_receive_cgc: true,
    can_send_cgc: false,
    // No custom filters by default
    action_whitelist: null,
    action_blacklist: [],
  };
}

// =============================================================================
// Validation Constants
// =============================================================================

/** Maximum length for sponsor DID */
export const MAX_SPONSOR_DID_LENGTH = 256;

/** Maximum length for agent ID */
export const MAX_AGENT_ID_LENGTH = 128;

/** Maximum KREDIT cap allowed */
export const MAX_KREDIT_CAP = 1_000_000_000;

/** Minimum KREDIT cap allowed */
export const MIN_KREDIT_CAP = 100;

/** Maximum pending escalations per agent */
export const MAX_PENDING_ESCALATIONS = 100;

/** Maximum action string length */
export const MAX_ACTION_STRING_LENGTH = 1024;

/** Maximum context string length */
export const MAX_CONTEXT_STRING_LENGTH = 4096;
