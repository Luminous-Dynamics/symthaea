// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civic hApp Type Definitions
 *
 * These types mirror the Rust structs defined in the integrity zomes.
 */

import type { ActionHash, AgentPubKey, EntryHash, Record } from '@holochain/client';

// ============================================================================
// Civic Knowledge Types
// ============================================================================

/** Civic domains matching Symthaea domains */
export type CivicDomain =
  | 'benefits'
  | 'permits'
  | 'tax'
  | 'voting'
  | 'justice'
  | 'housing'
  | 'employment'
  | 'education'
  | 'health'
  | 'emergency'
  | 'general';

/** Knowledge type classification */
export type KnowledgeType =
  | 'faq'
  | 'eligibility_rule'
  | 'contact_info'
  | 'process'
  | 'regulation'
  | 'resource'
  | 'form'
  | 'deadline';

/** A piece of civic knowledge stored on the DHT */
export interface CivicKnowledge {
  domain: CivicDomain;
  knowledge_type: KnowledgeType;
  title: string;
  content: string;
  geographic_scope?: string;
  keywords: string[];
  source?: string;
  last_verified?: number;
  expires_at?: number;
  links: string[];
  contact_phone?: string;
  address?: string;
}

/** An update to existing civic knowledge */
export interface KnowledgeUpdate {
  original_hash: ActionHash;
  updated_content: string;
  reason: string;
  submitter_note?: string;
}

/** A validation of civic knowledge by an authority */
export interface KnowledgeValidation {
  knowledge_hash: ActionHash;
  is_valid: boolean;
  authority_type: string;
  notes?: string;
  validated_at: number;
}

/** Input for creating civic knowledge */
export interface CreateKnowledgeInput {
  domain: CivicDomain;
  knowledge_type: KnowledgeType;
  title: string;
  content: string;
  geographic_scope?: string;
  keywords: string[];
  source?: string;
  expires_at?: number;
  links: string[];
  contact_phone?: string;
  address?: string;
}

/** Search result containing knowledge and its hash */
export interface KnowledgeSearchResult {
  action_hash: ActionHash;
  knowledge: CivicKnowledge;
}

/** Combined search input */
export interface SearchInput {
  domain?: CivicDomain;
  geo_scope?: string;
  keywords: string[];
  limit?: number;
}

/** Input for validating knowledge */
export interface ValidateKnowledgeInput {
  knowledge_hash: ActionHash;
  is_valid: boolean;
  authority_type: string;
  notes?: string;
}

/** Input for updating knowledge */
export interface UpdateKnowledgeInput {
  original_hash: ActionHash;
  updated_content: string;
  reason: string;
  submitter_note?: string;
}

// ============================================================================
// Agent Reputation Types
// ============================================================================

/** Agent type classification */
export type AgentType =
  | 'symthaea_ai'
  | 'human_agent'
  | 'hybrid'
  | 'authority'
  | 'community_validator';

/** Specialization domain for the agent */
export type AgentSpecialization =
  | 'general'
  | 'benefits'
  | 'permits'
  | 'tax'
  | 'voting'
  | 'justice'
  | 'emergency';

/** Profile for a civic AI agent */
export interface AgentProfile {
  agent_type: AgentType;
  name: string;
  specializations: AgentSpecialization[];
  description: string;
  is_active: boolean;
  created_at: number;
  last_active: number;
  config_metadata?: string;
}

/** Type of reputation event */
export type ReputationEventType =
  | 'helpful'
  | 'not_helpful'
  | 'accurate'
  | 'inaccurate'
  | 'proper_escalation'
  | 'failed_escalation'
  | 'fast_response'
  | 'slow_response'
  | 'authority_approved'
  | 'authority_rejected';

/** A reputation event for an agent */
export interface ReputationEvent {
  agent_pubkey: AgentPubKey;
  event_type: ReputationEventType;
  context?: string;
  conversation_id?: string;
  timestamp: number;
  weight: number;
}

/** MATL-style trust score snapshot */
export interface TrustScore {
  agent_pubkey: AgentPubKey;
  quality: number;
  consistency: number;
  reputation: number;
  composite: number;
  positive_count: number;
  negative_count: number;
  computed_at: number;
  confidence: number;
}

/** Input for registering an agent */
export interface RegisterAgentInput {
  agent_type: AgentType;
  name: string;
  specializations: AgentSpecialization[];
  description: string;
  config_metadata?: string;
}

/** Input for recording a reputation event */
export interface RecordEventInput {
  agent_pubkey: AgentPubKey;
  event_type: ReputationEventType;
  context?: string;
  conversation_id?: string;
  weight?: number;
}

/** Agent with their trust score */
export interface AgentWithScore {
  pubkey: AgentPubKey;
  profile: AgentProfile;
  trust_score?: TrustScore;
}
