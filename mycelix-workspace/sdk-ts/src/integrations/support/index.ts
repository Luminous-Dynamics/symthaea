// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Community Support Integration
 *
 * Domain-specific client for the community support zomes within the
 * mycelix-commons cluster DNA. Covers knowledge (articles, resolutions,
 * reputation), tickets (support tickets, escalation, preemptive alerts,
 * satisfaction surveys), and diagnostics (system checks, helpers,
 * cognitive updates, privacy preferences).
 *
 * All calls are dispatched through the commons_bridge coordinator zome.
 *
 * @packageDocumentation
 * @module integrations/support
 */

// ============================================================================
// Types — Shared Enums
// ============================================================================

export type SupportCategory =
  | 'Network'
  | 'Hardware'
  | 'Software'
  | 'Holochain'
  | 'Mycelix'
  | 'Security'
  | 'General';

export type TicketPriority = 'Low' | 'Medium' | 'High' | 'Critical';

export type TicketStatus =
  | 'Open'
  | 'InProgress'
  | 'AwaitingUser'
  | 'Resolved'
  | 'Closed';

export type AutonomyLevel = 'Advisory' | 'SemiAutonomous' | 'FullAutonomous';

export type ActionType =
  | 'RestartService'
  | 'ClearCache'
  | 'UpdateConfig'
  | 'RunDiagnostic'
  | { Custom: string };

export type SharingTier = 'LocalOnly' | 'Anonymized' | 'Full';

export type DiagnosticType =
  | 'NetworkCheck'
  | 'DiskSpace'
  | 'ServiceStatus'
  | 'HolochainHealth'
  | 'MemoryUsage'
  | { Custom: string };

export type DiagnosticSeverity = 'Healthy' | 'Warning' | 'Error' | 'Critical';

export type DifficultyLevel = 'Beginner' | 'Intermediate' | 'Advanced';

export type ArticleSource = 'Community' | 'PreSeeded' | 'SymthaeaGenerated';

export type FlagReason = 'Harmful' | 'Incorrect' | 'Outdated' | 'Spam';

export type EscalationLevel = 'Tier1' | 'Tier2' | 'Management' | 'Emergency';

export type EpistemicStatus =
  | 'Certain'
  | 'Probable'
  | 'Uncertain'
  | 'Unknown'
  | 'OutOfDomain';

export type LinkReason =
  | 'SuggestedFAQ'
  | 'DuplicateResolution'
  | 'RelatedKnowledge';

// ============================================================================
// Types — Knowledge
// ============================================================================

export interface KnowledgeArticle {
  title: string;
  content: string;
  category: SupportCategory;
  tags: string[];
  author: Uint8Array;
  source: ArticleSource;
  difficulty_level: DifficultyLevel;
  upvotes: number;
  verified: boolean;
  deprecated: boolean;
  deprecation_reason: string | null;
  version: number;
}

export interface UpdateArticleInput {
  original_hash: Uint8Array;
  updated: KnowledgeArticle;
}

export interface DeprecateInput {
  article_hash: Uint8Array;
  reason: string;
}

export interface ArticleFlag {
  article_hash: Uint8Array;
  flagger: Uint8Array;
  reason: FlagReason;
  description: string;
  created_at: number;
}

export interface Resolution {
  ticket_hash: Uint8Array;
  steps: string[];
  root_cause: string | null;
  time_to_resolve_mins: number | null;
  effectiveness_rating: number | null;
  helper: Uint8Array;
  requester: Uint8Array;
  anonymized: boolean;
  helper_signature: Uint8Array;
  requester_signature: Uint8Array;
}

export interface LinkArticleInput {
  article_hash: Uint8Array;
  ticket_hash: Uint8Array;
  link_reason: LinkReason;
}

// ============================================================================
// Types — Tickets
// ============================================================================

export interface SupportTicket {
  title: string;
  description: string;
  category: SupportCategory;
  priority: TicketPriority;
  status: TicketStatus;
  requester: Uint8Array;
  assignee: Uint8Array | null;
  autonomy_level: AutonomyLevel;
  system_info: string | null;
  is_preemptive: boolean;
  prediction_confidence: number | null;
  created_at: number;
  updated_at: number;
}

export interface UpdateTicketInput {
  original_hash: Uint8Array;
  updated: SupportTicket;
}

export interface TicketComment {
  ticket_hash: Uint8Array;
  author: Uint8Array;
  content: string;
  is_symthaea_response: boolean;
  confidence: number | null;
  epistemic_status: EpistemicStatus | null;
}

export interface AutonomousAction {
  ticket_hash: Uint8Array;
  action_type: ActionType;
  description: string;
  approved: boolean;
  executed: boolean;
  result: string | null;
  success: boolean | null;
  rollback_steps: string[] | null;
  rollback_state: string | null;
  rolled_back: boolean;
  created_at: number;
}

export interface UndoAction {
  original_action_hash: Uint8Array;
  reason: string;
  rollback_result: string;
  created_at: number;
}

export interface PreemptiveAlert {
  predicted_failure: string;
  expected_time_to_failure: string | null;
  free_energy: number;
  recommended_action: string;
  auto_generated_ticket: Uint8Array | null;
  created_at: number;
}

export interface PromoteAlertInput {
  alert_hash: Uint8Array;
  ticket: SupportTicket;
}

export interface EscalateInput {
  ticket_hash: Uint8Array;
  from_level: EscalationLevel;
  to_level: EscalationLevel;
  reason: string;
}

export interface SatisfactionSurvey {
  ticket_hash: Uint8Array;
  respondent: Uint8Array;
  rating: number;
  comment: string | null;
  would_recommend: boolean;
  submitted_at: number;
}

// ============================================================================
// Types — Diagnostics
// ============================================================================

export interface DiagnosticResult {
  ticket_hash: Uint8Array | null;
  diagnostic_type: DiagnosticType;
  findings: string;
  severity: DiagnosticSeverity;
  recommendations: string[];
  agent: Uint8Array;
  scrubbed: boolean;
  created_at: number;
}

export interface PrivacyPreference {
  agent: Uint8Array;
  sharing_tier: SharingTier;
  allowed_categories: SupportCategory[];
  share_system_info: boolean;
  share_resolution_patterns: boolean;
  share_cognitive_updates: boolean;
  updated_at: number;
}

export interface HelperProfile {
  agent: Uint8Array;
  expertise_categories: SupportCategory[];
  max_concurrent: number;
  difficulty_preference: DifficultyLevel;
  available: boolean;
  created_at: number;
}

export interface UpdateAvailInput {
  helper_hash: Uint8Array;
  available: boolean;
}

export interface CognitiveUpdate {
  category: SupportCategory;
  encoding: Uint8Array;
  phi: number;
  resolution_pattern: string;
  source_agent: Uint8Array;
  created_at: number;
}

// ============================================================================
// Holochain ZomeCallable interface (minimal)
// ============================================================================

interface ZomeCallable {
  callZome<T>(params: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<T>;
}

// ============================================================================
// Constants
// ============================================================================

const COMMONS_ROLE = 'commons_care';

export const SUPPORT_ZOMES = [
  'support_knowledge',
  'support_tickets',
  'support_diagnostics',
] as const;

// ============================================================================
// Support Client
// ============================================================================

/**
 * Client for the community support domain within the commons cluster.
 *
 * Provides typed access to all 3 support zomes: knowledge, tickets,
 * and diagnostics.
 */
export class SupportClient {
  constructor(private readonly client: ZomeCallable) {}

  // --- Knowledge ---

  async createArticle(article: KnowledgeArticle): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'create_article',
      payload: article,
    });
  }

  async updateArticle(input: UpdateArticleInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'update_article',
      payload: input,
    });
  }

  async deprecateArticle(input: DeprecateInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'deprecate_article',
      payload: input,
    });
  }

  async getArticle(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'get_article',
      payload: actionHash,
    });
  }

  async searchByCategory(category: SupportCategory): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'search_by_category',
      payload: category,
    });
  }

  async searchByTag(tag: string): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'search_by_tag',
      payload: tag,
    });
  }

  async listRecentArticles(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'list_recent_articles',
      payload: null,
    });
  }

  async upvoteArticle(articleHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'upvote_article',
      payload: articleHash,
    });
  }

  async verifyArticle(articleHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'verify_article',
      payload: articleHash,
    });
  }

  async createResolution(resolution: Resolution): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'create_resolution',
      payload: resolution,
    });
  }

  async flagArticle(flag: ArticleFlag): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'flag_article',
      payload: flag,
    });
  }

  async getFlags(articleHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'get_flags',
      payload: articleHash,
    });
  }

  async getAgentReputation(agent: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'get_agent_reputation',
      payload: agent,
    });
  }

  async linkArticleToTicket(input: LinkArticleInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'link_article_to_ticket',
      payload: input,
    });
  }

  async getSuggestedArticles(ticketHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'get_suggested_articles',
      payload: ticketHash,
    });
  }

  async getTicketsForArticle(articleHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_knowledge',
      fn_name: 'get_tickets_for_article',
      payload: articleHash,
    });
  }

  // --- Tickets ---

  async createTicket(ticket: SupportTicket): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'create_ticket',
      payload: ticket,
    });
  }

  async updateTicket(input: UpdateTicketInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'update_ticket',
      payload: input,
    });
  }

  async getTicket(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'get_ticket',
      payload: actionHash,
    });
  }

  async listTicketsByStatus(status: TicketStatus): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'list_tickets_by_status',
      payload: status,
    });
  }

  async listMyTickets(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'list_my_tickets',
      payload: null,
    });
  }

  async closeTicket(ticketHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'close_ticket',
      payload: ticketHash,
    });
  }

  async addComment(comment: TicketComment): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'add_comment',
      payload: comment,
    });
  }

  async getComments(ticketHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'get_comments',
      payload: ticketHash,
    });
  }

  async proposeAction(action: AutonomousAction): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'propose_action',
      payload: action,
    });
  }

  async approveAction(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'approve_action',
      payload: actionHash,
    });
  }

  async executeAction(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'execute_action',
      payload: actionHash,
    });
  }

  async rollbackAction(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'rollback_action',
      payload: actionHash,
    });
  }

  async createUndo(undo: UndoAction): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'create_undo',
      payload: undo,
    });
  }

  async createPreemptiveAlert(alert: PreemptiveAlert): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'create_preemptive_alert',
      payload: alert,
    });
  }

  async listPreemptiveAlerts(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'list_preemptive_alerts',
      payload: null,
    });
  }

  async promoteAlertToTicket(input: PromoteAlertInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'promote_alert_to_ticket',
      payload: input,
    });
  }

  async escalateTicket(input: EscalateInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'escalate_ticket',
      payload: input,
    });
  }

  async getEscalationHistory(ticketHash: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'get_escalation_history',
      payload: ticketHash,
    });
  }

  async submitSatisfaction(survey: SatisfactionSurvey): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'submit_satisfaction',
      payload: survey,
    });
  }

  async getTicketSatisfaction(ticketHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_tickets',
      fn_name: 'get_ticket_satisfaction',
      payload: ticketHash,
    });
  }

  // --- Diagnostics ---

  async runDiagnostic(diagnostic: DiagnosticResult): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'run_diagnostic',
      payload: diagnostic,
    });
  }

  async getDiagnostic(actionHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'get_diagnostic',
      payload: actionHash,
    });
  }

  async listDiagnostics(): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'list_diagnostics',
      payload: null,
    });
  }

  async setPrivacyPreference(pref: PrivacyPreference): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'set_privacy_preference',
      payload: pref,
    });
  }

  async getPrivacyPreference(agent: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'get_privacy_preference',
      payload: agent,
    });
  }

  async getShareableDiagnostics(agent: Uint8Array): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'get_shareable_diagnostics',
      payload: agent,
    });
  }

  async registerHelper(profile: HelperProfile): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'register_helper',
      payload: profile,
    });
  }

  async updateAvailability(input: UpdateAvailInput): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'update_availability',
      payload: input,
    });
  }

  async getAvailableHelpers(
    category: SupportCategory | null,
  ): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'get_available_helpers',
      payload: category,
    });
  }

  async publishCognitiveUpdate(update: CognitiveUpdate): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'publish_cognitive_update',
      payload: update,
    });
  }

  async getCognitiveUpdatesByCategory(
    category: SupportCategory,
  ): Promise<unknown[]> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'get_cognitive_updates_by_category',
      payload: category,
    });
  }

  async absorbCognitiveUpdate(updateHash: Uint8Array): Promise<unknown> {
    return this.client.callZome({
      role_name: COMMONS_ROLE,
      zome_name: 'support_diagnostics',
      fn_name: 'absorb_cognitive_update',
      payload: updateHash,
    });
  }
}

// ============================================================================
// Factory
// ============================================================================

/** Create a SupportClient from an AppWebsocket or compatible client */
export function createSupportClient(client: ZomeCallable): SupportClient {
  return new SupportClient(client);
}
