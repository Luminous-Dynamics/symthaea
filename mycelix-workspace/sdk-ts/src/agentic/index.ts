/**
 * @mycelix/sdk Agentic Framework
 *
 * TypeScript SDK for managing AI agents with epistemic-aware agency.
 * This module provides types and client functions for:
 * - Agent lifecycle management
 * - K-Vector trust profiles
 * - Moral uncertainty and escalation
 * - Gaming detection
 * - Coherence (Phi) measurement
 *
 * @example
 * ```typescript
 * import { agentic } from '@mycelix/sdk';
 *
 * // Create agent client
 * const client = new agentic.AgentApiClient('http://localhost:8080');
 *
 * // Create a supervised agent
 * const response = await client.createAgent({
 *   sponsor_did: 'did:mycelix:sponsor123',
 *   agent_class: agentic.AgentClass.Supervised,
 *   initial_kredit: 5000,
 * });
 *
 * // Get pending escalations
 * const escalations = await client.getPendingEscalations(response.agent.agent_id);
 *
 * // Resolve an escalation
 * await client.resolveEscalation(
 *   response.agent.agent_id,
 *   'transfer_funds',
 *   true, // approved
 *   'did:mycelix:sponsor123'
 * );
 * ```
 *
 * @packageDocumentation
 * @module agentic
 */

// Export all types
export * from './types.js';

// Re-export specific GIS types used in agentic context
export {
  MoralUncertaintyType,
  MoralActionGuidance,
  createMoralUncertainty,
  totalMoralUncertainty,
  getMoralActionGuidance,
  getMoralRecommendations,
  shouldPauseForReflection,
  type MoralUncertainty,
} from '../epistemic/gis.js';


import {
  AgentClass,
  type AgentStatus,
  MIN_KREDIT_CAP,
  MAX_KREDIT_CAP,
  MAX_SPONSOR_DID_LENGTH,
  MAX_AGENT_ID_LENGTH,
 type KVectorValues ,
  type CreateAgentRequest,
  type CreateAgentResponse,
  type UpdateAgentRequest,
  type AgentSummary,
  type ListAgentsResponse,
  type KVectorHistoryResponse,
  type EventsResponse,
  type EscalationSummary,
  type EscalationResolutionResponse,
  type CalibrationSummary,
  type ApiError} from './types.js';

// =============================================================================
// Validation
// =============================================================================

/**
 * Validation error for agent requests
 */
export class AgentValidationError extends Error {
  public readonly code: string;
  public readonly field?: string;

  constructor(message: string, code: string, field?: string) {
    super(message);
    this.name = 'AgentValidationError';
    this.code = code;
    this.field = field;
  }
}

/**
 * Validate a CreateAgentRequest
 * @throws {AgentValidationError} If validation fails
 */
export function validateCreateAgentRequest(request: CreateAgentRequest): void {
  // Validate sponsor_did
  if (!request.sponsor_did) {
    throw new AgentValidationError(
      'sponsor_did cannot be empty',
      'VALIDATION_ERROR',
      'sponsor_did'
    );
  }
  if (request.sponsor_did.length > MAX_SPONSOR_DID_LENGTH) {
    throw new AgentValidationError(
      `sponsor_did exceeds maximum length of ${MAX_SPONSOR_DID_LENGTH} characters`,
      'VALIDATION_ERROR',
      'sponsor_did'
    );
  }
  if (!request.sponsor_did.startsWith('did:')) {
    throw new AgentValidationError(
      "sponsor_did must be a valid DID (start with 'did:')",
      'VALIDATION_ERROR',
      'sponsor_did'
    );
  }

  // Validate agent_id if provided
  if (request.agent_id !== undefined) {
    if (request.agent_id === '') {
      throw new AgentValidationError(
        'agent_id cannot be empty if provided',
        'VALIDATION_ERROR',
        'agent_id'
      );
    }
    if (request.agent_id.length > MAX_AGENT_ID_LENGTH) {
      throw new AgentValidationError(
        `agent_id exceeds maximum length of ${MAX_AGENT_ID_LENGTH} characters`,
        'VALIDATION_ERROR',
        'agent_id'
      );
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(request.agent_id)) {
      throw new AgentValidationError(
        'agent_id can only contain alphanumeric characters, dashes, and underscores',
        'VALIDATION_ERROR',
        'agent_id'
      );
    }
  }

  // Validate KREDIT values
  const kreditCap = request.kredit_cap ?? 10000;
  if (kreditCap < MIN_KREDIT_CAP) {
    throw new AgentValidationError(
      `kredit_cap must be at least ${MIN_KREDIT_CAP}`,
      'VALIDATION_ERROR',
      'kredit_cap'
    );
  }
  if (kreditCap > MAX_KREDIT_CAP) {
    throw new AgentValidationError(
      `kredit_cap cannot exceed ${MAX_KREDIT_CAP}`,
      'VALIDATION_ERROR',
      'kredit_cap'
    );
  }

  const initialKredit = request.initial_kredit ?? 5000;
  if (initialKredit < 0) {
    throw new AgentValidationError(
      'initial_kredit cannot be negative',
      'VALIDATION_ERROR',
      'initial_kredit'
    );
  }
  if (initialKredit > kreditCap) {
    throw new AgentValidationError(
      'initial_kredit cannot exceed kredit_cap',
      'VALIDATION_ERROR',
      'initial_kredit'
    );
  }

  // Validate agent_class
  if (!Object.values(AgentClass).includes(request.agent_class)) {
    throw new AgentValidationError(
      `Invalid agent_class: ${request.agent_class}`,
      'VALIDATION_ERROR',
      'agent_class'
    );
  }
}

// =============================================================================
// API Client
// =============================================================================

/**
 * API client configuration
 */
export interface AgentApiClientConfig {
  /** Base URL for the API */
  baseUrl: string;
  /** Request timeout in milliseconds */
  timeout?: number;
  /** Custom headers */
  headers?: Record<string, string>;
}

/**
 * Agent API client for REST operations
 */
export class AgentApiClient {
  private readonly baseUrl: string;
  private readonly timeout: number;
  private readonly headers: Record<string, string>;

  constructor(configOrUrl: string | AgentApiClientConfig) {
    if (typeof configOrUrl === 'string') {
      this.baseUrl = configOrUrl.replace(/\/$/, '');
      this.timeout = 30000;
      this.headers = {};
    } else {
      this.baseUrl = configOrUrl.baseUrl.replace(/\/$/, '');
      this.timeout = configOrUrl.timeout ?? 30000;
      this.headers = configOrUrl.headers ?? {};
    }
  }

  /**
   * Make an API request
   */
  private async request<T>(
    method: string,
    path: string,
    body?: unknown
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...this.headers,
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const error: ApiError = await response.json().catch(() => ({
          code: 'UNKNOWN_ERROR',
          message: `HTTP ${response.status}: ${response.statusText}`,
        }));
        throw new AgentApiError(error.message, error.code, error.details);
      }

      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof AgentApiError) {
        throw error;
      }
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AgentApiError('Request timeout', 'TIMEOUT');
        }
        throw new AgentApiError(error.message, 'NETWORK_ERROR');
      }
      throw new AgentApiError('Unknown error', 'UNKNOWN_ERROR');
    }
  }

  // -------------------------------------------------------------------------
  // Agent CRUD
  // -------------------------------------------------------------------------

  /**
   * List all agents
   */
  async listAgents(offset = 0, limit = 100): Promise<ListAgentsResponse> {
    return this.request<ListAgentsResponse>(
      'GET',
      `/agents?offset=${offset}&limit=${limit}`
    );
  }

  /**
   * Get agent by ID
   */
  async getAgent(agentId: string): Promise<AgentSummary> {
    return this.request<AgentSummary>('GET', `/agents/${encodeURIComponent(agentId)}`);
  }

  /**
   * Create a new agent
   */
  async createAgent(request: CreateAgentRequest): Promise<CreateAgentResponse> {
    validateCreateAgentRequest(request);
    return this.request<CreateAgentResponse>('POST', '/agents', request);
  }

  /**
   * Update an agent
   */
  async updateAgent(
    agentId: string,
    request: UpdateAgentRequest
  ): Promise<AgentSummary> {
    return this.request<AgentSummary>(
      'PUT',
      `/agents/${encodeURIComponent(agentId)}`,
      request
    );
  }

  /**
   * Delete an agent
   */
  async deleteAgent(agentId: string): Promise<void> {
    await this.request<void>('DELETE', `/agents/${encodeURIComponent(agentId)}`);
  }

  // -------------------------------------------------------------------------
  // K-Vector History
  // -------------------------------------------------------------------------

  /**
   * Get K-Vector history for an agent
   */
  async getKVectorHistory(
    agentId: string,
    limit?: number
  ): Promise<KVectorHistoryResponse> {
    const query = limit !== undefined ? `?limit=${limit}` : '';
    return this.request<KVectorHistoryResponse>(
      'GET',
      `/agents/${encodeURIComponent(agentId)}/kvector/history${query}`
    );
  }

  // -------------------------------------------------------------------------
  // Events
  // -------------------------------------------------------------------------

  /**
   * Get events for an agent
   */
  async getAgentEvents(
    agentId: string,
    since?: number,
    limit?: number
  ): Promise<EventsResponse> {
    const params = new URLSearchParams();
    if (since !== undefined) params.set('since', since.toString());
    if (limit !== undefined) params.set('limit', limit.toString());
    const query = params.toString() ? `?${params}` : '';
    return this.request<EventsResponse>(
      'GET',
      `/agents/${encodeURIComponent(agentId)}/events${query}`
    );
  }

  // -------------------------------------------------------------------------
  // Escalation Management (GIS Integration)
  // -------------------------------------------------------------------------

  /**
   * Get pending escalations for an agent
   */
  async getPendingEscalations(agentId: string): Promise<EscalationSummary[]> {
    return this.request<EscalationSummary[]>(
      'GET',
      `/agents/${encodeURIComponent(agentId)}/escalations`
    );
  }

  /**
   * Resolve an escalation (sponsor decision)
   */
  async resolveEscalation(
    agentId: string,
    blockedAction: string,
    approved: boolean,
    sponsorDid: string
  ): Promise<EscalationResolutionResponse> {
    return this.request<EscalationResolutionResponse>(
      'POST',
      `/agents/${encodeURIComponent(agentId)}/escalations/resolve`,
      {
        blocked_action: blockedAction,
        approved,
        sponsor_did: sponsorDid,
      }
    );
  }

  /**
   * Get uncertainty calibration for an agent
   */
  async getCalibration(agentId: string): Promise<CalibrationSummary> {
    return this.request<CalibrationSummary>(
      'GET',
      `/agents/${encodeURIComponent(agentId)}/calibration`
    );
  }

  // -------------------------------------------------------------------------
  // Bulk Operations
  // -------------------------------------------------------------------------

  /**
   * Find agents by sponsor
   */
  async findBySponsor(sponsorDid: string): Promise<AgentSummary[]> {
    return this.request<AgentSummary[]>(
      'GET',
      `/agents?sponsor=${encodeURIComponent(sponsorDid)}`
    );
  }

  /**
   * Find agents by status
   */
  async findByStatus(status: AgentStatus): Promise<AgentSummary[]> {
    return this.request<AgentSummary[]>(
      'GET',
      `/agents?status=${encodeURIComponent(status)}`
    );
  }
}

/**
 * API error class
 */
export class AgentApiError extends Error {
  public readonly code: string;
  public readonly details?: string;

  constructor(message: string, code: string, details?: string) {
    super(message);
    this.name = 'AgentApiError';
    this.code = code;
    this.details = details;
  }

  /**
   * Check if this is a "not found" error
   */
  isNotFound(): boolean {
    return this.code === 'NOT_FOUND';
  }

  /**
   * Check if this is a validation error
   */
  isValidationError(): boolean {
    return this.code === 'VALIDATION_ERROR';
  }

  /**
   * Check if this is an authorization error
   */
  isForbidden(): boolean {
    return this.code === 'FORBIDDEN';
  }
}

// =============================================================================
// Trust Score Utilities
// =============================================================================


/**
 * Compute trust score from K-Vector values
 *
 * Formula: weighted average of K-Vector dimensions
 * Default weights emphasize reputation, integrity, and performance
 */
export function computeTrustScore(
  kvector: KVectorValues,
  weights?: Partial<Record<keyof KVectorValues, number>>
): number {
  const defaultWeights: Record<keyof KVectorValues, number> = {
    k_r: 0.20, // Reputation
    k_a: 0.10, // Activity
    k_i: 0.20, // Integrity
    k_p: 0.15, // Performance
    k_m: 0.10, // Membership
    k_s: 0.05, // Stake
    k_h: 0.10, // Historical
    k_topo: 0.05, // Topology
    k_v: 0.025, // Verification
    k_coherence: 0.025, // Coherence
  };

  const mergedWeights = { ...defaultWeights, ...weights };

  let score = 0;
  let totalWeight = 0;

  for (const [key, weight] of Object.entries(mergedWeights)) {
    const value = kvector[key as keyof KVectorValues];
    if (value !== undefined) {
      score += value * weight;
      totalWeight += weight;
    }
  }

  return totalWeight > 0 ? score / totalWeight : 0;
}

/**
 * Calculate KREDIT cap from trust score
 *
 * Higher trust = higher KREDIT cap
 */
export function calculateKreditFromTrust(
  trustScore: number,
  baseKredit = 5000,
  maxMultiplier = 10
): number {
  // Clamp trust score to [0, 1]
  const clampedTrust = Math.max(0, Math.min(1, trustScore));

  // Linear scaling: trust 0 = base, trust 1 = base * maxMultiplier
  const multiplier = 1 + (maxMultiplier - 1) * clampedTrust;

  return Math.floor(baseKredit * multiplier);
}

// =============================================================================
// Action Gating Utilities
// =============================================================================

import { getMoralActionGuidance, type MoralUncertainty } from '../epistemic/gis.js';
import { MoralActionGuidance } from '../epistemic/gis.js';

/**
 * Result of uncertainty gating check
 */
export type UncertaintyCheckResult =
  | { type: 'proceed' }
  | { type: 'proceed_with_monitoring'; reason: string }
  | { type: 'escalation_required'; escalation: EscalationSummary }
  | { type: 'blocked'; reason: string };

/**
 * Check if an action should proceed based on moral uncertainty
 */
export function gateActionOnUncertainty(
  agentId: string,
  uncertainty: MoralUncertainty,
  action: string,
  _context?: string  // Prefix with underscore to indicate intentionally unused
): UncertaintyCheckResult {
  const guidance = getMoralActionGuidance(uncertainty);
  const total = Math.sqrt(
    (uncertainty.epistemic ** 2 +
      uncertainty.axiological ** 2 +
      uncertainty.deontic ** 2) /
      3
  );
  const maxDimension = Math.max(
    uncertainty.epistemic,
    uncertainty.axiological,
    uncertainty.deontic
  );

  switch (guidance) {
    case MoralActionGuidance.ProceedConfidently:
      return { type: 'proceed' };

    case MoralActionGuidance.ProceedWithMonitoring:
      return {
        type: 'proceed_with_monitoring',
        reason: `Moderate uncertainty (total: ${total.toFixed(2)})`,
      };

    case MoralActionGuidance.PauseForReflection:
    case MoralActionGuidance.SeekConsultation:
      return {
        type: 'escalation_required',
        escalation: {
          agent_id: agentId,
          blocked_action: action,
          uncertainty_total: total,
          uncertainty_max: maxDimension,
          guidance: guidance,
          recommendations: getRecommendations(uncertainty),
          timestamp: Date.now(),
        },
      };

    case MoralActionGuidance.DeferAction:
      return {
        type: 'blocked',
        reason: `Uncertainty too high for action (total: ${total.toFixed(2)})`,
      };
  }
}

/**
 * Get recommendations based on uncertainty profile
 */
function getRecommendations(uncertainty: MoralUncertainty): string[] {
  const recommendations: string[] = [];

  if (uncertainty.epistemic > 0.5) {
    recommendations.push('Gather more factual information');
    recommendations.push('Consult domain experts');
  }

  if (uncertainty.axiological > 0.5) {
    recommendations.push('Reflect on relevant values');
    recommendations.push('Consider stakeholder perspectives');
  }

  if (uncertainty.deontic > 0.5) {
    recommendations.push('Explore alternative actions');
    recommendations.push('Consider reversibility');
  }

  return recommendations.length > 0
    ? recommendations
    : ['Uncertainty levels acceptable'];
}

// =============================================================================
// New Module Exports
// =============================================================================

// Differential Privacy
export * from './differential-privacy.js';

// Dashboard
export * from './dashboard.js';

// Adaptive Thresholds
export * from './adaptive-thresholds.js';

// Verification
export * from './verification.js';

// Integration Flows
export * from './integration.js';
