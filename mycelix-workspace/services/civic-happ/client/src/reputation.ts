/**
 * Agent Reputation Client
 *
 * Provides methods to interact with the agent_reputation zome.
 * Implements MATL-based trust scoring for Symthaea AI agents.
 */

import type { AppClient, ActionHash, AgentPubKey, RoleNameCallZomeRequest } from '@holochain/client';
import type {
  AgentProfile,
  AgentSpecialization,
  AgentWithScore,
  RecordEventInput,
  RegisterAgentInput,
  ReputationEvent,
  ReputationEventType,
  TrustScore,
} from './types.js';

export class AgentReputationClient {
  private client: AppClient;
  private roleName: string;
  private zomeName: string;

  constructor(client: AppClient, roleName = 'civic', zomeName = 'agent_reputation') {
    this.client = client;
    this.roleName = roleName;
    this.zomeName = zomeName;
  }

  private async callZome<T>(fnName: string, payload: unknown): Promise<T> {
    const result = await this.client.callZome({
      role_name: this.roleName,
      zome_name: this.zomeName,
      fn_name: fnName,
      payload,
    } as RoleNameCallZomeRequest);
    return result as T;
  }

  /**
   * Register a new agent profile
   */
  async registerAgent(input: RegisterAgentInput): Promise<ActionHash> {
    return this.callZome('register_agent', input);
  }

  /**
   * Get an agent's profile
   */
  async getAgentProfile(agentPubkey: AgentPubKey): Promise<AgentProfile | null> {
    return this.callZome('get_agent_profile', agentPubkey);
  }

  /**
   * Record a reputation event for an agent
   */
  async recordEvent(input: RecordEventInput): Promise<ActionHash> {
    return this.callZome('record_event', input);
  }

  /**
   * Get all reputation events for an agent
   */
  async getAgentEvents(agentPubkey: AgentPubKey): Promise<ReputationEvent[]> {
    return this.callZome('get_agent_events', agentPubkey);
  }

  /**
   * Compute and store a trust score for an agent
   */
  async computeTrustScore(agentPubkey: AgentPubKey): Promise<TrustScore> {
    return this.callZome('compute_trust_score', agentPubkey);
  }

  /**
   * Get the latest trust score for an agent
   */
  async getTrustScore(agentPubkey: AgentPubKey): Promise<TrustScore | null> {
    return this.callZome('get_trust_score', agentPubkey);
  }

  /**
   * Get all active agents with their trust scores
   */
  async getActiveAgents(): Promise<AgentWithScore[]> {
    return this.callZome('get_active_agents', null);
  }

  /**
   * Get agents that meet the MATL trust threshold
   */
  async getTrustworthyAgents(minComposite?: number): Promise<AgentWithScore[]> {
    return this.callZome('get_trustworthy_agents', minComposite ?? null);
  }

  /**
   * Get agents by specialization
   */
  async getAgentsBySpecialization(spec: AgentSpecialization): Promise<AgentWithScore[]> {
    return this.callZome('get_agents_by_specialization', spec);
  }

  /**
   * Get my own profile
   */
  async getMyProfile(): Promise<AgentProfile | null> {
    return this.callZome('get_my_profile', null);
  }

  /**
   * Get my own trust score (compute if needed)
   */
  async getMyTrustScore(): Promise<TrustScore> {
    return this.callZome('get_my_trust_score', null);
  }

  // ========================================================================
  // Convenience Methods
  // ========================================================================

  /**
   * Record a helpful feedback event
   */
  async recordHelpful(
    agentPubkey: AgentPubKey,
    conversationId?: string,
    context?: string,
  ): Promise<ActionHash> {
    return this.recordEvent({
      agent_pubkey: agentPubkey,
      event_type: 'helpful',
      conversation_id: conversationId,
      context,
    });
  }

  /**
   * Record a not helpful feedback event
   */
  async recordNotHelpful(
    agentPubkey: AgentPubKey,
    conversationId?: string,
    context?: string,
  ): Promise<ActionHash> {
    return this.recordEvent({
      agent_pubkey: agentPubkey,
      event_type: 'not_helpful',
      conversation_id: conversationId,
      context,
    });
  }

  /**
   * Record an accuracy event (verified correct response)
   */
  async recordAccurate(
    agentPubkey: AgentPubKey,
    context?: string,
  ): Promise<ActionHash> {
    return this.recordEvent({
      agent_pubkey: agentPubkey,
      event_type: 'accurate',
      context,
    });
  }

  /**
   * Record an inaccuracy event (verified incorrect response)
   */
  async recordInaccurate(
    agentPubkey: AgentPubKey,
    context?: string,
  ): Promise<ActionHash> {
    return this.recordEvent({
      agent_pubkey: agentPubkey,
      event_type: 'inaccurate',
      context,
    });
  }

  /**
   * Check if an agent is trustworthy (meets MATL threshold)
   */
  async isTrustworthy(agentPubkey: AgentPubKey, minComposite = 0.55): Promise<boolean> {
    const score = await this.getTrustScore(agentPubkey);
    if (!score) return false;
    return score.composite >= minComposite && score.confidence >= 0.3;
  }

  /**
   * Get the best agent for a specialization
   * Returns the agent with highest composite score
   */
  async getBestAgentForSpecialization(spec: AgentSpecialization): Promise<AgentWithScore | null> {
    const agents = await this.getAgentsBySpecialization(spec);
    if (agents.length === 0) return null;

    // Sort by composite score (highest first)
    const sorted = agents
      .filter(a => a.trust_score)
      .sort((a, b) => (b.trust_score?.composite ?? 0) - (a.trust_score?.composite ?? 0));

    return sorted[0] ?? null;
  }
}

/**
 * Create an agent reputation client from an AppClient
 */
export function createReputationClient(
  client: AppClient,
  roleName?: string,
): AgentReputationClient {
  return new AgentReputationClient(client, roleName);
}

/**
 * MATL trust score utilities
 */
export const MATL = {
  /** Default Byzantine tolerance threshold */
  TRUST_THRESHOLD: 0.55,

  /** Minimum confidence for trustworthiness */
  MIN_CONFIDENCE: 0.3,

  /** MATL weight for quality */
  QUALITY_WEIGHT: 0.4,

  /** MATL weight for consistency */
  CONSISTENCY_WEIGHT: 0.3,

  /** MATL weight for reputation */
  REPUTATION_WEIGHT: 0.3,

  /**
   * Compute composite score using MATL weights
   */
  computeComposite(quality: number, consistency: number, reputation: number): number {
    return Math.min(
      1.0,
      Math.max(
        0.0,
        this.QUALITY_WEIGHT * quality +
        this.CONSISTENCY_WEIGHT * consistency +
        this.REPUTATION_WEIGHT * reputation,
      ),
    );
  },

  /**
   * Check if a score meets MATL trustworthiness criteria
   */
  isTrustworthy(score: TrustScore): boolean {
    return score.composite >= this.TRUST_THRESHOLD && score.confidence >= this.MIN_CONFIDENCE;
  },

  /**
   * Get trust level description
   */
  getTrustLevel(score: TrustScore): 'untrusted' | 'low' | 'moderate' | 'high' | 'very_high' {
    if (score.confidence < this.MIN_CONFIDENCE) return 'untrusted';
    if (score.composite < 0.4) return 'low';
    if (score.composite < 0.55) return 'moderate';
    if (score.composite < 0.75) return 'high';
    return 'very_high';
  },
};
