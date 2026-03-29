// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Symthaea Agent System
 *
 * "Symthaea" = "growing together" (Greek: σύν + θάλλω)
 *
 * AI agents that grow with citizens, not above them. Symthaea agents:
 * - Empower rather than replace human connection
 * - Know their limits and escalate appropriately
 * - Respect privacy and dignity
 * - Serve all citizens equally
 *
 * Philosophy: When a grandmother texts "HELP" from a flip phone,
 * she deserves the same quality of assistance as someone with
 * a smartphone and broadband.
 *
 * @module symthaea
 */

// Types
export type {
  AgentConfig,
  AgentResponse,
  AgentStats,
  CivicAgentDomain,
  AgentCapabilityLevel,
  ConversationContext,
  ConversationTurn,
  ExtractedEntity,
  EntityType,
  Intent,
  KnowledgeSource,
  KnowledgeSourceType,
  SuggestedAction,
  ResponseMetadata,
  ModelConfig,
  KnowledgeBaseConfig,
  EscalationConfig,
  EscalationChannel,
  PersonalityConfig,
  RateLimitConfig,
  EscalationRequest,
} from './types.js';

// Agent Runner
export {
  AgentRunner,
  createAgentRunner,
  DEFAULT_AGENT_CONFIG,
  type KnowledgeRetriever,
  type ModelClient,
} from './agent-runner.js';

// Civic Agent Configurations
export {
  CIVIC_AGENT_CONFIGS,
  getAgentConfig,
  getAvailableDomains,
  detectDomain,
} from './civic-agents.js';

// Ollama Client
export {
  OllamaClient,
  createOllamaClient,
  createResilientModelClient,
  type OllamaClientConfig,
  DEFAULT_OLLAMA_CONFIG,
} from './ollama-client.js';

// Candle Client (Rust-native inference)
export {
  CandleClient,
  createCandleClient,
  createResilientCandleClient,
  type CandleClientConfig,
  DEFAULT_CANDLE_CONFIG,
} from './candle-client.js';

/**
 * Symthaea Agent Hub
 *
 * Central management for all civic domain agents.
 */
export class SymthaeaHub {
  private agents: Map<string, import('./agent-runner.js').AgentRunner> = new Map();
  private knowledgeRetriever: import('./agent-runner.js').KnowledgeRetriever | null = null;
  private modelClient: import('./agent-runner.js').ModelClient | null = null;

  /**
   * Initialize agents for specified domains
   */
  initialize(
    domains: import('./types.js').CivicAgentDomain[] = ['general']
  ): void {
    const { AgentRunner } = require('./agent-runner.js');
    const { CIVIC_AGENT_CONFIGS } = require('./civic-agents.js');

    for (const domain of domains) {
      const config = CIVIC_AGENT_CONFIGS[domain];
      if (config) {
        const runner = new AgentRunner(config);
        if (this.knowledgeRetriever) {
          runner.setKnowledgeRetriever(this.knowledgeRetriever);
        }
        if (this.modelClient) {
          runner.setModelClient(this.modelClient);
        }
        this.agents.set(domain, runner);
        console.log(`[Symthaea] Initialized ${domain} agent`);
      }
    }
  }

  /**
   * Initialize all civic domain agents
   */
  initializeAll(): void {
    const { getAvailableDomains } = require('./civic-agents.js');
    this.initialize(getAvailableDomains());
  }

  /**
   * Set knowledge retriever for all agents
   */
  setKnowledgeRetriever(
    retriever: import('./agent-runner.js').KnowledgeRetriever
  ): void {
    this.knowledgeRetriever = retriever;
    for (const agent of this.agents.values()) {
      agent.setKnowledgeRetriever(retriever);
    }
  }

  /**
   * Set model client for all agents
   */
  setModelClient(client: import('./agent-runner.js').ModelClient): void {
    this.modelClient = client;
    for (const agent of this.agents.values()) {
      agent.setModelClient(client);
    }
  }

  /**
   * Get agent for a domain
   */
  getAgent(
    domain: import('./types.js').CivicAgentDomain
  ): import('./agent-runner.js').AgentRunner | undefined {
    return this.agents.get(domain);
  }

  /**
   * Process a message, auto-routing to the appropriate agent
   */
  async processMessage(
    message: string,
    options: {
      conversationId: string;
      channel: import('./types.js').ConversationContext['channel'];
      citizenDid?: string;
      preferredDomain?: import('./types.js').CivicAgentDomain;
    }
  ): Promise<import('./types.js').AgentResponse> {
    const { detectDomain } = require('./civic-agents.js');

    // Determine domain
    const domain = options.preferredDomain || detectDomain(message);

    // Get or initialize agent
    let agent = this.agents.get(domain);
    if (!agent) {
      // Fallback to general agent
      agent = this.agents.get('general');
      if (!agent) {
        // Initialize general agent on demand
        this.initialize(['general']);
        agent = this.agents.get('general')!;
      }
    }

    return agent.processMessage(
      options.conversationId,
      message,
      options.channel,
      options.citizenDid
    );
  }

  /**
   * Get all pending escalations across all agents
   */
  getAllPendingEscalations(): import('./types.js').EscalationRequest[] {
    const escalations: import('./types.js').EscalationRequest[] = [];
    for (const agent of this.agents.values()) {
      escalations.push(...agent.getPendingEscalations());
    }
    return escalations.sort(
      (a, b) => b.requestedAt.getTime() - a.requestedAt.getTime()
    );
  }

  /**
   * Get combined statistics
   */
  getStats(): HubStats {
    const stats: HubStats = {
      totalConversations: 0,
      avgConfidence: 0,
      escalationRate: 0,
      byDomain: {} as Record<import('./types.js').CivicAgentDomain, import('./types.js').AgentStats>,
    };

    let totalConfidence = 0;
    let totalEscalations = 0;
    let agentCount = 0;

    for (const [domain, agent] of this.agents.entries()) {
      const agentStats = agent.getStats();
      stats.byDomain[domain as import('./types.js').CivicAgentDomain] = agentStats;
      stats.totalConversations += agentStats.totalConversations;
      totalConfidence += agentStats.avgConfidence;
      totalEscalations += agentStats.escalationRate * agentStats.totalConversations;
      agentCount++;
    }

    if (agentCount > 0) {
      stats.avgConfidence = totalConfidence / agentCount;
      stats.escalationRate =
        stats.totalConversations > 0
          ? totalEscalations / stats.totalConversations
          : 0;
    }

    return stats;
  }

  /**
   * Shutdown all agents
   */
  shutdown(): void {
    for (const [domain, _agent] of this.agents.entries()) {
      // End all active conversations
      console.log(`[Symthaea] Shutting down ${domain} agent`);
    }
    this.agents.clear();
  }
}

/**
 * Hub statistics
 */
export interface HubStats {
  totalConversations: number;
  avgConfidence: number;
  escalationRate: number;
  byDomain: Record<
    import('./types.js').CivicAgentDomain,
    import('./types.js').AgentStats
  >;
}

/**
 * Create a Symthaea hub instance
 */
export function createSymthaeaHub(): SymthaeaHub {
  return new SymthaeaHub();
}

/**
 * SMS Gateway Integration
 *
 * Provides a simple interface for the SMS Gateway to use Symthaea agents.
 */
export class SMSAgentAdapter {
  private hub: SymthaeaHub;
  private conversationMap: Map<string, string> = new Map(); // phone -> conversationId

  constructor(hub: SymthaeaHub) {
    this.hub = hub;
  }

  /**
   * Process an SMS message
   */
  async processMessage(
    phoneNumber: string,
    message: string,
    citizenDid?: string
  ): Promise<{ text: string; actions: import('./types.js').SuggestedAction[] }> {
    // Get or create conversation ID for this phone number
    let conversationId = this.conversationMap.get(phoneNumber);
    if (!conversationId) {
      conversationId = `sms-${phoneNumber}-${Date.now()}`;
      this.conversationMap.set(phoneNumber, conversationId);
    }

    const response = await this.hub.processMessage(message, {
      conversationId,
      channel: 'sms',
      citizenDid,
    });

    // Format response for SMS (160 char limit per message)
    let text = response.text;
    if (text.length > 160) {
      // Split into multiple messages
      text = this.splitForSMS(text);
    }

    // Add escalation info if needed
    if (response.escalateToHuman) {
      const actions = response.suggestedActions.filter((a) => a.type === 'call');
      if (actions.length > 0) {
        text += `\n\nCall ${actions[0].value} for help.`;
      }
    }

    return {
      text,
      actions: response.suggestedActions,
    };
  }

  /**
   * End a conversation for a phone number
   */
  endConversation(phoneNumber: string): void {
    const conversationId = this.conversationMap.get(phoneNumber);
    if (conversationId) {
      // Could notify the agent to cleanup
      this.conversationMap.delete(phoneNumber);
    }
  }

  private splitForSMS(text: string): string {
    // Simple split at sentence boundaries
    const sentences = text.split(/(?<=[.!?])\s+/);
    const messages: string[] = [];
    let current = '';

    for (const sentence of sentences) {
      if ((current + sentence).length <= 155) {
        current += (current ? ' ' : '') + sentence;
      } else {
        if (current) messages.push(current);
        current = sentence;
      }
    }
    if (current) messages.push(current);

    // Return first message (caller should handle pagination)
    return messages[0] + (messages.length > 1 ? ' (1/' + messages.length + ')' : '');
  }
}

/**
 * Create SMS adapter for Symthaea hub
 */
export function createSMSAdapter(hub: SymthaeaHub): SMSAgentAdapter {
  return new SMSAgentAdapter(hub);
}

/**
 * Dashboard Integration
 *
 * Provides chat widget support for the MY GOV dashboard.
 */
export class DashboardAgentAdapter {
  private hub: SymthaeaHub;

  constructor(hub: SymthaeaHub) {
    this.hub = hub;
  }

  /**
   * Process a dashboard chat message
   */
  async processMessage(
    conversationId: string,
    message: string,
    citizenDid: string,
    preferredDomain?: import('./types.js').CivicAgentDomain
  ): Promise<import('./types.js').AgentResponse> {
    return this.hub.processMessage(message, {
      conversationId,
      channel: 'dashboard',
      citizenDid,
      preferredDomain,
    });
  }

  /**
   * Get conversation history
   */
  getHistory(
    conversationId: string,
    domain: import('./types.js').CivicAgentDomain
  ): import('./types.js').ConversationTurn[] {
    const agent = this.hub.getAgent(domain);
    if (agent) {
      return agent.getConversationHistory(conversationId);
    }
    return [];
  }

  /**
   * Get available domains for UI
   */
  getAvailableDomains(): Array<{
    domain: import('./types.js').CivicAgentDomain;
    name: string;
    description: string;
  }> {
    const { CIVIC_AGENT_CONFIGS } = require('./civic-agents.js');
    return Object.entries(CIVIC_AGENT_CONFIGS).map(([domain, config]) => ({
      domain: domain as import('./types.js').CivicAgentDomain,
      name: (config as import('./types.js').AgentConfig).name,
      description: this.getDomainDescription(
        domain as import('./types.js').CivicAgentDomain
      ),
    }));
  }

  private getDomainDescription(
    domain: import('./types.js').CivicAgentDomain
  ): string {
    const descriptions: Record<import('./types.js').CivicAgentDomain, string> =
      {
        benefits: 'SNAP, Medicaid, housing assistance, and other benefits',
        permits: 'Business licenses, building permits, and more',
        tax: 'Tax questions, credits, and free filing resources',
        voting: 'Voter registration, polling places, and ballot info',
        justice: 'Court information, legal aid, and expungement',
        housing: 'Affordable housing, tenant rights, and assistance',
        employment: 'Jobs, unemployment, and workplace issues',
        education: 'Schools, financial aid, and adult education',
        health: 'Healthcare coverage, clinics, and mental health',
        emergency: 'Crisis resources and disaster assistance',
        general: 'General questions about government services',
      };
    return descriptions[domain];
  }
}

/**
 * Create dashboard adapter for Symthaea hub
 */
export function createDashboardAdapter(
  hub: SymthaeaHub
): DashboardAgentAdapter {
  return new DashboardAgentAdapter(hub);
}
