// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Symthaea Module Tests
 *
 * Tests for the Symthaea civic AI agent system including:
 * - SymthaeaHub lifecycle
 * - SMSAgentAdapter text splitting
 * - DashboardAgentAdapter creation
 * - AgentRunner and civic agent configs (imported directly)
 *
 * Note: SymthaeaHub.initialize() uses CommonJS require() internally,
 * so we test the AgentRunner and civic configs directly via ESM imports.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  SymthaeaHub,
  createSymthaeaHub,
  SMSAgentAdapter,
  createSMSAdapter,
  DashboardAgentAdapter,
  createDashboardAdapter,
} from '../src/symthaea/index.js';

import {
  AgentRunner,
  createAgentRunner,
  DEFAULT_AGENT_CONFIG,
} from '../src/symthaea/agent-runner.js';

import {
  CIVIC_AGENT_CONFIGS,
  getAgentConfig,
  getAvailableDomains,
  detectDomain,
} from '../src/symthaea/civic-agents.js';

// =============================================================================
// Civic Agent Config Tests
// =============================================================================

describe('civic-agents', () => {
  describe('getAvailableDomains', () => {
    it('should return array of domains', () => {
      const domains = getAvailableDomains();
      expect(Array.isArray(domains)).toBe(true);
      expect(domains.length).toBeGreaterThan(0);
    });

    it('should include general domain', () => {
      expect(getAvailableDomains()).toContain('general');
    });

    it('should include benefits domain', () => {
      expect(getAvailableDomains()).toContain('benefits');
    });

    it('should include voting domain', () => {
      expect(getAvailableDomains()).toContain('voting');
    });
  });

  describe('getAgentConfig', () => {
    it('should return config for general domain', () => {
      const config = getAgentConfig('general');
      expect(config).toBeDefined();
      expect(config!.name).toBeDefined();
      expect(config!.domain).toBe('general');
    });

    it('should return config for benefits domain', () => {
      const config = getAgentConfig('benefits');
      expect(config).toBeDefined();
      expect(config!.domain).toBe('benefits');
    });

    it('should return undefined for unknown domain', () => {
      const config = getAgentConfig('nonexistent' as any);
      expect(config).toBeUndefined();
    });
  });

  describe('CIVIC_AGENT_CONFIGS', () => {
    it('should have configs for all available domains', () => {
      const domains = getAvailableDomains();
      for (const domain of domains) {
        expect(CIVIC_AGENT_CONFIGS[domain]).toBeDefined();
        expect(CIVIC_AGENT_CONFIGS[domain].name).toBeDefined();
      }
    });

    it('should have personality configs', () => {
      const config = CIVIC_AGENT_CONFIGS.general;
      expect(config.personality).toBeDefined();
      expect(config.personality.tone).toBeDefined();
    });

    it('should have escalation configs', () => {
      const config = CIVIC_AGENT_CONFIGS.general;
      expect(config.escalation).toBeDefined();
      expect(config.escalation.channels).toBeDefined();
    });
  });

  describe('detectDomain', () => {
    it('should detect benefits-related messages', () => {
      expect(detectDomain('Am I eligible for SNAP benefits?')).toBe('benefits');
    });

    it('should detect voting-related messages', () => {
      expect(detectDomain('How do I register to vote?')).toBe('voting');
    });

    it('should detect tax-related messages', () => {
      expect(detectDomain('How do I file my taxes?')).toBe('tax');
    });

    it('should detect health-related messages', () => {
      expect(detectDomain('I need to find a health clinic')).toBe('health');
    });

    it('should detect housing-related messages', () => {
      expect(detectDomain('I need help finding affordable housing')).toBe('housing');
    });

    it('should detect employment-related messages', () => {
      expect(detectDomain('How do I file for unemployment?')).toBe('employment');
    });

    it('should detect justice-related messages', () => {
      expect(detectDomain('I need to find a lawyer for court')).toBe('justice');
    });

    it('should detect permits-related messages', () => {
      expect(detectDomain('How do I get a building permit?')).toBe('permits');
    });

    it('should detect emergency-related messages', () => {
      expect(detectDomain('This is an emergency crisis situation')).toBe('emergency');
    });

    it('should default to general for unrecognized messages', () => {
      expect(detectDomain('Hello there')).toBe('general');
    });
  });
});

// =============================================================================
// AgentRunner Tests
// =============================================================================

describe('AgentRunner', () => {
  let runner: AgentRunner;

  beforeEach(() => {
    runner = createAgentRunner(CIVIC_AGENT_CONFIGS.general);
  });

  describe('creation', () => {
    it('should create with factory function', () => {
      expect(runner).toBeInstanceOf(AgentRunner);
    });

    it('should create with constructor', () => {
      const r = new AgentRunner(CIVIC_AGENT_CONFIGS.general);
      expect(r).toBeInstanceOf(AgentRunner);
    });

    it('should create for different domains', () => {
      for (const domain of getAvailableDomains()) {
        const r = new AgentRunner(CIVIC_AGENT_CONFIGS[domain]);
        expect(r).toBeInstanceOf(AgentRunner);
      }
    });
  });

  describe('message processing', () => {
    it('should process a message', async () => {
      const response = await runner.processMessage('conv-1', 'Hello', 'web');
      expect(response).toBeDefined();
      expect(response.text).toBeDefined();
      expect(response.text.length).toBeGreaterThan(0);
      expect(response.confidence).toBeGreaterThanOrEqual(0);
      expect(response.confidence).toBeLessThanOrEqual(1);
    });

    it('should return suggested actions', async () => {
      const response = await runner.processMessage('conv-2', 'What can I do?', 'web');
      expect(response.suggestedActions).toBeDefined();
      expect(Array.isArray(response.suggestedActions)).toBe(true);
    });

    it('should handle SMS channel', async () => {
      const response = await runner.processMessage('conv-3', 'HELP', 'sms');
      expect(response.text).toBeDefined();
    });

    it('should handle dashboard channel', async () => {
      const response = await runner.processMessage('conv-4', 'Show me options', 'dashboard');
      expect(response.text).toBeDefined();
    });

    it('should include domain in response metadata', async () => {
      const response = await runner.processMessage('conv-5', 'Help', 'web');
      expect(response.metadata?.domain).toBe('general');
    });

    it('should accept citizen DID', async () => {
      const response = await runner.processMessage('conv-6', 'Hello', 'web', 'did:mycelix:citizen:1');
      expect(response.text).toBeDefined();
    });
  });

  describe('conversation history', () => {
    it('should track conversation history', async () => {
      await runner.processMessage('conv-7', 'Hello', 'web');
      await runner.processMessage('conv-7', 'What services?', 'web');
      const history = runner.getConversationHistory('conv-7');
      expect(history.length).toBeGreaterThanOrEqual(2);
    });

    it('should return empty for unknown conversation', () => {
      const history = runner.getConversationHistory('nonexistent');
      expect(history).toEqual([]);
    });
  });

  describe('statistics', () => {
    it('should return stats', () => {
      const stats = runner.getStats();
      expect(stats.totalConversations).toBeDefined();
      expect(stats.avgConfidence).toBeDefined();
      expect(stats.escalationRate).toBeDefined();
    });

    it('should update stats after messages', async () => {
      await runner.processMessage('conv-8', 'Hello', 'web');
      const stats = runner.getStats();
      expect(stats.totalConversations).toBeGreaterThan(0);
    });
  });

  describe('escalations', () => {
    it('should return pending escalations list', () => {
      const escalations = runner.getPendingEscalations();
      expect(Array.isArray(escalations)).toBe(true);
    });
  });

  describe('knowledge retriever', () => {
    it('should set and use knowledge retriever', async () => {
      const retrieveFn = vi.fn().mockResolvedValue([
        { title: 'Test', content: 'Test content', url: 'http://test.com', relevance: 0.9 },
      ]);
      const retriever = { retrieve: retrieveFn };
      runner.setKnowledgeRetriever(retriever as any);
      await runner.processMessage('conv-9', 'What help is available?', 'web');
      expect(retrieveFn).toHaveBeenCalled();
    });
  });

  describe('model client', () => {
    it('should set model client', () => {
      const client = {
        generate: vi.fn().mockResolvedValue('AI response'),
      };
      runner.setModelClient(client as any);
      // No error
    });
  });
});

// =============================================================================
// DEFAULT_AGENT_CONFIG Tests
// =============================================================================

describe('DEFAULT_AGENT_CONFIG', () => {
  it('should have required fields', () => {
    // DEFAULT_AGENT_CONFIG is Omit<AgentConfig, 'id' | 'name' | 'domain'>
    expect(DEFAULT_AGENT_CONFIG.personality).toBeDefined();
    expect(DEFAULT_AGENT_CONFIG.escalation).toBeDefined();
    expect(DEFAULT_AGENT_CONFIG.model).toBeDefined();
    expect(DEFAULT_AGENT_CONFIG.knowledgeBase).toBeDefined();
    expect(DEFAULT_AGENT_CONFIG.rateLimit).toBeDefined();
  });
});

// =============================================================================
// SymthaeaHub Tests (without initialize)
// =============================================================================

describe('SymthaeaHub', () => {
  it('should create with factory function', () => {
    const hub = createSymthaeaHub();
    expect(hub).toBeInstanceOf(SymthaeaHub);
  });

  it('should return undefined for uninitialized domain', () => {
    const hub = createSymthaeaHub();
    expect(hub.getAgent('benefits')).toBeUndefined();
  });

  it('should return empty escalations when no agents', () => {
    const hub = createSymthaeaHub();
    expect(hub.getAllPendingEscalations()).toEqual([]);
  });

  it('should return zero stats when no agents', () => {
    const hub = createSymthaeaHub();
    const stats = hub.getStats();
    expect(stats.totalConversations).toBe(0);
    expect(stats.avgConfidence).toBe(0);
  });

  it('should not throw on shutdown with no agents', () => {
    const hub = createSymthaeaHub();
    hub.shutdown();
  });
});

// =============================================================================
// SMSAgentAdapter Tests (without hub.initialize)
// =============================================================================

describe('SMSAgentAdapter', () => {
  it('should create with factory function', () => {
    const hub = createSymthaeaHub();
    const adapter = createSMSAdapter(hub);
    expect(adapter).toBeInstanceOf(SMSAgentAdapter);
  });

  it('should create with constructor', () => {
    const hub = createSymthaeaHub();
    const adapter = new SMSAgentAdapter(hub);
    expect(adapter).toBeInstanceOf(SMSAgentAdapter);
  });

  it('should end conversation without error', () => {
    const hub = createSymthaeaHub();
    const adapter = createSMSAdapter(hub);
    adapter.endConversation('+15559999999');
  });
});

// =============================================================================
// DashboardAgentAdapter Tests (without hub.initialize)
// =============================================================================

describe('DashboardAgentAdapter', () => {
  it('should create with factory function', () => {
    const hub = createSymthaeaHub();
    const adapter = createDashboardAdapter(hub);
    expect(adapter).toBeInstanceOf(DashboardAgentAdapter);
  });

  it('should create with constructor', () => {
    const hub = createSymthaeaHub();
    const adapter = new DashboardAgentAdapter(hub);
    expect(adapter).toBeInstanceOf(DashboardAgentAdapter);
  });

  it('should return empty history for unknown conversation', () => {
    const hub = createSymthaeaHub();
    const adapter = createDashboardAdapter(hub);
    // getHistory needs an agent - returns empty when agent not found
    const history = adapter.getHistory('nonexistent', 'general');
    expect(history).toEqual([]);
  });
});
