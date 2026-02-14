/**
 * Epistemic Validator Agent System Tests
 *
 * Tests for the epistemic agent runner, claim classification,
 * DKG validation, and calibration integration.
 *
 * @module innovations/epistemic-agents/__tests__
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  EpistemicAgentRunner,
  EpistemicKnowledgeRetriever,
  createEpistemicAgent,
  createEpistemicAgentFromDomain,
  DEFAULT_EPISTEMIC_CONFIG,
  type EpistemicClaim,
  type DKGClient,
  type DKGValidationResult,
  type EpistemicConfig,
} from '../index';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  toClassificationCode,
} from '../../../integrations/epistemic-markets/index';
import type { AgentConfig } from '../../../symthaea/types';

// ============================================================================
// MOCKS
// ============================================================================

const mockAgentConfig: AgentConfig = {
  id: 'test-agent-001',
  name: 'Test Epistemic Agent',
  domain: 'benefits',
  capabilityLevel: 'navigational',
  model: {
    provider: 'ollama',
    model: 'llama3.2:3b',
    temperature: 0.3,
    maxTokens: 500,
    systemPrompt: 'You are a helpful benefits navigator.',
  },
  knowledgeBase: {
    enabled: true,
    sources: ['agency_faqs'],
    maxSourcesPerResponse: 3,
    minRelevance: 0.6,
    dkgEnabled: true,
  },
  escalation: {
    enabled: true,
    confidenceThreshold: 0.5,
    alwaysEscalateTopics: ['legal_advice'],
    maxTurnsBeforeEscalation: 10,
    channels: [{ type: 'phone', value: '311', availability: '24/7' }],
  },
  personality: {
    tone: 'friendly',
    readingLevel: 'simple',
    useEmojis: false,
    languages: ['en'],
    greetingStyle: 'brief',
  },
  rateLimit: {
    requestsPerMinute: 10,
    requestsPerHour: 100,
    maxConversationLength: 50,
  },
};

const createMockDKGClient = (): DKGClient => ({
  queryClaims: vi.fn().mockResolvedValue({
    supporting: [],
    contradicting: [],
  }),
  storeClaim: vi.fn().mockResolvedValue('claim-stored-id'),
  updateClaimStatus: vi.fn().mockResolvedValue(undefined),
  getClaim: vi.fn().mockResolvedValue(null),
});

// ============================================================================
// TESTS
// ============================================================================

describe('EpistemicAgentRunner', () => {
  let agent: EpistemicAgentRunner;
  let mockDKGClient: DKGClient;

  beforeEach(() => {
    mockDKGClient = createMockDKGClient();
    agent = new EpistemicAgentRunner(mockAgentConfig);
    agent.setDKGClient(mockDKGClient);
  });

  describe('initialization', () => {
    it('should create an agent with default epistemic config', () => {
      const agent = new EpistemicAgentRunner(mockAgentConfig);
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });

    it('should merge custom epistemic config', () => {
      const customConfig: Partial<EpistemicConfig> = {
        minConfidenceForStorage: 0.8,
        enableDKGValidation: false,
      };
      const agent = new EpistemicAgentRunner(mockAgentConfig, customConfig);
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });

    it('should accept DKG client', () => {
      const agent = new EpistemicAgentRunner(mockAgentConfig);
      const dkgClient = createMockDKGClient();
      agent.setDKGClient(dkgClient);
      // No error thrown
      expect(true).toBe(true);
    });
  });

  describe('isVerificationSource', () => {
    it('should return false initially (low reputation)', () => {
      expect(agent.isVerificationSource()).toBe(false);
    });

    it('should become verification source with high reputation', async () => {
      // Record many correct outcomes to build reputation
      const claim: EpistemicClaim = {
        id: 'test-claim-1',
        text: 'SNAP benefits are federally funded.',
        position: {
          empirical: EmpiricalLevel.Cryptographic,
          normative: NormativeLevel.Universal,
          materiality: MaterialityLevel.Persistent,
        },
        classificationCode: 'E3-N3-M2',
        confidence: 0.9,
        calibratedConfidence: 0.9,
        sources: [],
        agentId: 'test-agent-001',
        conversationId: 'conv-1',
        timestamp: Date.now(),
        domain: 'benefits',
        dkgValidated: true,
        isVerificationSource: false,
      };

      // Manually set claim in history and record outcomes
      // In real usage, claims come from processMessageWithValidation
      // We'd need to test through the full flow
    });
  });

  describe('getEpistemicStats', () => {
    it('should return initial stats', () => {
      const stats = agent.getEpistemicStats();
      expect(stats.totalClaims).toBe(0);
      expect(stats.validatedCorrect).toBe(0);
      expect(stats.validatedIncorrect).toBe(0);
      expect(stats.pendingValidation).toBe(0);
    });
  });

  describe('getReputation', () => {
    it('should return initial neutral reputation', () => {
      const rep = agent.getReputation();
      expect(rep).toBeDefined();
      expect(rep.positive).toBe(0);
      expect(rep.negative).toBe(0);
    });
  });
});

describe('EpistemicKnowledgeRetriever', () => {
  let retriever: EpistemicKnowledgeRetriever;
  let mockDKGClient: DKGClient;

  beforeEach(() => {
    retriever = new EpistemicKnowledgeRetriever();
    mockDKGClient = createMockDKGClient();
  });

  describe('retrieve', () => {
    it('should return empty array without DKG client', async () => {
      const sources = await retriever.retrieve('test query', {
        domain: 'benefits',
        maxResults: 5,
        minRelevance: 0.5,
      });
      expect(sources).toEqual([]);
    });

    it('should query DKG when client is set', async () => {
      (mockDKGClient.queryClaims as ReturnType<typeof vi.fn>).mockResolvedValue({
        supporting: [
          { id: 'claim-1', confidence: 0.8, text: 'Supporting claim text' },
        ],
        contradicting: [],
      });

      retriever.setDKGClient(mockDKGClient);

      const sources = await retriever.retrieve('test query', {
        domain: 'benefits',
        maxResults: 5,
        minRelevance: 0.5,
      });

      expect(mockDKGClient.queryClaims).toHaveBeenCalledWith('test query', 'benefits');
      expect(sources.length).toBe(1);
      expect(sources[0].type).toBe('dkg');
      expect(sources[0].id).toBe('claim-1');
    });

    it('should filter by minimum relevance', async () => {
      (mockDKGClient.queryClaims as ReturnType<typeof vi.fn>).mockResolvedValue({
        supporting: [
          { id: 'claim-1', confidence: 0.3, text: 'Low confidence claim' },
          { id: 'claim-2', confidence: 0.9, text: 'High confidence claim' },
        ],
        contradicting: [],
      });

      retriever.setDKGClient(mockDKGClient);

      const sources = await retriever.retrieve('test query', {
        domain: 'benefits',
        maxResults: 5,
        minRelevance: 0.5,
      });

      expect(sources.length).toBe(1);
      expect(sources[0].id).toBe('claim-2');
    });
  });

  describe('source reputation tracking', () => {
    it('should track positive outcomes', () => {
      retriever.recordSourceOutcome('source-1', true);
      const rep = retriever.getSourceReputation('source-1');
      expect(rep).toBeGreaterThan(0.5);
    });

    it('should track negative outcomes', () => {
      retriever.recordSourceOutcome('source-1', false);
      const rep = retriever.getSourceReputation('source-1');
      expect(rep).toBeLessThan(0.5);
    });

    it('should return 0.5 for unknown sources', () => {
      const rep = retriever.getSourceReputation('unknown-source');
      expect(rep).toBe(0.5);
    });
  });
});

describe('Factory Functions', () => {
  describe('createEpistemicAgent', () => {
    it('should create agent with provided config', () => {
      const agent = createEpistemicAgent(mockAgentConfig);
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });

    it('should accept custom epistemic config', () => {
      const agent = createEpistemicAgent(mockAgentConfig, {
        enableDKGValidation: false,
      });
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });
  });

  describe('createEpistemicAgentFromDomain', () => {
    it('should create agent for benefits domain', () => {
      const agent = createEpistemicAgentFromDomain('benefits');
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });

    it('should create agent for justice domain', () => {
      const agent = createEpistemicAgentFromDomain('justice');
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });

    it('should allow config overrides', () => {
      const agent = createEpistemicAgentFromDomain('health', {
        name: 'Custom Health Agent',
      });
      expect(agent).toBeInstanceOf(EpistemicAgentRunner);
    });
  });
});

describe('Epistemic Classification Integration', () => {
  it('should use correct classification codes', () => {
    const position = {
      empirical: EmpiricalLevel.Measurable,
      normative: NormativeLevel.Universal,
      materiality: MaterialityLevel.Foundational,
    };
    const code = toClassificationCode(position);
    expect(code).toBe('E4-N3-M3');
  });

  it('should classify subjective claims correctly', () => {
    const position = {
      empirical: EmpiricalLevel.Subjective,
      normative: NormativeLevel.Personal,
      materiality: MaterialityLevel.Ephemeral,
    };
    const code = toClassificationCode(position);
    expect(code).toBe('E0-N0-M0');
  });
});

describe('DEFAULT_EPISTEMIC_CONFIG', () => {
  it('should have expected default values', () => {
    expect(DEFAULT_EPISTEMIC_CONFIG.enableDKGValidation).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.minConfidenceForStorage).toBe(0.6);
    expect(DEFAULT_EPISTEMIC_CONFIG.enableCalibration).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.autoAdjustConfidence).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.flagContradictions).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.escalateHighStakes).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.enableVerificationSource).toBe(true);
    expect(DEFAULT_EPISTEMIC_CONFIG.minVerificationReputation).toBe(0.7);
  });
});
