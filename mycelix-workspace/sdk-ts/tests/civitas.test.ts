/**
 * Civitas Module Tests
 *
 * Tests for civic AI agents:
 * - BenefitsCivitas (eligibility, applications, renewals)
 * - JusticeCivitas (rights, appeals, mediation)
 * - ParticipationCivitas (voting, delegation, budget)
 * - createCivitas factory
 * - CivitasAgent base class (conversation, error handling)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  BenefitsCivitas,
  JusticeCivitas,
  ParticipationCivitas,
  createCivitas,
  CivitasDefaults,
  type CivitasConfig,
  type CivitasMessage,
  type ConversationContext,
} from '../src/civitas/index.js';

const BASE_CONFIG: Omit<CivitasConfig, 'domain'> = {
  symthaeaUrl: 'http://localhost:9999',
  language: 'en',
  accessibilityMode: false,
  voiceEnabled: false,
};

// =============================================================================
// Factory Tests
// =============================================================================

describe('createCivitas', () => {
  it('should create BenefitsCivitas for benefits domain', () => {
    const agent = createCivitas('benefits', BASE_CONFIG);
    expect(agent).toBeInstanceOf(BenefitsCivitas);
  });

  it('should create JusticeCivitas for justice domain', () => {
    const agent = createCivitas('justice', BASE_CONFIG);
    expect(agent).toBeInstanceOf(JusticeCivitas);
  });

  it('should create ParticipationCivitas for participation domain', () => {
    const agent = createCivitas('participation', BASE_CONFIG);
    expect(agent).toBeInstanceOf(ParticipationCivitas);
  });

  it('should throw for unknown domain', () => {
    expect(() => createCivitas('unknown' as any, BASE_CONFIG)).toThrow('Unknown Civitas domain');
  });
});

describe('CivitasDefaults', () => {
  it('should have sensible defaults', () => {
    expect(CivitasDefaults.language).toBe('en');
    expect(CivitasDefaults.accessibilityMode).toBe(false);
    expect(CivitasDefaults.voiceEnabled).toBe(false);
  });
});

// =============================================================================
// BenefitsCivitas Tests
// =============================================================================

describe('BenefitsCivitas', () => {
  let agent: BenefitsCivitas;

  beforeEach(() => {
    agent = new BenefitsCivitas(BASE_CONFIG);
  });

  describe('conversation lifecycle', () => {
    it('should start a conversation and return context with greeting', async () => {
      const context = await agent.startConversation('citizen-1');

      expect(context.citizenId).toBe('citizen-1');
      expect(context.sessionId).toBeDefined();
      expect(context.domain).toBe('benefits');
      expect(context.history).toHaveLength(1);
      expect(context.history[0].role).toBe('civitas');
      expect(context.history[0].content).toContain('benefits');
    });

    it('should include suggested actions in greeting', async () => {
      const context = await agent.startConversation('citizen-1');
      const greeting = context.history[0];

      expect(greeting.actions).toBeDefined();
      expect(greeting.actions!.length).toBeGreaterThan(0);
      expect(greeting.actions!.some(a => a.type === 'learn')).toBe(true);
    });

    it('should throw when chatting without starting conversation', async () => {
      await expect(agent.chat('hello')).rejects.toThrow('No active conversation');
    });
  });

  describe('message routing', () => {
    beforeEach(async () => {
      await agent.startConversation('citizen-1');
    });

    it('should route eligibility questions to eligibility check', async () => {
      const response = await agent.chat('Am I eligible for any programs?');

      expect(response.role).toBe('civitas');
      expect(response.content).toContain('eligible');
      expect(response.epistemicStatus).toBeDefined();
      expect(response.sources).toBeDefined();
      expect(response.actions).toBeDefined();
    });

    it('should route status questions to application status', async () => {
      const response = await agent.chat('What is the status of my pending applications?');

      expect(response.content).toContain('pending');
      expect(response.epistemicStatus?.confidence).toBe('known');
    });

    it('should route renewal questions', async () => {
      const response = await agent.chat('Do I need to renew anything?');

      expect(response.content).toContain('Renewal');
      expect(response.actions).toBeDefined();
    });

    it('should fall back to Symthaea for general questions', async () => {
      // Mock fetch to simulate Symthaea being unavailable
      const mockFetch = vi.fn().mockRejectedValue(new Error('Connection refused'));
      vi.stubGlobal('fetch', mockFetch);

      const response = await agent.chat('Tell me about housing assistance');

      // Should get fallback response
      expect(response.role).toBe('civitas');
      expect(response.content).toBeDefined();
      expect(response.content.length).toBeGreaterThan(0);

      vi.unstubAllGlobals();
    });

    it('should add messages to conversation history', async () => {
      await agent.chat('Am I eligible?');
      await agent.chat('Tell me more about SNAP');

      // greeting + citizen msg + response + citizen msg + response
      const context = (agent as any).context as ConversationContext;
      expect(context.history.length).toBe(5);
    });
  });

  describe('Spanish language support', () => {
    it('should greet in Spanish when language is es', async () => {
      const spanishAgent = new BenefitsCivitas({
        ...BASE_CONFIG,
        language: 'es',
      });

      const context = await spanishAgent.startConversation('citizen-1');
      expect(context.history[0].content).toContain('Hola');
    });
  });
});

// =============================================================================
// JusticeCivitas Tests
// =============================================================================

describe('JusticeCivitas', () => {
  let agent: JusticeCivitas;

  beforeEach(async () => {
    agent = new JusticeCivitas(BASE_CONFIG);
    await agent.startConversation('citizen-1');
  });

  it('should greet with rights advocacy message', async () => {
    const freshAgent = new JusticeCivitas(BASE_CONFIG);
    const context = await freshAgent.startConversation('citizen-2');
    expect(context.history[0].content).toContain('rights');
    expect(context.history[0].content).toContain('legal advice');
  });

  it('should handle appeal requests', async () => {
    const response = await agent.chat('I want to appeal a decision that was unfair');

    expect(response.content).toContain('Appeal');
    expect(response.sources).toBeDefined();
    expect(response.sources!.some(s => s.type === 'law')).toBe(true);
    expect(response.actions).toBeDefined();
  });

  it('should handle rights questions', async () => {
    const response = await agent.chat('What are my rights in this situation?');

    expect(response.content).toContain('rights');
    expect(response.epistemicStatus).toBeDefined();
  });

  it('should handle dispute/mediation requests', async () => {
    const response = await agent.chat('I have a dispute with my neighbor');

    expect(response.content).toContain('Mediation');
    expect(response.content).toContain('Free');
    expect(response.actions).toBeDefined();
    expect(response.actions!.some(a => a.type === 'contact')).toBe(true);
  });

  it('should fall back to Symthaea for unrecognized topics', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('fail'));
    vi.stubGlobal('fetch', mockFetch);

    const response = await agent.chat('Something completely different');
    expect(response.role).toBe('civitas');
    expect(response.content).toBeDefined();

    vi.unstubAllGlobals();
  });
});

// =============================================================================
// ParticipationCivitas Tests
// =============================================================================

describe('ParticipationCivitas', () => {
  let agent: ParticipationCivitas;

  beforeEach(async () => {
    agent = new ParticipationCivitas(BASE_CONFIG);
    await agent.startConversation('citizen-1');
  });

  it('should greet with civic participation message', async () => {
    const freshAgent = new ParticipationCivitas(BASE_CONFIG);
    const context = await freshAgent.startConversation('citizen-2');
    expect(context.history[0].content).toContain('participation');
    expect(context.history[0].actions).toBeDefined();
  });

  it('should handle voting questions', async () => {
    const response = await agent.chat('What can I vote on?');

    expect(response.content).toContain('Proposals');
    expect(response.content).toContain('VOTE');
    expect(response.epistemicStatus?.confidence).toBe('known');
  });

  it('should handle delegation questions', async () => {
    const response = await agent.chat('How can I set up delegation for a proxy representative?');

    expect(response.content).toContain('Liquid Democracy');
    expect(response.content).toContain('delegate');
    expect(response.actions).toBeDefined();
  });

  it('should handle budget/HEARTH questions', async () => {
    const response = await agent.chat('Tell me about the community budget');

    expect(response.content).toContain('HEARTH');
    expect(response.content).toContain('CGC');
    expect(response.actions).toBeDefined();
  });

  it('should provide SMS commands in suggested actions', async () => {
    const response = await agent.chat('What votes are available?');

    expect(response.actions).toBeDefined();
    const hasSmsCommand = response.actions!.some(a => a.smsCommand);
    expect(hasSmsCommand).toBe(true);
  });
});

// =============================================================================
// Fallback Response Tests
// =============================================================================

describe('Symthaea fallback', () => {
  it('should return English fallback by default', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('fail'));
    vi.stubGlobal('fetch', mockFetch);

    const agent = new BenefitsCivitas(BASE_CONFIG);
    await agent.startConversation('citizen-1');
    const response = await agent.chat('random general question');

    expect(response.content).toContain('trouble connecting');

    vi.unstubAllGlobals();
  });

  it('should return Spanish fallback for es language', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('fail'));
    vi.stubGlobal('fetch', mockFetch);

    const agent = new BenefitsCivitas({ ...BASE_CONFIG, language: 'es' });
    await agent.startConversation('citizen-1');
    const response = await agent.chat('pregunta general');

    expect(response.content).toContain('problemas');

    vi.unstubAllGlobals();
  });
});
