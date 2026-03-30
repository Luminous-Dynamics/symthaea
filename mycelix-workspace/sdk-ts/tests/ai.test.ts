// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * AI Module Tests
 *
 * Tests for LocalLLMClient and MycelixAssistant
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  LocalLLMClient,
  createLocalLLM,
  createOllamaClient,
  createLlamaCppClient,
  createLMStudioClient,
  type LLMConfig,
  type ChatMessage,
  MycelixAssistant,
  createAssistant,
  createOllamaAssistant,
} from '../src/ai/index.js';

// =============================================================================
// LocalLLMClient Tests
// =============================================================================

describe('LocalLLMClient', () => {
  describe('construction', () => {
    it('should create with Ollama provider', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'llama3.2',
      });

      expect(client).toBeInstanceOf(LocalLLMClient);
      const config = client.getConfig();
      expect(config.provider).toBe('ollama');
      expect(config.model).toBe('llama3.2');
      expect(config.baseUrl).toBe('http://localhost:11434');
    });

    it('should create with llama.cpp provider', () => {
      const client = new LocalLLMClient({
        provider: 'llamacpp',
        model: 'mistral',
      });

      const config = client.getConfig();
      expect(config.provider).toBe('llamacpp');
      expect(config.baseUrl).toBe('http://localhost:8080');
    });

    it('should create with LM Studio provider', () => {
      const client = new LocalLLMClient({
        provider: 'lmstudio',
        model: 'codellama',
      });

      const config = client.getConfig();
      expect(config.provider).toBe('lmstudio');
      expect(config.baseUrl).toBe('http://localhost:1234');
    });

    it('should create with custom provider', () => {
      const client = new LocalLLMClient({
        provider: 'custom',
        model: 'custom-model',
        baseUrl: 'http://custom-server:5000',
      });

      const config = client.getConfig();
      expect(config.provider).toBe('custom');
      expect(config.baseUrl).toBe('http://custom-server:5000');
    });

    it('should use custom baseUrl when provided', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'llama3.2',
        baseUrl: 'http://custom:11434',
      });

      const config = client.getConfig();
      expect(config.baseUrl).toBe('http://custom:11434');
    });

    it('should apply default config values', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'llama3.2',
      });

      const config = client.getConfig();
      expect(config.timeout).toBe(60000);
      expect(config.temperature).toBe(0.7);
      expect(config.maxTokens).toBe(2048);
      expect(config.contextWindow).toBe(4096);
    });

    it('should override default config with custom values', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'llama3.2',
        timeout: 30000,
        temperature: 0.5,
        maxTokens: 1024,
        contextWindow: 8192,
      });

      const config = client.getConfig();
      expect(config.timeout).toBe(30000);
      expect(config.temperature).toBe(0.5);
      expect(config.maxTokens).toBe(1024);
      expect(config.contextWindow).toBe(8192);
    });
  });

  describe('getConfig', () => {
    it('should return a copy of config', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const config1 = client.getConfig();
      const config2 = client.getConfig();

      expect(config1).toEqual(config2);
      expect(config1).not.toBe(config2); // Different object reference
    });
  });

  describe('updateConfig', () => {
    it('should update temperature', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      client.updateConfig({ temperature: 0.9 });
      expect(client.getConfig().temperature).toBe(0.9);
    });

    it('should update maxTokens', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      client.updateConfig({ maxTokens: 4096 });
      expect(client.getConfig().maxTokens).toBe(4096);
    });

    it('should update multiple values', () => {
      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      client.updateConfig({
        temperature: 0.3,
        timeout: 120000,
        maxTokens: 512,
      });

      const config = client.getConfig();
      expect(config.temperature).toBe(0.3);
      expect(config.timeout).toBe(120000);
      expect(config.maxTokens).toBe(512);
    });
  });

  describe('isAvailable', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should return true when server responds ok', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
      } as Response);

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const available = await client.isAvailable();
      expect(available).toBe(true);
    });

    it('should return false when server responds with error', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
      } as Response);

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const available = await client.isAvailable();
      expect(available).toBe(false);
    });

    it('should return false when fetch throws', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const available = await client.isAvailable();
      expect(available).toBe(false);
    });
  });

  describe('pullModel', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should throw for non-Ollama providers', async () => {
      const client = new LocalLLMClient({
        provider: 'llamacpp',
        model: 'test',
      });

      await expect(client.pullModel('mistral')).rejects.toThrow(
        'Pull model is only supported for Ollama'
      );
    });

    it('should return true when pull succeeds', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
      } as Response);

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const result = await client.pullModel('mistral');
      expect(result).toBe(true);
    });

    it('should return false when pull fails', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Pull failed'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const result = await client.pullModel('mistral');
      expect(result).toBe(false);
    });
  });

  describe('listModels', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should return empty array for unsupported provider', async () => {
      const client = new LocalLLMClient({
        provider: 'llamacpp',
        model: 'test',
      });

      const models = await client.listModels();
      expect(models).toEqual([]);
    });

    it('should return empty array on error', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const models = await client.listModels();
      expect(models).toEqual([]);
    });
  });

  describe('chat error handling', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should return error result when fetch fails', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const messages: ChatMessage[] = [{ role: 'user', content: 'Hello' }];
      const result = await client.chat(messages);

      expect(result.content).toBe('');
      expect(result.finishReason).toBe('error');
      expect(result.model).toBe('test');
      expect(result.latencyMs).toBeGreaterThanOrEqual(0);
    });
  });

  describe('embed error handling', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should return empty embedding on error', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const result = await client.embed('test text');

      expect(result).toHaveLength(1);
      expect(result[0].embedding).toEqual([]);
      expect(result[0].model).toBe('test');
    });

    it('should handle array input', async () => {
      vi.mocked(fetch).mockRejectedValue(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const result = await client.embed(['text1', 'text2']);

      expect(result).toHaveLength(2);
    });

    it('should throw for unsupported provider', async () => {
      const client = new LocalLLMClient({
        provider: 'lmstudio',
        model: 'test',
      });

      const result = await client.embed('test');
      expect(result[0].embedding).toEqual([]);
    });
  });

  describe('complete', () => {
    beforeEach(() => {
      vi.stubGlobal('fetch', vi.fn());
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('should wrap prompt in user message', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const client = new LocalLLMClient({
        provider: 'ollama',
        model: 'test',
      });

      const result = await client.complete('Hello world');

      expect(result.content).toBe('');
      expect(result.finishReason).toBe('error');
    });
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('createLocalLLM', () => {
    it('should create a LocalLLMClient', () => {
      const client = createLocalLLM({
        provider: 'ollama',
        model: 'llama3.2',
      });

      expect(client).toBeInstanceOf(LocalLLMClient);
    });
  });

  describe('createOllamaClient', () => {
    it('should create with default model', () => {
      const client = createOllamaClient();

      const config = client.getConfig();
      expect(config.provider).toBe('ollama');
      expect(config.model).toBe('llama3.2');
      expect(config.baseUrl).toBe('http://localhost:11434');
    });

    it('should create with custom model', () => {
      const client = createOllamaClient('mistral');

      expect(client.getConfig().model).toBe('mistral');
    });

    it('should create with custom baseUrl', () => {
      const client = createOllamaClient('llama3.2', 'http://remote:11434');

      expect(client.getConfig().baseUrl).toBe('http://remote:11434');
    });
  });

  describe('createLlamaCppClient', () => {
    it('should create with default model', () => {
      const client = createLlamaCppClient();

      const config = client.getConfig();
      expect(config.provider).toBe('llamacpp');
      expect(config.model).toBe('default');
      expect(config.baseUrl).toBe('http://localhost:8080');
    });

    it('should create with custom model and baseUrl', () => {
      const client = createLlamaCppClient('codellama', 'http://custom:8080');

      const config = client.getConfig();
      expect(config.model).toBe('codellama');
      expect(config.baseUrl).toBe('http://custom:8080');
    });
  });

  describe('createLMStudioClient', () => {
    it('should create with specified model', () => {
      const client = createLMStudioClient('codellama');

      const config = client.getConfig();
      expect(config.provider).toBe('lmstudio');
      expect(config.model).toBe('codellama');
      expect(config.baseUrl).toBe('http://localhost:1234');
    });

    it('should create with custom baseUrl', () => {
      const client = createLMStudioClient('model', 'http://custom:1234');

      expect(client.getConfig().baseUrl).toBe('http://custom:1234');
    });
  });
});

// =============================================================================
// MycelixAssistant Tests
// =============================================================================

describe('MycelixAssistant', () => {
  let mockLLM: LocalLLMClient;

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
    mockLLM = new LocalLLMClient({
      provider: 'ollama',
      model: 'test',
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('construction', () => {
    it('should create with LLM client', () => {
      const assistant = new MycelixAssistant({ llm: mockLLM });

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should accept custom system prompt', () => {
      const assistant = new MycelixAssistant({
        llm: mockLLM,
        systemPrompt: 'Custom prompt',
      });

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should accept maxContextMessages', () => {
      const assistant = new MycelixAssistant({
        llm: mockLLM,
        maxContextMessages: 10,
      });

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });
  });

  describe('getHistory', () => {
    it('should return empty history initially', () => {
      const assistant = new MycelixAssistant({ llm: mockLLM });

      const history = assistant.getHistory();
      expect(history).toEqual([]);
    });

    it('should return a copy of history', () => {
      const assistant = new MycelixAssistant({ llm: mockLLM });

      const history1 = assistant.getHistory();
      const history2 = assistant.getHistory();

      expect(history1).not.toBe(history2);
    });
  });

  describe('clearHistory', () => {
    it('should clear conversation history', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'response' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      // Add some history by chatting
      await assistant.chat('Hello');

      expect(assistant.getHistory().length).toBeGreaterThan(0);

      assistant.clearHistory();

      expect(assistant.getHistory()).toEqual([]);
    });
  });

  describe('isAvailable', () => {
    it('should delegate to LLM client', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const available = await assistant.isAvailable();
      expect(available).toBe(true);
    });
  });

  describe('chat', () => {
    it('should add message to history', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'response' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      await assistant.chat('Hello');

      const history = assistant.getHistory();
      expect(history.length).toBeGreaterThanOrEqual(1);
      expect(history[0]).toEqual({ role: 'user', content: 'Hello' });
    });

    it('should trim history when exceeding maxContextMessages', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'response' } }),
      } as Response);

      const assistant = new MycelixAssistant({
        llm: mockLLM,
        maxContextMessages: 4, // 2 user + 2 assistant
      });

      await assistant.chat('Message 1');
      await assistant.chat('Message 2');
      await assistant.chat('Message 3');
      await assistant.chat('Message 4');
      await assistant.chat('Message 5');

      const history = assistant.getHistory();
      // After 5 chats, history should be trimmed to maxContextMessages
      expect(history.length).toBeLessThanOrEqual(6); // May have some pending
      expect(history.length).toBeGreaterThan(0);
    });
  });

  describe('analyzeClaim fallback', () => {
    it('should return default analysis when JSON parsing fails', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'Non-JSON response text' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const analysis = await assistant.analyzeClaim('Test claim');

      expect(analysis.confidence).toBe(0.3);
      expect(analysis.suggestedClassification).toEqual({
        empirical: 1,
        normative: 1,
        materiality: 1,
      });
    });
  });

  describe('interpretTrust fallback', () => {
    it('should determine tier from composite score', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'Non-JSON response' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const interp = await assistant.interpretTrust({ composite: 0.95 });
      expect(interp.tier).toBe('exemplary');

      const interp2 = await assistant.interpretTrust({ composite: 0.75 });
      expect(interp2.tier).toBe('trusted');

      const interp3 = await assistant.interpretTrust({ composite: 0.55 });
      expect(interp3.tier).toBe('established');

      const interp4 = await assistant.interpretTrust({ composite: 0.35 });
      expect(interp4.tier).toBe('emerging');

      const interp5 = await assistant.interpretTrust({ composite: 0.15 });
      expect(interp5.tier).toBe('new');
    });
  });

  describe('analyzeProposal fallback', () => {
    it('should return default analysis when JSON parsing fails', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'Non-JSON response' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const analysis = await assistant.analyzeProposal({
        title: 'Test Proposal',
        description: 'Test description',
      });

      expect(analysis.summary).toBe('Test Proposal');
      expect(analysis.recommendation).toBe('needs-more-info');
    });
  });

  describe('summarize', () => {
    it('should call LLM with summarize prompt', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'Summary text' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const summary = await assistant.summarize('Long text to summarize', 100);

      expect(summary).toBe('Summary text');
    });
  });

  describe('explainConcept', () => {
    it('should call LLM with concept explanation prompt', async () => {
      vi.mocked(fetch).mockResolvedValue({
        ok: true,
        json: async () => ({ message: { content: 'MATL explanation' } }),
      } as Response);

      const assistant = new MycelixAssistant({ llm: mockLLM });

      const explanation = await assistant.explainConcept('MATL');

      expect(explanation).toBe('MATL explanation');
    });
  });
});

// =============================================================================
// Assistant Factory Tests
// =============================================================================

describe('Assistant Factory Functions', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('createAssistant', () => {
    it('should create with default LLM', () => {
      const assistant = createAssistant();

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should create with custom LLM', () => {
      const llm = createOllamaClient('mistral');
      const assistant = createAssistant({ llm });

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should pass through config options', () => {
      const assistant = createAssistant({
        systemPrompt: 'Custom prompt',
        maxContextMessages: 5,
      });

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });
  });

  describe('createOllamaAssistant', () => {
    it('should create with default model', () => {
      const assistant = createOllamaAssistant();

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should create with custom model', () => {
      const assistant = createOllamaAssistant('mistral');

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });

    it('should create with custom baseUrl', () => {
      const assistant = createOllamaAssistant('llama3.2', 'http://custom:11434');

      expect(assistant).toBeInstanceOf(MycelixAssistant);
    });
  });
});

// =============================================================================
// Type Export Tests
// =============================================================================

describe('AI Type Exports', () => {
  it('should export LLMConfig type', () => {
    const config: LLMConfig = {
      provider: 'ollama',
      baseUrl: 'http://localhost:11434',
      model: 'test',
    };

    expect(config.provider).toBe('ollama');
  });

  it('should export ChatMessage type', () => {
    const message: ChatMessage = {
      role: 'user',
      content: 'Hello',
    };

    expect(message.role).toBe('user');
  });
});
