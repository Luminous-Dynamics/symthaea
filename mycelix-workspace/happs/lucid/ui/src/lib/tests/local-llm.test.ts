// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Local LLM Service Tests
 *
 * Tests for the local LLM integration including:
 * - Provider detection and switching
 * - Error handling and recovery
 * - API responses
 * - Configuration management
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { get } from 'svelte/store';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Import after mocking
import {
  checkLLMStatus,
  generate,
  classifyThought,
  summarizeThoughts,
  llmStatus,
  llmConfig,
  setLLMConfig,
  type LLMConfig,
  type CompletionRequest,
} from '../services/local-llm';

// ============================================================================
// TEST FIXTURES
// ============================================================================

function createMockOllamaResponse(text: string) {
  return {
    response: text,
    eval_count: 50,
    done: true,
  };
}

function createMockLlamaCppResponse(text: string) {
  return {
    content: text,
    tokens_predicted: 50,
    stop: true,
  };
}

function createMockLMStudioResponse(text: string) {
  return {
    choices: [{ message: { content: text } }],
    usage: { total_tokens: 50 },
    model: 'test-model',
  };
}

// ============================================================================
// TESTS
// ============================================================================

describe('Local LLM Service', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();

    // Reset config to defaults
    llmConfig.set({
      provider: 'ollama',
      endpoint: 'http://localhost:11434',
      model: 'llama3.2',
      temperature: 0.7,
      maxTokens: 512,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('checkLLMStatus', () => {
    it('should detect Ollama availability', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ models: [{ name: 'llama3.2' }] }),
      });

      const status = await checkLLMStatus();

      expect(status.available).toBe(true);
      expect(status.provider).toBe('ollama');
    });

    it('should handle Ollama unavailable', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

      const status = await checkLLMStatus();

      expect(status.available).toBe(false);
      expect(status.error).toBeDefined();
    });

    it('should detect llama.cpp availability', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'llamacpp', endpoint: 'http://localhost:8080' }));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok' }),
      });

      const status = await checkLLMStatus();

      expect(status.available).toBe(true);
      expect(status.provider).toBe('llamacpp');
    });

    it('should detect LM Studio availability', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'lmstudio', endpoint: 'http://localhost:1234' }));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: [{ id: 'test-model' }] }),
      });

      const status = await checkLLMStatus();

      expect(status.available).toBe(true);
      expect(status.provider).toBe('lmstudio');
    });

    it('should timeout on slow responses', async () => {
      mockFetch.mockImplementationOnce(
        () => new Promise((resolve) => setTimeout(resolve, 5000))
      );

      const status = await checkLLMStatus();

      expect(status.available).toBe(false);
    });
  });

  describe('generate', () => {
    it('should generate completion with Ollama', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => createMockOllamaResponse('Generated text'),
      });

      const request: CompletionRequest = {
        prompt: 'Test prompt',
        systemPrompt: 'You are a helpful assistant',
      };

      const response = await generate(request);

      expect(response.text).toBe('Generated text');
      expect(response.tokensUsed).toBe(50);
    });

    it('should generate completion with llama.cpp', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'llamacpp' }));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => createMockLlamaCppResponse('LlamaCpp output'),
      });

      const response = await generate({ prompt: 'Test' });

      expect(response.text).toBe('LlamaCpp output');
    });

    it('should generate completion with LM Studio', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'lmstudio' }));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => createMockLMStudioResponse('LM Studio output'),
      });

      const response = await generate({ prompt: 'Test' });

      expect(response.text).toBe('LM Studio output');
    });

    it('should throw descriptive error on Ollama failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'Model not found',
      });

      await expect(generate({ prompt: 'Test' })).rejects.toThrow(/Ollama request failed.*HTTP 404/);
    });

    it('should throw descriptive error on llama.cpp failure', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'llamacpp' }));

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'Internal error',
      });

      await expect(generate({ prompt: 'Test' })).rejects.toThrow(/llama.cpp request failed/);
    });

    it('should throw descriptive error on LM Studio failure', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'lmstudio' }));

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        text: async () => 'Model not loaded',
      });

      await expect(generate({ prompt: 'Test' })).rejects.toThrow(/LM Studio request failed/);
    });

    it('should respect custom temperature and maxTokens', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => createMockOllamaResponse('Response'),
      });

      await generate({
        prompt: 'Test',
        temperature: 0.2,
        maxTokens: 100,
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: expect.stringContaining('"temperature":0.2'),
        })
      );
    });
  });

  describe('classifyThought', () => {
    it('should classify thought content', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () =>
          createMockOllamaResponse(
            JSON.stringify({
              thoughtType: 'Claim',
              confidence: 0.85,
              suggestedTags: ['philosophy', 'ethics'],
            })
          ),
      });

      const result = await classifyThought('This is a claim about ethics');

      expect(result.thoughtType).toBe('Claim');
      expect(result.confidence).toBeGreaterThan(0);
    });

    it('should handle malformed JSON response gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => createMockOllamaResponse('Not valid JSON'),
      });

      const result = await classifyThought('Test content');

      // Should return a default classification
      expect(result.thoughtType).toBeDefined();
    });
  });

  describe('summarizeThoughts', () => {
    it('should summarize multiple thought contents', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () =>
          createMockOllamaResponse(
            JSON.stringify({
              summary: 'These thoughts discuss philosophy',
              keyPoints: ['Point 1', 'Point 2'],
              connections: ['Connection 1'],
            })
          ),
      });

      const contents = ['First thought content', 'Second thought content'];

      const result = await summarizeThoughts(contents);

      expect(result.summary).toBeDefined();
      expect(result.keyPoints).toBeInstanceOf(Array);
    });

    it('should handle empty contents array', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () =>
          createMockOllamaResponse(
            JSON.stringify({
              summary: 'No thoughts to summarize',
              keyPoints: [],
              connections: [],
            })
          ),
      });

      const result = await summarizeThoughts([]);

      expect(result.summary).toBeDefined();
      expect(result.keyPoints).toEqual([]);
    });
  });

  describe('setLLMConfig', () => {
    it('should update configuration', () => {
      setLLMConfig({
        provider: 'lmstudio',
        endpoint: 'http://localhost:1234',
        model: 'custom-model',
        temperature: 0.5,
        maxTokens: 1024,
      });

      const config = get(llmConfig);

      expect(config.provider).toBe('lmstudio');
      expect(config.endpoint).toBe('http://localhost:1234');
      expect(config.model).toBe('custom-model');
    });

    it('should partially update configuration', () => {
      const originalConfig = get(llmConfig);

      setLLMConfig({ model: 'new-model' } as LLMConfig);

      const config = get(llmConfig);

      expect(config.model).toBe('new-model');
      expect(config.provider).toBe(originalConfig.provider);
    });
  });

  describe('Provider Fallback', () => {
    it('should indicate when no provider is configured', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'none' }));

      const status = await checkLLMStatus();

      expect(status.available).toBe(false);
      expect(status.provider).toBe('none');
    });
  });

  describe('Error Recovery', () => {
    it('should include recovery suggestions in Ollama errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => 'model not found',
      });

      try {
        await generate({ prompt: 'Test' });
      } catch (error: any) {
        expect(error.message).toContain('ollama pull');
      }
    });

    it('should include recovery suggestions in llama.cpp errors', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'llamacpp' }));

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: async () => 'server error',
      });

      try {
        await generate({ prompt: 'Test' });
      } catch (error: any) {
        expect(error.message).toContain('./server');
      }
    });

    it('should include recovery suggestions in LM Studio errors', async () => {
      llmConfig.update((c) => ({ ...c, provider: 'lmstudio' }));

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503,
        text: async () => 'service unavailable',
      });

      try {
        await generate({ prompt: 'Test' });
      } catch (error: any) {
        expect(error.message).toContain('LM Studio');
        expect(error.message).toContain('settings');
      }
    });
  });
});
