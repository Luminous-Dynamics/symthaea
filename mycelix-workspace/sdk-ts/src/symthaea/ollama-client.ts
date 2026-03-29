// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Ollama Model Client for Symthaea
 *
 * Connects Symthaea agents to local Ollama LLM for intelligent responses.
 * Falls back gracefully when Ollama is unavailable.
 *
 * @module symthaea/ollama-client
 */

import type { ModelClient } from './agent-runner.js';

/**
 * Ollama client configuration
 */
export interface OllamaClientConfig {
  /** Ollama API URL */
  baseUrl: string;
  /** Default model to use */
  model: string;
  /** Request timeout in ms */
  timeoutMs: number;
  /** Retry attempts */
  retries: number;
  /** Enable streaming (for future use) */
  stream: boolean;
}

/**
 * Default Ollama configuration
 */
export const DEFAULT_OLLAMA_CONFIG: OllamaClientConfig = {
  baseUrl: 'http://localhost:11434',
  model: 'llama3.2:3b',
  timeoutMs: 30000,
  retries: 2,
  stream: false,
};

/**
 * Ollama API response
 */
interface OllamaGenerateResponse {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  context?: number[];
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

/**
 * Ollama Model Client
 *
 * Implements the ModelClient interface for Symthaea agents.
 */
export class OllamaClient implements ModelClient {
  private config: OllamaClientConfig;
  private isAvailable: boolean | null = null;
  private lastHealthCheck: number = 0;
  private healthCheckInterval = 60000; // 1 minute

  constructor(config: Partial<OllamaClientConfig> = {}) {
    this.config = { ...DEFAULT_OLLAMA_CONFIG, ...config };
  }

  /**
   * Check if Ollama is available
   */
  async checkHealth(): Promise<boolean> {
    const now = Date.now();
    if (this.isAvailable !== null && now - this.lastHealthCheck < this.healthCheckInterval) {
      return this.isAvailable;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      this.isAvailable = response.ok;
      this.lastHealthCheck = now;

      if (this.isAvailable) {
        const data = await response.json();
        const models = data.models?.map((m: { name: string }) => m.name) || [];
        console.log(`[Ollama] Connected. Available models: ${models.join(', ')}`);
      }

      return this.isAvailable;
    } catch (error) {
      this.isAvailable = false;
      this.lastHealthCheck = now;
      console.log('[Ollama] Not available, will use rule-based responses');
      return false;
    }
  }

  /**
   * Generate a response using Ollama
   */
  async generate(options: {
    systemPrompt: string;
    conversationHistory: string;
    knowledgeContext: string;
    maxTokens: number;
    temperature: number;
  }): Promise<{ text: string; confidence: number }> {
    // Check availability
    const available = await this.checkHealth();
    if (!available) {
      throw new Error('Ollama not available');
    }

    // Build the prompt
    const prompt = this.buildPrompt(options);

    // Make the request with retries
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.config.retries; attempt++) {
      try {
        const response = await this.callOllama(prompt, options);
        return this.parseResponse(response);
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (attempt < this.config.retries) {
          await this.sleep(1000 * (attempt + 1)); // Exponential backoff
        }
      }
    }

    throw lastError || new Error('Failed to generate response');
  }

  /**
   * Generate with a specific model override
   */
  async generateWithModel(
    model: string,
    options: {
      systemPrompt: string;
      conversationHistory: string;
      knowledgeContext: string;
      maxTokens: number;
      temperature: number;
    }
  ): Promise<{ text: string; confidence: number }> {
    const originalModel = this.config.model;
    this.config.model = model;
    try {
      return await this.generate(options);
    } finally {
      this.config.model = originalModel;
    }
  }

  /**
   * Get available models
   */
  async getModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`);
      if (!response.ok) return [];
      const data = await response.json();
      return data.models?.map((m: { name: string }) => m.name) || [];
    } catch {
      return [];
    }
  }

  // Private methods

  private buildPrompt(options: {
    systemPrompt: string;
    conversationHistory: string;
    knowledgeContext: string;
  }): string {
    let prompt = options.systemPrompt;

    if (options.knowledgeContext) {
      prompt += `\n\n## Relevant Information\n${options.knowledgeContext}`;
    }

    if (options.conversationHistory) {
      prompt += `\n\n## Conversation\n${options.conversationHistory}`;
    }

    prompt += '\n\nAgent:';

    return prompt;
  }

  private async callOllama(
    prompt: string,
    options: { maxTokens: number; temperature: number }
  ): Promise<OllamaGenerateResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(`${this.config.baseUrl}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.model,
          prompt,
          stream: false,
          options: {
            num_predict: options.maxTokens,
            temperature: options.temperature,
            top_p: 0.9,
            stop: ['Citizen:', '\n\nCitizen', '\nCitizen:'],
          },
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Ollama error: ${response.status} - ${errorText}`);
      }

      return response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private parseResponse(response: OllamaGenerateResponse): {
    text: string;
    confidence: number;
  } {
    let text = response.response.trim();

    // Clean up common artifacts
    text = text.replace(/^(Agent:|Assistant:)\s*/i, '');
    text = text.replace(/\n(Citizen:|Human:).*$/s, '');

    // Estimate confidence based on response characteristics
    let confidence = 0.75;

    // Higher confidence for shorter, more direct responses
    if (text.length < 200) confidence += 0.05;

    // Lower confidence if response contains uncertainty markers
    const uncertaintyMarkers = [
      'i\'m not sure',
      'i think',
      'might be',
      'could be',
      'possibly',
      'i believe',
      'it seems',
    ];
    if (uncertaintyMarkers.some((m) => text.toLowerCase().includes(m))) {
      confidence -= 0.15;
    }

    // Higher confidence if response contains specific information
    const specificityMarkers = [
      /\d{3}-\d{4}/, // Phone number
      /\d{5}/, // Zip code
      /\$[\d,]+/, // Dollar amount
      /\d+%/, // Percentage
    ];
    if (specificityMarkers.some((m) => m.test(text))) {
      confidence += 0.05;
    }

    return {
      text,
      confidence: Math.max(0.3, Math.min(0.95, confidence)),
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create an Ollama client
 */
export function createOllamaClient(
  config: Partial<OllamaClientConfig> = {}
): OllamaClient {
  return new OllamaClient(config);
}

/**
 * Create a model client that falls back to a simple response generator
 * when Ollama is unavailable
 */
export function createResilientModelClient(
  ollamaConfig: Partial<OllamaClientConfig> = {}
): ModelClient {
  const ollama = new OllamaClient(ollamaConfig);

  return {
    async generate(options) {
      try {
        return await ollama.generate(options);
      } catch {
        // Fallback: extract a simple response from the system prompt
        console.log('[Ollama] Falling back to rule-based response');
        return {
          text: 'I\'m here to help! Could you tell me more about what you need? I can assist with benefits, permits, voting, and other government services. For immediate help, call 311.',
          confidence: 0.5,
        };
      }
    },
  };
}
