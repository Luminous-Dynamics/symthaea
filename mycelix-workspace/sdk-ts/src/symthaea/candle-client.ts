// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Candle Inference Client for Symthaea
 *
 * Connects Symthaea agents to the local Rust/Candle inference engine.
 * Provides fallback to Ollama when Candle is unavailable.
 *
 * @module symthaea/candle-client
 */

import type { ModelClient } from './agent-runner.js';
import type { CivicAgentDomain } from './types.js';

/**
 * Candle client configuration
 */
export interface CandleClientConfig {
  /** Candle inference server URL (default: localhost:3600) */
  serverUrl: string;
  /** Request timeout in ms */
  timeoutMs: number;
  /** Retry attempts */
  retries: number;
  /** Fallback to Ollama if Candle unavailable */
  fallbackToOllama: boolean;
  /** Ollama URL for fallback */
  ollamaUrl?: string;
}

/**
 * Default Candle configuration
 */
export const DEFAULT_CANDLE_CONFIG: CandleClientConfig = {
  serverUrl: 'http://localhost:3600',
  timeoutMs: 15000, // Much faster than Ollama
  retries: 2,
  fallbackToOllama: true,
  ollamaUrl: 'http://localhost:11434',
};

/**
 * Candle inference request
 */
interface CandleRequest {
  message: string;
  domain: string;
  history?: string;
  knowledge_context?: string;
  max_tokens?: number;
  temperature?: number;
}

/**
 * Candle inference response
 */
interface CandleResponse {
  text: string;
  confidence: number;
  should_escalate: boolean;
  escalation_reason?: string;
  domain: string;
  generation_time_ms: number;
  tokens_generated: number;
}

/**
 * Candle Model Client
 *
 * Implements the ModelClient interface for Symthaea agents,
 * connecting to the Rust/Candle inference server.
 */
export class CandleClient implements ModelClient {
  private config: CandleClientConfig;
  private isAvailable: boolean | null = null;
  private lastHealthCheck: number = 0;
  private healthCheckInterval = 30000; // 30 seconds

  constructor(config: Partial<CandleClientConfig> = {}) {
    this.config = { ...DEFAULT_CANDLE_CONFIG, ...config };
  }

  /**
   * Check if Candle server is available
   */
  async checkHealth(): Promise<boolean> {
    const now = Date.now();
    if (
      this.isAvailable !== null &&
      now - this.lastHealthCheck < this.healthCheckInterval
    ) {
      return this.isAvailable;
    }

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      const response = await fetch(`${this.config.serverUrl}/health`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      this.isAvailable = response.ok;
      this.lastHealthCheck = now;

      if (this.isAvailable) {
        const data = (await response.json()) as {
          status: string;
          model: string;
          device: string;
        };
        console.log(
          `[Candle] Connected. Model: ${data.model}, Device: ${data.device}`
        );
      }

      return this.isAvailable;
    } catch {
      this.isAvailable = false;
      this.lastHealthCheck = now;
      console.log('[Candle] Server not available');
      return false;
    }
  }

  /**
   * Generate a response using Candle inference
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
      throw new Error('Candle not available');
    }

    // Build request
    const request: CandleRequest = {
      message: this.extractUserMessage(options.conversationHistory),
      domain: this.detectDomainFromPrompt(options.systemPrompt),
      history: options.conversationHistory,
      knowledge_context: options.knowledgeContext,
      max_tokens: options.maxTokens,
      temperature: options.temperature,
    };

    // Make the request with retries
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= this.config.retries; attempt++) {
      try {
        const response = await this.callCandle(request);
        return {
          text: response.text,
          confidence: response.confidence,
        };
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (attempt < this.config.retries) {
          await this.sleep(500 * (attempt + 1));
        }
      }
    }

    throw lastError || new Error('Failed to generate response');
  }

  /**
   * Generate with domain override
   */
  async generateForDomain(
    domain: CivicAgentDomain,
    message: string,
    options: {
      history?: string;
      knowledgeContext?: string;
      maxTokens?: number;
      temperature?: number;
    } = {}
  ): Promise<{ text: string; confidence: number; shouldEscalate: boolean }> {
    const available = await this.checkHealth();
    if (!available) {
      throw new Error('Candle not available');
    }

    const request: CandleRequest = {
      message,
      domain,
      history: options.history,
      knowledge_context: options.knowledgeContext,
      max_tokens: options.maxTokens ?? 256,
      temperature: options.temperature ?? 0.7,
    };

    const response = await this.callCandle(request);
    return {
      text: response.text,
      confidence: response.confidence,
      shouldEscalate: response.should_escalate,
    };
  }

  /**
   * Get server info
   */
  async getInfo(): Promise<{
    model: string;
    device: string;
    memoryUsed: number;
  } | null> {
    try {
      const response = await fetch(`${this.config.serverUrl}/info`);
      if (!response.ok) return null;
      return (await response.json()) as {
        model: string;
        device: string;
        memoryUsed: number;
      };
    } catch {
      return null;
    }
  }

  // Private methods

  private async callCandle(request: CandleRequest): Promise<CandleResponse> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.config.timeoutMs);

    try {
      const response = await fetch(`${this.config.serverUrl}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Candle error: ${response.status} - ${errorText}`);
      }

      return (await response.json()) as CandleResponse;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  private extractUserMessage(history: string): string {
    // Extract the last user message from conversation history
    const lines = history.split('\n');
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i].trim();
      if (line.startsWith('Citizen:') || line.startsWith('User:')) {
        return line.replace(/^(Citizen|User):\s*/, '');
      }
    }
    return history;
  }

  private detectDomainFromPrompt(systemPrompt: string): string {
    const lower = systemPrompt.toLowerCase();

    if (lower.includes('benefits') || lower.includes('snap') || lower.includes('medicaid')) {
      return 'benefits';
    }
    if (lower.includes('permit') || lower.includes('license')) {
      return 'permits';
    }
    if (lower.includes('tax')) {
      return 'tax';
    }
    if (lower.includes('voting') || lower.includes('election')) {
      return 'voting';
    }
    if (lower.includes('justice') || lower.includes('court') || lower.includes('legal')) {
      return 'justice';
    }
    if (lower.includes('housing') || lower.includes('rent') || lower.includes('shelter')) {
      return 'housing';
    }
    if (lower.includes('employment') || lower.includes('job') || lower.includes('unemploy')) {
      return 'employment';
    }
    if (lower.includes('education') || lower.includes('school') || lower.includes('college')) {
      return 'education';
    }
    if (lower.includes('health') || lower.includes('medical') || lower.includes('clinic')) {
      return 'health';
    }
    if (lower.includes('emergency') || lower.includes('crisis') || lower.includes('disaster')) {
      return 'emergency';
    }

    return 'general';
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/**
 * Create a Candle client
 */
export function createCandleClient(
  config: Partial<CandleClientConfig> = {}
): CandleClient {
  return new CandleClient(config);
}

/**
 * Create a resilient model client that tries Candle first, then Ollama, then rule-based
 */
export function createResilientCandleClient(
  candleConfig: Partial<CandleClientConfig> = {},
  ollamaConfig: { baseUrl?: string; model?: string } = {}
): ModelClient {
  const candle = new CandleClient(candleConfig);

  // Import Ollama client dynamically to avoid circular deps
  let ollamaClient: ModelClient | null = null;

  return {
    async generate(options) {
      // Try Candle first (fastest)
      try {
        const candleAvailable = await candle.checkHealth();
        if (candleAvailable) {
          console.log('[Symthaea] Using Candle inference');
          return await candle.generate(options);
        }
      } catch (error) {
        console.log('[Symthaea] Candle error, trying fallback:', error);
      }

      // Try Ollama second
      if (candleConfig.fallbackToOllama !== false) {
        try {
          if (!ollamaClient) {
            const { createOllamaClient } = await import('./ollama-client.js');
            ollamaClient = createOllamaClient({
              baseUrl: ollamaConfig.baseUrl ?? 'http://localhost:11434',
              model: ollamaConfig.model ?? 'llama3.2:3b',
            });
          }

          console.log('[Symthaea] Using Ollama fallback');
          return await ollamaClient.generate(options);
        } catch (error) {
          console.log('[Symthaea] Ollama error, using rule-based:', error);
        }
      }

      // Final fallback: rule-based response
      console.log('[Symthaea] Using rule-based fallback');
      return {
        text:
          "I'm here to help! Could you tell me more about what you need? " +
          'I can assist with benefits, permits, voting, taxes, housing, and other government services. ' +
          'For immediate help, call 311.',
        confidence: 0.5,
      };
    },
  };
}

/**
 * Candle inference server (for running in Node.js alongside the Rust engine)
 *
 * This is a stub - the actual server runs in Rust.
 * This interface is provided for type checking.
 */
export interface CandleServerConfig {
  port: number;
  host: string;
  modelPath?: string;
  useGpu: boolean;
}

export const DEFAULT_SERVER_CONFIG: CandleServerConfig = {
  port: 3600,
  host: '127.0.0.1',
  useGpu: true,
};
