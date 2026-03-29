// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Local LLM Integration Service
 *
 * Provides AI-powered features using local LLM backends:
 * - Ollama (primary)
 * - llama.cpp server
 * - LM Studio
 *
 * Features:
 * - Thought summarization
 * - Auto-classification
 * - Contradiction detection suggestions
 * - Belief synthesis
 */

import { writable, get } from 'svelte/store';

// ============================================================================
// TYPES
// ============================================================================

export interface LLMConfig {
  provider: 'ollama' | 'llamacpp' | 'lmstudio' | 'none';
  endpoint: string;
  model: string;
  temperature: number;
  maxTokens: number;
}

export interface LLMStatus {
  available: boolean;
  provider: string;
  model: string;
  error?: string;
}

export interface CompletionRequest {
  prompt: string;
  systemPrompt?: string;
  temperature?: number;
  maxTokens?: number;
  stopSequences?: string[];
}

export interface CompletionResponse {
  text: string;
  tokensUsed: number;
  model: string;
}

export interface ClassificationResult {
  thoughtType: string;
  confidence: number;
  suggestedTags: string[];
  domain?: string;
}

export interface SummaryResult {
  summary: string;
  keyPoints: string[];
  connections: string[];
}

// ============================================================================
// STATE
// ============================================================================

export const llmStatus = writable<LLMStatus>({
  available: false,
  provider: 'none',
  model: '',
});

export const llmConfig = writable<LLMConfig>({
  provider: 'ollama',
  endpoint: 'http://localhost:11434',
  model: 'llama3.2',
  temperature: 0.7,
  maxTokens: 512,
});

/**
 * Update LLM configuration
 */
export function setLLMConfig(config: Partial<LLMConfig>): void {
  llmConfig.update((current) => ({ ...current, ...config }));
}

// ============================================================================
// PROVIDER IMPLEMENTATIONS
// ============================================================================

async function ollamaGenerate(config: LLMConfig, request: CompletionRequest): Promise<CompletionResponse> {
  const response = await fetch(`${config.endpoint}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      prompt: request.systemPrompt
        ? `${request.systemPrompt}\n\nUser: ${request.prompt}\n\nAssistant:`
        : request.prompt,
      stream: false,
      options: {
        temperature: request.temperature ?? config.temperature,
        num_predict: request.maxTokens ?? config.maxTokens,
        stop: request.stopSequences,
      },
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `Ollama request failed (HTTP ${response.status}): ${errorBody || 'Unknown error'}. ` +
      `Check that Ollama is running at ${config.endpoint} and model '${config.model}' is available. ` +
      `Try: ollama pull ${config.model}`
    );
  }

  const data = await response.json();
  return {
    text: data.response,
    tokensUsed: data.eval_count || 0,
    model: config.model,
  };
}

async function llamaCppGenerate(config: LLMConfig, request: CompletionRequest): Promise<CompletionResponse> {
  const response = await fetch(`${config.endpoint}/completion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: request.systemPrompt
        ? `${request.systemPrompt}\n\nUser: ${request.prompt}\n\nAssistant:`
        : request.prompt,
      n_predict: request.maxTokens ?? config.maxTokens,
      temperature: request.temperature ?? config.temperature,
      stop: request.stopSequences || ['User:', '\n\n'],
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `llama.cpp request failed (HTTP ${response.status}): ${errorBody || 'Unknown error'}. ` +
      `Check that llama.cpp server is running at ${config.endpoint}. ` +
      `Start with: ./server -m model.gguf --port 8080`
    );
  }

  const data = await response.json();
  return {
    text: data.content,
    tokensUsed: data.tokens_predicted || 0,
    model: 'llama.cpp',
  };
}

async function lmStudioGenerate(config: LLMConfig, request: CompletionRequest): Promise<CompletionResponse> {
  const response = await fetch(`${config.endpoint}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: config.model,
      messages: [
        ...(request.systemPrompt ? [{ role: 'system', content: request.systemPrompt }] : []),
        { role: 'user', content: request.prompt },
      ],
      temperature: request.temperature ?? config.temperature,
      max_tokens: request.maxTokens ?? config.maxTokens,
      stop: request.stopSequences,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => '');
    throw new Error(
      `LM Studio request failed (HTTP ${response.status}): ${errorBody || 'Unknown error'}. ` +
      `Check that LM Studio is running with a loaded model at ${config.endpoint}. ` +
      `Enable the local server in LM Studio settings.`
    );
  }

  const data = await response.json();
  return {
    text: data.choices[0].message.content,
    tokensUsed: data.usage?.total_tokens || 0,
    model: data.model || config.model,
  };
}

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

/**
 * Check if local LLM is available
 */
export async function checkLLMStatus(): Promise<LLMStatus> {
  const config = get(llmConfig);

  try {
    let available = false;
    let model = '';

    switch (config.provider) {
      case 'ollama': {
        const response = await fetch(`${config.endpoint}/api/tags`, {
          signal: AbortSignal.timeout(3000),
        });
        if (response.ok) {
          const data = await response.json();
          available = data.models?.length > 0;
          model = config.model;
        }
        break;
      }
      case 'llamacpp': {
        const response = await fetch(`${config.endpoint}/health`, {
          signal: AbortSignal.timeout(3000),
        });
        available = response.ok;
        model = 'llama.cpp';
        break;
      }
      case 'lmstudio': {
        const response = await fetch(`${config.endpoint}/v1/models`, {
          signal: AbortSignal.timeout(3000),
        });
        if (response.ok) {
          const data = await response.json();
          available = data.data?.length > 0;
          model = data.data?.[0]?.id || config.model;
        }
        break;
      }
      default:
        available = false;
    }

    const status: LLMStatus = {
      available,
      provider: config.provider,
      model,
    };

    llmStatus.set(status);
    return status;
  } catch (error) {
    const status: LLMStatus = {
      available: false,
      provider: config.provider,
      model: '',
      error: error instanceof Error ? error.message : 'Connection failed',
    };
    llmStatus.set(status);
    return status;
  }
}

/**
 * Generate completion from local LLM
 */
export async function generate(request: CompletionRequest): Promise<CompletionResponse> {
  const config = get(llmConfig);

  switch (config.provider) {
    case 'ollama':
      return ollamaGenerate(config, request);
    case 'llamacpp':
      return llamaCppGenerate(config, request);
    case 'lmstudio':
      return lmStudioGenerate(config, request);
    default:
      throw new Error('No LLM provider configured');
  }
}

// ============================================================================
// HIGH-LEVEL FEATURES
// ============================================================================

/**
 * Auto-classify a thought
 */
export async function classifyThought(content: string): Promise<ClassificationResult> {
  const systemPrompt = `You are a thought classifier. Analyze the given text and classify it.
Respond in JSON format with these fields:
- thoughtType: one of "belief", "question", "evidence", "hypothesis", "note", "idea"
- confidence: 0-1 confidence score
- suggestedTags: array of 1-3 relevant tags
- domain: optional domain like "philosophy", "science", "personal", etc.

Only respond with valid JSON, no other text.`;

  const response = await generate({
    prompt: content,
    systemPrompt,
    temperature: 0.3,
    maxTokens: 200,
  });

  try {
    // Extract JSON from response
    const jsonMatch = response.text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (e) {
    console.error('Failed to parse classification:', e);
  }

  // Fallback
  return {
    thoughtType: 'note',
    confidence: 0.5,
    suggestedTags: [],
  };
}

/**
 * Summarize multiple thoughts
 */
export async function summarizeThoughts(contents: string[]): Promise<SummaryResult> {
  const systemPrompt = `You are a thought synthesizer. Given multiple thoughts, provide:
1. A concise summary (2-3 sentences)
2. Key points (3-5 bullet points)
3. Potential connections between the thoughts

Respond in JSON format with fields: summary, keyPoints (array), connections (array).
Only respond with valid JSON.`;

  const prompt = contents.map((c, i) => `Thought ${i + 1}: ${c}`).join('\n\n');

  const response = await generate({
    prompt,
    systemPrompt,
    temperature: 0.5,
    maxTokens: 500,
  });

  try {
    const jsonMatch = response.text.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (e) {
    console.error('Failed to parse summary:', e);
  }

  return {
    summary: 'Unable to generate summary',
    keyPoints: [],
    connections: [],
  };
}

/**
 * Suggest contradiction resolution
 */
export async function suggestResolution(
  thoughtA: string,
  thoughtB: string,
  description: string
): Promise<string> {
  const systemPrompt = `You are a philosophical counselor helping to resolve contradictions in beliefs.
Given two contradictory thoughts, suggest a thoughtful resolution or synthesis.
Be concise (2-4 sentences) and suggest how the person might reconcile or update their beliefs.`;

  const prompt = `Thought A: ${thoughtA}
Thought B: ${thoughtB}
Contradiction: ${description}

How might these be reconciled?`;

  const response = await generate({
    prompt,
    systemPrompt,
    temperature: 0.7,
    maxTokens: 300,
  });

  return response.text.trim();
}

/**
 * Generate questions to deepen understanding
 */
export async function generateQuestions(content: string): Promise<string[]> {
  const systemPrompt = `You are a Socratic teacher. Given a belief or idea, generate 3-5 probing questions
that would help deepen understanding or reveal hidden assumptions.
Respond as a JSON array of strings, nothing else.`;

  const response = await generate({
    prompt: content,
    systemPrompt,
    temperature: 0.8,
    maxTokens: 300,
  });

  try {
    const jsonMatch = response.text.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (e) {
    console.error('Failed to parse questions:', e);
  }

  return [];
}

/**
 * Suggest related concepts
 */
export async function suggestRelated(content: string): Promise<string[]> {
  const systemPrompt = `Given a thought or concept, suggest 3-5 related concepts, theories, or ideas
that might be worth exploring. Respond as a JSON array of strings.`;

  const response = await generate({
    prompt: content,
    systemPrompt,
    temperature: 0.7,
    maxTokens: 200,
  });

  try {
    const jsonMatch = response.text.match(/\[[\s\S]*\]/);
    if (jsonMatch) {
      return JSON.parse(jsonMatch[0]);
    }
  } catch (e) {
    console.error('Failed to parse related concepts:', e);
  }

  return [];
}

// ============================================================================
// INITIALIZATION
// ============================================================================

/**
 * Initialize LLM service
 */
export async function initializeLLM(): Promise<boolean> {
  const status = await checkLLMStatus();
  return status.available;
}

/**
 * Update LLM configuration
 */
export function updateConfig(config: Partial<LLMConfig>): void {
  llmConfig.update((current) => ({ ...current, ...config }));
}

// Auto-check on import (browser only)
if (typeof window !== 'undefined') {
  setTimeout(() => {
    checkLLMStatus().catch(console.error);
  }, 1000);
}
