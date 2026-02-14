/**
 * Local LLM Integration
 *
 * Privacy-preserving AI using local language models.
 * Supports Ollama, llama.cpp, and other local inference providers.
 *
 * All inference happens on-device - no data leaves the user's machine.
 */

// =============================================================================
// Types
// =============================================================================

export type LLMProvider = 'ollama' | 'llamacpp' | 'lmstudio' | 'custom';

export interface LLMConfig {
  provider: LLMProvider;
  baseUrl: string;
  model: string;
  timeout?: number;
  temperature?: number;
  maxTokens?: number;
  contextWindow?: number;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface CompletionOptions {
  temperature?: number;
  maxTokens?: number;
  stopSequences?: string[];
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
  seed?: number;
}

export interface CompletionResult {
  content: string;
  model: string;
  promptTokens?: number;
  completionTokens?: number;
  totalTokens?: number;
  finishReason?: 'stop' | 'length' | 'error';
  latencyMs: number;
}

export interface EmbeddingResult {
  embedding: number[];
  model: string;
  latencyMs: number;
}

export interface ModelInfo {
  name: string;
  size: number;
  quantization?: string;
  family?: string;
  parameterSize?: string;
  modified?: Date;
}

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_CONFIG: Partial<LLMConfig> = {
  timeout: 60000,
  temperature: 0.7,
  maxTokens: 2048,
  contextWindow: 4096,
};

const DEFAULT_OLLAMA_URL = 'http://localhost:11434';
const DEFAULT_LLAMACPP_URL = 'http://localhost:8080';
const DEFAULT_LMSTUDIO_URL = 'http://localhost:1234';

// =============================================================================
// Local LLM Client
// =============================================================================

export class LocalLLMClient {
  private config: Required<LLMConfig>;

  constructor(config: Partial<LLMConfig> & Pick<LLMConfig, 'provider' | 'model'>) {
    const baseUrl = config.baseUrl ?? this.getDefaultUrl(config.provider);
    this.config = {
      timeout: DEFAULT_CONFIG.timeout!,
      temperature: DEFAULT_CONFIG.temperature!,
      maxTokens: DEFAULT_CONFIG.maxTokens!,
      contextWindow: DEFAULT_CONFIG.contextWindow!,
      ...config,
      baseUrl,
    };
  }

  /**
   * Generate a chat completion
   */
  async chat(messages: ChatMessage[], options?: CompletionOptions): Promise<CompletionResult> {
    const startTime = Date.now();

    try {
      switch (this.config.provider) {
        case 'ollama':
          return await this.ollamaChat(messages, options);
        case 'llamacpp':
          return await this.llamacppChat(messages, options);
        case 'lmstudio':
          return await this.openaiCompatChat(messages, options);
        case 'custom':
          return await this.openaiCompatChat(messages, options);
        default: {
          // Type assertion for exhaustive switch - this code is unreachable
          const _exhaustive: never = this.config.provider;
          throw new Error(`Unsupported provider: ${_exhaustive as string}`);
        }
      }
    } catch (error) {
      return {
        content: '',
        model: this.config.model,
        finishReason: 'error',
        latencyMs: Date.now() - startTime,
      };
    }
  }

  /**
   * Generate a simple completion (no chat context)
   */
  async complete(prompt: string, options?: CompletionOptions): Promise<CompletionResult> {
    const messages: ChatMessage[] = [{ role: 'user', content: prompt }];
    return this.chat(messages, options);
  }

  /**
   * Generate embeddings for text
   */
  async embed(text: string | string[]): Promise<EmbeddingResult[]> {
    const texts = Array.isArray(text) ? text : [text];
    const startTime = Date.now();

    try {
      switch (this.config.provider) {
        case 'ollama':
          return await this.ollamaEmbed(texts);
        case 'llamacpp':
          return await this.llamacppEmbed(texts);
        default:
          throw new Error(`Embeddings not supported for provider: ${this.config.provider as string}`);
      }
    } catch (error) {
      console.error('[LocalLLM] Embedding error:', error);
      return texts.map(() => ({
        embedding: [],
        model: this.config.model,
        latencyMs: Date.now() - startTime,
      }));
    }
  }

  /**
   * Check if the LLM service is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.config.baseUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * List available models
   */
  async listModels(): Promise<ModelInfo[]> {
    try {
      switch (this.config.provider) {
        case 'ollama':
          return await this.ollamaListModels();
        case 'lmstudio':
          return await this.openaiListModels();
        default:
          return [];
      }
    } catch (error) {
      console.error('[LocalLLM] List models error:', error);
      return [];
    }
  }

  /**
   * Pull a model (Ollama only)
   */
  async pullModel(modelName: string): Promise<boolean> {
    if (this.config.provider !== 'ollama') {
      throw new Error('Pull model is only supported for Ollama');
    }

    try {
      const response = await fetch(`${this.config.baseUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: modelName }),
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<LLMConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration
   */
  updateConfig(config: Partial<LLMConfig>): void {
    Object.assign(this.config, config);
  }

  // =============================================================================
  // Provider-Specific Implementations
  // =============================================================================

  private async ollamaChat(
    messages: ChatMessage[],
    options?: CompletionOptions
  ): Promise<CompletionResult> {
    const startTime = Date.now();

    const response = await fetch(`${this.config.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        messages: messages.map((m) => ({ role: m.role, content: m.content })),
        stream: false,
        options: {
          temperature: options?.temperature ?? this.config.temperature,
          num_predict: options?.maxTokens ?? this.config.maxTokens,
          top_p: options?.topP,
          top_k: options?.topK,
          repeat_penalty: options?.repeatPenalty,
          seed: options?.seed,
          stop: options?.stopSequences,
        },
      }),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`Ollama error: ${response.statusText}`);
    }

    const data = (await response.json()) as {
      message: { content: string };
      prompt_eval_count?: number;
      eval_count?: number;
      done_reason?: string;
    };

    return {
      content: data.message.content,
      model: this.config.model,
      promptTokens: data.prompt_eval_count,
      completionTokens: data.eval_count,
      totalTokens: (data.prompt_eval_count ?? 0) + (data.eval_count ?? 0),
      finishReason: data.done_reason === 'stop' ? 'stop' : 'length',
      latencyMs: Date.now() - startTime,
    };
  }

  private async llamacppChat(
    messages: ChatMessage[],
    options?: CompletionOptions
  ): Promise<CompletionResult> {
    const startTime = Date.now();

    // llama.cpp uses /completion endpoint with formatted prompt
    const prompt = this.formatChatPrompt(messages);

    const response = await fetch(`${this.config.baseUrl}/completion`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        temperature: options?.temperature ?? this.config.temperature,
        n_predict: options?.maxTokens ?? this.config.maxTokens,
        top_p: options?.topP ?? 0.9,
        top_k: options?.topK ?? 40,
        repeat_penalty: options?.repeatPenalty ?? 1.1,
        seed: options?.seed ?? -1,
        stop: options?.stopSequences ?? ['</s>', 'User:', 'Human:'],
      }),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`llama.cpp error: ${response.statusText}`);
    }

    const data = (await response.json()) as {
      content: string;
      tokens_predicted?: number;
      tokens_evaluated?: number;
      stop: boolean;
    };

    return {
      content: data.content.trim(),
      model: this.config.model,
      promptTokens: data.tokens_evaluated,
      completionTokens: data.tokens_predicted,
      totalTokens: (data.tokens_evaluated ?? 0) + (data.tokens_predicted ?? 0),
      finishReason: data.stop ? 'stop' : 'length',
      latencyMs: Date.now() - startTime,
    };
  }

  private async openaiCompatChat(
    messages: ChatMessage[],
    options?: CompletionOptions
  ): Promise<CompletionResult> {
    const startTime = Date.now();

    const response = await fetch(`${this.config.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        messages,
        temperature: options?.temperature ?? this.config.temperature,
        max_tokens: options?.maxTokens ?? this.config.maxTokens,
        top_p: options?.topP,
        stop: options?.stopSequences,
      }),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`OpenAI compat error: ${response.statusText}`);
    }

    const data = (await response.json()) as {
      choices: Array<{
        message: { content: string };
        finish_reason: string;
      }>;
      usage?: {
        prompt_tokens: number;
        completion_tokens: number;
        total_tokens: number;
      };
    };

    const choice = data.choices[0];
    return {
      content: choice?.message.content ?? '',
      model: this.config.model,
      promptTokens: data.usage?.prompt_tokens,
      completionTokens: data.usage?.completion_tokens,
      totalTokens: data.usage?.total_tokens,
      finishReason: choice?.finish_reason === 'stop' ? 'stop' : 'length',
      latencyMs: Date.now() - startTime,
    };
  }

  private async ollamaEmbed(texts: string[]): Promise<EmbeddingResult[]> {
    const results: EmbeddingResult[] = [];

    for (const text of texts) {
      const startTime = Date.now();
      const response = await fetch(`${this.config.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.model,
          prompt: text,
        }),
        signal: AbortSignal.timeout(this.config.timeout),
      });

      if (!response.ok) {
        throw new Error(`Ollama embedding error: ${response.statusText}`);
      }

      const data = (await response.json()) as { embedding: number[] };
      results.push({
        embedding: data.embedding,
        model: this.config.model,
        latencyMs: Date.now() - startTime,
      });
    }

    return results;
  }

  private async llamacppEmbed(texts: string[]): Promise<EmbeddingResult[]> {
    const results: EmbeddingResult[] = [];

    for (const text of texts) {
      const startTime = Date.now();
      const response = await fetch(`${this.config.baseUrl}/embedding`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: text }),
        signal: AbortSignal.timeout(this.config.timeout),
      });

      if (!response.ok) {
        throw new Error(`llama.cpp embedding error: ${response.statusText}`);
      }

      const data = (await response.json()) as { embedding: number[] };
      results.push({
        embedding: data.embedding,
        model: this.config.model,
        latencyMs: Date.now() - startTime,
      });
    }

    return results;
  }

  private async ollamaListModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${this.config.baseUrl}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error('Failed to list models');
    }

    const data = (await response.json()) as {
      models: Array<{
        name: string;
        size: number;
        details?: {
          family?: string;
          parameter_size?: string;
          quantization_level?: string;
        };
        modified_at?: string;
      }>;
    };

    return data.models.map((m) => ({
      name: m.name,
      size: m.size,
      family: m.details?.family,
      parameterSize: m.details?.parameter_size,
      quantization: m.details?.quantization_level,
      modified: m.modified_at ? new Date(m.modified_at) : undefined,
    }));
  }

  private async openaiListModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${this.config.baseUrl}/v1/models`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error('Failed to list models');
    }

    const data = (await response.json()) as {
      data: Array<{ id: string; owned_by?: string }>;
    };

    return data.data.map((m) => ({
      name: m.id,
      size: 0,
      family: m.owned_by,
    }));
  }

  private getDefaultUrl(provider: LLMProvider): string {
    switch (provider) {
      case 'ollama':
        return DEFAULT_OLLAMA_URL;
      case 'llamacpp':
        return DEFAULT_LLAMACPP_URL;
      case 'lmstudio':
        return DEFAULT_LMSTUDIO_URL;
      default:
        return DEFAULT_OLLAMA_URL;
    }
  }

  private formatChatPrompt(messages: ChatMessage[]): string {
    // Simple ChatML-style formatting for llama.cpp
    let prompt = '';
    for (const msg of messages) {
      switch (msg.role) {
        case 'system':
          prompt += `<|system|>\n${msg.content}\n`;
          break;
        case 'user':
          prompt += `<|user|>\n${msg.content}\n`;
          break;
        case 'assistant':
          prompt += `<|assistant|>\n${msg.content}\n`;
          break;
      }
    }
    prompt += '<|assistant|>\n';
    return prompt;
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

export function createLocalLLM(
  config: Partial<LLMConfig> & Pick<LLMConfig, 'provider' | 'model'>
): LocalLLMClient {
  return new LocalLLMClient(config);
}

/**
 * Create an Ollama client with sensible defaults
 */
export function createOllamaClient(
  model: string = 'llama3.2',
  baseUrl?: string
): LocalLLMClient {
  return new LocalLLMClient({
    provider: 'ollama',
    model,
    baseUrl: baseUrl ?? DEFAULT_OLLAMA_URL,
  });
}

/**
 * Create a llama.cpp client
 */
export function createLlamaCppClient(
  model: string = 'default',
  baseUrl?: string
): LocalLLMClient {
  return new LocalLLMClient({
    provider: 'llamacpp',
    model,
    baseUrl: baseUrl ?? DEFAULT_LLAMACPP_URL,
  });
}

/**
 * Create an LM Studio client
 */
export function createLMStudioClient(
  model: string,
  baseUrl?: string
): LocalLLMClient {
  return new LocalLLMClient({
    provider: 'lmstudio',
    model,
    baseUrl: baseUrl ?? DEFAULT_LMSTUDIO_URL,
  });
}
