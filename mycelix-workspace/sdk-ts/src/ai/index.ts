/**
 * Mycelix AI - Privacy-Preserving Local AI
 *
 * All AI processing happens locally using on-device language models.
 * Supports Ollama, llama.cpp, LM Studio, and other local inference providers.
 *
 * @example Using the assistant
 * ```typescript
 * import { ai } from '@mycelix/sdk';
 *
 * // Create an assistant with default Ollama backend
 * const assistant = ai.createAssistant();
 *
 * // Check availability
 * if (await assistant.isAvailable()) {
 *   // Chat with the assistant
 *   const response = await assistant.chat("What is MATL?");
 *   console.log(response);
 *
 *   // Analyze an epistemic claim
 *   const analysis = await assistant.analyzeClaim(
 *     "The transaction was verified by 3 independent validators"
 *   );
 *   console.log(analysis.suggestedClassification);
 *
 *   // Interpret a trust score
 *   const interpretation = await assistant.interpretTrust({
 *     composite: 0.75,
 *     quality: 0.8,
 *     consistency: 0.7,
 *     reputation: 0.75,
 *   });
 *   console.log(interpretation.tier); // "trusted"
 * }
 * ```
 *
 * @example Using the low-level LLM client
 * ```typescript
 * import { ai } from '@mycelix/sdk';
 *
 * // Create an Ollama client
 * const llm = ai.createOllamaClient('mistral', 'http://localhost:11434');
 *
 * // Generate a completion
 * const result = await llm.chat([
 *   { role: 'system', content: 'You are a helpful assistant.' },
 *   { role: 'user', content: 'Explain Byzantine fault tolerance.' }
 * ]);
 *
 * console.log(result.content);
 * console.log(`Latency: ${result.latencyMs}ms`);
 *
 * // Generate embeddings
 * const embeddings = await llm.embed('Trust is the foundation of cooperation.');
 * console.log(`Embedding dimension: ${embeddings[0].embedding.length}`);
 *
 * // List available models
 * const models = await llm.listModels();
 * models.forEach(m => console.log(`${m.name}: ${m.parameterSize}`));
 * ```
 *
 * @module ai
 */

// Local LLM client
export {
  LocalLLMClient,
  createLocalLLM,
  createOllamaClient,
  createLlamaCppClient,
  createLMStudioClient,
  type LLMProvider,
  type LLMConfig,
  type ChatMessage,
  type CompletionOptions,
  type CompletionResult,
  type EmbeddingResult,
  type ModelInfo,
} from './local-llm.js';

// Mycelix-specific assistant
export {
  MycelixAssistant,
  createAssistant,
  createOllamaAssistant,
  type AssistantConfig,
  type ClaimAnalysis,
  type TrustInterpretation,
  type ProposalAnalysis,
} from './assistant.js';
