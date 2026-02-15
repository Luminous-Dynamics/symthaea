/**
 * Mycelix AI Assistant
 *
 * Privacy-preserving AI assistant for the Mycelix ecosystem.
 * Uses local LLMs for all processing - no data leaves the device.
 *
 * Capabilities:
 * - Epistemic claim analysis
 * - Trust score interpretation
 * - Natural language queries
 * - Document summarization
 * - Governance proposal analysis
 */

import {
  type LocalLLMClient,
  createOllamaClient,
  type ChatMessage,
  type CompletionOptions,
} from './local-llm.js';

// =============================================================================
// Types
// =============================================================================

export interface AssistantConfig {
  llm: LocalLLMClient;
  systemPrompt?: string;
  maxContextMessages?: number;
}

export interface ClaimAnalysis {
  summary: string;
  empiricalAssessment: string;
  normativeContext: string;
  materialityLevel: string;
  suggestedClassification: {
    empirical: number; // 0-4
    normative: number; // 0-3
    materiality: number; // 0-3
  };
  confidence: number;
  reasoning: string;
}

export interface TrustInterpretation {
  summary: string;
  strengths: string[];
  concerns: string[];
  recommendation: string;
  tier: 'new' | 'emerging' | 'established' | 'trusted' | 'exemplary';
}

export interface ProposalAnalysis {
  summary: string;
  keyPoints: string[];
  potentialImpact: string;
  risks: string[];
  stakeholders: string[];
  recommendation: 'support' | 'oppose' | 'neutral' | 'needs-more-info';
  reasoning: string;
}

// =============================================================================
// System Prompts
// =============================================================================

const MYCELIX_SYSTEM_PROMPT = `You are an AI assistant for the Mycelix ecosystem, a decentralized trust network built on Holochain.

Key concepts you understand:
- MATL (Mycelix Adaptive Trust Layer): A reputation system with composite scores from 0-1
- Epistemic Charter: A 3D classification system for claims:
  - Empirical (E0-E4): How verifiable is the claim?
  - Normative (N0-N3): What level of agreement exists?
  - Materiality (M0-M3): How long-lasting is its importance?
- Byzantine Fault Tolerance: The system tolerates up to 34% validated malicious actors
- Holochain: Agent-centric distributed ledger technology

You prioritize:
1. Privacy: All analysis happens locally, no data is shared
2. Accuracy: Provide honest assessments, acknowledge uncertainty
3. Helpfulness: Give actionable insights and clear explanations

When analyzing claims, trust scores, or proposals, structure your responses clearly and cite specific factors that influence your assessment.`;

const CLAIM_ANALYSIS_PROMPT = `Analyze the following claim using the Epistemic Charter framework.

Classify it along three axes:
- Empirical (E0-E4): E0=Subjective, E1=Testimonial, E2=Private verification, E3=Cryptographic proof, E4=Publicly reproducible
- Normative (N0-N3): N0=Personal preference, N1=Communal agreement, N2=Network consensus, N3=Axiomatic/universal
- Materiality (M0-M3): M0=Ephemeral, M1=Temporal, M2=Persistent, M3=Foundational

Provide your analysis as JSON with these fields:
- summary: One-sentence summary
- empiricalAssessment: Why you chose this E level
- normativeContext: Why you chose this N level
- materialityLevel: Why you chose this M level
- suggestedClassification: { empirical: number, normative: number, materiality: number }
- confidence: 0-1 how confident you are
- reasoning: Overall reasoning

Claim to analyze:`;

const TRUST_INTERPRETATION_PROMPT = `Interpret the following MATL trust score for a user.

The composite score ranges from 0 to 1:
- 0.0-0.3: Low trust (new or problematic history)
- 0.3-0.5: Emerging trust (limited positive history)
- 0.5-0.7: Established trust (consistent positive behavior)
- 0.7-0.9: High trust (extensive positive track record)
- 0.9-1.0: Exemplary trust (exceptional contribution and reliability)

Components:
- Quality (40%): Based on Proof of Gradient Quality
- Consistency (30%): Behavioral consistency over time
- Reputation (30%): Network-wide peer assessments

Provide your interpretation as JSON with:
- summary: Overall trust assessment
- strengths: Array of positive factors
- concerns: Array of potential concerns
- recommendation: Action recommendation
- tier: One of "new", "emerging", "established", "trusted", "exemplary"

Trust data to interpret:`;

const PROPOSAL_ANALYSIS_PROMPT = `Analyze the following governance proposal for the Mycelix ecosystem.

Consider:
- Impact on the community
- Technical feasibility
- Economic implications
- Potential risks
- Affected stakeholders

Provide your analysis as JSON with:
- summary: Brief description
- keyPoints: Array of main points
- potentialImpact: Description of expected impact
- risks: Array of potential risks
- stakeholders: Array of affected groups
- recommendation: One of "support", "oppose", "neutral", "needs-more-info"
- reasoning: Why you made this recommendation

Proposal to analyze:`;

// =============================================================================
// Mycelix Assistant
// =============================================================================

export class MycelixAssistant {
  private llm: LocalLLMClient;
  private systemPrompt: string;
  private maxContextMessages: number;
  private conversationHistory: ChatMessage[] = [];

  constructor(config: AssistantConfig) {
    this.llm = config.llm;
    this.systemPrompt = config.systemPrompt ?? MYCELIX_SYSTEM_PROMPT;
    this.maxContextMessages = config.maxContextMessages ?? 20;
  }

  /**
   * Chat with the assistant
   */
  async chat(message: string, options?: CompletionOptions): Promise<string> {
    // Add user message to history
    this.conversationHistory.push({ role: 'user', content: message });

    // Trim history if too long
    while (this.conversationHistory.length > this.maxContextMessages) {
      this.conversationHistory.shift();
    }

    // Build messages with system prompt
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      ...this.conversationHistory,
    ];

    const result = await this.llm.chat(messages, options);

    // Add assistant response to history
    if (result.content) {
      this.conversationHistory.push({ role: 'assistant', content: result.content });
    }

    return result.content;
  }

  /**
   * Analyze an epistemic claim
   */
  async analyzeClaim(claim: string): Promise<ClaimAnalysis> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: `${CLAIM_ANALYSIS_PROMPT}\n\n"${claim}"` },
    ];

    const result = await this.llm.chat(messages, {
      temperature: 0.3, // Lower temperature for more consistent analysis
    });

    try {
      // Extract JSON from response
      const jsonMatch = result.content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]) as ClaimAnalysis;
      }
    } catch {
      // Fall back to structured extraction
    }

    // Default response if JSON parsing fails
    return {
      summary: result.content.slice(0, 200),
      empiricalAssessment: 'Unable to extract structured assessment',
      normativeContext: 'Unable to extract structured context',
      materialityLevel: 'Unable to extract structured level',
      suggestedClassification: { empirical: 1, normative: 1, materiality: 1 },
      confidence: 0.3,
      reasoning: result.content,
    };
  }

  /**
   * Interpret a trust score
   */
  async interpretTrust(trustData: {
    composite: number;
    quality?: number;
    consistency?: number;
    reputation?: number;
    interactions?: number;
    sources?: string[];
  }): Promise<TrustInterpretation> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: `${TRUST_INTERPRETATION_PROMPT}\n\n${JSON.stringify(trustData, null, 2)}` },
    ];

    const result = await this.llm.chat(messages, {
      temperature: 0.3,
    });

    try {
      const jsonMatch = result.content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]) as TrustInterpretation;
      }
    } catch {
      // Fall back
    }

    // Determine tier from composite
    const tier = this.determineTier(trustData.composite);

    return {
      summary: result.content.slice(0, 200),
      strengths: [],
      concerns: [],
      recommendation: result.content,
      tier,
    };
  }

  /**
   * Analyze a governance proposal
   */
  async analyzeProposal(proposal: {
    title: string;
    description: string;
    proposer?: string;
    category?: string;
    budget?: number;
  }): Promise<ProposalAnalysis> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      { role: 'user', content: `${PROPOSAL_ANALYSIS_PROMPT}\n\n${JSON.stringify(proposal, null, 2)}` },
    ];

    const result = await this.llm.chat(messages, {
      temperature: 0.4,
    });

    try {
      const jsonMatch = result.content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]) as ProposalAnalysis;
      }
    } catch {
      // Fall back
    }

    return {
      summary: proposal.title,
      keyPoints: [],
      potentialImpact: 'Unable to assess',
      risks: [],
      stakeholders: [],
      recommendation: 'needs-more-info',
      reasoning: result.content,
    };
  }

  /**
   * Summarize a document or text
   */
  async summarize(text: string, maxLength: number = 200): Promise<string> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      {
        role: 'user',
        content: `Summarize the following text in ${maxLength} words or less:\n\n${text}`,
      },
    ];

    const result = await this.llm.chat(messages, {
      temperature: 0.3,
      maxTokens: maxLength * 2, // Rough estimate of tokens per word
    });

    return result.content;
  }

  /**
   * Answer a question about Mycelix concepts
   */
  async explainConcept(concept: string): Promise<string> {
    const messages: ChatMessage[] = [
      { role: 'system', content: this.systemPrompt },
      {
        role: 'user',
        content: `Explain the Mycelix concept: "${concept}". Be concise and practical.`,
      },
    ];

    const result = await this.llm.chat(messages, {
      temperature: 0.5,
    });

    return result.content;
  }

  /**
   * Clear conversation history
   */
  clearHistory(): void {
    this.conversationHistory = [];
  }

  /**
   * Get conversation history
   */
  getHistory(): Readonly<ChatMessage[]> {
    return [...this.conversationHistory];
  }

  /**
   * Check if the LLM is available
   */
  async isAvailable(): Promise<boolean> {
    return this.llm.isAvailable();
  }

  // =============================================================================
  // Private Methods
  // =============================================================================

  private determineTier(
    composite: number
  ): 'new' | 'emerging' | 'established' | 'trusted' | 'exemplary' {
    if (composite >= 0.9) return 'exemplary';
    if (composite >= 0.7) return 'trusted';
    if (composite >= 0.5) return 'established';
    if (composite >= 0.3) return 'emerging';
    return 'new';
  }
}

// =============================================================================
// Factory
// =============================================================================

export function createAssistant(config?: Partial<AssistantConfig>): MycelixAssistant {
  const llm = config?.llm ?? createOllamaClient('llama3.2');
  return new MycelixAssistant({
    ...config,
    llm,
  });
}

/**
 * Create an assistant with a specific Ollama model
 */
export function createOllamaAssistant(
  model: string = 'llama3.2',
  baseUrl?: string
): MycelixAssistant {
  const llm = createOllamaClient(model, baseUrl);
  return new MycelixAssistant({ llm });
}
