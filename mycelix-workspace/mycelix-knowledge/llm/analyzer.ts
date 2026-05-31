// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - LLM Integration for Enhanced Analysis
 *
 * AI-powered epistemic classification, evidence extraction, and fact analysis
 */

// ============================================================================
// Types
// ============================================================================

export interface LLMConfig {
  provider: 'openai' | 'anthropic' | 'ollama' | 'custom';
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export interface ClassificationResult {
  empirical: number;
  normative: number;
  mythic: number;
  confidence: number;
  reasoning: string;
  keywords: {
    empirical: string[];
    normative: string[];
    mythic: string[];
  };
}

export interface EvidenceExtractionResult {
  claims: ExtractedClaim[];
  sources: ExtractedSource[];
  statistics: ExtractedStatistic[];
  quotes: ExtractedQuote[];
  relationships: ExtractedRelationship[];
}

export interface ExtractedClaim {
  content: string;
  confidence: number;
  epistemicType: 'empirical' | 'normative' | 'mythic' | 'mixed';
  supportingEvidence: string[];
}

export interface ExtractedSource {
  name: string;
  type: 'academic' | 'news' | 'government' | 'expert' | 'primary' | 'unknown';
  credibilityIndicators: string[];
}

export interface ExtractedStatistic {
  value: string;
  context: string;
  source?: string;
}

export interface ExtractedQuote {
  text: string;
  attribution?: string;
  context: string;
}

export interface ExtractedRelationship {
  sourceIndex: number;
  targetIndex: number;
  type: 'supports' | 'contradicts' | 'refines' | 'depends_on';
  strength: number;
}

export interface ContradictionResult {
  hasContradiction: boolean;
  contradictions: Contradiction[];
  overallConsistency: number;
}

export interface Contradiction {
  claim1: string;
  claim2: string;
  type: 'direct' | 'implicit' | 'scope' | 'temporal';
  explanation: string;
  severity: number;
}

export interface FactCheckAnalysis {
  verdict: 'TRUE' | 'MOSTLY_TRUE' | 'MIXED' | 'MOSTLY_FALSE' | 'FALSE' | 'UNVERIFIABLE';
  confidence: number;
  explanation: string;
  evidenceAssessment: {
    supporting: number;
    contradicting: number;
    quality: number;
  };
  suggestedSources: string[];
  caveats: string[];
}

export interface SummaryResult {
  summary: string;
  keyPoints: string[];
  epistemicMix: ClassificationResult;
  mainClaims: string[];
  uncertainties: string[];
}

// ============================================================================
// LLM Analyzer Class
// ============================================================================

export class LLMAnalyzer {
  private config: Required<LLMConfig>;

  constructor(config: LLMConfig) {
    this.config = {
      provider: config.provider,
      apiKey: config.apiKey || '',
      baseUrl: config.baseUrl || this.getDefaultBaseUrl(config.provider),
      model: config.model || this.getDefaultModel(config.provider),
      temperature: config.temperature ?? 0.3,
      maxTokens: config.maxTokens ?? 2000,
    };
  }

  private getDefaultBaseUrl(provider: LLMConfig['provider']): string {
    switch (provider) {
      case 'openai':
        return 'https://api.openai.com/v1';
      case 'anthropic':
        return 'https://api.anthropic.com/v1';
      case 'ollama':
        return 'http://localhost:11434/api';
      default:
        return '';
    }
  }

  private getDefaultModel(provider: LLMConfig['provider']): string {
    switch (provider) {
      case 'openai':
        return 'gpt-4-turbo-preview';
      case 'anthropic':
        return 'claude-3-sonnet-20240229';
      case 'ollama':
        return 'llama2';
      default:
        return '';
    }
  }

  // ==========================================================================
  // Epistemic Classification
  // ==========================================================================

  async classifyEpistemic(text: string): Promise<ClassificationResult> {
    const prompt = `Analyze the following text and classify it along three epistemic dimensions:

1. EMPIRICAL (0-1): How verifiable through observation, experiment, or data?
   - High: Scientific claims, statistics, measurable facts
   - Low: Opinions, values, beliefs

2. NORMATIVE (0-1): How much does it express values, ethics, or "should" statements?
   - High: Moral judgments, prescriptions, evaluations
   - Low: Descriptive facts without value loading

3. MYTHIC (0-1): How much does it involve meaning-making, narrative, or cultural significance?
   - High: Stories, traditions, existential claims
   - Low: Technical, procedural statements

The three values should sum to approximately 1.0.

TEXT TO ANALYZE:
"""
${text}
"""

Respond in JSON format:
{
  "empirical": <0-1>,
  "normative": <0-1>,
  "mythic": <0-1>,
  "confidence": <0-1>,
  "reasoning": "<explanation of classification>",
  "keywords": {
    "empirical": ["<keywords supporting empirical classification>"],
    "normative": ["<keywords supporting normative classification>"],
    "mythic": ["<keywords supporting mythic classification>"]
  }
}`;

    const response = await this.complete(prompt);
    return this.parseJSON<ClassificationResult>(response);
  }

  // ==========================================================================
  // Evidence Extraction
  // ==========================================================================

  async extractEvidence(text: string): Promise<EvidenceExtractionResult> {
    const prompt = `Extract all claims, sources, statistics, and relationships from the following text.

TEXT:
"""
${text}
"""

Respond in JSON format:
{
  "claims": [
    {
      "content": "<the claim>",
      "confidence": <0-1>,
      "epistemicType": "empirical|normative|mythic|mixed",
      "supportingEvidence": ["<evidence from text>"]
    }
  ],
  "sources": [
    {
      "name": "<source name>",
      "type": "academic|news|government|expert|primary|unknown",
      "credibilityIndicators": ["<indicators>"]
    }
  ],
  "statistics": [
    {
      "value": "<the statistic>",
      "context": "<surrounding context>",
      "source": "<source if mentioned>"
    }
  ],
  "quotes": [
    {
      "text": "<the quote>",
      "attribution": "<who said it>",
      "context": "<context>"
    }
  ],
  "relationships": [
    {
      "sourceIndex": <claim index>,
      "targetIndex": <claim index>,
      "type": "supports|contradicts|refines|depends_on",
      "strength": <0-1>
    }
  ]
}`;

    const response = await this.complete(prompt);
    return this.parseJSON<EvidenceExtractionResult>(response);
  }

  // ==========================================================================
  // Contradiction Detection
  // ==========================================================================

  async detectContradictions(claims: string[]): Promise<ContradictionResult> {
    const prompt = `Analyze the following claims for contradictions or inconsistencies.

CLAIMS:
${claims.map((c, i) => `${i + 1}. "${c}"`).join('\n')}

Look for:
- Direct contradictions (claim A says X, claim B says not-X)
- Implicit contradictions (claims that can't both be true)
- Scope contradictions (claims true in different contexts)
- Temporal contradictions (claims true at different times)

Respond in JSON format:
{
  "hasContradiction": <boolean>,
  "contradictions": [
    {
      "claim1": "<first claim>",
      "claim2": "<second claim>",
      "type": "direct|implicit|scope|temporal",
      "explanation": "<why they contradict>",
      "severity": <0-1>
    }
  ],
  "overallConsistency": <0-1>
}`;

    const response = await this.complete(prompt);
    return this.parseJSON<ContradictionResult>(response);
  }

  // ==========================================================================
  // Fact Check Analysis
  // ==========================================================================

  async analyzeFactCheck(
    statement: string,
    evidence: {
      supporting: string[];
      contradicting: string[];
      uncertain: string[];
    }
  ): Promise<FactCheckAnalysis> {
    const prompt = `Analyze the following statement against the provided evidence and determine its veracity.

STATEMENT TO CHECK:
"${statement}"

SUPPORTING EVIDENCE:
${evidence.supporting.map((e) => `- ${e}`).join('\n') || '(none)'}

CONTRADICTING EVIDENCE:
${evidence.contradicting.map((e) => `- ${e}`).join('\n') || '(none)'}

UNCERTAIN/CONTEXTUAL EVIDENCE:
${evidence.uncertain.map((e) => `- ${e}`).join('\n') || '(none)'}

Provide a verdict and detailed analysis. Consider:
- Quality and reliability of evidence
- Scope and context of claims
- Potential nuances or caveats
- What additional evidence would help

Respond in JSON format:
{
  "verdict": "TRUE|MOSTLY_TRUE|MIXED|MOSTLY_FALSE|FALSE|UNVERIFIABLE",
  "confidence": <0-1>,
  "explanation": "<detailed explanation>",
  "evidenceAssessment": {
    "supporting": <0-1 strength>,
    "contradicting": <0-1 strength>,
    "quality": <0-1 overall quality>
  },
  "suggestedSources": ["<sources that could help verify>"],
  "caveats": ["<important caveats or limitations>"]
}`;

    const response = await this.complete(prompt);
    return this.parseJSON<FactCheckAnalysis>(response);
  }

  // ==========================================================================
  // Summarization
  // ==========================================================================

  async summarize(text: string, maxLength: number = 200): Promise<SummaryResult> {
    const prompt = `Summarize the following text, preserving the key epistemic content.

TEXT:
"""
${text}
"""

Provide:
1. A concise summary (max ${maxLength} characters)
2. Key points
3. Epistemic classification of the overall content
4. Main claims made
5. Areas of uncertainty or ambiguity

Respond in JSON format:
{
  "summary": "<concise summary>",
  "keyPoints": ["<key point 1>", "<key point 2>"],
  "epistemicMix": {
    "empirical": <0-1>,
    "normative": <0-1>,
    "mythic": <0-1>,
    "confidence": <0-1>,
    "reasoning": "<brief reasoning>"
  },
  "mainClaims": ["<main claim 1>", "<main claim 2>"],
  "uncertainties": ["<uncertainty 1>", "<uncertainty 2>"]
}`;

    const response = await this.complete(prompt);
    return this.parseJSON<SummaryResult>(response);
  }

  // ==========================================================================
  // Similarity Analysis
  // ==========================================================================

  async findSimilarConcepts(
    text: string,
    candidateClaims: Array<{ id: string; content: string }>
  ): Promise<Array<{ id: string; similarity: number; relationship: string }>> {
    const prompt = `Analyze the semantic similarity between the following text and candidate claims.

TEXT:
"${text}"

CANDIDATE CLAIMS:
${candidateClaims.map((c, i) => `${i + 1}. [${c.id}] "${c.content}"`).join('\n')}

For each candidate, assess:
- Semantic similarity (0-1)
- Relationship type (supports, contradicts, refines, related, unrelated)

Respond in JSON format:
{
  "results": [
    {
      "id": "<claim id>",
      "similarity": <0-1>,
      "relationship": "<relationship type>"
    }
  ]
}`;

    const response = await this.complete(prompt);
    const result = this.parseJSON<{ results: Array<{ id: string; similarity: number; relationship: string }> }>(response);
    return result.results;
  }

  // ==========================================================================
  // Question Generation
  // ==========================================================================

  async generateVerificationQuestions(claim: string): Promise<string[]> {
    const prompt = `Generate questions that would help verify or refute the following claim.

CLAIM:
"${claim}"

Generate 5-7 specific, answerable questions that would:
- Test the empirical basis of the claim
- Identify potential sources of evidence
- Explore edge cases or limitations
- Consider alternative explanations

Respond in JSON format:
{
  "questions": [
    "<question 1>",
    "<question 2>"
  ]
}`;

    const response = await this.complete(prompt);
    const result = this.parseJSON<{ questions: string[] }>(response);
    return result.questions;
  }

  // ==========================================================================
  // Bias Detection
  // ==========================================================================

  async detectBias(text: string): Promise<{
    overallBias: number;
    biases: Array<{
      type: string;
      severity: number;
      examples: string[];
    }>;
    suggestions: string[];
  }> {
    const prompt = `Analyze the following text for potential biases.

TEXT:
"""
${text}
"""

Look for:
- Confirmation bias (cherry-picking supporting evidence)
- Selection bias (non-representative examples)
- Framing bias (loaded language, emotional appeals)
- Source bias (over-reliance on particular sources)
- Omission bias (missing relevant information)

Respond in JSON format:
{
  "overallBias": <0-1>,
  "biases": [
    {
      "type": "<bias type>",
      "severity": <0-1>,
      "examples": ["<example from text>"]
    }
  ],
  "suggestions": ["<how to address the bias>"]
}`;

    const response = await this.complete(prompt);
    return this.parseJSON(response);
  }

  // ==========================================================================
  // API Communication
  // ==========================================================================

  private async complete(prompt: string): Promise<string> {
    switch (this.config.provider) {
      case 'openai':
        return this.completeOpenAI(prompt);
      case 'anthropic':
        return this.completeAnthropic(prompt);
      case 'ollama':
        return this.completeOllama(prompt);
      default:
        return this.completeCustom(prompt);
    }
  }

  private async completeOpenAI(prompt: string): Promise<string> {
    const response = await fetch(`${this.config.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        model: this.config.model,
        messages: [
          {
            role: 'system',
            content:
              'You are an expert epistemic analyst specializing in claim classification, evidence extraction, and fact-checking. Always respond with valid JSON.',
          },
          { role: 'user', content: prompt },
        ],
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
        response_format: { type: 'json_object' },
      }),
    });

    const data = await response.json();
    return data.choices[0].message.content;
  }

  private async completeAnthropic(prompt: string): Promise<string> {
    const response = await fetch(`${this.config.baseUrl}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this.config.apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: this.config.model,
        max_tokens: this.config.maxTokens,
        messages: [{ role: 'user', content: prompt }],
        system:
          'You are an expert epistemic analyst. Respond only with valid JSON, no other text.',
      }),
    });

    const data = await response.json();
    return data.content[0].text;
  }

  private async completeOllama(prompt: string): Promise<string> {
    const response = await fetch(`${this.config.baseUrl}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.config.model,
        prompt: `You are an expert epistemic analyst. Respond only with valid JSON.\n\n${prompt}`,
        stream: false,
        options: {
          temperature: this.config.temperature,
        },
      }),
    });

    const data = await response.json();
    return data.response;
  }

  private async completeCustom(prompt: string): Promise<string> {
    const response = await fetch(this.config.baseUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(this.config.apiKey && { Authorization: `Bearer ${this.config.apiKey}` }),
      },
      body: JSON.stringify({
        prompt,
        model: this.config.model,
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
      }),
    });

    const data = await response.json();
    return data.response || data.text || data.content;
  }

  private parseJSON<T>(text: string): T {
    // Extract JSON from potentially wrapped response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON found in response');
    }
    return JSON.parse(jsonMatch[0]);
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createAnalyzer(config: LLMConfig): LLMAnalyzer {
  return new LLMAnalyzer(config);
}

// ============================================================================
// Batch Processing Utilities
// ============================================================================

export async function batchClassify(
  analyzer: LLMAnalyzer,
  texts: string[],
  concurrency: number = 3
): Promise<ClassificationResult[]> {
  const results: ClassificationResult[] = [];
  const queue = [...texts];

  const worker = async () => {
    while (queue.length > 0) {
      const text = queue.shift();
      if (text) {
        const result = await analyzer.classifyEpistemic(text);
        results.push(result);
      }
    }
  };

  await Promise.all(Array(concurrency).fill(null).map(() => worker()));
  return results;
}

export async function batchExtract(
  analyzer: LLMAnalyzer,
  texts: string[],
  concurrency: number = 3
): Promise<EvidenceExtractionResult[]> {
  const results: EvidenceExtractionResult[] = [];
  const queue = [...texts];

  const worker = async () => {
    while (queue.length > 0) {
      const text = queue.shift();
      if (text) {
        const result = await analyzer.extractEvidence(text);
        results.push(result);
      }
    }
  };

  await Promise.all(Array(concurrency).fill(null).map(() => worker()));
  return results;
}

export default LLMAnalyzer;
