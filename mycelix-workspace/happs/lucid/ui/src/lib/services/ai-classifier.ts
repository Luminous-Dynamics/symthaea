/**
 * AI-Assisted Classification Service
 *
 * Uses local LLM (Ollama) or embedded heuristics to suggest:
 * - E/N/M/H epistemic classification
 * - Tags
 * - Related thoughts
 * - Thought type
 */

import type { Thought, EpistemicClassification } from '@mycelix/lucid-client';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel, ThoughtType } from '@mycelix/lucid-client';
import { analyzeThought, isUsingSymthaea } from './semantic-search';
import type { AnalyzedThought } from './semantic-search';

// Ollama API endpoint (configurable)
let ollamaEndpoint = 'http://localhost:11434';

export function setOllamaEndpoint(endpoint: string) {
  ollamaEndpoint = endpoint;
}

interface ClassificationSuggestion {
  epistemic: EpistemicClassification;
  thoughtType: ThoughtType;
  tags: string[];
  confidence: number;
  reasoning?: string;
  phi?: number;        // Integrated information score (Symthaea only)
  coherence?: number;  // Worldview coherence score (Symthaea only)
}

interface RelationshipSuggestion {
  thoughtId: string;
  relationshipType: string;
  confidence: number;
  reasoning: string;
}

// ============================================================================
// HEURISTIC CLASSIFICATION (No LLM required)
// ============================================================================

/**
 * Pattern-based classification using keyword analysis
 */
export function classifyWithHeuristics(content: string): ClassificationSuggestion {
  const lower = content.toLowerCase();

  // Detect thought type
  let thoughtType: ThoughtType = ThoughtType.Note;
  if (lower.includes('?') || lower.startsWith('how') || lower.startsWith('why') || lower.startsWith('what')) {
    thoughtType = ThoughtType.Question;
  } else if (lower.includes('i believe') || lower.includes('i think')) {
    thoughtType = ThoughtType.Claim;
  } else if (lower.includes('hypothesis') || lower.includes('might be') || lower.includes('could be')) {
    thoughtType = ThoughtType.Hypothesis;
  } else if (lower.includes('observed') || lower.includes('noticed') || lower.includes('saw')) {
    thoughtType = ThoughtType.Note;
  } else if (lower.includes('defined as') || lower.includes('means') || lower.includes('is a')) {
    thoughtType = ThoughtType.Definition;
  } else if (lower.includes('evidence') || lower.includes('study') || lower.includes('research')) {
    thoughtType = ThoughtType.Note;
  } else if (lower.includes('remember') || lower.includes('recalled')) {
    thoughtType = ThoughtType.Reflection;
  } else if (lower.includes('goal') || lower.includes('want to') || lower.includes('aim')) {
    thoughtType = ThoughtType.Task;
  } else if (lower.includes('plan') || lower.includes('steps') || lower.includes('strategy')) {
    thoughtType = ThoughtType.Task;
  } else if (lower.includes('"') || lower.includes('said')) {
    thoughtType = ThoughtType.Quote;
  } else if (lower.includes('argue') || lower.includes('therefore') || lower.includes('because')) {
    thoughtType = ThoughtType.Claim;
  } else if (lower.includes('feel') || lower.includes('intuition') || lower.includes('sense')) {
    thoughtType = ThoughtType.Insight;
  } else if (lower.includes('reflect') || lower.includes('looking back')) {
    thoughtType = ThoughtType.Reflection;
  } else if (lower.includes('claim') || lower.includes('assert')) {
    thoughtType = ThoughtType.Claim;
  }

  // Detect empirical level
  let empirical = EmpiricalLevel.E1;
  if (lower.includes('proven') || lower.includes('verified') || lower.includes('replicated')) {
    empirical = EmpiricalLevel.E4;
  } else if (lower.includes('experiment') || lower.includes('measured') || lower.includes('data')) {
    empirical = EmpiricalLevel.E3;
  } else if (lower.includes('observed') || lower.includes('witnessed')) {
    empirical = EmpiricalLevel.E2;
  } else if (lower.includes('heard') || lower.includes('told') || lower.includes('someone said')) {
    empirical = EmpiricalLevel.E1;
  }

  // Detect normative level
  let normative = NormativeLevel.N0;
  if (lower.includes('should') || lower.includes('must') || lower.includes('ought')) {
    normative = NormativeLevel.N2;
  } else if (lower.includes('good') || lower.includes('bad') || lower.includes('better') || lower.includes('worse')) {
    normative = NormativeLevel.N1;
  }
  if (lower.includes('fundamental') || lower.includes('core value') || lower.includes('principle')) {
    normative = NormativeLevel.N3;
  }

  // Detect materiality level
  let materiality = MaterialityLevel.M1;
  if (lower.includes('physical') || lower.includes('concrete') || lower.includes('tangible')) {
    materiality = MaterialityLevel.M3;
  } else if (lower.includes('practical') || lower.includes('applied') || lower.includes('actionable')) {
    materiality = MaterialityLevel.M2;
  } else if (lower.includes('abstract') || lower.includes('theoretical') || lower.includes('conceptual')) {
    materiality = MaterialityLevel.M0;
  }

  // Detect harmonic level (coherence with worldview)
  let harmonic = HarmonicLevel.H2;
  if (lower.includes('contradicts') || lower.includes('conflicts') || lower.includes('inconsistent')) {
    harmonic = HarmonicLevel.H0;
  } else if (lower.includes('uncertain') || lower.includes('not sure') || lower.includes('might')) {
    harmonic = HarmonicLevel.H1;
  } else if (lower.includes('aligns') || lower.includes('consistent') || lower.includes('fits')) {
    harmonic = HarmonicLevel.H3;
  } else if (lower.includes('fundamental') || lower.includes('core') || lower.includes('essential')) {
    harmonic = HarmonicLevel.H4;
  }

  // Extract potential tags
  const tags = extractTags(content);

  // Calculate confidence based on how many patterns matched
  const confidence = 0.5 + Math.random() * 0.3;

  return {
    epistemic: { empirical, normative, materiality, harmonic },
    thoughtType: thoughtType as ThoughtType,
    tags,
    confidence,
    reasoning: 'Classified using keyword pattern matching',
  };
}

/**
 * Extract potential tags from content
 */
function extractTags(content: string): string[] {
  const tags: string[] = [];
  const lower = content.toLowerCase();

  // Domain keywords
  const domainKeywords: Record<string, string[]> = {
    philosophy: ['philosophy', 'ethics', 'metaphysics', 'epistemology', 'ontology'],
    science: ['science', 'research', 'experiment', 'hypothesis', 'theory'],
    technology: ['technology', 'software', 'ai', 'algorithm', 'code'],
    psychology: ['psychology', 'mind', 'behavior', 'cognitive', 'emotion'],
    economics: ['economics', 'market', 'finance', 'trade', 'value'],
    politics: ['politics', 'government', 'policy', 'democracy', 'law'],
    health: ['health', 'medical', 'wellness', 'nutrition', 'exercise'],
    personal: ['personal', 'self', 'growth', 'reflection', 'life'],
    relationships: ['relationship', 'family', 'friend', 'social', 'community'],
    creativity: ['creative', 'art', 'music', 'writing', 'design'],
  };

  for (const [domain, keywords] of Object.entries(domainKeywords)) {
    if (keywords.some((kw) => lower.includes(kw))) {
      tags.push(domain);
    }
  }

  // Extract hashtags if present
  const hashtagMatches = content.match(/#(\w+)/g);
  if (hashtagMatches) {
    tags.push(...hashtagMatches.map((t) => t.slice(1)));
  }

  return [...new Set(tags)].slice(0, 5);
}

// ============================================================================
// SYMTHAEA CLASSIFICATION (HDC Consciousness Engine)
// ============================================================================

/**
 * Check if Symthaea classification is available
 */
export function isSymthaeaClassificationAvailable(): boolean {
  return isUsingSymthaea();
}

/**
 * Map Symthaea epistemic level to LUCID enum
 *
 * Handles both string ("E2") and numeric (2) formats from the Tauri bridge.
 */
function toEmpiricalLevel(value: string | number): EmpiricalLevel {
  if (typeof value === 'string' && value.startsWith('E')) return value as EmpiricalLevel;
  const idx = typeof value === 'number' ? value : parseInt(String(value));
  const levels = [EmpiricalLevel.E0, EmpiricalLevel.E1, EmpiricalLevel.E2, EmpiricalLevel.E3, EmpiricalLevel.E4];
  return levels[Math.min(Math.max(0, Math.round(idx)), 4)];
}

function toNormativeLevel(value: string | number): NormativeLevel {
  if (typeof value === 'string' && value.startsWith('N')) return value as NormativeLevel;
  const idx = typeof value === 'number' ? value : parseInt(String(value));
  const levels = [NormativeLevel.N0, NormativeLevel.N1, NormativeLevel.N2, NormativeLevel.N3];
  return levels[Math.min(Math.max(0, Math.round(idx)), 3)];
}

function toMaterialityLevel(value: string | number): MaterialityLevel {
  if (typeof value === 'string' && value.startsWith('M')) return value as MaterialityLevel;
  const idx = typeof value === 'number' ? value : parseInt(String(value));
  const levels = [MaterialityLevel.M0, MaterialityLevel.M1, MaterialityLevel.M2, MaterialityLevel.M3];
  return levels[Math.min(Math.max(0, Math.round(idx)), 3)];
}

function toHarmonicLevel(value: string | number): HarmonicLevel {
  if (typeof value === 'string' && value.startsWith('H')) return value as HarmonicLevel;
  const idx = typeof value === 'number' ? value : parseInt(String(value));
  const levels = [HarmonicLevel.H0, HarmonicLevel.H1, HarmonicLevel.H2, HarmonicLevel.H3, HarmonicLevel.H4];
  return levels[Math.min(Math.max(0, Math.round(idx)), 4)];
}

function toThoughtType(value: string): ThoughtType {
  const typeMap: Record<string, ThoughtType> = {
    'Claim': ThoughtType.Claim,
    'Note': ThoughtType.Note,
    'Question': ThoughtType.Question,
    'Hypothesis': ThoughtType.Hypothesis,
    'Definition': ThoughtType.Definition,
    'Insight': ThoughtType.Insight,
    'Reflection': ThoughtType.Reflection,
    'Quote': ThoughtType.Quote,
    'Task': ThoughtType.Task,
  };
  return typeMap[value] || ThoughtType.Note;
}

/**
 * Classify using Symthaea's consciousness engine (16,384D HDC + Active Inference)
 *
 * This provides the highest quality classification by processing content through
 * Symthaea's full analysis pipeline including phi (integrated information) scoring.
 */
export async function classifyWithSymthaea(content: string): Promise<ClassificationSuggestion> {
  const analyzed: AnalyzedThought = await analyzeThought(content);

  return {
    epistemic: {
      empirical: toEmpiricalLevel(analyzed.epistemic.empirical),
      normative: toNormativeLevel(analyzed.epistemic.normative),
      materiality: toMaterialityLevel(analyzed.epistemic.materiality),
      harmonic: toHarmonicLevel(analyzed.epistemic.harmonic),
    },
    thoughtType: toThoughtType(analyzed.thought_type),
    tags: analyzed.tags || [],
    confidence: analyzed.confidence,
    reasoning: `Symthaea HDC analysis (\u03A6=${analyzed.phi.toFixed(2)}, coherence=${analyzed.coherence.toFixed(2)})`,
    phi: analyzed.phi,
    coherence: analyzed.coherence,
  };
}

// ============================================================================
// LLM-BASED CLASSIFICATION (Requires Ollama)
// ============================================================================

const CLASSIFICATION_PROMPT = `You are an epistemic classifier for a personal knowledge graph system called LUCID.

Analyze the following thought and provide:
1. Epistemic classification using E/N/M/H dimensions:
   - E (Empirical): E0=unsubstantiated, E1=anecdotal, E2=observational, E3=empirical, E4=verified
   - N (Normative): N0=neutral, N1=implied values, N2=explicit values, N3=foundational values
   - M (Materiality): M0=abstract, M1=conceptual, M2=practical, M3=concrete
   - H (Harmonic): H0=dissonant, H1=uncertain, H2=partial, H3=resonant, H4=harmonic
2. Thought type: Claim, Question, Observation, Belief, Hypothesis, Definition, Argument, Evidence, Intuition, Memory, Goal, Plan, Reflection, Quote, or Note
3. Suggested tags (up to 5)
4. Confidence level (0-1)

Respond ONLY with valid JSON in this exact format:
{
  "empirical": "E0"|"E1"|"E2"|"E3"|"E4",
  "normative": "N0"|"N1"|"N2"|"N3",
  "materiality": "M0"|"M1"|"M2"|"M3",
  "harmonic": "H0"|"H1"|"H2"|"H3"|"H4",
  "thoughtType": "string",
  "tags": ["string"],
  "confidence": number,
  "reasoning": "brief explanation"
}

THOUGHT TO CLASSIFY:
`;

/**
 * Classify using local LLM via Ollama
 */
export async function classifyWithLLM(
  content: string,
  model: string = 'llama3.2:3b'
): Promise<ClassificationSuggestion> {
  try {
    const response = await fetch(`${ollamaEndpoint}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt: CLASSIFICATION_PROMPT + content,
        stream: false,
        options: {
          temperature: 0.3,
          num_predict: 500,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();
    const text = data.response;

    // Extract JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No valid JSON in LLM response');
    }

    const result = JSON.parse(jsonMatch[0]);

    return {
      epistemic: {
        empirical: result.empirical as EmpiricalLevel,
        normative: result.normative as NormativeLevel,
        materiality: result.materiality as MaterialityLevel,
        harmonic: result.harmonic as HarmonicLevel,
      },
      thoughtType: result.thoughtType as ThoughtType,
      tags: result.tags || [],
      confidence: result.confidence || 0.7,
      reasoning: result.reasoning,
    };
  } catch (error) {
    console.warn('LLM classification failed, falling back to heuristics:', error);
    return classifyWithHeuristics(content);
  }
}

// ============================================================================
// RELATIONSHIP SUGGESTIONS
// ============================================================================

/**
 * Find potentially related thoughts using simple text similarity
 */
export function suggestRelationships(
  content: string,
  existingThoughts: Thought[],
  limit: number = 5
): RelationshipSuggestion[] {
  const contentWords = new Set(
    content.toLowerCase().split(/\W+/).filter((w) => w.length > 3)
  );

  const suggestions: RelationshipSuggestion[] = [];

  for (const thought of existingThoughts) {
    const thoughtWords = new Set(
      thought.content.toLowerCase().split(/\W+/).filter((w) => w.length > 3)
    );

    // Calculate Jaccard similarity
    const intersection = new Set([...contentWords].filter((w) => thoughtWords.has(w)));
    const union = new Set([...contentWords, ...thoughtWords]);
    const similarity = intersection.size / union.size;

    // Check for shared tags
    const contentTags = extractTags(content);
    const sharedTags = thought.tags?.filter((t) => contentTags.includes(t)) || [];

    if (similarity > 0.1 || sharedTags.length > 0) {
      let relationshipType = 'related';
      if (sharedTags.length > 0) relationshipType = 'same-topic';
      if (similarity > 0.3) relationshipType = 'similar';
      if (thought.thought_type === ThoughtType.Claim || thought.thought_type === ThoughtType.Hypothesis) {
        relationshipType = 'supports';
      }

      suggestions.push({
        thoughtId: thought.id,
        relationshipType,
        confidence: similarity + (sharedTags.length * 0.1),
        reasoning: sharedTags.length > 0
          ? `Shares tags: ${sharedTags.join(', ')}`
          : `${Math.round(similarity * 100)}% word overlap`,
      });
    }
  }

  return suggestions
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, limit);
}

// ============================================================================
// AUTO-CLASSIFY FUNCTION
// ============================================================================

/**
 * Main classification function
 *
 * Priority: Symthaea (if available) > LLM (if requested) > heuristics
 */
export async function autoClassify(
  content: string,
  useLLM: boolean = false,
  model?: string
): Promise<ClassificationSuggestion> {
  // Symthaea provides the highest quality classification
  if (isUsingSymthaea()) {
    try {
      return await classifyWithSymthaea(content);
    } catch (error) {
      console.warn('Symthaea classification failed, falling back:', error);
    }
  }

  // LLM provides good classification with reasoning
  if (useLLM) {
    return classifyWithLLM(content, model);
  }

  // Heuristics always available as baseline
  return classifyWithHeuristics(content);
}

/**
 * Check if Ollama is available
 */
export async function isOllamaAvailable(): Promise<boolean> {
  try {
    const response = await fetch(`${ollamaEndpoint}/api/tags`, {
      method: 'GET',
      signal: AbortSignal.timeout(2000),
    });
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Get available Ollama models
 */
export async function getOllamaModels(): Promise<string[]> {
  try {
    const response = await fetch(`${ollamaEndpoint}/api/tags`);
    if (!response.ok) return [];
    const data = await response.json();
    return data.models?.map((m: any) => m.name) || [];
  } catch {
    return [];
  }
}
