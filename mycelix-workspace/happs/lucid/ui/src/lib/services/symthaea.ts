/**
 * Symthaea Integration Service
 *
 * Connects to the Symthaea consciousness engine for coherence checking,
 * worldview analysis, and harmonic resonance calculation.
 *
 * This service provides both Tauri-based (real Symthaea) and local fallback
 * implementations for all consciousness features.
 */

import type { Thought } from '@mycelix/lucid-client';
import { lucidClient } from '../stores/holochain';
import { get } from 'svelte/store';

// Tauri invoke - will be undefined in browser-only mode
let invoke: ((cmd: string, args?: Record<string, unknown>) => Promise<unknown>) | null = null;

// Try to import Tauri's invoke function
try {
  // Dynamic import for Tauri compatibility
  import('@tauri-apps/api/core').then(tauri => {
    invoke = tauri.invoke;
  }).catch(() => {
    console.log('Tauri not available, using fallback mode');
  });
} catch {
  console.log('Running in browser mode without Tauri');
}

/**
 * Check if Symthaea is available via Tauri
 */
export async function isSymthaeaAvailable(): Promise<boolean> {
  if (!invoke) return false;
  try {
    return await invoke('symthaea_ready') as boolean;
  } catch {
    return false;
  }
}

/**
 * Initialize Symthaea (call early in app lifecycle)
 */
export async function initializeSymthaea(): Promise<{ success: boolean; message: string }> {
  if (!invoke) {
    return { success: false, message: 'Tauri not available' };
  }
  try {
    const result = await invoke('initialize_symthaea') as { success: boolean; message: string };
    return result;
  } catch (error) {
    return { success: false, message: String(error) };
  }
}

// ============================================================================
// TYPES
// ============================================================================

export interface CoherenceResult {
  overallScore: number;
  dimensions: {
    logical: number; // Logical consistency
    temporal: number; // Consistency over time
    epistemic: number; // Epistemological coherence
    harmonic: number; // Resonance with core values
  };
  contradictions: Contradiction[];
  suggestions: string[];
}

export interface Contradiction {
  thoughtA: string;
  thoughtB: string;
  type: 'direct' | 'implicit' | 'temporal';
  severity: 'low' | 'medium' | 'high';
  description: string;
}

export interface WorldviewProfile {
  coreBeliefs: ThoughtCluster[];
  valueHierarchy: string[];
  epistemicProfile: {
    preferredSources: string[];
    confidenceDistribution: number[];
    domainExpertise: string[];
  };
  harmonicCenter: {
    x: number;
    y: number;
    z: number;
  };
}

export interface ThoughtCluster {
  name: string;
  thoughts: string[];
  coherenceScore: number;
  tags: string[];
}

// ============================================================================
// COHERENCE CHECKING
// ============================================================================

/**
 * Auto-coherence input for Tauri
 */
interface AutoCoherenceInput {
  new_thought_content: string;
  new_thought_id: string;
  existing_thoughts: Array<{
    id: string;
    content: string;
    embedding?: number[];
  }>;
  similarity_threshold?: number;
}

/**
 * Auto-coherence result from Tauri
 */
interface AutoCoherenceResult {
  thought_id: string;
  coherence_score: number;
  phi_score: number;
  is_coherent: boolean;
  embedding: number[];
  contradictions: Array<{
    conflicting_thought_id: string;
    conflicting_content_preview: string;
    severity: number;
    contradiction_type: string;
    description: string;
  }>;
  suggestions: string[];
  analyzed_thought_ids: string[];
}

/**
 * Check coherence for a newly created thought via Tauri/Symthaea
 *
 * This is the primary entry point for the coherence feedback loop.
 * Call this after creating a thought to check for contradictions.
 */
export async function checkNewThoughtCoherence(
  newThought: { id: string; content: string },
  existingThoughts: Array<{ id: string; content: string; embedding?: number[] }>
): Promise<AutoCoherenceResult | null> {
  if (!invoke) return null;

  try {
    const input: AutoCoherenceInput = {
      new_thought_content: newThought.content,
      new_thought_id: newThought.id,
      existing_thoughts: existingThoughts,
      similarity_threshold: 0.3,
    };

    const result = await invoke('auto_check_coherence', { input }) as AutoCoherenceResult;
    return result;
  } catch (error) {
    console.warn('Symthaea auto-coherence failed:', error);
    return null;
  }
}

/**
 * Get embedding for text via Symthaea
 */
export async function getEmbedding(text: string): Promise<number[] | null> {
  if (!invoke) return null;

  try {
    return await invoke('embed_text', { text }) as number[];
  } catch (error) {
    console.warn('Failed to get embedding:', error);
    return null;
  }
}

/**
 * Batch compute embeddings for multiple texts
 */
export async function batchGetEmbeddings(texts: string[]): Promise<number[][] | null> {
  if (!invoke) return null;

  try {
    return await invoke('batch_embed', { texts }) as number[][];
  } catch (error) {
    console.warn('Failed to batch embed:', error);
    return null;
  }
}

/**
 * Compute similarity between two embeddings
 */
export async function computeSimilarity(embeddingA: number[], embeddingB: number[]): Promise<number | null> {
  if (!invoke) return null;

  try {
    return await invoke('compute_similarity', { embedding_a: embeddingA, embedding_b: embeddingB }) as number;
  } catch (error) {
    console.warn('Failed to compute similarity:', error);
    return null;
  }
}

/**
 * Check coherence via the bridge zome (if connected to Symthaea)
 * Falls back to local analysis if Symthaea is unavailable.
 */
export async function checkCoherenceWithSymthaea(thoughtIds: string[]): Promise<CoherenceResult | null> {
  const client = get(lucidClient);
  if (!client) return null;

  // First try Tauri/Symthaea directly
  if (invoke) {
    try {
      // Get thought contents for Symthaea analysis
      const thoughts: string[] = [];
      for (const id of thoughtIds) {
        const record = await client.lucid.getThought(id);
        if (record) {
          thoughts.push(record.content);
        }
      }

      if (thoughts.length > 0) {
        const result = await invoke('check_coherence', { thought_contents: thoughts }) as {
          overall: number;
          logical: number;
          temporal: number;
          epistemic: number;
          harmonic: number;
          contradictions: Array<{
            thought_a: string;
            thought_b: string;
            description: string;
            severity: number;
          }>;
        };

        return {
          overallScore: result.overall,
          dimensions: {
            logical: result.logical,
            temporal: result.temporal,
            epistemic: result.epistemic,
            harmonic: result.harmonic,
          },
          contradictions: result.contradictions.map(c => ({
            thoughtA: thoughtIds[parseInt(c.thought_a)] || c.thought_a,
            thoughtB: thoughtIds[parseInt(c.thought_b)] || c.thought_b,
            type: 'direct' as const,
            severity: c.severity > 0.7 ? 'high' : c.severity > 0.4 ? 'medium' : 'low',
            description: c.description,
          })),
          suggestions: result.overall < 0.5
            ? ['Review your beliefs for coherence', 'Check for potential contradictions']
            : [],
        };
      }
    } catch (error) {
      console.warn('Symthaea coherence check failed:', error);
    }
  }

  // Fall back to bridge zome
  try {
    const result = await client.bridge.checkCoherenceWithSymthaea(thoughtIds);
    return result as unknown as CoherenceResult;
  } catch (error) {
    console.warn('Symthaea integration not available, using local analysis');
    return null;
  }
}

/**
 * Local coherence analysis (fallback when Symthaea is unavailable)
 */
export function analyzeCoherenceLocally(thoughts: Thought[]): CoherenceResult {
  const contradictions: Contradiction[] = [];
  const suggestions: string[] = [];

  // Simple contradiction detection based on content
  for (let i = 0; i < thoughts.length; i++) {
    for (let j = i + 1; j < thoughts.length; j++) {
      const a = thoughts[i];
      const b = thoughts[j];

      // Check for potential contradictions
      const contradiction = detectContradiction(a, b);
      if (contradiction) {
        contradictions.push(contradiction);
      }
    }
  }

  // Calculate dimension scores
  const logicalScore = Math.max(0, 1 - (contradictions.filter(c => c.type === 'direct').length * 0.1));
  const temporalScore = calculateTemporalCoherence(thoughts);
  const epistemicScore = calculateEpistemicCoherence(thoughts);
  const harmonicScore = calculateHarmonicCoherence(thoughts);

  // Overall score (weighted average)
  const overallScore = (
    logicalScore * 0.3 +
    temporalScore * 0.2 +
    epistemicScore * 0.25 +
    harmonicScore * 0.25
  );

  // Generate suggestions
  if (contradictions.length > 0) {
    suggestions.push(`Review ${contradictions.length} potential contradictions in your beliefs`);
  }
  if (epistemicScore < 0.5) {
    suggestions.push('Consider strengthening evidence for low-confidence beliefs');
  }
  if (harmonicScore < 0.5) {
    suggestions.push('Some beliefs may not align with your core values');
  }

  return {
    overallScore,
    dimensions: {
      logical: logicalScore,
      temporal: temporalScore,
      epistemic: epistemicScore,
      harmonic: harmonicScore,
    },
    contradictions,
    suggestions,
  };
}

/**
 * Detect potential contradiction between two thoughts
 */
function detectContradiction(a: Thought, b: Thought): Contradiction | null {
  const aLower = a.content.toLowerCase();
  const bLower = b.content.toLowerCase();

  // Simple negation patterns
  const negationPatterns = [
    { positive: /\bis\b/, negative: /\bis not\b|\bisn't\b/ },
    { positive: /\bshould\b/, negative: /\bshould not\b|\bshouldn't\b/ },
    { positive: /\bcan\b/, negative: /\bcannot\b|\bcan't\b/ },
    { positive: /\balways\b/, negative: /\bnever\b/ },
    { positive: /\btrue\b/, negative: /\bfalse\b/ },
    { positive: /\bgood\b/, negative: /\bbad\b/ },
  ];

  // Check for shared topics (by tags or keywords)
  const sharedTags = a.tags?.filter(t => b.tags?.includes(t)) || [];
  if (sharedTags.length === 0) return null;

  // Check for negation patterns
  for (const pattern of negationPatterns) {
    const aHasPositive = pattern.positive.test(aLower);
    const aHasNegative = pattern.negative.test(aLower);
    const bHasPositive = pattern.positive.test(bLower);
    const bHasNegative = pattern.negative.test(bLower);

    if ((aHasPositive && bHasNegative) || (aHasNegative && bHasPositive)) {
      return {
        thoughtA: a.id,
        thoughtB: b.id,
        type: 'direct',
        severity: 'medium',
        description: `Potential contradiction in beliefs about "${sharedTags[0]}"`,
      };
    }
  }

  // Check for significantly different confidence on similar topics
  if (sharedTags.length >= 2 && Math.abs(a.confidence - b.confidence) > 0.5) {
    return {
      thoughtA: a.id,
      thoughtB: b.id,
      type: 'implicit',
      severity: 'low',
      description: `Inconsistent confidence levels on related topics`,
    };
  }

  return null;
}

/**
 * Calculate temporal coherence (consistency of beliefs over time)
 */
function calculateTemporalCoherence(thoughts: Thought[]): number {
  if (thoughts.length < 2) return 1;

  // Sort by creation time
  const sorted = [...thoughts].sort((a, b) => a.created_at - b.created_at);

  let consistencyScore = 1;
  const topicHistory = new Map<string, number[]>();

  for (const thought of sorted) {
    for (const tag of thought.tags || []) {
      if (!topicHistory.has(tag)) {
        topicHistory.set(tag, []);
      }
      topicHistory.get(tag)!.push(thought.confidence);
    }
  }

  // Check for large swings in confidence on same topics
  for (const [, confidences] of topicHistory) {
    if (confidences.length > 1) {
      for (let i = 1; i < confidences.length; i++) {
        const swing = Math.abs(confidences[i] - confidences[i - 1]);
        if (swing > 0.3) {
          consistencyScore -= 0.1;
        }
      }
    }
  }

  return Math.max(0, consistencyScore);
}

/**
 * Calculate epistemic coherence (quality and consistency of evidence)
 */
function calculateEpistemicCoherence(thoughts: Thought[]): number {
  if (thoughts.length === 0) return 1;

  let score = 0;

  for (const thought of thoughts) {
    const e = parseInt(thought.epistemic.empirical.slice(1));
    const n = parseInt(thought.epistemic.normative.slice(1));
    const m = parseInt(thought.epistemic.materiality.slice(1));

    // Higher empirical levels with appropriate confidence
    const empiricalMatch = Math.abs(thought.confidence - (e / 4)) < 0.3 ? 1 : 0.5;

    // Normative claims should acknowledge their nature
    const normativeAwareness = n > 0 || !hasNormativeContent(thought.content) ? 1 : 0.7;

    score += (empiricalMatch + normativeAwareness) / 2;
  }

  return score / thoughts.length;
}

/**
 * Check if content has normative language
 */
function hasNormativeContent(content: string): boolean {
  const normativeWords = /\bshould\b|\bmust\b|\bought\b|\bright\b|\bwrong\b|\bgood\b|\bbad\b|\bbetter\b|\bworse\b/i;
  return normativeWords.test(content);
}

/**
 * Calculate harmonic coherence (alignment with expressed values)
 */
function calculateHarmonicCoherence(thoughts: Thought[]): number {
  if (thoughts.length === 0) return 1;

  // Calculate average harmonic level
  let totalHarmonic = 0;
  for (const thought of thoughts) {
    const h = parseInt(thought.epistemic.harmonic.slice(1));
    totalHarmonic += h / 4;
  }

  return totalHarmonic / thoughts.length;
}

// ============================================================================
// WORLDVIEW ANALYSIS
// ============================================================================

/**
 * Generate worldview profile from thoughts
 */
export function generateWorldviewProfile(thoughts: Thought[]): WorldviewProfile {
  // Cluster thoughts by domain/tags
  const clusters = clusterThoughts(thoughts);

  // Extract core beliefs (high confidence, high harmonic)
  const coreBeliefs = clusters.filter(c => c.coherenceScore > 0.7);

  // Build value hierarchy from normative statements
  const valueHierarchy = extractValues(thoughts);

  // Analyze epistemic preferences
  const epistemicProfile = analyzeEpistemicProfile(thoughts);

  // Calculate harmonic center (for visualization)
  const harmonicCenter = calculateHarmonicCenter(thoughts);

  return {
    coreBeliefs,
    valueHierarchy,
    epistemicProfile,
    harmonicCenter,
  };
}

/**
 * Cluster thoughts by similarity
 */
function clusterThoughts(thoughts: Thought[]): ThoughtCluster[] {
  const tagGroups = new Map<string, Thought[]>();

  for (const thought of thoughts) {
    for (const tag of thought.tags || ['uncategorized']) {
      if (!tagGroups.has(tag)) {
        tagGroups.set(tag, []);
      }
      tagGroups.get(tag)!.push(thought);
    }
  }

  const clusters: ThoughtCluster[] = [];

  for (const [tag, groupThoughts] of tagGroups) {
    if (groupThoughts.length >= 2) {
      const avgConfidence = groupThoughts.reduce((sum, t) => sum + t.confidence, 0) / groupThoughts.length;
      const avgHarmonic = groupThoughts.reduce((sum, t) => sum + parseInt(t.epistemic.harmonic.slice(1)), 0) / groupThoughts.length / 4;

      clusters.push({
        name: tag,
        thoughts: groupThoughts.map(t => t.id),
        coherenceScore: (avgConfidence + avgHarmonic) / 2,
        tags: [tag],
      });
    }
  }

  return clusters.sort((a, b) => b.coherenceScore - a.coherenceScore);
}

/**
 * Extract value hierarchy from normative statements
 */
function extractValues(thoughts: Thought[]): string[] {
  const valuePatterns = [
    { pattern: /\bmost important\b/i, weight: 3 },
    { pattern: /\bfundamental\b/i, weight: 3 },
    { pattern: /\bcore value\b/i, weight: 3 },
    { pattern: /\bshould\b/i, weight: 1 },
    { pattern: /\bmust\b/i, weight: 2 },
  ];

  const valueMentions = new Map<string, number>();

  for (const thought of thoughts) {
    const n = parseInt(thought.epistemic.normative.slice(1));
    if (n >= 2) {
      for (const tag of thought.tags || []) {
        const current = valueMentions.get(tag) || 0;
        valueMentions.set(tag, current + n);
      }
    }
  }

  return Array.from(valueMentions.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([tag]) => tag);
}

/**
 * Analyze epistemic preferences
 */
function analyzeEpistemicProfile(thoughts: Thought[]) {
  const empiricalCounts = [0, 0, 0, 0, 0];
  const domains = new Map<string, number>();

  for (const thought of thoughts) {
    const e = parseInt(thought.epistemic.empirical.slice(1));
    empiricalCounts[e]++;

    if (thought.domain) {
      domains.set(thought.domain, (domains.get(thought.domain) || 0) + 1);
    }
  }

  const total = thoughts.length || 1;

  return {
    preferredSources: ['direct observation', 'research', 'experience'],
    confidenceDistribution: empiricalCounts.map(c => c / total),
    domainExpertise: Array.from(domains.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([d]) => d),
  };
}

/**
 * Calculate harmonic center for 3D visualization
 */
function calculateHarmonicCenter(thoughts: Thought[]): { x: number; y: number; z: number } {
  if (thoughts.length === 0) return { x: 0, y: 0, z: 0 };

  let sumE = 0, sumN = 0, sumH = 0;

  for (const thought of thoughts) {
    sumE += parseInt(thought.epistemic.empirical.slice(1)) / 4;
    sumN += parseInt(thought.epistemic.normative.slice(1)) / 3;
    sumH += parseInt(thought.epistemic.harmonic.slice(1)) / 4;
  }

  return {
    x: sumE / thoughts.length,
    y: sumN / thoughts.length,
    z: sumH / thoughts.length,
  };
}
