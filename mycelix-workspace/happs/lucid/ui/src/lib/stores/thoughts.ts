// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Thoughts Store
 */

import { writable, derived, get } from 'svelte/store';
import type { Thought, SearchThoughtsInput } from '@mycelix/lucid-client';
import { EmpiricalLevel, NormativeLevel, MaterialityLevel, HarmonicLevel, ThoughtType } from '@mycelix/lucid-client';
import { lucidClient, demoMode } from './holochain';
import { checkNewThoughtCoherence, isSymthaeaAvailable } from '../services/symthaea';
import { DEMO_THOUGHTS, generateDemoCoherence } from '$lib/demo-data';

// Thoughts state
export const thoughts = writable<Thought[]>([]);
export const selectedThought = writable<Thought | null>(null);
export const isLoading = writable(false);
export const error = writable<string | null>(null);

// ============================================================================
// COHERENCE TRACKING
// ============================================================================

export interface ThoughtCoherenceStatus {
  thoughtId: string;
  coherenceScore: number;
  phiScore: number;
  isCoherent: boolean;
  contradictions: Array<{
    conflictingThoughtId: string;
    conflictingContentPreview: string;
    severity: number;
    contradictionType: string;
    description: string;
  }>;
  suggestions: string[];
  checkedAt: number;
}

// Map of thought ID -> coherence status
export const coherenceStatus = writable<Map<string, ThoughtCoherenceStatus>>(new Map());

// Thoughts with detected contradictions
export const thoughtsWithContradictions = derived(coherenceStatus, ($status) => {
  return Array.from($status.entries())
    .filter(([_, s]) => s.contradictions.length > 0)
    .map(([id, s]) => ({ id, ...s }));
});

// Get coherence status for a specific thought
export function getCoherenceStatus(thoughtId: string): ThoughtCoherenceStatus | undefined {
  return get(coherenceStatus).get(thoughtId);
}

// Search/filter state
export const searchQuery = writable('');
export const selectedTags = writable<string[]>([]);
export const minConfidence = writable(0);

// Derived: filtered thoughts based on local search
export const filteredThoughts = derived(
  [thoughts, searchQuery, selectedTags, minConfidence],
  ([$thoughts, $query, $tags, $minConf]) => {
    return $thoughts.filter((thought) => {
      // Text search
      if ($query) {
        const q = $query.toLowerCase();
        const matchesContent = thought.content.toLowerCase().includes(q);
        const matchesTags = thought.tags.some((t) => t.toLowerCase().includes(q));
        if (!matchesContent && !matchesTags) return false;
      }

      // Tag filter
      if ($tags.length > 0) {
        const hasAllTags = $tags.every((tag) => thought.tags.includes(tag));
        if (!hasAllTags) return false;
      }

      // Confidence filter
      if (thought.confidence < $minConf) return false;

      return true;
    });
  }
);

// All unique tags from thoughts
export const allTags = derived(thoughts, ($thoughts) => {
  const tagSet = new Set<string>();
  $thoughts.forEach((t) => t.tags.forEach((tag) => tagSet.add(tag)));
  return Array.from(tagSet).sort();
});

/**
 * Load all thoughts for current user
 */
export async function loadThoughts(): Promise<void> {
  // Demo mode: load pre-populated demo thoughts + coherence data
  if (get(demoMode)) {
    thoughts.set([...DEMO_THOUGHTS]);
    coherenceStatus.set(generateDemoCoherence());
    return;
  }

  let client: any;
  lucidClient.subscribe((c) => (client = c))();

  if (!client) {
    error.set('Not connected to Holochain');
    return;
  }

  isLoading.set(true);
  error.set(null);

  try {
    const result = await client.getMyThoughts();
    thoughts.set(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to load thoughts';
    error.set(message);
    console.error('Failed to load thoughts:', err);
  } finally {
    isLoading.set(false);
  }
}

/**
 * Search thoughts with server-side filtering
 */
export async function searchThoughts(input: SearchThoughtsInput): Promise<void> {
  // Demo mode: use local filtering (the filteredThoughts derived store handles this)
  if (get(demoMode)) {
    if (input.content_contains) {
      searchQuery.set(input.content_contains);
    }
    return;
  }

  let client: any;
  lucidClient.subscribe((c) => (client = c))();

  if (!client) {
    error.set('Not connected to Holochain');
    return;
  }

  isLoading.set(true);
  error.set(null);

  try {
    const result = await client.searchThoughts(input);
    thoughts.set(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Search failed';
    error.set(message);
    console.error('Search failed:', err);
  } finally {
    isLoading.set(false);
  }
}

/**
 * Create a new thought
 */
export async function createThought(input: Parameters<typeof import('@mycelix/lucid-client').LucidClient.prototype.createThought>[0]): Promise<Thought | null> {
  // Demo mode: create thought locally without Holochain
  if (get(demoMode)) {
    const now = Date.now() * 1000;
    const thought: Thought = {
      id: `demo-thought-${Date.now()}`,
      content: input.content,
      thought_type: input.thought_type ?? ThoughtType.Note,
      epistemic: input.epistemic ?? { empirical: EmpiricalLevel.E0, normative: NormativeLevel.N0, materiality: MaterialityLevel.M0, harmonic: HarmonicLevel.H1 },
      confidence: input.confidence ?? 0.5,
      tags: input.tags || [],
      domain: input.domain || 'general',
      related_thoughts: input.related_thoughts || [],
      source_hashes: [],
      parent_thought: input.parent_thought || null,
      created_at: now,
      updated_at: now,
      version: 1,
    };
    thoughts.update((list) => [thought, ...list]);

    // Run local coherence check against demo thoughts
    demoCoherenceCheck(thought);

    return thought;
  }

  let client: any;
  lucidClient.subscribe((c) => (client = c))();

  if (!client) {
    error.set('Not connected to Holochain');
    return null;
  }

  isLoading.set(true);
  error.set(null);

  try {
    const thought = await client.createThought(input);
    thoughts.update((list) => [thought, ...list]);

    // Check coherence with existing thoughts (async, non-blocking)
    checkCoherenceForNewThought(thought).catch((err) => {
      console.warn('Coherence check failed:', err);
    });

    return thought;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to create thought';
    error.set(message);
    console.error('Failed to create thought:', err);
    return null;
  } finally {
    isLoading.set(false);
  }
}

/**
 * Check coherence for a newly created thought against existing thoughts
 */
async function checkCoherenceForNewThought(newThought: Thought): Promise<void> {
  // Check if Symthaea is available
  const symthaeaAvailable = await isSymthaeaAvailable();
  if (!symthaeaAvailable) {
    return;
  }

  // Get existing thoughts
  const existingThoughts = get(thoughts).filter((t) => t.id !== newThought.id);
  if (existingThoughts.length === 0) {
    return;
  }

  // Prepare data for coherence check
  const existingData = existingThoughts.slice(0, 50).map((t) => ({
    id: t.id,
    content: t.content,
    embedding: (t as any).embedding, // May have been cached
  }));

  // Run coherence check via Symthaea
  const result = await checkNewThoughtCoherence(
    { id: newThought.id, content: newThought.content },
    existingData
  );

  if (result) {
    // Store the coherence status
    const status: ThoughtCoherenceStatus = {
      thoughtId: newThought.id,
      coherenceScore: result.coherence_score,
      phiScore: result.phi_score,
      isCoherent: result.is_coherent,
      contradictions: result.contradictions.map((c) => ({
        conflictingThoughtId: c.conflicting_thought_id,
        conflictingContentPreview: c.conflicting_content_preview,
        severity: c.severity,
        contradictionType: c.contradiction_type,
        description: c.description,
      })),
      suggestions: result.suggestions,
      checkedAt: Date.now(),
    };

    coherenceStatus.update((map) => {
      map.set(newThought.id, status);
      return new Map(map);
    });
  }
}

/**
 * Simple demo-mode coherence check using keyword overlap
 */
function demoCoherenceCheck(newThought: Thought): void {
  const existing = get(thoughts).filter((t) => t.id !== newThought.id);
  const newWords = new Set(newThought.content.toLowerCase().split(/\s+/));

  // Assign a basic coherence score based on tag/domain overlap
  let bestOverlap = 0;
  let contradictionTarget: Thought | null = null;

  for (const t of existing) {
    const tWords = new Set(t.content.toLowerCase().split(/\s+/));
    const overlap = [...newWords].filter((w) => tWords.has(w) && w.length > 4).length;
    if (overlap > bestOverlap) bestOverlap = overlap;

    // Simple contradiction heuristic: high word overlap but opposing normative levels
    const normDiff = Math.abs(
      (newThought.epistemic?.normative ?? 0) - (t.epistemic?.normative ?? 0)
    );
    if (overlap > 5 && normDiff >= 2 && !contradictionTarget) {
      contradictionTarget = t;
    }
  }

  const phi = 0.3 + Math.random() * 0.5;
  const status: ThoughtCoherenceStatus = {
    thoughtId: newThought.id,
    coherenceScore: contradictionTarget ? 0.35 : 0.7 + Math.random() * 0.25,
    phiScore: phi,
    isCoherent: !contradictionTarget,
    contradictions: contradictionTarget
      ? [
          {
            conflictingThoughtId: contradictionTarget.id,
            conflictingContentPreview: contradictionTarget.content.slice(0, 80) + '...',
            severity: 0.6,
            contradictionType: 'normative_divergence',
            description: 'Divergent normative framing on a closely related topic.',
          },
        ]
      : [],
    suggestions: contradictionTarget
      ? ['Potential tension detected with an existing thought. Review for coherence.']
      : [],
    checkedAt: Date.now(),
  };

  coherenceStatus.update((map) => {
    map.set(newThought.id, status);
    return new Map(map);
  });
}

/**
 * Delete a thought
 */
export async function deleteThought(thoughtId: string): Promise<boolean> {
  // Demo mode: delete locally
  if (get(demoMode)) {
    thoughts.update((list) => list.filter((t) => t.id !== thoughtId));
    selectedThought.update((t) => (t?.id === thoughtId ? null : t));
    coherenceStatus.update((map) => {
      map.delete(thoughtId);
      return new Map(map);
    });
    return true;
  }

  let client: any;
  lucidClient.subscribe((c) => (client = c))();

  if (!client) {
    error.set('Not connected to Holochain');
    return false;
  }

  try {
    await client.deleteThought(thoughtId);
    thoughts.update((list) => list.filter((t) => t.id !== thoughtId));
    selectedThought.update((t) => (t?.id === thoughtId ? null : t));
    return true;
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Failed to delete thought';
    error.set(message);
    console.error('Failed to delete thought:', err);
    return false;
  }
}
