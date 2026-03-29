// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Bridge Zome Integration Tests
 *
 * Tests for Symthaea integration including embedding storage,
 * coherence analysis, and HDC operations.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  bridgeZome,
  lucidZome,
  LucidPlayer,
  randomEmbedding,
} from './setup';

describe('Bridge Zome (Symthaea Integration)', () => {
  let scenario: Scenario;
  let player: LucidPlayer;

  beforeAll(async () => {
    const setup = await setupLucidScenario(1);
    scenario = setup.scenario;
    player = setup.players[0];
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  describe('Embedding Storage', () => {
    it('should store an embedding for a thought', async () => {
      // Create a thought
      const thought = await lucidZome.createThought(player, {
        content: 'A thought to embed',
        thought_type: 'note',
      });

      // Store its embedding
      const embedding = randomEmbedding(16384);
      await bridgeZome.storeEmbedding(player, thought.id, embedding);

      // Retrieve and verify
      const retrieved = await bridgeZome.getEmbedding(player, thought.id);

      expect(retrieved).toBeDefined();
      expect(retrieved!.length).toBe(16384);
      // Check first few values match
      expect(retrieved![0]).toBeCloseTo(embedding[0], 5);
      expect(retrieved![1]).toBeCloseTo(embedding[1], 5);
    });

    it('should return null for non-existent embedding', async () => {
      const result = await bridgeZome.getEmbedding(player, 'non-existent-id');
      expect(result).toBeNull();
    });

    it('should handle different embedding dimensions', async () => {
      // 384D transformers.js fallback dimension
      const thought = await lucidZome.createThought(player, {
        content: 'Fallback embedding test',
        thought_type: 'note',
      });

      const embedding = randomEmbedding(384);
      await bridgeZome.storeEmbedding(player, thought.id, embedding);

      const retrieved = await bridgeZome.getEmbedding(player, thought.id);
      expect(retrieved!.length).toBe(384);
    });
  });

  describe('Coherence Analysis Storage', () => {
    it('should store coherence analysis results', async () => {
      // Create some thoughts
      const thought1 = await lucidZome.createThought(player, {
        content: 'The sky is blue',
        thought_type: 'belief',
      });

      const thought2 = await lucidZome.createThought(player, {
        content: 'Water is transparent',
        thought_type: 'belief',
      });

      const thoughtIds = [thought1.id, thought2.id];

      // Store coherence analysis
      const coherenceResult = {
        overall: 0.85,
        logical: 0.9,
        temporal: 1.0,
        epistemic: 0.8,
        harmonic: 0.7,
        phi: 0.75,
        contradictions: [],
      };

      await bridgeZome.storeCoherenceAnalysis(player, thoughtIds, coherenceResult);

      // Note: Retrieval depends on zome implementation
      // This test verifies storage doesn't throw
    });

    it('should store coherence with detected contradictions', async () => {
      const thought1 = await lucidZome.createThought(player, {
        content: 'All cats are mammals',
        thought_type: 'belief',
      });

      const thought2 = await lucidZome.createThought(player, {
        content: 'My pet Felix is not a mammal',
        thought_type: 'belief',
      });

      const thought3 = await lucidZome.createThought(player, {
        content: 'Felix is a cat',
        thought_type: 'belief',
      });

      const thoughtIds = [thought1.id, thought2.id, thought3.id];

      const coherenceResult = {
        overall: 0.3,
        logical: 0.2,
        temporal: 1.0,
        epistemic: 0.5,
        harmonic: 0.4,
        phi: 0.35,
        contradictions: [
          {
            thought_a: thought1.id,
            thought_b: thought2.id,
            description: 'If Felix is a cat and all cats are mammals, Felix must be a mammal',
            severity: 0.9,
          },
        ],
      };

      await bridgeZome.storeCoherenceAnalysis(player, thoughtIds, coherenceResult);
    });
  });

  describe('Embedding Similarity', () => {
    it('should find similar thoughts by embedding', async () => {
      // Create thoughts with related embeddings
      const thought1 = await lucidZome.createThought(player, {
        content: 'Machine learning models',
        thought_type: 'note',
      });

      const thought2 = await lucidZome.createThought(player, {
        content: 'Deep neural networks',
        thought_type: 'note',
      });

      const thought3 = await lucidZome.createThought(player, {
        content: 'Cooking recipes',
        thought_type: 'note',
      });

      // Create embeddings where thought1 and thought2 are similar
      const baseEmbedding = randomEmbedding(16384);

      // Thought 1 and 2 are similar (add small noise)
      const embedding1 = baseEmbedding.map((v) => v + (Math.random() - 0.5) * 0.1);
      const embedding2 = baseEmbedding.map((v) => v + (Math.random() - 0.5) * 0.1);

      // Thought 3 is different
      const embedding3 = randomEmbedding(16384);

      await bridgeZome.storeEmbedding(player, thought1.id, embedding1);
      await bridgeZome.storeEmbedding(player, thought2.id, embedding2);
      await bridgeZome.storeEmbedding(player, thought3.id, embedding3);

      // Verify embeddings are stored correctly
      const retrieved1 = await bridgeZome.getEmbedding(player, thought1.id);
      const retrieved2 = await bridgeZome.getEmbedding(player, thought2.id);

      expect(retrieved1).toBeDefined();
      expect(retrieved2).toBeDefined();

      // Compute similarity (cosine)
      const similarity = cosineSimilarity(retrieved1!, retrieved2!);
      expect(similarity).toBeGreaterThan(0.9); // Should be very similar
    });
  });
});

// Helper function for computing cosine similarity
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
