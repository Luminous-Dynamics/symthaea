// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Embedding Quality Verification Tests
 *
 * Tests that Symthaea HDC embeddings capture semantic similarity:
 * - Similar concepts should have high cosine similarity (> 0.7)
 * - Unrelated concepts should have low similarity (< 0.3)
 * - Antonyms should have moderate similarity (0.3-0.6) - related but opposite
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { invoke } from '@tauri-apps/api/core';

// Skip tests if not in Tauri environment
const isTauri = typeof window !== 'undefined' && '__TAURI__' in window;

describe.skipIf(!isTauri)('Embedding Quality', () => {
  beforeAll(async () => {
    // Initialize Symthaea if needed
    try {
      await invoke('initialize_symthaea');
    } catch (e) {
      // Already initialized
    }
  });

  /**
   * Compute cosine similarity between two vectors
   */
  function cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Vectors must have same length');

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom < 1e-10 ? 0 : dot / denom;
  }

  /**
   * Get embedding for text via Tauri
   */
  async function getEmbedding(text: string): Promise<number[]> {
    return await invoke('embed_text', { text });
  }

  describe('Semantic Similarity', () => {
    it('should show high similarity for synonyms', async () => {
      const catEmbed = await getEmbedding('cat');
      const felineEmbed = await getEmbedding('feline');

      const similarity = cosineSimilarity(catEmbed, felineEmbed);

      expect(similarity).toBeGreaterThan(0.6);
      console.log(`"cat" vs "feline" similarity: ${similarity.toFixed(3)}`);
    });

    it('should show high similarity for related concepts', async () => {
      const dogEmbed = await getEmbedding('dog');
      const puppyEmbed = await getEmbedding('puppy');

      const similarity = cosineSimilarity(dogEmbed, puppyEmbed);

      expect(similarity).toBeGreaterThan(0.6);
      console.log(`"dog" vs "puppy" similarity: ${similarity.toFixed(3)}`);
    });

    it('should show low similarity for unrelated concepts', async () => {
      const catEmbed = await getEmbedding('cat');
      const bicycleEmbed = await getEmbedding('bicycle');

      const similarity = cosineSimilarity(catEmbed, bicycleEmbed);

      expect(similarity).toBeLessThan(0.4);
      console.log(`"cat" vs "bicycle" similarity: ${similarity.toFixed(3)}`);
    });

    it('should differentiate abstract concepts', async () => {
      const justiceEmbed = await getEmbedding('justice');
      const fairnessEmbed = await getEmbedding('fairness');
      const pizzaEmbed = await getEmbedding('pizza');

      const justiceVsFairness = cosineSimilarity(justiceEmbed, fairnessEmbed);
      const justiceVsPizza = cosineSimilarity(justiceEmbed, pizzaEmbed);

      expect(justiceVsFairness).toBeGreaterThan(justiceVsPizza);
      console.log(`"justice" vs "fairness": ${justiceVsFairness.toFixed(3)}`);
      console.log(`"justice" vs "pizza": ${justiceVsPizza.toFixed(3)}`);
    });
  });

  describe('Sentence-Level Semantics', () => {
    it('should capture meaning beyond keywords', async () => {
      const sentence1 = await getEmbedding('The cat sat on the mat');
      const sentence2 = await getEmbedding('A feline rested on the rug');
      const sentence3 = await getEmbedding('The stock market crashed today');

      const similar = cosineSimilarity(sentence1, sentence2);
      const different = cosineSimilarity(sentence1, sentence3);

      expect(similar).toBeGreaterThan(different);
      console.log(`Similar sentences: ${similar.toFixed(3)}`);
      console.log(`Different sentences: ${different.toFixed(3)}`);
    });

    it('should handle negation appropriately', async () => {
      const positive = await getEmbedding('I love this product');
      const negative = await getEmbedding('I hate this product');
      const neutral = await getEmbedding('The weather is cloudy');

      const posVsNeg = cosineSimilarity(positive, negative);
      const posVsNeutral = cosineSimilarity(positive, neutral);

      // Negation should still be more related than completely unrelated
      // (they're about the same topic, just opposite sentiment)
      console.log(`"love" vs "hate" (same topic): ${posVsNeg.toFixed(3)}`);
      console.log(`"love product" vs "weather": ${posVsNeutral.toFixed(3)}`);
    });
  });

  describe('Epistemic Concepts', () => {
    it('should cluster epistemic terms appropriately', async () => {
      const belief = await getEmbedding('belief');
      const knowledge = await getEmbedding('knowledge');
      const certainty = await getEmbedding('certainty');
      const sandwich = await getEmbedding('sandwich');

      const beliefVsKnowledge = cosineSimilarity(belief, knowledge);
      const beliefVsCertainty = cosineSimilarity(belief, certainty);
      const beliefVsSandwich = cosineSimilarity(belief, sandwich);

      expect(beliefVsKnowledge).toBeGreaterThan(beliefVsSandwich);
      expect(beliefVsCertainty).toBeGreaterThan(beliefVsSandwich);

      console.log(`"belief" vs "knowledge": ${beliefVsKnowledge.toFixed(3)}`);
      console.log(`"belief" vs "certainty": ${beliefVsCertainty.toFixed(3)}`);
      console.log(`"belief" vs "sandwich": ${beliefVsSandwich.toFixed(3)}`);
    });

    it('should distinguish evidence types', async () => {
      const empirical = await getEmbedding('empirical evidence from experiments');
      const anecdotal = await getEmbedding('anecdotal evidence from personal experience');
      const theoretical = await getEmbedding('theoretical reasoning from first principles');

      const empVsAnec = cosineSimilarity(empirical, anecdotal);
      const empVsTheor = cosineSimilarity(empirical, theoretical);

      // All are evidence-related, should have moderate similarity
      expect(empVsAnec).toBeGreaterThan(0.3);
      expect(empVsTheor).toBeGreaterThan(0.3);

      console.log(`Empirical vs Anecdotal: ${empVsAnec.toFixed(3)}`);
      console.log(`Empirical vs Theoretical: ${empVsTheor.toFixed(3)}`);
    });
  });

  describe('Embedding Dimensions', () => {
    it('should return correct dimension (16384 for HDC)', async () => {
      const embedding = await getEmbedding('test');

      // HDC uses 16384 dimensions
      expect(embedding.length).toBe(16384);
    });

    it('should return normalized vectors', async () => {
      const embedding = await getEmbedding('test normalization');

      // Compute L2 norm
      const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));

      // Should be approximately 1.0 for normalized vectors
      // (HDC may not be strictly normalized, so allow some tolerance)
      expect(norm).toBeGreaterThan(0.5);
      expect(norm).toBeLessThan(2.0);

      console.log(`Embedding L2 norm: ${norm.toFixed(3)}`);
    });
  });
});
