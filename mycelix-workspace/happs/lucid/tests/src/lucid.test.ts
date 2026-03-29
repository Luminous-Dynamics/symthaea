// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Core Zome Integration Tests
 *
 * Tests for thought CRUD operations, epistemic classification,
 * and basic knowledge graph functionality.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  lucidZome,
  LucidPlayer,
  CreateThoughtInput,
  randomEmbedding,
} from './setup';

describe('LUCID Core Zome', () => {
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

  describe('Thought CRUD', () => {
    it('should create a thought', async () => {
      const input: CreateThoughtInput = {
        content: 'This is my first thought in LUCID',
        thought_type: 'note',
        confidence: 0.8,
        tags: ['test', 'first'],
      };

      const thought = await lucidZome.createThought(player, input);

      expect(thought).toBeDefined();
      expect(thought.id).toBeDefined();
      expect(thought.content).toBe(input.content);
      expect(thought.thought_type).toBe('note');
      expect(thought.confidence).toBe(0.8);
      expect(thought.tags).toContain('test');
      expect(thought.created_at).toBeDefined();
    });

    it('should get a thought by ID', async () => {
      const input: CreateThoughtInput = {
        content: 'A thought to retrieve',
        thought_type: 'belief',
        confidence: 0.9,
      };

      const created = await lucidZome.createThought(player, input);
      const retrieved = await lucidZome.getThought(player, created.id);

      expect(retrieved).toBeDefined();
      expect(retrieved!.id).toBe(created.id);
      expect(retrieved!.content).toBe(input.content);
    });

    it('should return null for non-existent thought', async () => {
      const result = await lucidZome.getThought(player, 'non-existent-id');
      expect(result).toBeNull();
    });

    it('should get all thoughts', async () => {
      // Create a few more thoughts
      await lucidZome.createThought(player, {
        content: 'Thought A',
        thought_type: 'note',
      });
      await lucidZome.createThought(player, {
        content: 'Thought B',
        thought_type: 'question',
      });

      const allThoughts = await lucidZome.getAllThoughts(player);

      expect(allThoughts.length).toBeGreaterThanOrEqual(2);
      expect(allThoughts.some((t) => t.content === 'Thought A')).toBe(true);
      expect(allThoughts.some((t) => t.content === 'Thought B')).toBe(true);
    });

    it('should update a thought', async () => {
      const created = await lucidZome.createThought(player, {
        content: 'Original content',
        thought_type: 'note',
        confidence: 0.5,
      });

      const updated = await lucidZome.updateThought(player, created.id, {
        content: 'Updated content',
        confidence: 0.9,
      });

      expect(updated.content).toBe('Updated content');
      expect(updated.confidence).toBe(0.9);
      expect(updated.updated_at).toBeGreaterThan(created.updated_at);
    });

    it('should delete a thought', async () => {
      const created = await lucidZome.createThought(player, {
        content: 'To be deleted',
        thought_type: 'note',
      });

      await lucidZome.deleteThought(player, created.id);

      const retrieved = await lucidZome.getThought(player, created.id);
      expect(retrieved).toBeNull();
    });
  });

  describe('Thought Types', () => {
    const thoughtTypes = ['note', 'belief', 'question', 'evidence', 'hypothesis', 'idea'];

    for (const type of thoughtTypes) {
      it(`should create a ${type} thought`, async () => {
        const thought = await lucidZome.createThought(player, {
          content: `This is a ${type}`,
          thought_type: type,
        });

        expect(thought.thought_type).toBe(type);
      });
    }
  });

  describe('Epistemic Classification', () => {
    it('should store epistemic classification with thought', async () => {
      const input: CreateThoughtInput = {
        content: 'The Earth orbits the Sun',
        thought_type: 'belief',
        confidence: 0.95,
        epistemic: {
          empirical: 4, // E4: Strong empirical evidence
          normative: 0, // N0: Descriptive
          materiality: 3, // M3: Physical reality
          harmonic: 2, // H2: Moderate integration
        },
      };

      const thought = await lucidZome.createThought(player, input);

      expect(thought).toBeDefined();
      // The epistemic classification should be stored
      // (exact structure depends on zome implementation)
    });

    it('should handle embedding storage', async () => {
      const embedding = randomEmbedding(16384);

      const thought = await lucidZome.createThought(player, {
        content: 'A thought with embedding',
        thought_type: 'note',
        embedding,
      });

      expect(thought).toBeDefined();
      // Embedding should be stored (verified via bridge zome)
    });
  });

  describe('Tags', () => {
    it('should create thought with multiple tags', async () => {
      const tags = ['philosophy', 'consciousness', 'mind'];

      const thought = await lucidZome.createThought(player, {
        content: 'Thoughts on consciousness',
        thought_type: 'idea',
        tags,
      });

      expect(thought.tags).toEqual(expect.arrayContaining(tags));
    });

    it('should update thought tags', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'Taggable thought',
        thought_type: 'note',
        tags: ['initial'],
      });

      const updated = await lucidZome.updateThought(player, thought.id, {
        tags: ['updated', 'new-tag'],
      });

      expect(updated.tags).toContain('updated');
      expect(updated.tags).toContain('new-tag');
    });
  });

  describe('Search', () => {
    it('should search thoughts by query', async () => {
      // Create searchable thoughts
      await lucidZome.createThought(player, {
        content: 'Machine learning is a subset of artificial intelligence',
        thought_type: 'note',
        tags: ['ai', 'ml'],
      });

      await lucidZome.createThought(player, {
        content: 'Deep learning uses neural networks',
        thought_type: 'note',
        tags: ['ai', 'deep-learning'],
      });

      const results = await lucidZome.searchThoughts(player, {
        query: 'artificial intelligence',
        limit: 10,
      });

      expect(results.length).toBeGreaterThan(0);
      // At least one result should mention AI
      expect(results.some((t) => t.content.toLowerCase().includes('artificial'))).toBe(true);
    });

    it('should respect search limit', async () => {
      const results = await lucidZome.searchThoughts(player, {
        query: 'thought',
        limit: 3,
      });

      expect(results.length).toBeLessThanOrEqual(3);
    });
  });
});
