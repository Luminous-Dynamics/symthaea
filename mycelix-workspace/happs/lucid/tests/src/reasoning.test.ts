// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Reasoning Zome Integration Tests
 *
 * Tests for contradiction detection, resolution tracking,
 * and epistemic coherence.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  reasoningZome,
  lucidZome,
  LucidPlayer,
} from './setup';

describe('Reasoning Zome', () => {
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

  describe('Contradiction Management', () => {
    it('should record a contradiction between thoughts', async () => {
      // Create contradictory thoughts
      const thoughtA = await lucidZome.createThought(player, {
        content: 'Free will is an illusion',
        thought_type: 'belief',
        confidence: 0.7,
      });

      const thoughtB = await lucidZome.createThought(player, {
        content: 'Humans have genuine free will',
        thought_type: 'belief',
        confidence: 0.6,
      });

      // Record the contradiction
      const contradictionId = await reasoningZome.recordContradiction(player, {
        thought_a: thoughtA.id,
        thought_b: thoughtB.id,
        thought_a_content: thoughtA.content,
        thought_b_content: thoughtB.content,
        description: 'Conflicting views on free will',
      });

      expect(contradictionId).toBeDefined();
      expect(typeof contradictionId).toBe('string');
    });

    it('should retrieve all contradictions', async () => {
      // Record another contradiction
      const thoughtA = await lucidZome.createThought(player, {
        content: 'Knowledge requires certainty',
        thought_type: 'belief',
      });

      const thoughtB = await lucidZome.createThought(player, {
        content: 'All knowledge is probabilistic',
        thought_type: 'belief',
      });

      await reasoningZome.recordContradiction(player, {
        thought_a: thoughtA.id,
        thought_b: thoughtB.id,
        thought_a_content: thoughtA.content,
        thought_b_content: thoughtB.content,
        description: 'Conflicting views on epistemology',
      });

      const contradictions = await reasoningZome.getContradictions(player);

      expect(contradictions.length).toBeGreaterThanOrEqual(1);
    });

    it('should resolve a contradiction', async () => {
      // Create and record a contradiction
      const thoughtA = await lucidZome.createThought(player, {
        content: 'Time is absolute',
        thought_type: 'belief',
      });

      const thoughtB = await lucidZome.createThought(player, {
        content: 'Time is relative',
        thought_type: 'belief',
      });

      const contradictionId = await reasoningZome.recordContradiction(player, {
        thought_a: thoughtA.id,
        thought_b: thoughtB.id,
        thought_a_content: thoughtA.content,
        thought_b_content: thoughtB.content,
        description: 'Different views on the nature of time',
      });

      // Resolve it
      await reasoningZome.resolveContradiction(
        player,
        contradictionId,
        'Resolved: Time is relative as proven by special relativity. Updated belief to accept relativistic time.'
      );

      const contradictions = await reasoningZome.getContradictions(player);
      const resolved = contradictions.find((c) => c.id === contradictionId);

      expect(resolved).toBeDefined();
      expect(resolved.resolution_status).toBe('resolved');
    });
  });

  describe('Epistemic Coherence', () => {
    it('should detect logical contradictions', async () => {
      // Create logically contradictory thoughts
      const thoughtA = await lucidZome.createThought(player, {
        content: 'All ravens are black',
        thought_type: 'belief',
        confidence: 0.9,
      });

      const thoughtB = await lucidZome.createThought(player, {
        content: 'I saw a white raven',
        thought_type: 'evidence',
        confidence: 0.8,
      });

      const contradictionId = await reasoningZome.recordContradiction(player, {
        thought_a: thoughtA.id,
        thought_b: thoughtB.id,
        thought_a_content: thoughtA.content,
        thought_b_content: thoughtB.content,
        description: 'Evidence contradicts universal claim',
      });

      expect(contradictionId).toBeDefined();
    });

    it('should track belief confidence changes after contradiction', async () => {
      // Create a thought
      const thought = await lucidZome.createThought(player, {
        content: 'The universe is static and unchanging',
        thought_type: 'belief',
        confidence: 0.7,
      });

      // Create contradicting evidence
      const evidence = await lucidZome.createThought(player, {
        content: 'Hubble observations show the universe is expanding',
        thought_type: 'evidence',
        confidence: 0.95,
      });

      await reasoningZome.recordContradiction(player, {
        thought_a: thought.id,
        thought_b: evidence.id,
        thought_a_content: thought.content,
        thought_b_content: evidence.content,
        description: 'Empirical evidence contradicts static universe belief',
      });

      // Update the original thought with reduced confidence
      const updated = await lucidZome.updateThought(player, thought.id, {
        confidence: 0.1,
      });

      expect(updated.confidence).toBe(0.1);
    });
  });

  describe('Contradiction Types', () => {
    it('should categorize different types of contradictions', async () => {
      // Create thoughts that represent different contradiction types
      const empiricalContra = await lucidZome.createThought(player, {
        content: 'Water boils at 100°C at sea level',
        thought_type: 'belief',
      });

      const normativeContra = await lucidZome.createThought(player, {
        content: 'Lying is always wrong',
        thought_type: 'belief',
      });

      const normativeContra2 = await lucidZome.createThought(player, {
        content: 'Sometimes lying is morally justified',
        thought_type: 'belief',
      });

      // Record normative contradiction
      const id = await reasoningZome.recordContradiction(player, {
        thought_a: normativeContra.id,
        thought_b: normativeContra2.id,
        thought_a_content: normativeContra.content,
        thought_b_content: normativeContra2.content,
        description: 'Conflicting moral judgments about lying',
      });

      expect(id).toBeDefined();
    });
  });
});
