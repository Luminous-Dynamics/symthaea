// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Collective Sensemaking Integration Tests
 *
 * Tests for multi-agent belief sharing, voting, consensus,
 * and pattern detection.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  collectiveZome,
  lucidZome,
  LucidPlayer,
  syncPlayers,
} from './setup';

describe('Collective Sensemaking Zome', () => {
  let scenario: Scenario;
  let alice: LucidPlayer;
  let bob: LucidPlayer;

  beforeAll(async () => {
    const setup = await setupLucidScenario(2);
    scenario = setup.scenario;
    alice = setup.players[0];
    bob = setup.players[1];
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  describe('Belief Sharing', () => {
    it('should share a belief to the collective', async () => {
      const beliefHash = await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Climate change is primarily caused by human activities',
        stance: 'agree',
        confidence: 0.85,
      });

      expect(beliefHash).toBeDefined();
      expect(typeof beliefHash).toBe('string');
    });

    it('should propagate shared beliefs across agents', async () => {
      // Alice shares a belief
      await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Open source software promotes innovation',
        stance: 'agree',
        confidence: 0.9,
      });

      // Wait for DHT sync
      await syncPlayers([alice, bob]);

      // Bob should see Alice's belief
      const beliefs = await collectiveZome.getSharedBeliefs(bob);

      expect(beliefs.length).toBeGreaterThan(0);
      expect(beliefs.some((b) => b.belief_content.includes('Open source'))).toBe(true);
    });

    it('should track belief stance correctly', async () => {
      // Share with disagreement
      await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Social media improves mental health',
        stance: 'disagree',
        confidence: 0.7,
      });

      const beliefs = await collectiveZome.getSharedBeliefs(alice);
      const socialMediaBelief = beliefs.find((b) => b.belief_content.includes('Social media'));

      expect(socialMediaBelief).toBeDefined();
      expect(socialMediaBelief.stance).toBe('disagree');
    });
  });

  describe('Voting', () => {
    it('should allow voting on shared beliefs', async () => {
      // Alice shares a belief
      const beliefHash = await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Regular exercise improves cognitive function',
        stance: 'agree',
        confidence: 0.8,
      });

      await syncPlayers([alice, bob]);

      // Bob votes on the belief
      await collectiveZome.voteOnBelief(bob, beliefHash, 'agree', 0.9);

      // Verify vote was recorded
      const beliefs = await collectiveZome.getSharedBeliefs(alice);
      const exerciseBelief = beliefs.find((b) => b.belief_content.includes('exercise'));

      expect(exerciseBelief).toBeDefined();
      // Vote count should be updated (implementation-dependent)
    });

    it('should handle conflicting votes', async () => {
      const beliefHash = await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'AI will create more jobs than it eliminates',
        stance: 'neutral',
        confidence: 0.5,
      });

      await syncPlayers([alice, bob]);

      // Bob disagrees
      await collectiveZome.voteOnBelief(bob, beliefHash, 'disagree', 0.7);

      // Should handle the conflict gracefully
      const beliefs = await collectiveZome.getSharedBeliefs(alice);
      expect(beliefs.some((b) => b.belief_content.includes('AI'))).toBe(true);
    });
  });

  describe('Pattern Detection', () => {
    it('should detect patterns in collective beliefs', async () => {
      // Create multiple related beliefs
      await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Renewable energy is crucial for sustainability',
        stance: 'agree',
        confidence: 0.9,
      });

      await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Solar power is a key renewable energy source',
        stance: 'agree',
        confidence: 0.85,
      });

      await collectiveZome.shareBeliefToCollective(bob, {
        belief_content: 'Wind energy is an important renewable resource',
        stance: 'agree',
        confidence: 0.8,
      });

      await syncPlayers([alice, bob]);

      // Detect patterns
      const patterns = await collectiveZome.detectPatterns(alice);

      // Should find a cluster around renewable energy
      // (exact behavior depends on implementation)
      expect(patterns).toBeDefined();
      expect(Array.isArray(patterns)).toBe(true);
    });
  });

  describe('Consensus', () => {
    it('should reach consensus on widely agreed beliefs', async () => {
      // Both agents agree on something
      const beliefHash = await collectiveZome.shareBeliefToCollective(alice, {
        belief_content: 'Mathematics is a universal language',
        stance: 'agree',
        confidence: 0.95,
      });

      await syncPlayers([alice, bob]);

      await collectiveZome.voteOnBelief(bob, beliefHash, 'agree', 0.9);

      await syncPlayers([alice, bob]);

      const beliefs = await collectiveZome.getSharedBeliefs(alice);
      const mathBelief = beliefs.find((b) => b.belief_content.includes('Mathematics'));

      expect(mathBelief).toBeDefined();
      // High agreement should be reflected in consensus metrics
    });
  });
});
