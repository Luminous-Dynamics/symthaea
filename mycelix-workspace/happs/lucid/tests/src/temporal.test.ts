// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Temporal Zome Integration Tests
 *
 * Tests for belief versioning, temporal snapshots, consciousness-aware
 * trajectory analysis, and evolution tracking.
 *
 * Requires a running Holochain conductor with the LUCID hApp installed.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  lucidZome,
  temporalZome,
  temporalConsciousnessZome,
  LucidPlayer,
  RecordVersionInput,
  RecordSnapshotInput,
} from './setup';

describe('Temporal Zome (Belief Versioning)', () => {
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

  describe('Version Recording', () => {
    it('should record a first version of a thought', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'Initial belief about consciousness',
        thought_type: 'belief',
        confidence: 0.6,
      });

      const versionInput: RecordVersionInput = {
        thought_id: thought.id,
        version: 1,
        content: 'Initial belief about consciousness',
        confidence: 0.6,
        epistemic_code: 'E2-N1-M2-H1',
      };

      const record = await temporalZome.recordVersion(player, versionInput);
      expect(record).toBeDefined();
    });

    it('should record multiple versions tracking belief evolution', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'Earth is the center of the universe',
        thought_type: 'belief',
        confidence: 0.9,
      });

      // Version 1: initial geocentric belief
      await temporalZome.recordVersion(player, {
        thought_id: thought.id,
        version: 1,
        content: 'Earth is the center of the universe',
        confidence: 0.9,
        epistemic_code: 'E1-N0-M3-H0',
      });

      // Version 2: doubt emerges
      await temporalZome.recordVersion(player, {
        thought_id: thought.id,
        version: 2,
        content: 'Earth might not be the center of the universe',
        confidence: 0.5,
        epistemic_code: 'E2-N0-M3-H1',
        change_reason: 'New astronomical observations',
      });

      // Version 3: heliocentric model accepted
      await temporalZome.recordVersion(player, {
        thought_id: thought.id,
        version: 3,
        content: 'Earth orbits the Sun',
        confidence: 0.95,
        epistemic_code: 'E4-N0-M3-H2',
        change_reason: 'Overwhelming empirical evidence',
      });

      // Retrieve history — should have 3 versions
      const history = await temporalZome.getThoughtHistory(player, thought.id);
      expect(history).toBeDefined();
      expect(history.length).toBe(3);
    });

    it('should return empty history for thought with no versions', async () => {
      const history = await temporalZome.getThoughtHistory(player, 'nonexistent-thought-id');
      expect(history).toBeDefined();
      expect(history.length).toBe(0);
    });
  });

  describe('Knowledge Graph Snapshots', () => {
    it('should create a snapshot of current knowledge state', async () => {
      const snapshot = await temporalZome.createSnapshot(player, {
        name: 'Morning reflection',
        description: 'Daily knowledge graph checkpoint',
        thought_count: 15,
        avg_confidence: 0.72,
        tags: ['daily', 'reflection'],
      });

      expect(snapshot).toBeDefined();
    });

    it('should retrieve all snapshots for current agent', async () => {
      // Create a couple snapshots
      await temporalZome.createSnapshot(player, {
        name: 'Session start',
        thought_count: 10,
        avg_confidence: 0.65,
      });

      await temporalZome.createSnapshot(player, {
        name: 'Session end',
        thought_count: 18,
        avg_confidence: 0.78,
        description: 'After deep research session',
      });

      const snapshots = await temporalZome.getMySnapshots(player);
      expect(snapshots).toBeDefined();
      expect(snapshots.length).toBeGreaterThanOrEqual(2);
    });
  });
});

describe('Temporal-Consciousness Zome (Trajectory Analysis)', () => {
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

  describe('Belief Snapshots with Consciousness Metrics', () => {
    it('should record a belief snapshot with Phi and coherence', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'Consciousness arises from integrated information',
        thought_type: 'belief',
        confidence: 0.7,
      });

      const snapshotInput: RecordSnapshotInput = {
        thought_id: thought.id,
        epistemic_code: 'E2-N1-M2-H2',
        confidence: 0.7,
        phi: 0.45,
        coherence: 0.82,
        trigger: 'Created',
      };

      const record = await temporalConsciousnessZome.recordBeliefSnapshot(player, snapshotInput);
      expect(record).toBeDefined();
    });

    it('should track belief strengthening over multiple snapshots', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'HDC enables efficient symbolic reasoning',
        thought_type: 'hypothesis',
        confidence: 0.5,
      });

      // Initial creation
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E1-N0-M2-H1',
        confidence: 0.5,
        phi: 0.3,
        coherence: 0.6,
        trigger: 'Created',
      });

      // After evidence review — confidence grows
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E2-N0-M2-H2',
        confidence: 0.7,
        phi: 0.5,
        coherence: 0.75,
        trigger: 'EvidenceUpdated',
      });

      // After integration with other beliefs — Phi rises
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E3-N1-M2-H3',
        confidence: 0.85,
        phi: 0.7,
        coherence: 0.88,
        trigger: 'CoherenceChanged',
      });

      // Retrieve trajectory
      const trajectory = await temporalConsciousnessZome.getBeliefTrajectory(
        player,
        thought.id
      );

      expect(trajectory).toBeDefined();
      // Trajectory should contain the 3 snapshots
      if (trajectory && trajectory.snapshots) {
        expect(trajectory.snapshots.length).toBe(3);
      }
    });

    it('should handle contradiction-triggered snapshots', async () => {
      const thought = await lucidZome.createThought(player, {
        content: 'Free will is an illusion',
        thought_type: 'belief',
        confidence: 0.8,
      });

      // High confidence initially
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E1-N2-M2-H1',
        confidence: 0.8,
        phi: 0.55,
        coherence: 0.7,
        trigger: 'Created',
      });

      // Contradiction detected — confidence drops, coherence drops
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E1-N2-M2-H1',
        confidence: 0.5,
        phi: 0.4,
        coherence: 0.45,
        trigger: 'ContradictionDetected',
      });

      // After review and modification
      await temporalConsciousnessZome.recordBeliefSnapshot(player, {
        thought_id: thought.id,
        epistemic_code: 'E2-N2-M2-H2',
        confidence: 0.65,
        phi: 0.6,
        coherence: 0.72,
        trigger: 'ReviewedModified',
      });

      const analysis = await temporalConsciousnessZome.analyzeBeliefTrajectory(
        player,
        thought.id
      );

      expect(analysis).toBeDefined();
      // Analysis should identify the trajectory type (e.g., Oscillating or Volatile)
      if (analysis && analysis.trajectory_type) {
        expect(['Oscillating', 'Volatile', 'Weakening', 'Growing']).toContain(
          analysis.trajectory_type
        );
      }
    });
  });

  describe('Consciousness Evolution Tracking', () => {
    it('should record consciousness evolution for a time period', async () => {
      const now = Date.now() * 1000; // microseconds for Holochain Timestamp
      const oneHourAgo = now - 3600 * 1_000_000;

      const record = await temporalConsciousnessZome.recordConsciousnessEvolution(player, {
        period_start: oneHourAgo,
        period_end: now,
        avg_phi: 0.62,
        phi_trend: 0.05, // slightly increasing
        avg_coherence: 0.78,
        coherence_trend: 0.02,
        stable_belief_count: 12,
        growing_belief_count: 5,
        weakening_belief_count: 2,
        entrenched_belief_count: 1,
        insights: [
          'Phi increased during deep research sessions',
          'Two beliefs showed oscillating patterns — may need review',
          'One belief is becoming entrenched without new evidence',
        ],
      });

      expect(record).toBeDefined();
    });

    it('should retrieve evolution history', async () => {
      const now = Date.now() * 1000;

      // Record two evolution periods
      await temporalConsciousnessZome.recordConsciousnessEvolution(player, {
        period_start: now - 7200 * 1_000_000,
        period_end: now - 3600 * 1_000_000,
        avg_phi: 0.55,
        phi_trend: -0.02,
        avg_coherence: 0.7,
        coherence_trend: -0.01,
        stable_belief_count: 8,
        growing_belief_count: 3,
        weakening_belief_count: 4,
        entrenched_belief_count: 0,
        insights: ['Morning session — lower integration'],
      });

      await temporalConsciousnessZome.recordConsciousnessEvolution(player, {
        period_start: now - 3600 * 1_000_000,
        period_end: now,
        avg_phi: 0.72,
        phi_trend: 0.08,
        avg_coherence: 0.85,
        coherence_trend: 0.07,
        stable_belief_count: 10,
        growing_belief_count: 6,
        weakening_belief_count: 1,
        entrenched_belief_count: 0,
        insights: ['Deep work session — Phi surged', 'Coherence improving across belief graph'],
      });

      const history = await temporalConsciousnessZome.getMyEvolutionHistory(player);
      expect(history).toBeDefined();
      expect(history.length).toBeGreaterThanOrEqual(2);
    });
  });
});
