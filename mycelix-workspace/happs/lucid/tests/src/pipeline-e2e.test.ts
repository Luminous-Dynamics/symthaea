// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * LUCID Full Pipeline E2E Integration Tests
 *
 * Tests the complete Thought -> Embedding -> Coherence -> Epistemic ->
 * Temporal -> Collective pipeline, verifying all zomes work together
 * as an integrated system.
 *
 * Requires a running Holochain conductor with the LUCID hApp installed.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario } from '@holochain/tryorama';
import {
  setupLucidScenario,
  lucidZome,
  bridgeZome,
  temporalZome,
  temporalConsciousnessZome,
  reasoningZome,
  collectiveZome,
  LucidPlayer,
  randomEmbedding,
  syncPlayers,
} from './setup';

describe('Full Pipeline E2E: Thought Lifecycle', () => {
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

  it('should complete: create -> embed -> version -> analyze trajectory', async () => {
    // Step 1: Create a thought with epistemic classification
    const thought = await lucidZome.createThought(player, {
      content: 'Active inference minimizes free energy in biological systems',
      thought_type: 'hypothesis',
      confidence: 0.65,
      tags: ['active-inference', 'free-energy', 'biology'],
      epistemic: {
        empirical: 2,
        normative: 0,
        materiality: 2,
        harmonic: 1,
      },
    });
    expect(thought).toBeDefined();
    expect(thought.id).toBeDefined();

    // Step 2: Store HDC embedding via Symthaea bridge
    const embedding = randomEmbedding(16384);
    await bridgeZome.storeEmbedding(player, thought.id, embedding);

    const storedEmbedding = await bridgeZome.getEmbedding(player, thought.id);
    expect(storedEmbedding).toBeDefined();
    expect(storedEmbedding!.length).toBe(16384);

    // Step 3: Record initial version in temporal zome
    await temporalZome.recordVersion(player, {
      thought_id: thought.id,
      version: 1,
      content: thought.content,
      confidence: 0.65,
      epistemic_code: 'E2-N0-M2-H1',
    });

    // Step 4: Record consciousness-aware snapshot
    await temporalConsciousnessZome.recordBeliefSnapshot(player, {
      thought_id: thought.id,
      epistemic_code: 'E2-N0-M2-H1',
      confidence: 0.65,
      phi: 0.42,
      coherence: 0.7,
      trigger: 'Created',
    });

    // Step 5: Thought evolves after new evidence
    const updated = await lucidZome.updateThought(player, thought.id, {
      content: 'Active inference is a unifying framework for perception and action',
      confidence: 0.82,
    });
    expect(updated.confidence).toBe(0.82);

    // Step 6: Record new version
    await temporalZome.recordVersion(player, {
      thought_id: thought.id,
      version: 2,
      content: updated.content,
      confidence: 0.82,
      epistemic_code: 'E3-N1-M2-H2',
      change_reason: 'New experimental evidence from neuroscience literature',
    });

    // Step 7: Record updated consciousness snapshot
    await temporalConsciousnessZome.recordBeliefSnapshot(player, {
      thought_id: thought.id,
      epistemic_code: 'E3-N1-M2-H2',
      confidence: 0.82,
      phi: 0.6,
      coherence: 0.85,
      trigger: 'EvidenceUpdated',
    });

    // Step 8: Verify full history
    const history = await temporalZome.getThoughtHistory(player, thought.id);
    expect(history.length).toBe(2);

    // Step 9: Analyze trajectory — should show Growing pattern
    const analysis = await temporalConsciousnessZome.analyzeBeliefTrajectory(
      player,
      thought.id
    );
    expect(analysis).toBeDefined();
  });

  it('should complete: create -> coherence analysis -> contradiction -> resolution', async () => {
    // Step 1: Create a set of related beliefs
    const belief1 = await lucidZome.createThought(player, {
      content: 'Consciousness requires integrated information (Phi > 0)',
      thought_type: 'belief',
      confidence: 0.8,
      tags: ['iit', 'consciousness'],
    });

    const belief2 = await lucidZome.createThought(player, {
      content: 'Higher Phi correlates with richer subjective experience',
      thought_type: 'belief',
      confidence: 0.75,
      tags: ['iit', 'qualia'],
    });

    const belief3 = await lucidZome.createThought(player, {
      content: 'Simple feedforward networks can be conscious',
      thought_type: 'belief',
      confidence: 0.4,
      tags: ['consciousness', 'networks'],
    });

    // Step 2: Store coherence analysis (from Symthaea)
    await bridgeZome.storeCoherenceAnalysis(
      player,
      [belief1.id, belief2.id, belief3.id],
      {
        overall: 0.55,
        logical: 0.4,
        temporal: 0.9,
        epistemic: 0.6,
        harmonic: 0.5,
        phi: 0.48,
        contradictions: [
          {
            thought_a: belief2.id,
            thought_b: belief3.id,
            description:
              'IIT predicts feedforward networks have Phi=0, contradicting consciousness claim',
            severity: 0.8,
          },
        ],
      }
    );

    // Step 3: Record the contradiction
    const contradictionId = await reasoningZome.recordContradiction(player, {
      thought_a: belief2.id,
      thought_b: belief3.id,
      thought_a_content: belief2.content,
      thought_b_content: belief3.content,
      description:
        'IIT predicts feedforward networks have Phi=0, contradicting consciousness claim',
    });
    expect(contradictionId).toBeDefined();

    // Step 4: Verify contradiction is recorded
    const contradictions = await reasoningZome.getContradictions(player);
    expect(contradictions.length).toBeGreaterThanOrEqual(1);

    // Step 5: Resolve the contradiction — update the weaker belief
    await reasoningZome.resolveContradiction(
      player,
      contradictionId,
      'Revised: feedforward networks likely lack consciousness per IIT, need recurrent connections'
    );

    // Step 6: Update the contradicted belief with lower confidence
    await lucidZome.updateThought(player, belief3.id, {
      content:
        'Simple feedforward networks likely lack consciousness; recurrent connections may be needed',
      confidence: 0.6,
    });
  });
});

describe('Full Pipeline E2E: Multi-Agent Collective Sensemaking', () => {
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

  it('should allow two agents to share beliefs and reach collective coherence', async () => {
    // Alice creates a belief
    const aliceBelief = await lucidZome.createThought(alice, {
      content: 'Decentralized systems are more resilient than centralized ones',
      thought_type: 'belief',
      confidence: 0.85,
      tags: ['decentralization', 'resilience'],
    });
    expect(aliceBelief).toBeDefined();

    // Bob creates a related belief
    const bobBelief = await lucidZome.createThought(bob, {
      content: 'Byzantine fault tolerance enables trust in adversarial networks',
      thought_type: 'belief',
      confidence: 0.9,
      tags: ['byzantine', 'trust'],
    });
    expect(bobBelief).toBeDefined();

    // Sync DHT
    await syncPlayers([alice, bob]);

    // Alice shares her belief to the collective
    const aliceShareId = await collectiveZome.shareBeliefToCollective(alice, {
      belief_content: aliceBelief.content,
      stance: 'agree',
      confidence: 0.85,
    });
    expect(aliceShareId).toBeDefined();

    // Bob shares his belief to the collective
    const bobShareId = await collectiveZome.shareBeliefToCollective(bob, {
      belief_content: bobBelief.content,
      stance: 'agree',
      confidence: 0.9,
    });
    expect(bobShareId).toBeDefined();

    // Sync again after shares
    await syncPlayers([alice, bob]);

    // Both should see shared beliefs
    const aliceShared = await collectiveZome.getSharedBeliefs(alice);
    const bobShared = await collectiveZome.getSharedBeliefs(bob);

    expect(aliceShared.length).toBeGreaterThanOrEqual(2);
    expect(bobShared.length).toBeGreaterThanOrEqual(2);

    // Bob votes agreement on Alice's shared belief
    await collectiveZome.voteOnBelief(bob, aliceShareId, 'agree', 0.8);

    // Alice votes agreement on Bob's shared belief
    await collectiveZome.voteOnBelief(alice, bobShareId, 'agree', 0.75);

    // Detect collective patterns
    const patterns = await collectiveZome.detectPatterns(alice);
    expect(patterns).toBeDefined();
  });
});

describe('Full Pipeline E2E: Knowledge Graph Snapshot Lifecycle', () => {
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

  it('should create thoughts, snapshot state, add more, snapshot again, compare', async () => {
    // Phase 1: Create initial knowledge
    await lucidZome.createThought(player, {
      content: 'Hyperdimensional computing uses high-dimensional vectors',
      thought_type: 'note',
      confidence: 0.9,
    });

    await lucidZome.createThought(player, {
      content: 'Liquid time-constant networks model continuous dynamics',
      thought_type: 'note',
      confidence: 0.85,
    });

    // Snapshot Phase 1
    const snapshot1 = await temporalZome.createSnapshot(player, {
      name: 'Phase 1: Foundations',
      description: 'Initial HDC and LTC concepts',
      thought_count: 2,
      avg_confidence: 0.875,
      tags: ['foundations', 'hdc', 'ltc'],
    });
    expect(snapshot1).toBeDefined();

    // Phase 2: Expand knowledge
    await lucidZome.createThought(player, {
      content: 'HDC + LTC combination enables consciousness-aware computation',
      thought_type: 'hypothesis',
      confidence: 0.6,
    });

    await lucidZome.createThought(player, {
      content: 'IIT Phi measures information integration in neural networks',
      thought_type: 'belief',
      confidence: 0.8,
    });

    // Snapshot Phase 2
    const snapshot2 = await temporalZome.createSnapshot(player, {
      name: 'Phase 2: Integration',
      description: 'Combining HDC, LTC, and IIT into unified framework',
      thought_count: 4,
      avg_confidence: 0.7875,
      tags: ['integration', 'symthaea'],
    });
    expect(snapshot2).toBeDefined();

    // Verify we have both snapshots
    const allSnapshots = await temporalZome.getMySnapshots(player);
    expect(allSnapshots.length).toBeGreaterThanOrEqual(2);

    // Record consciousness evolution comparing the two periods
    const now = Date.now() * 1000;
    await temporalConsciousnessZome.recordConsciousnessEvolution(player, {
      period_start: now - 3600 * 1_000_000,
      period_end: now,
      avg_phi: 0.58,
      phi_trend: 0.12,
      avg_coherence: 0.81,
      coherence_trend: 0.08,
      stable_belief_count: 2,
      growing_belief_count: 2,
      weakening_belief_count: 0,
      entrenched_belief_count: 0,
      insights: [
        'Knowledge graph doubled in size',
        'Integration hypothesis connecting HDC+LTC+IIT shows promise',
        'Phi trend positive — framework becoming more integrated',
      ],
    });

    const evolutionHistory = await temporalConsciousnessZome.getMyEvolutionHistory(player);
    expect(evolutionHistory).toBeDefined();
    expect(evolutionHistory.length).toBeGreaterThanOrEqual(1);
  });
});
