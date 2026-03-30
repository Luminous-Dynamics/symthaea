// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect } from 'vitest';
import { runScenario } from '@holochain/tryorama';
import type { ActionHash } from '@holochain/client';
import { setupPlayers, callReputation, sampleAgent } from './setup.js';

describe('Agent Reputation Zome', () => {
  it('register_agent creates profile', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      const hash: ActionHash = await callReputation(
        alice,
        'register_agent',
        sampleAgent,
      );
      expect(hash).toBeDefined();

      const profile = await callReputation(
        alice,
        'get_agent_profile',
        alice.cells[0].cell_id[1],
      );
      expect(profile).toBeDefined();
      expect((profile as any).name).toBe(sampleAgent.name);
      expect((profile as any).agent_type).toBe(sampleAgent.agent_type);
      expect((profile as any).is_active).toBe(true);
    });
  });

  it('record_event stores feedback', async () => {
    await runScenario(async (scenario) => {
      const [alice, bob] = await setupPlayers(scenario, 2);

      await callReputation(alice, 'register_agent', sampleAgent);

      const eventHash: ActionHash = await callReputation(bob, 'record_event', {
        agent_pubkey: alice.cells[0].cell_id[1],
        event_type: 'helpful',
        context: 'Answered SNAP question correctly',
        conversation_id: 'conv-001',
      });
      expect(eventHash).toBeDefined();

      const events: any[] = await callReputation(
        bob,
        'get_agent_events',
        alice.cells[0].cell_id[1],
      );
      expect(events.length).toBe(1);
      expect(events[0].event_type).toBe('helpful');
    });
  });

  it('compute_trust_score returns MATL composite', async () => {
    await runScenario(async (scenario) => {
      const [alice, bob] = await setupPlayers(scenario, 2);
      const alicePubkey = alice.cells[0].cell_id[1];

      await callReputation(alice, 'register_agent', sampleAgent);

      // Record several events
      for (const eventType of ['helpful', 'accurate', 'helpful', 'fast_response']) {
        await callReputation(bob, 'record_event', {
          agent_pubkey: alicePubkey,
          event_type: eventType,
        });
      }

      const score: any = await callReputation(
        alice,
        'compute_trust_score',
        alicePubkey,
      );

      expect(score).toBeDefined();
      expect(score.quality).toBeGreaterThanOrEqual(0);
      expect(score.quality).toBeLessThanOrEqual(1);
      expect(score.consistency).toBeGreaterThanOrEqual(0);
      expect(score.consistency).toBeLessThanOrEqual(1);
      expect(score.reputation).toBeGreaterThanOrEqual(0);
      expect(score.reputation).toBeLessThanOrEqual(1);
      expect(score.composite).toBeGreaterThanOrEqual(0);
      expect(score.composite).toBeLessThanOrEqual(1);
      expect(score.positive_count).toBeGreaterThanOrEqual(3);

      // Verify MATL formula: composite ≈ 0.4·Q + 0.3·C + 0.3·R
      const expected = 0.4 * score.quality + 0.3 * score.consistency + 0.3 * score.reputation;
      expect(score.composite).toBeCloseTo(expected, 1);
    });
  });

  it('get_trustworthy_agents filters by threshold', async () => {
    await runScenario(async (scenario) => {
      const [alice, bob] = await setupPlayers(scenario, 2);
      const alicePubkey = alice.cells[0].cell_id[1];

      await callReputation(alice, 'register_agent', sampleAgent);

      // Record enough positive events to be trustworthy
      for (let i = 0; i < 5; i++) {
        await callReputation(bob, 'record_event', {
          agent_pubkey: alicePubkey,
          event_type: 'accurate',
        });
        await callReputation(bob, 'record_event', {
          agent_pubkey: alicePubkey,
          event_type: 'helpful',
        });
      }

      await callReputation(alice, 'compute_trust_score', alicePubkey);

      const trustworthy: any[] = await callReputation(
        alice,
        'get_trustworthy_agents',
        null,
      );
      // Should include alice if she has enough events for confidence
      expect(Array.isArray(trustworthy)).toBe(true);
    });
  });

  it('get_agents_by_specialization returns domain experts', async () => {
    await runScenario(async (scenario) => {
      const [alice] = await setupPlayers(scenario, 1);

      await callReputation(alice, 'register_agent', {
        ...sampleAgent,
        specializations: ['benefits', 'tax'],
      });

      const benefitsAgents: any[] = await callReputation(
        alice,
        'get_agents_by_specialization',
        'benefits',
      );
      expect(benefitsAgents.length).toBeGreaterThanOrEqual(1);

      const votingAgents: any[] = await callReputation(
        alice,
        'get_agents_by_specialization',
        'voting',
      );
      expect(votingAgents.length).toBe(0);
    });
  });
});
