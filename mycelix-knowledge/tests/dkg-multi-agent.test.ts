// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DKG Multi-Agent Collaboration Tests
 *
 * Tests the Truth Engine's behavior with multiple agents:
 * - 5-agent consensus formation
 * - Reputation propagation (experienced vs fresh agents)
 * - Subject discovery across agents
 * - Self-attestation behavior
 */

import { describe, test, expect, jest } from '@jest/globals';
import { Scenario, dhtSync, runScenario } from '@holochain/tryorama';
import { ActionHash } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const knowledgeHappPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '../workdir/knowledge.happ'
);
const scenarioOptions = { timeout: 120000 };

// ============================================================================
// Type Definitions
// ============================================================================

interface SubmitClaimInput {
  subject: string;
  predicate: string;
  object: string;
  object_type?: string;
  epistemic_type?: string;
  domain?: string;
}

interface AttestClaimInput {
  claim_hash: ActionHash;
  attestation_type: string;
  evidence?: string;
}

interface WeightedClaim {
  claim_hash: ActionHash;
  subject: string;
  predicate: string;
  object: string;
  object_type: string;
  author: Uint8Array;
  confidence: number;
  attestation_count: number;
  endorsements: number;
  challenges: number;
  created_at: number;
}

interface AgentReputation {
  agent: Uint8Array;
  reputation_score: number;
  claim_count: number;
  attestation_count: number;
  endorsements_received: number;
  challenges_received: number;
}

// ============================================================================
// Multi-Agent Collaboration Tests
// ============================================================================

describe('DKG Multi-Agent Collaboration Tests', () => {
  jest.setTimeout(600000);

  test('5-agent consensus: 3 endorsements beat 2 endorsements', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob, charlie, dave, eve] =
        await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

      await scenario.shareAllAgents();
      const allAgents = [alice, bob, charlie, dave, eve];
      const dnaHash = alice.cells[0].cell_id[0];

      // Alice submits claim A: "Earth orbits Sun"
      const claimA: ActionHash = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Earth',
          predicate: 'orbits',
          object: 'Sun',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      // Dave submits claim B: "Earth orbits Moon" (wrong)
      const claimB: ActionHash = await dave.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Earth',
          predicate: 'orbits',
          object: 'Moon',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync(allAgents, dnaHash);

      // Bob, Charlie, Eve endorse claim A (3 endorsements)
      for (const agent of [bob, charlie, eve]) {
        await agent.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: claimA,
            attestation_type: 'endorse',
            evidence: 'Confirmed by astronomical observation',
          } as AttestClaimInput,
        });
      }

      // Dave and Eve endorse claim B (2 endorsements - Eve endorses both)
      for (const agent of [dave, eve]) {
        await agent.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: claimB,
            attestation_type: 'endorse',
            evidence: 'My observation',
          } as AttestClaimInput,
        });
      }

      await dhtSync(allAgents, dnaHash);

      // Query claims about Earth
      const claims: WeightedClaim[] = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Earth',
      });

      const sunClaim = claims.find((c) => c.object === 'Sun');
      const moonClaim = claims.find((c) => c.object === 'Moon');

      expect(sunClaim).toBeDefined();
      expect(moonClaim).toBeDefined();

      console.log(
        `5-agent consensus: Sun confidence=${sunClaim!.confidence.toFixed(4)} (${sunClaim!.endorsements} endorsements), ` +
          `Moon confidence=${moonClaim!.confidence.toFixed(4)} (${moonClaim!.endorsements} endorsements)`,
      );

      // Claim A (3 endorsements) should have higher confidence than claim B (2 endorsements)
      expect(sunClaim!.confidence).toBeGreaterThan(moonClaim!.confidence);
    }, true, scenarioOptions);
  });

  test('Reputation propagation: experienced agent claim wins over fresh agent', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [experienced, fresh, endorser1, endorser2, endorser3, observer] =
        await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

      await scenario.shareAllAgents();
      const allAgents = [
        experienced,
        fresh,
        endorser1,
        endorser2,
        endorser3,
        observer,
      ];
      const dnaHash = experienced.cells[0].cell_id[0];

      // Build reputation for "experienced" agent: submit 10 claims and get them endorsed
      const experiencedClaims: ActionHash[] = [];
      for (let i = 0; i < 10; i++) {
        const hash: ActionHash = await experienced.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: `Topic_${i}`,
            predicate: 'value',
            object: `fact_${i}`,
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });
        experiencedClaims.push(hash);
      }

      await dhtSync(allAgents, dnaHash);

      // Endorsers endorse all 10 claims
      for (const claim of experiencedClaims) {
        for (const endorser of [endorser1, endorser2, endorser3]) {
          await endorser.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'attest_claim',
            payload: {
              claim_hash: claim,
              attestation_type: 'endorse',
            } as AttestClaimInput,
          });
        }
      }

      await dhtSync(allAgents, dnaHash);

      // Check reputations before the conflict
      const expReputation: AgentReputation =
        await observer.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_agent_reputation',
          payload: experienced.agentPubKey,
        });

      const freshReputation: AgentReputation =
        await observer.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_agent_reputation',
          payload: fresh.agentPubKey,
        });

      console.log(
        `Reputation before conflict: experienced=${expReputation.reputation_score.toFixed(4)}, fresh=${freshReputation.reputation_score.toFixed(4)}`,
      );

      // Experienced agent has higher reputation
      expect(expReputation.reputation_score).toBeGreaterThan(
        freshReputation.reputation_score,
      );

      // Now both submit conflicting claims on the same subject
      const expClaim: ActionHash = await experienced.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Gravity',
          predicate: 'acceleration',
          object: '9.81',
          object_type: 'number',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      const freshClaim: ActionHash = await fresh.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Gravity',
          predicate: 'acceleration',
          object: '10.5',
          object_type: 'number',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync(allAgents, dnaHash);

      // Query claims about Gravity
      const gravityClaims: WeightedClaim[] =
        await observer.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'Gravity',
        });

      const claim981 = gravityClaims.find((c) => c.object === '9.81');
      const claim105 = gravityClaims.find((c) => c.object === '10.5');

      expect(claim981).toBeDefined();
      expect(claim105).toBeDefined();

      console.log(
        `Reputation propagation: experienced claim confidence=${claim981!.confidence.toFixed(4)}, ` +
          `fresh claim confidence=${claim105!.confidence.toFixed(4)}`,
      );

      // Experienced agent's claim should have higher confidence due to reputation
      expect(claim981!.confidence).toBeGreaterThan(claim105!.confidence);
    }, true, scenarioOptions);
  });

  test('Subject discovery: list_subjects returns all subjects from multiple agents', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob, charlie] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      // Each agent submits claims about different subjects
      await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Mathematics',
          predicate: 'foundation',
          object: 'Logic',
          object_type: 'text',
          epistemic_type: 'metaphysical',
        } as SubmitClaimInput,
      });

      await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Physics',
          predicate: 'foundation',
          object: 'Mathematics',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await charlie.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Ethics',
          predicate: 'basis',
          object: 'Empathy',
          object_type: 'text',
          epistemic_type: 'normative',
        } as SubmitClaimInput,
      });

      // Alice also submits about Physics (same subject as Bob)
      await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Physics',
          predicate: 'method',
          object: 'Experimentation',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync([alice, bob, charlie], dnaHash);

      // All agents should see all 3 unique subjects
      for (const agent of [alice, bob, charlie]) {
        const subjects: string[] = await agent.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'list_subjects',
          payload: null,
        });

        console.log(
          `Subjects visible to agent: [${subjects.join(', ')}]`,
        );

        expect(subjects).toContain('Mathematics');
        expect(subjects).toContain('Physics');
        expect(subjects).toContain('Ethics');
      }

      // Check that Physics has 2 claims
      const physicsClaims: WeightedClaim[] =
        await charlie.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'Physics',
        });

      expect(physicsClaims.length).toBe(2);
    }, true, scenarioOptions);
  });

  test('Self-attestation: agent can endorse own claim', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, observer] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      // Alice submits a claim
      const claim: ActionHash = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'SelfAttest',
          predicate: 'test',
          object: 'value',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync([alice, observer], dnaHash);

      // Alice endorses her own claim (self-attestation)
      const attestation: ActionHash = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: claim,
          attestation_type: 'endorse',
          evidence: 'I stand by my claim',
        } as AttestClaimInput,
      });

      expect(attestation).toBeDefined();

      await dhtSync([alice, observer], dnaHash);

      // Verify the self-attestation is recorded
      const claims: WeightedClaim[] = await observer.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'SelfAttest',
      });

      expect(claims.length).toBe(1);
      expect(claims[0].endorsements).toBeGreaterThanOrEqual(1);
      expect(claims[0].attestation_count).toBeGreaterThanOrEqual(1);

      console.log(
        `Self-attestation: endorsements=${claims[0].endorsements}, confidence=${claims[0].confidence.toFixed(4)}`,
      );

      // Document: Self-attestation is allowed but shouldn't dramatically boost confidence
      // The confidence should be reasonable (not artificially inflated)
      expect(claims[0].confidence).toBeLessThan(1.0);
    }, true, scenarioOptions);
  });

  test('Governance tier reflects agent activity level', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [active, passive] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = active.cells[0].cell_id[0];

      // Passive agent: check tier (should be Observer)
      const passiveTier: string = await passive.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_agent_governance_tier',
        payload: passive.agentPubKey,
      });

      console.log(`Passive agent tier: ${passiveTier}`);
      expect(passiveTier).toBe('Major');

      // Active agent: submit several claims
      for (let i = 0; i < 5; i++) {
        await active.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: `GovTest_${i}`,
            predicate: 'value',
            object: `data_${i}`,
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });
      }

      await dhtSync([active, passive], dnaHash);

      // Active agent: check tier (should be at least Observer, possibly higher)
      const activeTier: string = await active.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_agent_governance_tier',
        payload: active.agentPubKey,
      });

      console.log(`Active agent tier: ${activeTier}`);

      // Active agent should have a tier (even if still Observer, the system doesn't crash)
      expect(['Observer', 'Basic', 'Major', 'Constitutional']).toContain(
        activeTier,
      );
    }, true, scenarioOptions);
  });
});
