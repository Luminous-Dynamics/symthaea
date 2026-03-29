// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DKG Adversarial Tests
 *
 * Tests the Truth Engine's resilience against:
 * - Sybil attacks (many agents claiming falsehoods)
 * - Challenge warfare (cascading challenges/counter-challenges)
 * - Spam floods (rapid claim submission)
 * - Invalid inputs (malformed data)
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
// Adversarial Tests
// ============================================================================

describe('DKG Adversarial Tests', () => {
  jest.setTimeout(1200000);

  test('Sybil resistance: reputation-weighted confidence beats raw claim count', async () => {
    await runScenario(async (scenario: Scenario) => {
      // 5 sybil agents + 2 honest agents = 7 total
      const [sybil1, sybil2, sybil3, sybil4, sybil5, alice, bob] =
        await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      // All 5 sybils claim "Sky is Green"
      const sybilAgents = [sybil1, sybil2, sybil3, sybil4, sybil5];
      const sybilClaims: ActionHash[] = [];

      for (const sybil of sybilAgents) {
        const claim: ActionHash = await sybil.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Sky',
            predicate: 'color',
            object: 'Green',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });
        sybilClaims.push(claim);
      }

      // Alice submits truth
      const aliceClaim: ActionHash = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Sky',
          predicate: 'color',
          object: 'Blue',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync(
        [sybil1, sybil2, sybil3, sybil4, sybil5, alice, bob],
        dnaHash,
      );

      // Bob endorses Alice's claim with evidence
      await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: aliceClaim,
          attestation_type: 'endorse',
          evidence: 'Verified by direct observation - the sky is blue',
        } as AttestClaimInput,
      });

      // Alice endorses her own claim (self-attestation)
      await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: aliceClaim,
          attestation_type: 'endorse',
          evidence: 'Observable fact confirmed by atmospheric science',
        } as AttestClaimInput,
      });

      await dhtSync(
        [sybil1, sybil2, sybil3, sybil4, sybil5, alice, bob],
        dnaHash,
      );

      // Query truth - endorsed claim should have higher confidence
      const claims: WeightedClaim[] = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Sky',
      });

      const blueClaims = claims.filter((c) => c.object === 'Blue');
      const greenClaims = claims.filter((c) => c.object === 'Green');

      expect(blueClaims.length).toBeGreaterThanOrEqual(1);
      expect(greenClaims.length).toBeGreaterThanOrEqual(1);

      // The endorsed Blue claim should have higher confidence than any individual Green claim
      const maxBlueConfidence = Math.max(...blueClaims.map((c) => c.confidence));
      const maxGreenConfidence = Math.max(
        ...greenClaims.map((c) => c.confidence),
      );

      console.log(
        `Sybil test: Blue confidence=${maxBlueConfidence.toFixed(4)}, Green max confidence=${maxGreenConfidence.toFixed(4)}`,
      );
      console.log(
        `Green claims count: ${greenClaims.length}, Blue endorsements: ${blueClaims[0]?.endorsements}`,
      );

      // Reputation-weighted confidence should favor the endorsed claim
      expect(maxBlueConfidence).toBeGreaterThan(maxGreenConfidence);
    }, true, scenarioOptions);
  });

  test('Challenge warfare: confidence degrades with challenges', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob, charlie] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
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
          subject: 'Water',
          predicate: 'boiling_point',
          object: '100',
          object_type: 'number',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync([alice, bob, charlie], dnaHash);

      // Get baseline confidence (no attestations)
      let claims: WeightedClaim[] = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Water',
      });
      const baselineConfidence = claims[0].confidence;

      // Bob challenges the claim
      await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: claim,
          attestation_type: 'challenge',
          evidence: 'Depends on altitude and pressure',
        } as AttestClaimInput,
      });

      await dhtSync([alice, bob, charlie], dnaHash);

      // Get confidence after challenge
      claims = await charlie.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Water',
      });
      const challengedConfidence = claims[0].confidence;

      // Charlie also challenges
      await charlie.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: claim,
          attestation_type: 'challenge',
          evidence: 'At altitude it boils at lower temps',
        } as AttestClaimInput,
      });

      await dhtSync([alice, bob, charlie], dnaHash);

      // Get confidence after two challenges
      claims = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Water',
      });
      const doubleChallengedConfidence = claims[0].confidence;

      console.log(
        `Challenge warfare: baseline=${baselineConfidence.toFixed(4)}, ` +
          `after 1 challenge=${challengedConfidence.toFixed(4)}, ` +
          `after 2 challenges=${doubleChallengedConfidence.toFixed(4)}`,
      );

      // Confidence should degrade (or at least not increase) with challenges
      expect(challengedConfidence).toBeLessThanOrEqual(baselineConfidence);
      expect(doubleChallengedConfidence).toBeLessThanOrEqual(
        challengedConfidence,
      );
    }, true, scenarioOptions);
  });

  test('Spam flood: 50 rapid claims cause no crash, reputation stays bounded', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [spammer, observer] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = spammer.cells[0].cell_id[0];

      // Submit 50 claims rapidly
      const claimHashes: ActionHash[] = [];
      for (let i = 0; i < 50; i++) {
        const hash: ActionHash = await spammer.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: `Spam_${i}`,
            predicate: 'value',
            object: `spam_value_${i}`,
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });
        claimHashes.push(hash);
      }

      // All 50 should have succeeded
      expect(claimHashes.length).toBe(50);

      await dhtSync([spammer, observer], dnaHash);

      // Check reputation - should not inflate beyond bounds
      const reputation: AgentReputation = await observer.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_agent_reputation',
        payload: spammer.agentPubKey,
      });

      expect(reputation.claim_count).toBe(50);
      // Reputation should stay in [0, 1] range
      expect(reputation.reputation_score).toBeGreaterThanOrEqual(0);
      expect(reputation.reputation_score).toBeLessThanOrEqual(1);

      console.log(
        `Spam flood: 50 claims submitted, reputation=${reputation.reputation_score.toFixed(4)} (bounded: ${reputation.reputation_score >= 0 && reputation.reputation_score <= 1})`,
      );

      // Verify subjects are discoverable
      const subjects: string[] = await observer.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'list_subjects',
        payload: null,
      });

      // All 50 unique subjects should be listed
      const spamSubjects = subjects.filter((s) => s.startsWith('Spam_'));
      expect(spamSubjects.length).toBe(50);
    }, true, scenarioOptions);
  });

  test('Invalid inputs: empty subject rejected', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Empty subject should fail validation
      await expect(
        alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: '',
            predicate: 'color',
            object: 'Blue',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        }),
      ).rejects.toThrow();
    }, true, scenarioOptions);
  });

  test('Invalid inputs: oversized object (>1024 bytes) rejected', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const oversizedObject = 'x'.repeat(1025);

      await expect(
        alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Test',
            predicate: 'value',
            object: oversizedObject,
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        }),
      ).rejects.toThrow();
    }, true, scenarioOptions);
  });

  test('Invalid inputs: invalid epistemic type rejected', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await expect(
        alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Test',
            predicate: 'value',
            object: 'data',
            object_type: 'text',
            epistemic_type: 'invalid_type',
          } as SubmitClaimInput,
        }),
      ).rejects.toThrow();
    }, true, scenarioOptions);
  });

  test('Invalid inputs: attestation for non-existent claim hash rejected', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create a fake ActionHash (39 bytes of zeros)
      const fakeHash = new Uint8Array(39);
      fakeHash[0] = 0x84; // ActionHash prefix
      fakeHash[1] = 0x21;
      fakeHash[2] = 0x24;

      await expect(
        alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: fakeHash,
            attestation_type: 'endorse',
            evidence: 'Endorsing non-existent claim',
          } as AttestClaimInput,
        }),
      ).rejects.toThrow();
    }, true, scenarioOptions);
  });

  test('Invalid inputs: invalid attestation type rejected', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      const claim: ActionHash = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'submit_claim',
        payload: {
          subject: 'Test',
          predicate: 'value',
          object: 'data',
          object_type: 'text',
          epistemic_type: 'empirical',
        } as SubmitClaimInput,
      });

      await dhtSync([alice, bob], dnaHash);

      await expect(
        bob.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: claim,
            attestation_type: 'invalid_type',
            evidence: 'Testing invalid type',
          } as AttestClaimInput,
        }),
      ).rejects.toThrow();
    }, true, scenarioOptions);
  });
});
