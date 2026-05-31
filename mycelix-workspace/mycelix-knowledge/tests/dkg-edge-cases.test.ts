// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DKG Edge Case Tests
 *
 * Tests boundary conditions, Unicode handling, and concurrent operations:
 * - Boundary values (max-length subjects, evidence)
 * - Unicode characters (CJK, emoji, Arabic)
 * - Object type variations (numeric, boolean, integer)
 * - Concurrent operations from multiple agents
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

// ============================================================================
// Edge Case Tests
// ============================================================================

describe('DKG Edge Case Tests', () => {
  jest.setTimeout(600000);

  // --------------------------------------------------------------------------
  // Boundary Values
  // --------------------------------------------------------------------------

  describe('Boundary values', () => {
    test('Subject at exactly 256 bytes accepted', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        // Create a subject that is exactly 256 ASCII characters (= 256 bytes)
        const maxSubject = 'A'.repeat(256);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: maxSubject,
            predicate: 'test',
            object: 'value',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        // Verify we can retrieve it
        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: maxSubject,
        });

        expect(claims.length).toBe(1);
        expect(claims[0].subject).toBe(maxSubject);
      }, true, scenarioOptions);
    });

    test('Subject over 256 bytes rejected', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const oversizedSubject = 'A'.repeat(257);

        await expect(
          alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject: oversizedSubject,
              predicate: 'test',
              object: 'value',
              object_type: 'text',
              epistemic_type: 'empirical',
            } as SubmitClaimInput,
          }),
        ).rejects.toThrow();
      }, true, scenarioOptions);
    });

    test('Evidence at exactly 2000 bytes accepted', async () => {
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
            subject: 'EvidenceBoundary',
            predicate: 'test',
            object: 'value',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        await dhtSync([alice, bob], dnaHash);

        // Evidence at exactly 2000 bytes
        const maxEvidence = 'E'.repeat(2000);

        const attestation: ActionHash = await bob.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: claim,
            attestation_type: 'endorse',
            evidence: maxEvidence,
          } as AttestClaimInput,
        });

        expect(attestation).toBeDefined();
      }, true, scenarioOptions);
    });

    test('Evidence over 2000 bytes rejected', async () => {
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
            subject: 'EvidenceOverflow',
            predicate: 'test',
            object: 'value',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        await dhtSync([alice, bob], dnaHash);

        const oversizedEvidence = 'E'.repeat(2001);

        await expect(
          bob.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'attest_claim',
            payload: {
              claim_hash: claim,
              attestation_type: 'endorse',
              evidence: oversizedEvidence,
            } as AttestClaimInput,
          }),
        ).rejects.toThrow();
      }, true, scenarioOptions);
    });

    test('Object at exactly 1024 bytes accepted', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const maxObject = 'O'.repeat(1024);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'ObjectBoundary',
            predicate: 'test',
            object: maxObject,
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();
      }, true, scenarioOptions);
    });
  });

  // --------------------------------------------------------------------------
  // Unicode Support
  // --------------------------------------------------------------------------

  describe('Unicode support', () => {
    test('CJK characters as subject', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: '\u5929\u7a7a',  // "Sky" in Chinese
            predicate: '\u989c\u8272',  // "color" in Chinese
            object: '\u84dd\u8272',  // "blue" in Chinese
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: '\u5929\u7a7a',
        });

        expect(claims.length).toBe(1);
        expect(claims[0].subject).toBe('\u5929\u7a7a');
        expect(claims[0].object).toBe('\u84dd\u8272');
      }, true, scenarioOptions);
    });

    test('Emoji as subject', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: '\ud83c\udf0d',  // Earth emoji
            predicate: 'is',
            object: '\ud83c\udf0e',  // Globe emoji
            object_type: 'text',
            epistemic_type: 'metaphysical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: '\ud83c\udf0d',
        });

        expect(claims.length).toBe(1);
      }, true, scenarioOptions);
    });

    test('Arabic text as subject', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: '\u0627\u0644\u0633\u0645\u0627\u0621',  // "The sky" in Arabic
            predicate: '\u0644\u0648\u0646',  // "color" in Arabic
            object: '\u0623\u0632\u0631\u0642',  // "blue" in Arabic
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: '\u0627\u0644\u0633\u0645\u0627\u0621',
        });

        expect(claims.length).toBe(1);
        expect(claims[0].object).toBe('\u0623\u0632\u0631\u0642');
      }, true, scenarioOptions);
    });

    test('Mixed Unicode subject in list_subjects', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const subjects = ['Hello', '\u4f60\u597d', '\u0645\u0631\u062d\u0628\u0627', '\ud83d\udc4b'];

        for (const subject of subjects) {
          await alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject,
              predicate: 'greeting',
              object: 'true',
              object_type: 'boolean',
              epistemic_type: 'normative',
            } as SubmitClaimInput,
          });
        }

        const listed: string[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'list_subjects',
          payload: null,
        });

        for (const subject of subjects) {
          expect(listed).toContain(subject);
        }

        console.log(`Unicode subjects listed: [${listed.join(', ')}]`);
      }, true, scenarioOptions);
    });
  });

  // --------------------------------------------------------------------------
  // Object Types
  // --------------------------------------------------------------------------

  describe('Object types', () => {
    test('Numeric object type', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Pi',
            predicate: 'approximation',
            object: '3.14159',
            object_type: 'number',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'Pi',
        });

        expect(claims[0].object_type).toBe('number');
        expect(claims[0].object).toBe('3.14159');
      }, true, scenarioOptions);
    });

    test('Boolean object type', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'EarthIsRound',
            predicate: 'is_true',
            object: 'true',
            object_type: 'boolean',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'EarthIsRound',
        });

        expect(claims[0].object_type).toBe('boolean');
      }, true, scenarioOptions);
    });

    test('Integer object type', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const hash: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Planets',
            predicate: 'count_in_solar_system',
            object: '8',
            object_type: 'integer',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        expect(hash).toBeDefined();

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'Planets',
        });

        expect(claims[0].object_type).toBe('integer');
        expect(claims[0].object).toBe('8');
      }, true, scenarioOptions);
    });

    test('All three epistemic types accepted', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        const types = [
          { type: 'empirical', subject: 'EmpiricalTest' },
          { type: 'normative', subject: 'NormativeTest' },
          { type: 'metaphysical', subject: 'MetaphysicalTest' },
        ];

        for (const { type, subject } of types) {
          const hash: ActionHash = await alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject,
              predicate: 'type_test',
              object: 'value',
              object_type: 'text',
              epistemic_type: type,
            } as SubmitClaimInput,
          });

          expect(hash).toBeDefined();
        }

        // Verify all subjects are listed
        const subjects: string[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'list_subjects',
          payload: null,
        });

        expect(subjects).toContain('EmpiricalTest');
        expect(subjects).toContain('NormativeTest');
        expect(subjects).toContain('MetaphysicalTest');
      }, true, scenarioOptions);
    });
  });

  // --------------------------------------------------------------------------
  // Concurrent Operations
  // --------------------------------------------------------------------------

  describe('Concurrent operations', () => {
    test('Two agents submit claims simultaneously, both visible after sync', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        await scenario.shareAllAgents();
        const dnaHash = alice.cells[0].cell_id[0];

        // Both submit claims about the same subject simultaneously
        const [aliceClaim, bobClaim] = await Promise.all([
          alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject: 'ConcurrentSubject',
              predicate: 'value',
              object: 'from_alice',
              object_type: 'text',
              epistemic_type: 'empirical',
            } as SubmitClaimInput,
          }),
          bob.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject: 'ConcurrentSubject',
              predicate: 'value',
              object: 'from_bob',
              object_type: 'text',
              epistemic_type: 'empirical',
            } as SubmitClaimInput,
          }),
        ]);

        expect(aliceClaim).toBeDefined();
        expect(bobClaim).toBeDefined();

        await dhtSync([alice, bob], dnaHash);

        // Both claims should be visible
        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'ConcurrentSubject',
        });

        expect(claims.length).toBe(2);

        const objects = claims.map((c) => c.object).sort();
        expect(objects).toEqual(['from_alice', 'from_bob']);

        console.log(
          `Concurrent claims: ${claims.length} claims visible after sync`,
        );
      }, true, scenarioOptions);
    });

    test('Concurrent endorsement and challenge on same claim', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob, charlie] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        await scenario.shareAllAgents();
        const dnaHash = alice.cells[0].cell_id[0];

        const claim: ActionHash = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'submit_claim',
          payload: {
            subject: 'Disputed',
            predicate: 'claim',
            object: 'contested_value',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        await dhtSync([alice, bob, charlie], dnaHash);

        // Bob endorses and Charlie challenges simultaneously
        const [endorsement, challenge] = await Promise.all([
          bob.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'attest_claim',
            payload: {
              claim_hash: claim,
              attestation_type: 'endorse',
              evidence: 'I agree',
            } as AttestClaimInput,
          }),
          charlie.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'attest_claim',
            payload: {
              claim_hash: claim,
              attestation_type: 'challenge',
              evidence: 'I disagree',
            } as AttestClaimInput,
          }),
        ]);

        expect(endorsement).toBeDefined();
        expect(challenge).toBeDefined();

        await dhtSync([alice, bob, charlie], dnaHash);

        // Both attestations should be reflected
        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'Disputed',
        });

        expect(claims.length).toBe(1);
        expect(claims[0].endorsements).toBe(1);
        expect(claims[0].challenges).toBe(1);
        expect(claims[0].attestation_count).toBe(2);

        console.log(
          `Concurrent attestation: endorsements=${claims[0].endorsements}, challenges=${claims[0].challenges}, confidence=${claims[0].confidence.toFixed(4)}`,
        );
      }, true, scenarioOptions);
    });

    test('Multiple subjects from single agent do not interfere', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        ]);

        // Submit claims about many distinct subjects
        const subjectCount = 10;
        for (let i = 0; i < subjectCount; i++) {
          await alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'submit_claim',
            payload: {
              subject: `IsolatedSubject_${i}`,
              predicate: 'index',
              object: `${i}`,
              object_type: 'integer',
              epistemic_type: 'empirical',
            } as SubmitClaimInput,
          });
        }

        // Each subject should have exactly 1 claim
        for (let i = 0; i < subjectCount; i++) {
          const claims: WeightedClaim[] = await alice.cells[0].callZome({
            zome_name: 'dkg',
            fn_name: 'get_claims',
            payload: `IsolatedSubject_${i}`,
          });

          expect(claims.length).toBe(1);
          expect(claims[0].object).toBe(`${i}`);
        }

        // All subjects listed
        const subjects: string[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'list_subjects',
          payload: null,
        });

        const isolated = subjects.filter((s) => s.startsWith('IsolatedSubject_'));
        expect(isolated.length).toBe(subjectCount);
      }, true, scenarioOptions);
    });
  });

  // --------------------------------------------------------------------------
  // Acknowledge attestation type
  // --------------------------------------------------------------------------

  describe('Acknowledge attestation', () => {
    test('Acknowledge attestation does not affect endorsement/challenge counts', async () => {
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
            subject: 'AckTest',
            predicate: 'value',
            object: 'data',
            object_type: 'text',
            epistemic_type: 'empirical',
          } as SubmitClaimInput,
        });

        await dhtSync([alice, bob], dnaHash);

        // Bob acknowledges (neutral attestation)
        await bob.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'attest_claim',
          payload: {
            claim_hash: claim,
            attestation_type: 'acknowledge',
            evidence: 'I note this claim',
          } as AttestClaimInput,
        });

        await dhtSync([alice, bob], dnaHash);

        const claims: WeightedClaim[] = await alice.cells[0].callZome({
          zome_name: 'dkg',
          fn_name: 'get_claims',
          payload: 'AckTest',
        });

        expect(claims.length).toBe(1);
        expect(claims[0].attestation_count).toBe(1);
        // Acknowledge should not count as endorsement or challenge
        expect(claims[0].endorsements).toBe(0);
        expect(claims[0].challenges).toBe(0);

        console.log(
          `Acknowledge: attestation_count=${claims[0].attestation_count}, endorsements=${claims[0].endorsements}, challenges=${claims[0].challenges}`,
        );
      }, true, scenarioOptions);
    });
  });
});
