// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Full Lifecycle Integration Tests
 *
 * End-to-end tests exercising the complete knowledge graph lifecycle:
 * - Claim -> Classification -> Evidence -> Endorsement -> Query -> Fact-Check
 * - Multi-zome data flow verification
 * - Bridge GIS classification of a live claim
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

interface Claim {
  id: string;
  content: string;
  classification: { empirical: number; normative: number; mythic: number };
  author: string;
  sources: string[];
  tags: string[];
  claim_type: string;
  confidence: number;
  expires: number | null;
  created: number;
  updated: number;
  version: number;
}

interface Evidence {
  id: string;
  claim_id: string;
  evidence_type: string;
  source_uri: string;
  content: string;
  strength: number;
  submitted_by: string;
  submitted_at: number;
}

interface Relationship {
  id: string;
  source: string;
  target: string;
  relationship_type: string;
  weight: number;
  properties: string | null;
  creator: string;
  created: number;
}

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

// Helper to create a claim via claims coordinator
async function submitClaimViaClaimsZome(cell: any, id: string, content: string, tags: string[] = []) {
  const now = Date.now() * 1000;
  return cell.callZome({
    zome_name: 'claims',
    fn_name: 'submit_claim',
    payload: {
      id,
      content,
      classification: { empirical: 0.7, normative: 0.3, mythic: 0.1 },
      author: 'did:key:lifecycle_user',
      sources: ['https://example.org/source'],
      tags,
      claim_type: 'Fact',
      confidence: 0.7,
      expires: null,
      created: now,
      updated: now,
      version: 1,
    } as Claim,
  });
}

// Helper to submit a DKG claim
async function submitDkgClaim(cell: any, subject: string, predicate: string, object: string) {
  return cell.callZome({
    zome_name: 'dkg',
    fn_name: 'submit_claim',
    payload: {
      subject,
      predicate,
      object,
      object_type: 'text',
      epistemic_type: 'empirical',
    } as SubmitClaimInput,
  });
}

// ============================================================================
// Full Lifecycle Tests
// ============================================================================

describe('Full Lifecycle Integration Tests', () => {
  jest.setTimeout(600000);

  test('End-to-end: claim -> evidence -> endorsement -> query', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob, charlie] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      console.log('=== FULL LIFECYCLE TEST ===');

      // Step 1: Alice submits a DKG claim
      console.log('1. Alice submits a claim via DKG...');
      const claimHash: ActionHash = await submitDkgClaim(
        alice.cells[0],
        'Climate',
        'global_temperature_trend',
        'Rising',
      );
      expect(claimHash).toBeDefined();

      await dhtSync([alice, bob, charlie], dnaHash);

      // Step 2: Bob endorses with evidence
      console.log('2. Bob endorses with evidence...');
      const endorsement: ActionHash = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: claimHash,
          attestation_type: 'endorse',
          evidence: 'NASA GISS temperature record shows consistent warming trend since 1880',
        } as AttestClaimInput,
      });
      expect(endorsement).toBeDefined();

      // Step 3: Charlie also endorses
      console.log('3. Charlie endorses...');
      await charlie.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'attest_claim',
        payload: {
          claim_hash: claimHash,
          attestation_type: 'endorse',
          evidence: 'IPCC AR6 confirms anthropogenic warming',
        } as AttestClaimInput,
      });

      await dhtSync([alice, bob, charlie], dnaHash);

      // Step 4: Query the truth
      console.log('4. Querying the Truth Engine...');
      const truth: WeightedClaim[] = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_truth',
        payload: 'Climate',
      });

      expect(truth.length).toBeGreaterThanOrEqual(1);
      const climateClaim = truth.find((c) => c.object === 'Rising');
      expect(climateClaim).toBeDefined();
      expect(climateClaim!.endorsements).toBeGreaterThanOrEqual(2);
      expect(climateClaim!.confidence).toBeGreaterThan(0);

      console.log(
        `Result: confidence=${climateClaim!.confidence.toFixed(4)}, ` +
          `endorsements=${climateClaim!.endorsements}`,
      );

      // Step 5: Check reputations
      console.log('5. Checking agent reputations...');
      const aliceRep = await charlie.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_agent_reputation',
        payload: alice.agentPubKey,
      }) as any;

      expect(aliceRep.claim_count).toBeGreaterThanOrEqual(1);

      console.log(
        `Alice reputation: score=${aliceRep.reputation_score.toFixed(4)}, claims=${aliceRep.claim_count}`,
      );

      console.log('=== LIFECYCLE COMPLETE ===\n');
    }, true, scenarioOptions);
  });

  test('Multi-zome data flow: claims + graph + DKG', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      console.log('=== MULTI-ZOME DATA FLOW ===');

      // Step 1: Submit claims via claims zome
      console.log('1. Creating claims via claims zome...');
      await submitClaimViaClaimsZome(
        alice.cells[0],
        'premise_claim',
        'All metals conduct electricity',
        ['physics', 'materials'],
      );
      await submitClaimViaClaimsZome(
        alice.cells[0],
        'evidence_claim',
        'Copper wire carries current efficiently',
        ['physics', 'copper'],
      );

      // Step 2: Create graph relationship
      console.log('2. Creating graph relationship...');
      const now = Date.now() * 1000;
      await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: {
          id: 'rel_copper_metals',
          source: 'evidence_claim',
          target: 'premise_claim',
          relationship_type: 'Supports',
          weight: 0.85,
          properties: null,
          creator: 'did:key:lifecycle_user',
          created: now,
        } as Relationship,
      });

      // Step 3: Submit via DKG for truth scoring
      console.log('3. Creating DKG claim for truth scoring...');
      const dkgHash = await submitDkgClaim(
        alice.cells[0],
        'Conductivity',
        'metal_property',
        'High',
      );

      // Step 4: Verify graph has the relationship
      console.log('4. Verifying graph data...');
      const outgoing = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_outgoing_relationships',
        payload: 'evidence_claim',
      }) as any[];
      expect(outgoing.length).toBe(1);

      // Step 5: Verify DKG has the claim
      console.log('5. Verifying DKG data...');
      const dkgClaims: WeightedClaim[] = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'Conductivity',
      });
      expect(dkgClaims.length).toBeGreaterThanOrEqual(1);

      // Step 6: Verify claims zome data
      console.log('6. Verifying claims zome data...');
      const retrievedClaim = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim',
        payload: 'premise_claim',
      });
      expect(retrievedClaim).not.toBeNull();

      const taggedClaims = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claims_by_tag',
        payload: 'physics',
      }) as any[];
      expect(taggedClaims.length).toBe(2);

      console.log('=== MULTI-ZOME DATA FLOW VERIFIED ===\n');
    }, true, scenarioOptions);
  });

  test('DKG + Claims coexistence: independent claim systems', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      // Submit via DKG (simple triple)
      const dkgClaim = await submitDkgClaim(
        alice.cells[0],
        'TestSubject',
        'has_property',
        'testvalue',
      );

      // Submit via claims (rich structured claim)
      const now = Date.now() * 1000;
      const claimsRecord = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: {
          id: 'structured_claim',
          content: 'TestSubject has property testvalue',
          classification: { empirical: 0.8, normative: 0.2, mythic: 0.1 },
          author: 'did:key:alice',
          sources: ['https://source.example.org'],
          tags: ['coexist_test'],
          claim_type: 'Fact',
          confidence: 0.8,
          expires: null,
          created: now,
          updated: now,
          version: 1,
        } as Claim,
      });

      await dhtSync([alice, bob], dnaHash);

      // Both should be independently queryable
      const dkgResults: WeightedClaim[] = await bob.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'get_claims',
        payload: 'TestSubject',
      });
      expect(dkgResults.length).toBeGreaterThanOrEqual(1);

      const claimsResults = await bob.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claims_by_tag',
        payload: 'coexist_test',
      }) as any[];
      expect(claimsResults.length).toBeGreaterThanOrEqual(1);

      console.log(
        `Coexistence: DKG claims=${dkgResults.length}, Claims zome=${claimsResults.length}`,
      );
    }, true, scenarioOptions);
  });

  test('Graph stats work with populated graph', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create claims and relationships
      await submitClaimViaClaimsZome(alice.cells[0], 'stats_claim1', 'Claim 1');
      await submitClaimViaClaimsZome(alice.cells[0], 'stats_claim2', 'Claim 2');
      await submitClaimViaClaimsZome(alice.cells[0], 'stats_claim3', 'Claim 3');

      const now = Date.now() * 1000;
      for (const [src, tgt, id] of [
        ['stats_claim1', 'stats_claim2', 'rel_s12'],
        ['stats_claim2', 'stats_claim3', 'rel_s23'],
      ]) {
        await alice.cells[0].callZome({
          zome_name: 'graph',
          fn_name: 'create_relationship',
          payload: {
            id,
            source: src,
            target: tgt,
            relationship_type: 'Supports',
            weight: 0.7,
            properties: null,
            creator: 'did:key:lifecycle_user',
            created: now,
          } as Relationship,
        });
      }

      // Get graph stats
      const stats = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_graph_stats',
        payload: null,
      }) as any;

      expect(stats.relationship_count).toBeGreaterThanOrEqual(2);
      console.log(`Graph stats: ${stats.relationship_count} relationships`);
    }, true, scenarioOptions);
  });

  test('Inference credibility assessment', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Assess credibility of a subject
      const credibilityRecord = await alice.cells[0].callZome({
        zome_name: 'inference',
        fn_name: 'assess_credibility',
        payload: {
          subject: 'test_subject',
          subject_type: 'Claim',
          expires_at: null,
        },
      });

      expect(credibilityRecord).toBeDefined();

      // Retrieve credibility
      const retrieved = await alice.cells[0].callZome({
        zome_name: 'inference',
        fn_name: 'get_credibility',
        payload: 'test_subject',
      });

      expect(retrieved).not.toBeNull();
      console.log('Credibility assessment created and retrieved');
    }, true, scenarioOptions);
  });

  test('Bridge: register external claim', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Register a claim from an external hApp
      const bridgeRecord = await alice.cells[0].callZome({
        zome_name: 'knowledge_bridge',
        fn_name: 'register_external_claim',
        payload: {
          source_happ: 'mycelix-health',
          subject: 'VitaminD',
          predicate: 'recommended_daily',
          object: '600 IU',
          epistemic_e: 0.85,
          epistemic_n: 0.3,
          epistemic_m: 0.1,
        },
      });

      expect(bridgeRecord).toBeDefined();

      // Query claims by hApp
      const happClaims = await alice.cells[0].callZome({
        zome_name: 'knowledge_bridge',
        fn_name: 'get_claims_by_happ',
        payload: 'mycelix-health',
      }) as any[];

      expect(happClaims.length).toBeGreaterThanOrEqual(1);

      console.log(
        `Bridge: registered external claim, ${happClaims.length} claims from mycelix-health`,
      );
    }, true, scenarioOptions);
  });

  test('All zomes healthy: ping equivalents', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // DKG ping
      const pong = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'ping',
        payload: null,
      });
      expect(pong).toBe('pong');

      // Graph stats (verifies graph zome loads)
      const stats = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_graph_stats',
        payload: null,
      });
      expect(stats).toBeDefined();

      // Claims query (verifies claims zome loads)
      const claimSearch = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim',
        payload: 'nonexistent_id',
      });
      expect(claimSearch).toBeNull();

      // DKG list subjects (verifies DKG is functional)
      const subjects = await alice.cells[0].callZome({
        zome_name: 'dkg',
        fn_name: 'list_subjects',
        payload: null,
      });
      expect(Array.isArray(subjects)).toBe(true);

      console.log('All zomes responding correctly');
    }, true, scenarioOptions);
  });
});
