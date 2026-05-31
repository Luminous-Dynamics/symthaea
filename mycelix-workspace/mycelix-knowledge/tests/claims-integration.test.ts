// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Claims Integration Tests
 *
 * Tests the claims zome's full lifecycle:
 * - Submit classified claims (E/N/M)
 * - Retrieve by ID and tag
 * - Add evidence and challenges
 * - Register dependencies with cycle detection
 * - Cascade update propagation
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
// Type Definitions (matching claims zome types)
// ============================================================================

interface EpistemicPosition {
  empirical: number;
  normative: number;
  mythic: number;
}

interface Claim {
  id: string;
  content: string;
  classification: EpistemicPosition;
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

interface ChallengeClaimInput {
  claim_id: string;
  challenger_did: string;
  reason: string;
  counter_evidence: string[];
}

interface RegisterDependencyInput {
  dependent_claim_id: string;
  dependency_claim_id: string;
  dependency_type: string;
  weight: number;
  influence: string;
  justification: string | null;
}

interface UpdateClaimInput {
  claim_id: string;
  content?: string;
  classification?: EpistemicPosition;
  sources?: string[];
  tags?: string[];
  confidence?: number;
  expires?: number | null;
}

interface CascadeResult {
  updated_claims: string[];
  trigger_claim_id: string;
  depth_reached: number;
  circular_detected: string[];
  processed_at: number;
  processing_time_ms: number;
}

// ============================================================================
// Claims Integration Tests
// ============================================================================

describe('Claims Integration Tests', () => {
  jest.setTimeout(600000);

  test('Submit classified claim with E/N/M position, retrieve by ID', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000; // microseconds
      const claim: Claim = {
        id: 'claim_gravity_001',
        content: 'Gravitational acceleration on Earth is approximately 9.81 m/s²',
        classification: {
          empirical: 0.95,
          normative: 0.1,
          mythic: 0.05,
        },
        author: 'did:key:alice001',
        sources: ['https://physics.nist.gov/gravity'],
        tags: ['physics', 'gravity', 'earth'],
        claim_type: 'Fact',
        confidence: 0.95,
        expires: null,
        created: now,
        updated: now,
        version: 1,
      };

      const record = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: claim,
      });

      expect(record).toBeDefined();

      // Retrieve by ID
      const retrieved = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim',
        payload: 'claim_gravity_001',
      });

      expect(retrieved).not.toBeNull();
    }, true, scenarioOptions);
  });

  test('Retrieve claims by tag', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000;

      // Submit two claims with shared tag
      for (const id of ['claim_tag_001', 'claim_tag_002']) {
        await alice.cells[0].callZome({
          zome_name: 'claims',
          fn_name: 'submit_claim',
          payload: {
            id,
            content: `Claim ${id}`,
            classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
            author: 'did:key:alice',
            sources: [],
            tags: ['shared_tag', id],
            claim_type: 'Fact',
            confidence: 0.5,
            expires: null,
            created: now,
            updated: now,
            version: 1,
          } as Claim,
        });
      }

      // Retrieve by shared tag
      const results = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claims_by_tag',
        payload: 'shared_tag',
      }) as any[];

      expect(results.length).toBe(2);
    }, true, scenarioOptions);
  });

  test('Add evidence to a claim', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000;

      // Submit claim
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: {
          id: 'claim_evidence_test',
          content: 'Water boils at 100°C at sea level',
          classification: { empirical: 0.9, normative: 0.1, mythic: 0.0 },
          author: 'did:key:alice',
          sources: [],
          tags: ['chemistry'],
          claim_type: 'Fact',
          confidence: 0.8,
          expires: null,
          created: now,
          updated: now,
          version: 1,
        } as Claim,
      });

      // Add evidence
      const evidence: Evidence = {
        id: 'ev_001',
        claim_id: 'claim_evidence_test',
        evidence_type: 'Experiment',
        source_uri: 'https://doi.org/10.1234/water-boiling',
        content: 'Controlled experiment confirming boiling point at 1 atm',
        strength: 0.9,
        submitted_by: 'did:key:alice',
        submitted_at: now,
      };

      const evidenceRecord = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'add_evidence',
        payload: evidence,
      });

      expect(evidenceRecord).toBeDefined();

      // Retrieve evidence for claim
      const claimEvidence = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim_evidence',
        payload: 'claim_evidence_test',
      }) as any[];

      expect(claimEvidence.length).toBe(1);
    }, true, scenarioOptions);
  });

  test('Challenge a claim', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      const now = Date.now() * 1000;

      // Alice submits a claim
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: {
          id: 'claim_challenge_test',
          content: 'Pluto is a planet',
          classification: { empirical: 0.3, normative: 0.5, mythic: 0.2 },
          author: 'did:key:alice',
          sources: [],
          tags: ['astronomy'],
          claim_type: 'Fact',
          confidence: 0.3,
          expires: null,
          created: now,
          updated: now,
          version: 1,
        } as Claim,
      });

      await dhtSync([alice, bob], dnaHash);

      // Bob challenges the claim
      const challengeInput: ChallengeClaimInput = {
        claim_id: 'claim_challenge_test',
        challenger_did: 'did:key:bob',
        reason: 'IAU reclassified Pluto as a dwarf planet in 2006',
        counter_evidence: ['https://www.iau.org/public/themes/pluto/'],
      };

      const challengeRecord = await bob.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'challenge_claim',
        payload: challengeInput,
      });

      expect(challengeRecord).toBeDefined();

      await dhtSync([alice, bob], dnaHash);

      // Get challenges for the claim
      const challenges = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim_challenges',
        payload: 'claim_challenge_test',
      }) as any[];

      expect(challenges.length).toBeGreaterThanOrEqual(1);
    }, true, scenarioOptions);
  });

  test('Register dependency and detect cycles', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000;

      // Create three claims: A depends on B, B depends on C
      for (const id of ['claim_A', 'claim_B', 'claim_C']) {
        await alice.cells[0].callZome({
          zome_name: 'claims',
          fn_name: 'submit_claim',
          payload: {
            id,
            content: `Claim ${id}`,
            classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
            author: 'did:key:alice',
            sources: [],
            tags: ['dep_test'],
            claim_type: 'Hypothesis',
            confidence: 0.5,
            expires: null,
            created: now,
            updated: now,
            version: 1,
          } as Claim,
        });
      }

      // A depends on B
      const depAB = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'register_dependency',
        payload: {
          dependent_claim_id: 'claim_A',
          dependency_claim_id: 'claim_B',
          dependency_type: 'EvidentialSupport',
          weight: 0.8,
          influence: 'Positive',
          justification: 'A relies on B for evidence',
        } as RegisterDependencyInput,
      });

      expect(depAB).toBeDefined();

      // B depends on C
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'register_dependency',
        payload: {
          dependent_claim_id: 'claim_B',
          dependency_claim_id: 'claim_C',
          dependency_type: 'Premise',
          weight: 0.6,
          influence: 'Positive',
          justification: 'B builds on C',
        } as RegisterDependencyInput,
      });

      // Try to create cycle: C depends on A (should fail)
      await expect(
        alice.cells[0].callZome({
          zome_name: 'claims',
          fn_name: 'register_dependency',
          payload: {
            dependent_claim_id: 'claim_C',
            dependency_claim_id: 'claim_A',
            dependency_type: 'EvidentialSupport',
            weight: 0.5,
            influence: 'Positive',
            justification: 'This would create a cycle',
          } as RegisterDependencyInput,
        }),
      ).rejects.toThrow();

      // Verify dependencies exist
      const depsA = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'get_claim_dependencies',
        payload: 'claim_A',
      }) as any[];

      expect(depsA.length).toBe(1);
    }, true, scenarioOptions);
  });

  test('Cascade update propagation', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000;

      // Create root and dependent claims
      for (const id of ['cascade_root', 'cascade_dep1', 'cascade_dep2']) {
        await alice.cells[0].callZome({
          zome_name: 'claims',
          fn_name: 'submit_claim',
          payload: {
            id,
            content: `Cascade test: ${id}`,
            classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
            author: 'did:key:alice',
            sources: [],
            tags: ['cascade_test'],
            claim_type: 'Hypothesis',
            confidence: 0.5,
            expires: null,
            created: now,
            updated: now,
            version: 1,
          } as Claim,
        });
      }

      // dep1 depends on root
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'register_dependency',
        payload: {
          dependent_claim_id: 'cascade_dep1',
          dependency_claim_id: 'cascade_root',
          dependency_type: 'EvidentialSupport',
          weight: 0.9,
          influence: 'Positive',
          justification: null,
        } as RegisterDependencyInput,
      });

      // dep2 depends on root
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'register_dependency',
        payload: {
          dependent_claim_id: 'cascade_dep2',
          dependency_claim_id: 'cascade_root',
          dependency_type: 'Premise',
          weight: 0.7,
          influence: 'Positive',
          justification: null,
        } as RegisterDependencyInput,
      });

      // Trigger cascade update from root
      const cascadeResult: CascadeResult = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'cascade_update',
        payload: 'cascade_root',
      });

      expect(cascadeResult).toBeDefined();
      expect(cascadeResult.trigger_claim_id).toBe('cascade_root');
      expect(cascadeResult.circular_detected.length).toBe(0);

      console.log(
        `Cascade: ${cascadeResult.updated_claims.length} claims updated, ` +
          `depth=${cascadeResult.depth_reached}, ${cascadeResult.processing_time_ms}ms`,
      );
    }, true, scenarioOptions);
  });

  test('Search claims by epistemic range', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      const now = Date.now() * 1000;

      // Submit claims with different epistemic positions
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: {
          id: 'high_empirical',
          content: 'Highly empirical claim',
          classification: { empirical: 0.9, normative: 0.1, mythic: 0.1 },
          author: 'did:key:alice',
          sources: [],
          tags: ['epistemic_test'],
          claim_type: 'Fact',
          confidence: 0.9,
          expires: null,
          created: now,
          updated: now,
          version: 1,
        } as Claim,
      });

      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'submit_claim',
        payload: {
          id: 'high_normative',
          content: 'Highly normative claim',
          classification: { empirical: 0.1, normative: 0.9, mythic: 0.1 },
          author: 'did:key:alice',
          sources: [],
          tags: ['epistemic_test'],
          claim_type: 'Normative',
          confidence: 0.7,
          expires: null,
          created: now,
          updated: now,
          version: 1,
        } as Claim,
      });

      // Search for highly empirical claims
      const highEmpirical = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'search_by_epistemic_range',
        payload: {
          e_min: 0.7,
          e_max: 1.0,
          n_min: 0.0,
          n_max: 1.0,
          m_min: 0.0,
          m_max: 1.0,
        },
      }) as any[];

      expect(highEmpirical.length).toBeGreaterThanOrEqual(1);
    }, true, scenarioOptions);
  });
});
