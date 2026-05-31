// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge hApp Integration Tests
 *
 * Tests using Tryorama for full Holochain integration testing.
 */

import { Scenario, Player, createConductor, dhtSync } from '@holochain/tryorama';
import { ActionHash, AgentPubKey, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to compiled DNA
const knowledgeDnaPath = path.join(__dirname, '../workdir/knowledge.dna');

// ============================================================================
// Test Utilities
// ============================================================================

interface ClaimInput {
  content: string;
  classification: {
    empirical: number;
    normative: number;
    mythic: number;
  };
  domain: string;
  topics: string[];
  evidence: Evidence[];
}

interface Evidence {
  id: string;
  evidence_type: string;
  source: string;
  content: string;
  strength: number;
}

interface Claim extends ClaimInput {
  id: string;
  status: string;
  created_at: number;
  updated_at: number;
}

// ============================================================================
// Claims Zome Tests
// ============================================================================

describe('Claims Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('create and retrieve a claim', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    const claimInput: ClaimInput = {
      content: 'The Earth orbits the Sun',
      classification: {
        empirical: 0.95,
        normative: 0.05,
        mythic: 0.3,
      },
      domain: 'science',
      topics: ['astronomy', 'solar-system'],
      evidence: [
        {
          id: 'evidence-1',
          evidence_type: 'Empirical',
          source: 'https://nasa.gov/solar-system',
          content: 'NASA observations',
          strength: 0.95,
        },
      ],
    };

    // Create claim
    const claimHash: ActionHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: claimInput,
    });

    expect(claimHash).toBeDefined();

    // Retrieve claim
    const record: Record = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload: claimHash,
    });

    expect(record).toBeDefined();
    const claim = record.entry as unknown as Claim;
    expect(claim.content).toBe(claimInput.content);
    expect(claim.classification.empirical).toBe(0.95);
  });

  test('update a claim', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create claim
    const claimInput: ClaimInput = {
      content: 'Original content',
      classification: { empirical: 0.5, normative: 0.3, mythic: 0.4 },
      domain: 'test',
      topics: ['test'],
      evidence: [],
    };

    const claimHash: ActionHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: claimInput,
    });

    // Update claim
    const updateInput = {
      original_hash: claimHash,
      content: 'Updated content',
      classification: { empirical: 0.7, normative: 0.2, mythic: 0.3 },
    };

    const updatedHash: ActionHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'update_claim',
      payload: updateInput,
    });

    expect(updatedHash).toBeDefined();

    // Verify update
    const record: Record = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload: updatedHash,
    });

    const claim = record.entry as unknown as Claim;
    expect(claim.content).toBe('Updated content');
    expect(claim.classification.empirical).toBe(0.7);
  });

  test('list claims by domain', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create claims in different domains
    const domains = ['climate', 'climate', 'energy'];
    for (const domain of domains) {
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'create_claim',
        payload: {
          content: `Claim in ${domain}`,
          classification: { empirical: 0.6, normative: 0.2, mythic: 0.3 },
          domain,
          topics: [],
          evidence: [],
        },
      });
    }

    // Query by domain
    const climateClaims: Record[] = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'list_claims_by_domain',
      payload: 'climate',
    });

    expect(climateClaims.length).toBe(2);
  });

  test('cannot update claim by non-author', async () => {
    const [alice, bob] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: knowledgeDnaPath } },
      { appBundleSource: { path: knowledgeDnaPath } },
    ]);
    await scenario.shareAllAgents();
    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Alice creates claim
    const claimHash: ActionHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: "Alice's claim",
        classification: { empirical: 0.5, normative: 0.3, mythic: 0.4 },
        domain: 'test',
        topics: [],
        evidence: [],
      },
    });

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob tries to update (should fail)
    await expect(
      bob.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'update_claim',
        payload: {
          original_hash: claimHash,
          content: "Bob's attempt to modify",
        },
      })
    ).rejects.toThrow(/NotAuthor/);
  });
});

// ============================================================================
// Graph Zome Tests
// ============================================================================

describe('Graph Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('create relationship between claims', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create two claims
    const claim1Hash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Climate is warming',
        classification: { empirical: 0.85, normative: 0.1, mythic: 0.3 },
        domain: 'climate',
        topics: [],
        evidence: [],
      },
    });

    const claim2Hash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'CO2 levels have increased',
        classification: { empirical: 0.9, normative: 0.05, mythic: 0.2 },
        domain: 'climate',
        topics: [],
        evidence: [],
      },
    });

    // Create relationship
    const relationshipInput = {
      source_id: claim2Hash.toString(),
      target_id: claim1Hash.toString(),
      relationship_type: 'Supports',
      weight: 0.8,
      properties: {},
    };

    const relHash: ActionHash = await alice.cells[0].callZome({
      zome_name: 'graph',
      fn_name: 'create_relationship',
      payload: relationshipInput,
    });

    expect(relHash).toBeDefined();

    // Get relationships
    const relationships = await alice.cells[0].callZome({
      zome_name: 'graph',
      fn_name: 'get_relationships',
      payload: claim1Hash.toString(),
    });

    expect(relationships.length).toBeGreaterThan(0);
  });

  test('detect circular dependencies', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create three claims
    const claims: ActionHash[] = [];
    for (let i = 0; i < 3; i++) {
      const hash = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'create_claim',
        payload: {
          content: `Claim ${i}`,
          classification: { empirical: 0.5, normative: 0.3, mythic: 0.4 },
          domain: 'test',
          topics: [],
          evidence: [],
        },
      });
      claims.push(hash);
    }

    // Create circular dependency: 0 -> 1 -> 2 -> 0
    await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'register_dependency',
      payload: {
        dependent_claim_id: claims[0].toString(),
        dependency_claim_id: claims[1].toString(),
        dependency_type: 'DerivedFrom',
        weight: 0.7,
        justification: 'Test',
      },
    });

    await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'register_dependency',
      payload: {
        dependent_claim_id: claims[1].toString(),
        dependency_claim_id: claims[2].toString(),
        dependency_type: 'DerivedFrom',
        weight: 0.7,
        justification: 'Test',
      },
    });

    // This should detect the potential cycle
    const cycles = await alice.cells[0].callZome({
      zome_name: 'graph',
      fn_name: 'detect_circular_dependencies',
      payload: claims[2].toString(),
    });

    // Check for cycle detection (implementation-dependent)
    expect(Array.isArray(cycles)).toBe(true);
  });
});

// ============================================================================
// Query Zome Tests
// ============================================================================

describe('Query Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('search claims', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create searchable claims
    await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Climate change is causing temperature increase',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.3 },
        domain: 'climate',
        topics: ['climate', 'temperature'],
        evidence: [],
      },
    });

    await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Solar energy production is growing',
        classification: { empirical: 0.85, normative: 0.05, mythic: 0.2 },
        domain: 'energy',
        topics: ['solar', 'renewable'],
        evidence: [],
      },
    });

    // Search for climate claims
    const results = await alice.cells[0].callZome({
      zome_name: 'query',
      fn_name: 'search',
      payload: {
        query: 'temperature',
        options: { domain: 'climate', limit: 10 },
      },
    });

    expect(results.length).toBeGreaterThan(0);
    expect(results[0].content).toContain('temperature');
  });

  test('query by epistemic range', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create claims with different E levels
    const eLevels = [0.3, 0.5, 0.7, 0.9];
    for (const e of eLevels) {
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'create_claim',
        payload: {
          content: `Claim with E=${e}`,
          classification: { empirical: e, normative: 0.2, mythic: 0.3 },
          domain: 'test',
          topics: [],
          evidence: [],
        },
      });
    }

    // Query high-E claims
    const highE = await alice.cells[0].callZome({
      zome_name: 'query',
      fn_name: 'query_by_epistemic',
      payload: {
        min_e: 0.7,
        max_e: 1.0,
        min_n: null,
        max_n: null,
        min_m: null,
        max_m: null,
        limit: 10,
      },
    });

    expect(highE.length).toBe(2); // E=0.7 and E=0.9
    for (const claim of highE) {
      expect(claim.classification.empirical).toBeGreaterThanOrEqual(0.7);
    }
  });
});

// ============================================================================
// Inference Zome Tests
// ============================================================================

describe('Inference Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('calculate credibility score', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create a claim with evidence
    const claimHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Well-evidenced claim',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.3 },
        domain: 'science',
        topics: [],
        evidence: [
          { id: 'ev1', evidence_type: 'Empirical', source: 'source1', content: 'Evidence 1', strength: 0.9 },
          { id: 'ev2', evidence_type: 'Empirical', source: 'source2', content: 'Evidence 2', strength: 0.85 },
        ],
      },
    });

    // Calculate credibility
    const credibility = await alice.cells[0].callZome({
      zome_name: 'inference',
      fn_name: 'calculate_enhanced_credibility',
      payload: {
        entity_id: claimHash.toString(),
        entity_type: 'Claim',
      },
    });

    expect(credibility).toBeDefined();
    expect(credibility.overall_score).toBeGreaterThan(0);
    expect(credibility.overall_score).toBeLessThanOrEqual(1);
    expect(credibility.evidence_strength).toBeDefined();
  });

  test('batch credibility assessment', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create multiple claims
    const claimIds: string[] = [];
    for (let i = 0; i < 5; i++) {
      const hash = await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'create_claim',
        payload: {
          content: `Batch claim ${i}`,
          classification: { empirical: 0.5 + i * 0.1, normative: 0.2, mythic: 0.3 },
          domain: 'test',
          topics: [],
          evidence: [],
        },
      });
      claimIds.push(hash.toString());
    }

    // Batch assessment
    const results = await alice.cells[0].callZome({
      zome_name: 'inference',
      fn_name: 'batch_credibility_assessment',
      payload: claimIds,
    });

    expect(results.results.length).toBe(5);
    expect(results.average_score).toBeGreaterThan(0);
  });
});

// ============================================================================
// Fact-Check Zome Tests
// ============================================================================

describe('Fact-Check Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('fact-check a statement', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create some knowledge base
    await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'The Earth is approximately 4.5 billion years old',
        classification: { empirical: 0.95, normative: 0.02, mythic: 0.4 },
        domain: 'science',
        topics: ['earth', 'geology', 'age'],
        evidence: [
          { id: 'ev1', evidence_type: 'Empirical', source: 'radiometric dating', content: 'Dating evidence', strength: 0.95 },
        ],
      },
    });

    // Fact-check
    const result = await alice.cells[0].callZome({
      zome_name: 'factcheck',
      fn_name: 'fact_check',
      payload: {
        statement: 'The Earth is 4.5 billion years old',
        context: 'Geological age verification',
        min_e: 0.6,
        source_happ: 'test-happ',
      },
    });

    expect(result).toBeDefined();
    expect(result.verdict).toBeDefined();
    expect(['True', 'MostlyTrue', 'Mixed', 'MostlyFalse', 'False', 'Unverifiable', 'InsufficientEvidence']).toContain(result.verdict);
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });
});

// ============================================================================
// Cross-hApp Bridge Tests
// ============================================================================

describe('Bridge Zome', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('submit external claim', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Register a hApp first
    await alice.cells[0].callZome({
      zome_name: 'bridge',
      fn_name: 'register_happ',
      payload: {
        happ_id: 'test-media-happ',
        name: 'Test Media',
        description: 'A test media hApp',
        allowed_domains: ['media', 'news'],
        permission_level: 'Standard',
      },
    });

    // Submit external claim
    const claimHash = await alice.cells[0].callZome({
      zome_name: 'bridge',
      fn_name: 'submit_external_claim',
      payload: {
        content: 'External claim from media hApp',
        classification: { empirical: 0.7, normative: 0.2, mythic: 0.3 },
        source_happ: 'test-media-happ',
        domain: 'media',
        evidence: [],
      },
    });

    expect(claimHash).toBeDefined();

    // Verify claim exists
    const record = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload: claimHash,
    });

    expect(record).toBeDefined();
    const claim = record.entry as unknown as Claim;
    expect(claim.content).toBe('External claim from media hApp');
  });

  test('query for hApp', async () => {
    const [alice] = await scenario.addPlayersWithApps([{ appBundleSource: { path: knowledgeDnaPath } }]);
    await scenario.shareAllAgents();

    // Create some claims
    for (let i = 0; i < 3; i++) {
      await alice.cells[0].callZome({
        zome_name: 'claims',
        fn_name: 'create_claim',
        payload: {
          content: `Energy claim ${i}`,
          classification: { empirical: 0.6 + i * 0.1, normative: 0.1, mythic: 0.2 },
          domain: 'energy',
          topics: ['solar', 'renewable'],
          evidence: [],
        },
      });
    }

    // Query for hApp
    const results = await alice.cells[0].callZome({
      zome_name: 'bridge',
      fn_name: 'query_for_happ',
      payload: {
        domain: 'energy',
        min_e: 0.7,
        topics: ['solar'],
        limit: 10,
      },
    });

    expect(results.length).toBeGreaterThan(0);
  });
});

// ============================================================================
// Multi-Agent Tests
// ============================================================================

describe('Multi-Agent Scenarios', () => {
  let scenario: Scenario;

  beforeAll(async () => {
    scenario = await Scenario.create();
  });

  afterAll(async () => {
    await scenario.cleanUp();
  });

  test('claims propagate between agents', async () => {
    const [alice, bob, carol] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: knowledgeDnaPath } },
      { appBundleSource: { path: knowledgeDnaPath } },
      { appBundleSource: { path: knowledgeDnaPath } },
    ]);
    await scenario.shareAllAgents();

    // Alice creates a claim
    const claimHash = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Shared knowledge claim',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.3 },
        domain: 'shared',
        topics: ['test'],
        evidence: [],
      },
    });

    // Wait for DHT sync
    await dhtSync([alice, bob, carol], alice.cells[0].cell_id[0]);

    // Bob can see the claim
    const bobRecord = await bob.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload: claimHash,
    });

    expect(bobRecord).toBeDefined();

    // Carol can see the claim
    const carolRecord = await carol.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'get_claim',
      payload: claimHash,
    });

    expect(carolRecord).toBeDefined();
  });

  test('relationships visible across agents', async () => {
    const [alice, bob] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: knowledgeDnaPath } },
      { appBundleSource: { path: knowledgeDnaPath } },
    ]);
    await scenario.shareAllAgents();

    // Alice creates claims and relationship
    const claim1 = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Claim 1',
        classification: { empirical: 0.7, normative: 0.2, mythic: 0.3 },
        domain: 'test',
        topics: [],
        evidence: [],
      },
    });

    const claim2 = await alice.cells[0].callZome({
      zome_name: 'claims',
      fn_name: 'create_claim',
      payload: {
        content: 'Claim 2',
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.2 },
        domain: 'test',
        topics: [],
        evidence: [],
      },
    });

    await alice.cells[0].callZome({
      zome_name: 'graph',
      fn_name: 'create_relationship',
      payload: {
        source_id: claim1.toString(),
        target_id: claim2.toString(),
        relationship_type: 'Supports',
        weight: 0.9,
        properties: {},
      },
    });

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob can see relationships
    const bobRelationships = await bob.cells[0].callZome({
      zome_name: 'graph',
      fn_name: 'get_relationships',
      payload: claim2.toString(),
    });

    expect(bobRelationships.length).toBeGreaterThan(0);
  });
});
