// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Graph Integration Tests
 *
 * Tests the knowledge graph zome:
 * - Create claims + relationships (Supports, Contradicts)
 * - Path finding between connected claims
 * - Belief propagation with contradictions
 * - Circular dependency detection
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

interface Relationship {
  id: string;
  source: string;
  target: string;
  relationship_type: string | { Custom: string };
  weight: number;
  properties: string | null;
  creator: string;
  created: number;
}

interface FindPathInput {
  source: string;
  target: string;
  max_depth: number;
}

interface DependencyTreeInput {
  claim_id: string;
  max_depth: number;
}

interface DependencyTree {
  root_claim_id: string;
  nodes: Array<{
    claim_id: string;
    depth: number;
    weight: number;
    children: string[];
    is_leaf: boolean;
  }>;
  depth: number;
  total_dependencies: number;
  aggregate_weight: number;
}

interface PropagationResult {
  source_claim_id: string;
  nodes_affected: number;
  iterations: number;
  converged: boolean;
  max_delta: number;
  processing_time_ms: number;
  updated_nodes: string[];
}

interface CascadeImpact {
  claim_id: string;
  total_affected: number;
  affected_by_depth: number[];
  max_depth: number;
  impact_score: number;
  high_impact_claims: string[];
  assessed_at: number;
}

// Helper: create a claim via the claims zome
async function createClaim(cell: any, id: string, content: string, tags: string[] = []) {
  const now = Date.now() * 1000;
  return cell.callZome({
    zome_name: 'claims',
    fn_name: 'submit_claim',
    payload: {
      id,
      content,
      classification: { empirical: 0.5, normative: 0.5, mythic: 0.5 },
      author: 'did:key:testuser',
      sources: [],
      tags,
      claim_type: 'Fact',
      confidence: 0.5,
      expires: null,
      created: now,
      updated: now,
      version: 1,
    } as Claim,
  });
}

// ============================================================================
// Graph Integration Tests
// ============================================================================

describe('Graph Integration Tests', () => {
  jest.setTimeout(600000);

  test('Create claims and Supports relationship', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create two claims via the claims zome
      await createClaim(alice.cells[0], 'gravity_claim', 'Objects fall due to gravity');
      await createClaim(alice.cells[0], 'apple_claim', 'Apples fall from trees');

      // Create a Supports relationship via graph zome
      const now = Date.now() * 1000;
      const relationship: Relationship = {
        id: 'rel_001',
        source: 'apple_claim',
        target: 'gravity_claim',
        relationship_type: 'Supports',
        weight: 0.8,
        properties: null,
        creator: 'did:key:testuser',
        created: now,
      };

      const relRecord = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: relationship,
      });

      expect(relRecord).toBeDefined();

      // Get outgoing relationships from apple_claim
      const outgoing = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_outgoing_relationships',
        payload: 'apple_claim',
      }) as any[];

      expect(outgoing.length).toBe(1);

      // Get incoming relationships to gravity_claim
      const incoming = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_incoming_relationships',
        payload: 'gravity_claim',
      }) as any[];

      expect(incoming.length).toBe(1);
    }, true, scenarioOptions);
  });

  test('Create Contradicts relationship', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await createClaim(alice.cells[0], 'flat_earth', 'The Earth is flat');
      await createClaim(alice.cells[0], 'round_earth', 'The Earth is roughly spherical');

      const now = Date.now() * 1000;
      const contradiction: Relationship = {
        id: 'rel_contradiction',
        source: 'round_earth',
        target: 'flat_earth',
        relationship_type: 'Contradicts',
        weight: 0.95,
        properties: null,
        creator: 'did:key:testuser',
        created: now,
      };

      const record = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: contradiction,
      });

      expect(record).toBeDefined();
    }, true, scenarioOptions);
  });

  test('Path finding between connected claims', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create a chain: A -> B -> C
      await createClaim(alice.cells[0], 'path_A', 'Claim A');
      await createClaim(alice.cells[0], 'path_B', 'Claim B');
      await createClaim(alice.cells[0], 'path_C', 'Claim C');

      const now = Date.now() * 1000;

      // A -> B
      await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: {
          id: 'rel_ab',
          source: 'path_A',
          target: 'path_B',
          relationship_type: 'DerivedFrom',
          weight: 0.7,
          properties: null,
          creator: 'did:key:testuser',
          created: now,
        } as Relationship,
      });

      // B -> C
      await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: {
          id: 'rel_bc',
          source: 'path_B',
          target: 'path_C',
          relationship_type: 'DerivedFrom',
          weight: 0.6,
          properties: null,
          creator: 'did:key:testuser',
          created: now,
        } as Relationship,
      });

      // Find path from A to C
      const pathResult: string[] = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'find_path',
        payload: {
          source: 'path_A',
          target: 'path_C',
          max_depth: 5,
        } as FindPathInput,
      });

      expect(pathResult.length).toBeGreaterThan(0);
      expect(pathResult[0]).toBe('path_A');
      expect(pathResult[pathResult.length - 1]).toBe('path_C');

      console.log(`Path found: ${pathResult.join(' -> ')}`);
    }, true, scenarioOptions);
  });

  test('No path returns empty array', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await createClaim(alice.cells[0], 'isolated_A', 'Isolated claim A');
      await createClaim(alice.cells[0], 'isolated_B', 'Isolated claim B');

      // No relationships between them
      const pathResult: string[] = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'find_path',
        payload: {
          source: 'isolated_A',
          target: 'isolated_B',
          max_depth: 5,
        } as FindPathInput,
      });

      expect(pathResult.length).toBe(0);
    }, true, scenarioOptions);
  });

  test('Belief propagation through Supports chain', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await createClaim(alice.cells[0], 'belief_source', 'Strong source claim');
      await createClaim(alice.cells[0], 'belief_target', 'Target to be updated');

      const now = Date.now() * 1000;
      await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: {
          id: 'rel_belief',
          source: 'belief_source',
          target: 'belief_target',
          relationship_type: 'Supports',
          weight: 0.9,
          properties: null,
          creator: 'did:key:testuser',
          created: now,
        } as Relationship,
      });

      // Propagate belief from source
      const result: PropagationResult = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'propagate_belief',
        payload: 'belief_source',
      });

      expect(result).toBeDefined();
      expect(result.source_claim_id).toBe('belief_source');
      expect(result.nodes_affected).toBeGreaterThanOrEqual(1);

      console.log(
        `Belief propagation: ${result.nodes_affected} nodes affected, ` +
          `${result.iterations} iterations, converged=${result.converged}, ` +
          `max_delta=${result.max_delta.toFixed(4)}`,
      );
    }, true, scenarioOptions);
  });

  test('Circular dependency detection', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create a cycle: X -> Y -> Z -> X
      await createClaim(alice.cells[0], 'cycle_X', 'Cycle X');
      await createClaim(alice.cells[0], 'cycle_Y', 'Cycle Y');
      await createClaim(alice.cells[0], 'cycle_Z', 'Cycle Z');

      const now = Date.now() * 1000;

      for (const [src, tgt, id] of [
        ['cycle_X', 'cycle_Y', 'rel_xy'],
        ['cycle_Y', 'cycle_Z', 'rel_yz'],
        ['cycle_Z', 'cycle_X', 'rel_zx'],
      ]) {
        await alice.cells[0].callZome({
          zome_name: 'graph',
          fn_name: 'create_relationship',
          payload: {
            id,
            source: src,
            target: tgt,
            relationship_type: 'RelatedTo',
            weight: 0.5,
            properties: null,
            creator: 'did:key:testuser',
            created: now,
          } as Relationship,
        });
      }

      // Detect cycles
      const cycles: string[][] = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'detect_circular_dependencies',
        payload: 'cycle_X',
      });

      expect(cycles.length).toBeGreaterThan(0);
      console.log(`Cycles detected: ${cycles.length}`);
    }, true, scenarioOptions);
  });

  test('Dependency tree construction', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      // Create tree: root -> child1, root -> child2, child1 -> grandchild
      await createClaim(alice.cells[0], 'tree_root', 'Root claim');
      await createClaim(alice.cells[0], 'tree_child1', 'Child 1');
      await createClaim(alice.cells[0], 'tree_child2', 'Child 2');
      await createClaim(alice.cells[0], 'tree_grandchild', 'Grandchild');

      const now = Date.now() * 1000;

      for (const [src, tgt, id] of [
        ['tree_root', 'tree_child1', 'rel_rc1'],
        ['tree_root', 'tree_child2', 'rel_rc2'],
        ['tree_child1', 'tree_grandchild', 'rel_c1g'],
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
            creator: 'did:key:testuser',
            created: now,
          } as Relationship,
        });
      }

      // Get dependency tree
      const tree: DependencyTree = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_dependency_tree',
        payload: {
          claim_id: 'tree_root',
          max_depth: 5,
        } as DependencyTreeInput,
      });

      expect(tree.root_claim_id).toBe('tree_root');
      expect(tree.nodes.length).toBeGreaterThanOrEqual(3);
      expect(tree.depth).toBeGreaterThanOrEqual(1);

      console.log(
        `Dependency tree: ${tree.nodes.length} nodes, depth=${tree.depth}, ` +
          `total_deps=${tree.total_dependencies}`,
      );
    }, true, scenarioOptions);
  });

  test('Cascade impact assessment', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await createClaim(alice.cells[0], 'impact_root', 'High-impact claim');
      await createClaim(alice.cells[0], 'impact_dep1', 'Dependent 1');
      await createClaim(alice.cells[0], 'impact_dep2', 'Dependent 2');

      const now = Date.now() * 1000;

      // dep1 and dep2 depend on root (incoming to root)
      for (const [src, tgt, id] of [
        ['impact_dep1', 'impact_root', 'rel_d1r'],
        ['impact_dep2', 'impact_root', 'rel_d2r'],
      ]) {
        await alice.cells[0].callZome({
          zome_name: 'graph',
          fn_name: 'create_relationship',
          payload: {
            id,
            source: src,
            target: tgt,
            relationship_type: 'DerivedFrom',
            weight: 0.85,
            properties: null,
            creator: 'did:key:testuser',
            created: now,
          } as Relationship,
        });
      }

      const impact: CascadeImpact = await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'calculate_cascade_impact',
        payload: 'impact_root',
      });

      expect(impact.claim_id).toBe('impact_root');
      expect(impact.impact_score).toBeGreaterThanOrEqual(0);

      console.log(
        `Cascade impact: ${impact.total_affected} affected, score=${impact.impact_score.toFixed(4)}, ` +
          `max_depth=${impact.max_depth}`,
      );
    }, true, scenarioOptions);
  });

  test('Multi-agent relationship visibility', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
        { appBundleSource: { type: 'path', value: knowledgeHappPath } },
      ]);

      await scenario.shareAllAgents();
      const dnaHash = alice.cells[0].cell_id[0];

      // Alice creates claims and relationship
      await createClaim(alice.cells[0], 'multi_claim1', 'Multi agent claim 1');
      await createClaim(alice.cells[0], 'multi_claim2', 'Multi agent claim 2');

      const now = Date.now() * 1000;
      await alice.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'create_relationship',
        payload: {
          id: 'rel_multi',
          source: 'multi_claim1',
          target: 'multi_claim2',
          relationship_type: 'Supports',
          weight: 0.7,
          properties: null,
          creator: 'did:key:alice',
          created: now,
        } as Relationship,
      });

      await dhtSync([alice, bob], dnaHash);

      // Bob should see the relationship
      const outgoing = await bob.cells[0].callZome({
        zome_name: 'graph',
        fn_name: 'get_outgoing_relationships',
        payload: 'multi_claim1',
      }) as any[];

      expect(outgoing.length).toBe(1);
    }, true, scenarioOptions);
  });
});
