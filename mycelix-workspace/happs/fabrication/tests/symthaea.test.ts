// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect } from 'vitest';
import { runScenario, Scenario } from '@holochain/tryorama';
import { Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Symthaea Zome - HDC Operations', () => {
  it('should generate intent vector from natural language', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Generate intent vector
      const result = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'generate_intent_vector',
        payload: {
          description: 'I need a bracket for a 12mm pipe that is weatherproof',
          language: 'en',
        },
      });

      expect(result).toBeDefined();
      expect(result.record).toBeDefined();
      expect(result.bindings).toBeDefined();
      expect(result.vector_hash).toBeDefined();
      expect(result.vector_hash.length).toBeGreaterThan(0);
    });
  });

  it('should perform lateral binding of two intents', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create base intent
      const base = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'generate_intent_vector',
        payload: {
          description: 'A bracket for mounting',
          language: null,
        },
      });

      const baseHash = base.record.signed_action.hashed.hash;

      // Lateral bind with modifier
      const bound = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'lateral_bind',
        payload: {
          base_intent_hash: baseHash,
          modifier_descriptions: ['weatherproof', 'stainless steel'],
        },
      });

      expect(bound).toBeDefined();
      expect(bound.record).toBeDefined();
      expect(bound.vector_hash).not.toEqual(base.vector_hash);
    });
  });

  it('should search intents by semantic similarity', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create two similar intents
      const intent1 = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'generate_intent_vector',
        payload: {
          description: 'A cup holder for a desk',
          language: null,
        },
      });

      await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'generate_intent_vector',
        payload: {
          description: 'A mug stand for an office',
          language: null,
        },
      });

      // Search from intent1
      const results = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'semantic_search',
        payload: {
          intent_hash: intent1.record.signed_action.hashed.hash,
          threshold: 0.1,
          limit: 10,
          record_matches: false,
        },
      });

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });
  });

  it('should perform LSH-accelerated search', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create some intents first
      for (const desc of [
        'A bracket for pipes',
        'A shelf support',
        'A wall mount for monitors',
      ]) {
        await alice.cells[0].callZome({
          zome_name: 'symthaea',
          fn_name: 'generate_intent_vector',
          payload: { description: desc, language: null },
        });
      }

      // LSH search
      const results = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'semantic_search_lsh',
        payload: {
          query_description: 'A mounting bracket',
          threshold: 0.1,
          limit: 5,
        },
      });

      expect(results).toBeDefined();
      expect(Array.isArray(results)).toBe(true);
    });
  });

  it('should get my intents', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create an intent
      await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'generate_intent_vector',
        payload: {
          description: 'A phone stand',
          language: null,
        },
      });

      // Get my intents
      const result = await alice.cells[0].callZome({
        zome_name: 'symthaea',
        fn_name: 'get_my_intents',
        payload: { pagination: null },
      });

      expect(result.total).toBeGreaterThanOrEqual(1);
      expect(result.items.length).toBeGreaterThanOrEqual(1);
    });
  });
});
