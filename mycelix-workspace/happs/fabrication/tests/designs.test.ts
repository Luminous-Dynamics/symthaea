import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { runScenario, Scenario, Player } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Path to the hApp bundle (adjust after packaging)
const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Designs Zome', () => {
  it('should create and retrieve a design', async () => {
    await runScenario(async (scenario: Scenario) => {
      // Set up the player with the fabrication hApp
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create a design
      const createInput = {
        title: 'Test Bracket',
        description: 'A simple bracket for testing',
        category: 'Parts',
        intent_vector: null,
        parametric_schema: null,
        constraint_graph: null,
        material_compatibility: [],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: null,
        license: 'PublicDomain',
        safety_class: 'Class1Functional',
      };

      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: createInput,
      });

      expect(designRecord).toBeDefined();
      expect(designRecord.signed_action).toBeDefined();

      // Retrieve the design
      const actionHash: ActionHash = designRecord.signed_action.hashed.hash;
      const retrievedRecord: Record | null = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'get_design',
        payload: actionHash,
      });

      expect(retrievedRecord).toBeDefined();
      expect(retrievedRecord?.entry).toBeDefined();
    });
  });

  it('should list designs by category', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create designs in different categories
      const toolDesign = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Wrench Holder',
          description: 'Tool organizer',
          category: 'Tools',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class0Decorative',
        },
      });

      const partDesign = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Gear',
          description: 'Replacement gear',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class1Functional',
        },
      });

      // Query by category
      const toolDesigns = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'get_designs_by_category',
        payload: { category: 'Tools', pagination: null },
      });

      expect(toolDesigns.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should fork a design', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Alice creates a design
      const originalDesign: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Original Design',
          description: 'The original',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: { CreativeCommons: 'BY' },
          safety_class: 'Class1Functional',
        },
      });

      const originalHash = originalDesign.signed_action.hashed.hash;

      // Bob forks it
      const forkedDesign: Record = await bob.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'fork_design',
        payload: {
          parent_hash: originalHash,
          modification_notes: 'Improved version with better tolerances',
          title: 'Forked Design',
          description: 'My improved version',
          intent_modifications: null,
        },
      });

      expect(forkedDesign).toBeDefined();

      // Check fork lineage
      const forks = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'get_design_forks',
        payload: { hash: originalHash, pagination: null },
      });

      expect(forks.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should search designs', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create a design with specific keywords
      await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Raspberry Pi Case',
          description: 'A protective case for Raspberry Pi 4',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class0Decorative',
        },
      });

      // Search for it
      const results = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'search_designs',
        payload: {
          query: 'raspberry',
          category: null,
          safety_class: null,
          min_circularity: null,
          license: null,
          limit: 10,
          pagination: null,
        },
      });

      expect(results.items.length).toBeGreaterThanOrEqual(1);
    });
  });
});
