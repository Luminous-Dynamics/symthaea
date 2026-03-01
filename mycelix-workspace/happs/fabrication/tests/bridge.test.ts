import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { runScenario, Scenario, Player } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Bridge Zome - Anticipatory Repair', () => {
  it('should create a repair prediction from property asset', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Simulate a repair prediction from Property hApp
      // In real use, this would come from a digital twin sensor analysis
      const predictionRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_prediction',
        payload: {
          property_asset_hash: new Uint8Array(39).fill(1), // Simulated hash
          asset_model: 'Bosch GSR 18V-60',
          predicted_failure_component: 'Battery clip',
          failure_probability: 0.85,
          estimated_failure_date: Date.now() * 1000 + 30 * 24 * 60 * 60 * 1000000, // 30 days from now
          confidence_interval_days: 7,
          sensor_data_summary: JSON.stringify({
            vibration_increase: 0.15,
            stress_cycles: 12500,
            material_fatigue_index: 0.72,
          }),
        },
      });

      expect(predictionRecord).toBeDefined();
      expect(predictionRecord.signed_action).toBeDefined();
    });
  });

  it('should create repair workflow from high-probability prediction', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create high-probability prediction (auto-creates workflow)
      const predictionRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_prediction',
        payload: {
          property_asset_hash: new Uint8Array(39).fill(2),
          asset_model: 'DeWalt DCD791',
          predicted_failure_component: 'Chuck mechanism',
          failure_probability: 0.92, // High enough to auto-create workflow
          estimated_failure_date: Date.now() * 1000 + 14 * 24 * 60 * 60 * 1000000,
          confidence_interval_days: 3,
          sensor_data_summary: '{"torque_variance": 0.25}',
        },
      });

      const predictionHash = predictionRecord.signed_action.hashed.hash;

      // Get active workflows - should include auto-created one
      const workflows = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'get_active_workflows',
        payload: { pagination: { offset: 0, limit: 100 } },
      });

      expect(workflows.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should update repair workflow through full lifecycle', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create prediction
      const predictionRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_prediction',
        payload: {
          property_asset_hash: new Uint8Array(39).fill(3),
          asset_model: 'Makita LXT',
          predicted_failure_component: 'Trigger switch',
          failure_probability: 0.75,
          estimated_failure_date: Date.now() * 1000 + 45 * 24 * 60 * 60 * 1000000,
          confidence_interval_days: 10,
          sensor_data_summary: '{}',
        },
      });

      const predictionHash = predictionRecord.signed_action.hashed.hash;

      // Create workflow manually
      const workflowRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_workflow',
        payload: predictionHash,
      });

      const workflowHash = workflowRecord.signed_action.hashed.hash;

      // Create a design for the repair part
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Makita Trigger Switch Replacement',
          description: 'Replacement trigger mechanism',
          category: 'Repair',
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

      // Update workflow: design found
      const updatedWorkflow1: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'update_repair_workflow',
        payload: {
          workflow_hash: workflowHash,
          status: 'DesignFound',
          design_hash: designRecord.signed_action.hashed.hash,
          printer_hash: null,
          hearth_funding_hash: null,
          print_job_hash: null,
        },
      });

      expect(updatedWorkflow1).toBeDefined();

      // Register a printer
      const printerRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Community Printer',
          location: null,
          printer_type: 'FDM',
          capabilities: {
            build_volume: { x: 200, y: 200, z: 200 },
            layer_heights: [0.2],
            nozzle_diameters: [0.4],
            heated_bed: true,
            enclosure: false,
            multi_material: null,
            max_temp_hotend: 260,
            max_temp_bed: 100,
            features: [],
          },
          materials_available: ['PETG'],
          rates: null,
        },
      });

      // Update workflow: printer matched
      const updatedWorkflow2: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'update_repair_workflow',
        payload: {
          workflow_hash: workflowHash,
          status: 'PrinterMatched',
          design_hash: designRecord.signed_action.hashed.hash,
          printer_hash: printerRecord.signed_action.hashed.hash,
          hearth_funding_hash: null,
          print_job_hash: null,
        },
      });

      expect(updatedWorkflow2).toBeDefined();

      // Update workflow: installed (complete)
      const completedWorkflow: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'update_repair_workflow',
        payload: {
          workflow_hash: workflowHash,
          status: 'Installed',
          design_hash: designRecord.signed_action.hashed.hash,
          printer_hash: printerRecord.signed_action.hashed.hash,
          hearth_funding_hash: null,
          print_job_hash: null,
        },
      });

      expect(completedWorkflow).toBeDefined();

      // Verify it's no longer in active workflows
      const activeWorkflows = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'get_active_workflows',
        payload: { pagination: { offset: 0, limit: 100 } },
      });

      const activeHashes = activeWorkflows.items.map((r: Record) => r.signed_action.hashed.hash);
      expect(activeHashes).not.toContain(workflowHash);
    });
  });

  it('should emit and retrieve fabrication events', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Emit design published event
      const eventRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'emit_fabrication_event',
        payload: {
          event_type: 'DesignPublished',
          design_id: null,
          payload: JSON.stringify({
            title: 'New Design',
            category: 'Tools',
          }),
        },
      });

      expect(eventRecord).toBeDefined();

      // Get recent events
      const events = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'get_recent_events',
        payload: { since: null, pagination: { offset: 0, limit: 100 } },
      });

      expect(events.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should list design on marketplace', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create a design
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Premium Widget',
          description: 'High quality widget for sale',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'Proprietary',
          safety_class: 'Class1Functional',
        },
      });

      const designHash = designRecord.signed_action.hashed.hash;

      // List on marketplace
      const listingRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'list_design_on_marketplace',
        payload: {
          design_hash: designHash,
          price: 500, // 5.00 in cents
          listing_type: 'DesignSale',
        },
      });

      expect(listingRecord).toBeDefined();
    });
  });

  it('should link material to supplier', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create a material
      const materialRecord: Record = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'create_material',
        payload: {
          name: 'Recycled PETG',
          material_type: 'PETG',
          properties: {
            print_temp_min: 230,
            print_temp_max: 250,
            bed_temp_min: 70,
            bed_temp_max: 85,
            density_g_cm3: 1.27,
            tensile_strength_mpa: 50.0,
            elongation_percent: 23.0,
            food_safe: false,
            uv_resistant: true,
            water_resistant: true,
            recyclable: true,
          },
          certifications: [],
          safety_data_sheet: null,
        },
      });

      const materialHash = materialRecord.signed_action.hashed.hash;

      // Link to supplier via bridge
      const linkRecord: Record = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'link_material_to_supplier',
        payload: {
          material_hash: materialHash,
          supplier_did: 'did:web:ecoplastics.example.com',
          supplychain_item_hash: null,
        },
      });

      expect(linkRecord).toBeDefined();
    });
  });

  it('should log audit trail entries for state-changing operations', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create a prediction (triggers audit logging)
      await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_prediction',
        payload: {
          property_asset_hash: new Uint8Array(39).fill(10),
          asset_model: 'Audit-Test-Model',
          predicted_failure_component: 'test-component',
          failure_probability: 0.5,
          estimated_failure_date: Date.now() * 1000 + 30 * 24 * 60 * 60 * 1000000,
          confidence_interval_days: 7,
          sensor_data_summary: '{"test": true}',
        },
      });

      // Query audit trail — should have at least 1 entry
      const auditTrail = await alice.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'get_audit_trail',
        payload: {
          domain: 'Bridge',
          agent: null,
          after: null,
          before: null,
          limit: 50,
          pagination: null,
        },
      });

      expect(auditTrail).toBeDefined();
      expect(auditTrail.items.length).toBeGreaterThanOrEqual(1);
    });
  });
});
