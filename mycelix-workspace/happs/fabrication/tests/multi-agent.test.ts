// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { describe, it, expect } from 'vitest';
import { runScenario, Scenario } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Multi-Agent Fabrication Workflow', () => {
  it('should allow designer to create design visible to printer operator', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [designer, printerOp] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Designer creates a design
      const designRecord: Record = await designer.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Replacement Bracket',
          description: 'Load-bearing bracket for shelf mount',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: { CreativeCommons: 'BY' },
          safety_class: 'Class2LoadBearing',
        },
      });

      const designHash: ActionHash = designRecord.signed_action.hashed.hash;

      // Printer operator can retrieve the design via DHT
      // (after a brief propagation delay in real conductor)
      const retrieved: Record | null = await printerOp.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'get_design',
        payload: designHash,
      });

      expect(retrieved).toBeDefined();
      expect(retrieved?.signed_action.hashed.hash).toEqual(designHash);
    });
  });

  it('should support full designer → operator → requester handoff', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [designer, printerOp, requester] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // 1. Designer creates a design
      const designRecord: Record = await designer.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Trigger Switch Housing',
          description: 'Replacement housing for Makita LXT trigger mechanism',
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

      const designHash = designRecord.signed_action.hashed.hash;

      // 2. Printer operator registers a printer
      const printerRecord: Record = await printerOp.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Workshop Printer Alpha',
          location: null,
          printer_type: 'FDM',
          capabilities: {
            build_volume: { x: 250, y: 250, z: 300 },
            layer_heights: [0.1, 0.2, 0.3],
            nozzle_diameters: [0.4],
            heated_bed: true,
            enclosure: true,
            multi_material: null,
            max_temp_hotend: 300,
            max_temp_bed: 110,
            features: ['AutoLeveling'],
          },
          materials_available: ['PLA', 'PETG', 'ABS'],
          rates: {
            hourly_rate: 150,
            material_rate: 5,
            currency: 'USD',
            minimum_order: null,
          },
        },
      });

      const printerHash = printerRecord.signed_action.hashed.hash;

      // 3. Requester creates a print job
      const jobRecord: Record = await requester.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'create_print_job',
        payload: {
          design_hash: designHash,
          printer_hash: printerHash,
          settings: {
            layer_height: 0.2,
            infill_percent: 60,
            material: 'PETG',
            supports: true,
            raft: false,
            print_speed: null,
            temperatures: { hotend: 240, bed: 80, chamber: null },
            custom_gcode: null,
          },
          energy_source: 'GridMix',
          material_passport: null,
        },
      });

      expect(jobRecord).toBeDefined();
      const jobHash = jobRecord.signed_action.hashed.hash;

      // 4. Printer operator can see their jobs
      const myJobs = await printerOp.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'get_printer_jobs',
        payload: { hash: printerHash, pagination: null },
      });

      expect(myJobs.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should propagate repair prediction across agents', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [sensorAgent, repairCoord] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Sensor agent creates repair prediction (simulating IoT digital twin)
      const predictionRecord: Record = await sensorAgent.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'create_repair_prediction',
        payload: {
          property_asset_hash: new Uint8Array(39).fill(42),
          asset_model: 'DeWalt DCD791',
          predicted_failure_component: 'Chuck mechanism',
          failure_probability: 0.88,
          estimated_failure_date: Date.now() * 1000 + 14 * 24 * 60 * 60 * 1000000,
          confidence_interval_days: 5,
          sensor_data_summary: JSON.stringify({
            vibration_rms: 8.5,
            vibration_trend_slope: 1.2,
            temp_max: 72,
            torque_variance: 0.35,
            degradation_score: 0.78,
          }),
        },
      });

      expect(predictionRecord).toBeDefined();

      // Repair coordinator can see active workflows created from high-prob prediction
      const workflows = await repairCoord.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'get_active_workflows',
        payload: { pagination: { offset: 0, limit: 100 } },
      });

      // Auto-created workflow from prediction > 0.7
      expect(workflows.items.length).toBeGreaterThanOrEqual(1);
    });
  });

  it('should allow verification by a different agent than the designer', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [designer, verifier] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Designer creates design
      const designRecord: Record = await designer.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Safety Bracket',
          description: 'Critical load path bracket',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class5Critical',
        },
      });

      const designHash = designRecord.signed_action.hashed.hash;

      // Verifier submits verification
      const verificationRecord: Record = await verifier.cells[0].callZome({
        zome_name: 'verification',
        fn_name: 'submit_verification',
        payload: {
          design_hash: designHash,
          verification_type: 'StructuralAnalysis',
          result: {
            Passed: {
              confidence: 0.92,
              notes: 'FEA analysis confirms safety factor > 2.5',
            },
          },
          evidence: [],
          credentials: ['PE_Mechanical', 'ASTM_F2792'],
        },
      });

      expect(verificationRecord).toBeDefined();
    });
  });

  it('should support multi-agent material verification chain', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [materialSupplier, fabricator] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Supplier creates material entry
      const materialRecord: Record = await materialSupplier.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'create_material',
        payload: {
          name: 'Ocean-Recycled PETG',
          material_type: 'PETG',
          properties: {
            print_temp_min: 230,
            print_temp_max: 250,
            bed_temp_min: 70,
            bed_temp_max: 85,
            density_g_cm3: 1.27,
            tensile_strength_mpa: 48.0,
            elongation_percent: 20.0,
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

      // Fabricator can see the material
      const retrieved: Record | null = await fabricator.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'get_material',
        payload: materialHash,
      });

      expect(retrieved).toBeDefined();

      // Fabricator links material to their supply chain
      const linkRecord: Record = await fabricator.cells[0].callZome({
        zome_name: 'bridge',
        fn_name: 'link_material_to_supplier',
        payload: {
          material_hash: materialHash,
          supplier_did: 'did:web:ocean-plastics.example.com',
          supplychain_item_hash: null,
        },
      });

      expect(linkRecord).toBeDefined();
    });
  });
});
