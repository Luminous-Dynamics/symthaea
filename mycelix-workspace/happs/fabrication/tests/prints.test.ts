import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { runScenario, Scenario, Player } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Prints Zome', () => {
  it('should create and manage a print job lifecycle', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Alice creates a design
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Widget',
          description: 'A useful widget',
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
      const designHash = designRecord.signed_action.hashed.hash;

      // Bob registers a printer
      const printerRecord: Record = await bob.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Bob Print Service',
          location: null,
          printer_type: 'FDM',
          capabilities: {
            build_volume: { x: 250, y: 210, z: 210 },
            layer_heights: [0.1, 0.2],
            nozzle_diameters: [0.4],
            heated_bed: true,
            enclosure: false,
            multi_material: null,
            max_temp_hotend: 280,
            max_temp_bed: 110,
            features: [],
          },
          materials_available: ['PLA', 'PETG'],
          rates: { hourly_rate: 2, material_rate: 0.05, currency: 'USD', minimum_order: null },
        },
      });
      const printerHash = printerRecord.signed_action.hashed.hash;

      // Alice creates a print job
      const jobRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'create_print_job',
        payload: {
          design_hash: designHash,
          printer_hash: printerHash,
          settings: {
            layer_height: 0.2,
            infill_percent: 20,
            material: 'PLA',
            supports: false,
            raft: false,
            print_speed: null,
            temperatures: {
              hotend: 210,
              bed: 60,
              chamber: null,
            },
            custom_gcode: null,
          },
          energy_source: null,
          material_passport: null,
        },
      });

      expect(jobRecord).toBeDefined();
      const jobHash = jobRecord.signed_action.hashed.hash;

      // Bob accepts the job
      const acceptedJob: Record = await bob.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'accept_print_job',
        payload: jobHash,
      });

      expect(acceptedJob).toBeDefined();

      // Bob starts printing
      const startedJob: Record = await bob.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'start_print',
        payload: jobHash,
      });

      expect(startedJob).toBeDefined();

      // Bob completes the print
      const completedRecord: Record = await bob.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'complete_print',
        payload: {
          job_hash: jobHash,
          result: 'Success',
        },
      });

      expect(completedRecord).toBeDefined();
    });
  });

  it('should record PoGF metrics for sustainable prints', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create design and printer (abbreviated)
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Eco Part',
          description: 'Sustainable part',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.8,
          embodied_energy_kwh: 0.5,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class1Functional',
        },
      });

      const printerRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Solar Printer',
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
          materials_available: ['PLA'],
          rates: null,
        },
      });

      // Create job with renewable energy source
      const jobRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'create_print_job',
        payload: {
          design_hash: designRecord.signed_action.hashed.hash,
          printer_hash: printerRecord.signed_action.hashed.hash,
          settings: {
            layer_height: 0.2,
            infill_percent: 15,
            material: 'PLA',
            supports: false,
            raft: false,
            print_speed: null,
            temperatures: { hotend: 200, bed: 55, chamber: null },
            custom_gcode: null,
          },
          energy_source: 'Solar',
          material_passport: null,
        },
      });

      expect(jobRecord).toBeDefined();

      // Record print with PoGF metrics
      const printRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'record_print_result',
        payload: {
          job_hash: jobRecord.signed_action.hashed.hash,
          result: 'Success',
          energy_used_kwh: 0.15,
          photos: [],
          notes: 'Solar powered, recycled PLA',
          issues: [],
          dimensional_measurements: null,
          attestations: null,
        },
      });

      expect(printRecord).toBeDefined();
    });
  });

  it('should track Cincinnati monitoring for quality assurance', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create design and printer (abbreviated)
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Precision Part',
          description: 'Requires quality monitoring',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class2LoadBearing',
        },
      });

      const printerRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Monitored Printer',
          location: null,
          printer_type: 'FDM',
          capabilities: {
            build_volume: { x: 200, y: 200, z: 200 },
            layer_heights: [0.1, 0.2],
            nozzle_diameters: [0.4],
            heated_bed: true,
            enclosure: true,
            multi_material: null,
            max_temp_hotend: 280,
            max_temp_bed: 110,
            features: [{ Other: 'CincinnatiMonitoring' }],
          },
          materials_available: ['PETG'],
          rates: null,
        },
      });

      const jobRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'create_print_job',
        payload: {
          design_hash: designRecord.signed_action.hashed.hash,
          printer_hash: printerRecord.signed_action.hashed.hash,
          settings: {
            layer_height: 0.1,
            infill_percent: 50,
            material: 'PETG',
            supports: true,
            raft: false,
            print_speed: 40,
            temperatures: { hotend: 240, bed: 80, chamber: null },
            custom_gcode: null,
          },
          energy_source: null,
          material_passport: null,
        },
      });

      const jobHash = jobRecord.signed_action.hashed.hash;

      // Start Cincinnati monitoring session
      const sessionRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'start_cincinnati_monitoring',
        payload: {
          job_hash: jobHash,
          total_layers: 200,
          sampling_rate_hz: 100,
        },
      });

      expect(sessionRecord).toBeDefined();

      // Update Cincinnati session with monitoring data
      const sessionHash = sessionRecord.signed_action.hashed.hash;
      await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'update_cincinnati_session',
        payload: {
          session_hash: sessionHash,
          current_layer: 50,
          health_score: 0.97,
          anomaly_count: 0,
        },
      });

      // Complete the print
      const completedRecord: Record = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'complete_print',
        payload: {
          job_hash: jobHash,
          result: 'Success',
        },
      });

      expect(completedRecord).toBeDefined();
    });
  });

  it('should get print statistics for a design', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Popular Widget',
          description: 'Gets printed a lot',
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

      const designHash = designRecord.signed_action.hashed.hash;

      // Get stats (should be empty initially)
      const stats = await alice.cells[0].callZome({
        zome_name: 'prints',
        fn_name: 'get_print_statistics',
        payload: designHash,
      });

      expect(stats).toBeDefined();
      expect(stats.total_prints).toBe(0);
    });
  });
});
