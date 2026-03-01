import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { runScenario, Scenario, Player } from '@holochain/tryorama';
import { ActionHash, Record } from '@holochain/client';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const HAPP_PATH = path.join(__dirname, '../workdir/fabrication.happ');

describe('Printers Zome', () => {
  it('should register and retrieve a printer', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Register a printer
      const registerInput = {
        name: 'Prusa MK3S+',
        location: {
          geohash: '9q8yyk',
          lat: 37.7749,
          lon: -122.4194,
          city: 'San Francisco',
          region: 'CA',
          country: 'USA',
        },
        printer_type: 'FDM',
        capabilities: {
          build_volume: { x: 250, y: 210, z: 210 },
          layer_heights: [0.05, 0.1, 0.15, 0.2, 0.3],
          nozzle_diameters: [0.4, 0.6],
          heated_bed: true,
          enclosure: false,
          multi_material: null,
          max_temp_hotend: 300,
          max_temp_bed: 120,
          features: ['AutoLeveling', 'PowerRecovery'],
        },
        materials_available: ['PLA', 'PETG', 'ABS'],
        rates: null,
      };

      const printerRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: registerInput,
      });

      expect(printerRecord).toBeDefined();
      expect(printerRecord.signed_action).toBeDefined();

      // Retrieve the printer
      const actionHash = printerRecord.signed_action.hashed.hash;
      const retrieved: Record | null = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'get_printer',
        payload: actionHash,
      });

      expect(retrieved).toBeDefined();
    });
  });

  it('should find printers by capability', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Register a high-temp printer
      await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'High Temp Printer',
          location: null,
          printer_type: 'FDM',
          capabilities: {
            build_volume: { x: 300, y: 300, z: 400 },
            layer_heights: [0.1, 0.2],
            nozzle_diameters: [0.4],
            heated_bed: true,
            enclosure: true,
            multi_material: null,
            max_temp_hotend: 450,
            max_temp_bed: 150,
            features: [],
          },
          materials_available: ['PEEK', { Custom: 'PEI' }, 'Nylon'],
          rates: null,
        },
      });

      // Find printers that can handle PEEK (high temp)
      const matches = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'find_printers_by_capability',
        payload: {
          requirements: {
            min_build_volume: null,
            material: 'PEEK',
            printer_type: 'FDM',
            min_layer_height: null,
            max_layer_height: null,
            heated_bed_required: true,
            enclosure_required: true,
            min_hotend_temp: 400,
          },
          pagination: null,
        },
      });

      expect(matches.items.length).toBeGreaterThanOrEqual(1);
      expect(matches.items[0].compatibility_score).toBeGreaterThan(0);
    });
  });

  it('should update printer availability', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Register a printer
      const printerRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Test Printer',
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

      const printerHash = printerRecord.signed_action.hashed.hash;

      // Update availability to busy
      const statusRecord: Record = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'update_availability',
        payload: {
          printer_hash: printerHash,
          status: 'Busy',
          message: 'Currently printing a large job',
          eta_available: 120, // minutes
          current_job: null,
        },
      });

      expect(statusRecord).toBeDefined();

      // Check available printers list
      const available = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'get_available_printers',
        payload: { pagination: null },
      });

      // The printer should not be in available list
      const hashes = available.items.map((r: Record) => r.signed_action.hashed.hash);
      expect(hashes).not.toContain(printerHash);
    });
  });

  it('should get printers owned by agent', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Alice registers 2 printers
      await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Alice Printer 1',
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

      await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Alice Printer 2',
          location: null,
          printer_type: 'SLA',
          capabilities: {
            build_volume: { x: 120, y: 68, z: 155 },
            layer_heights: [0.025, 0.05, 0.1],
            nozzle_diameters: [0.05],
            heated_bed: false,
            enclosure: true,
            multi_material: null,
            max_temp_hotend: 40,
            max_temp_bed: 0,
            features: [],
          },
          materials_available: ['StandardResin'],
          rates: null,
        },
      });

      // Bob registers 1 printer
      await bob.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'register_printer',
        payload: {
          name: 'Bob Printer',
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

      // Alice should see 2 printers
      const alicePrinters = await alice.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'get_my_printers',
        payload: { pagination: null },
      });

      expect(alicePrinters.items.length).toBe(2);

      // Bob should see 1 printer
      const bobPrinters = await bob.cells[0].callZome({
        zome_name: 'printers',
        fn_name: 'get_my_printers',
        payload: { pagination: null },
      });

      expect(bobPrinters.items.length).toBe(1);
    });
  });
});
