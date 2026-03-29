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

describe('Materials Zome', () => {
  it('should create, retrieve, update, and delete a material', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create
      const record: Record = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'create_material',
        payload: {
          name: 'Prusament PETG',
          material_type: 'PETG',
          properties: {
            print_temp_min: 220,
            print_temp_max: 250,
            bed_temp_min: 70,
            bed_temp_max: 85,
            density_g_cm3: 1.27,
            tensile_strength_mpa: 50.0,
            elongation_percent: 7.6,
            food_safe: true,
            uv_resistant: false,
            water_resistant: true,
            recyclable: true,
          },
          certifications: [
            {
              cert_type: 'FoodSafe',
              issuer: 'EU Regulation 10/2011',
              valid_until: null,
              document_cid: null,
            },
          ],
          safety_data_sheet: 'https://example.com/sds/petg.pdf',
        },
      });

      expect(record).toBeDefined();
      const materialHash = record.signed_action.hashed.hash;

      // Retrieve
      const retrieved: Record | null = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'get_material',
        payload: materialHash,
      });

      expect(retrieved).toBeDefined();

      // Update
      const updated: Record = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'update_material',
        payload: {
          original_action_hash: materialHash,
          name: 'Prusament PETG v2',
          material_type: null,
          properties: null,
          certifications: null,
          safety_data_sheet: null,
        },
      });

      expect(updated).toBeDefined();
      expect(updated.signed_action.hashed.hash).not.toEqual(materialHash);

      // Delete
      const deleteHash = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'delete_material',
        payload: materialHash,
      });

      expect(deleteHash).toBeDefined();
    });
  });

  it('should filter materials by type with pagination', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create 3 PLA materials
      for (let i = 0; i < 3; i++) {
        await alice.cells[0].callZome({
          zome_name: 'materials',
          fn_name: 'create_material',
          payload: {
            name: `PLA Brand #${i}`,
            material_type: 'PLA',
            properties: {
              print_temp_min: 190,
              print_temp_max: 220,
              bed_temp_min: 50,
              bed_temp_max: 70,
              density_g_cm3: 1.24,
              tensile_strength_mpa: 37.0 + i,
              elongation_percent: 6.0,
              food_safe: false,
              uv_resistant: false,
              water_resistant: false,
              recyclable: true,
            },
            certifications: [],
            safety_data_sheet: null,
          },
        });
      }

      // Query by type (all)
      const result = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'get_materials_by_type',
        payload: { material_type: 'PLA', pagination: null },
      });

      expect(result.total).toBeGreaterThanOrEqual(3);

      // Query with pagination
      const page = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'get_materials_by_type',
        payload: {
          material_type: 'PLA',
          pagination: { offset: 0, limit: 2 },
        },
      });

      expect(page.items.length).toBe(2);
    });
  });

  it('should filter food-safe materials', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Create food-safe
      await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'create_material',
        payload: {
          name: 'Food-Safe PLA',
          material_type: 'PLA',
          properties: {
            print_temp_min: 190,
            print_temp_max: 220,
            bed_temp_min: 50,
            bed_temp_max: 70,
            density_g_cm3: 1.24,
            tensile_strength_mpa: null,
            elongation_percent: null,
            food_safe: true,
            uv_resistant: false,
            water_resistant: false,
            recyclable: true,
          },
          certifications: [],
          safety_data_sheet: null,
        },
      });

      // Create non-food-safe
      await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'create_material',
        payload: {
          name: 'Regular ABS',
          material_type: 'ABS',
          properties: {
            print_temp_min: 230,
            print_temp_max: 260,
            bed_temp_min: 90,
            bed_temp_max: 110,
            density_g_cm3: 1.04,
            tensile_strength_mpa: null,
            elongation_percent: null,
            food_safe: false,
            uv_resistant: false,
            water_resistant: false,
            recyclable: false,
          },
          certifications: [],
          safety_data_sheet: null,
        },
      });

      const result = await alice.cells[0].callZome({
        zome_name: 'materials',
        fn_name: 'get_food_safe_materials',
        payload: { pagination: null },
      });

      expect(result.items.length).toBeGreaterThanOrEqual(1);
    });
  });
});
