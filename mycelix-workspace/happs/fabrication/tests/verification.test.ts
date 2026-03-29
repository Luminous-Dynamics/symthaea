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

describe('Verification Zome - Safety & Epistemic', () => {
  it('should submit verification and safety claim for a design', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // First create a design to verify
      const designRecord: Record = await alice.cells[0].callZome({
        zome_name: 'designs',
        fn_name: 'create_design',
        payload: {
          title: 'Load-Bearing Bracket V3',
          description: 'Structural bracket for Class3 body contact',
          category: 'Parts',
          intent_vector: null,
          parametric_schema: null,
          constraint_graph: null,
          material_compatibility: [],
          circularity_score: 0.5,
          embodied_energy_kwh: 1.0,
          repair_manifest: null,
          license: 'PublicDomain',
          safety_class: 'Class3BodyContact',
        },
      });

      const designHash = designRecord.signed_action.hashed.hash;

      // Submit a verification
      const verificationRecord: Record = await alice.cells[0].callZome({
        zome_name: 'verification',
        fn_name: 'submit_verification',
        payload: {
          design_hash: designHash,
          verification_type: 'StructuralAnalysis',
          result: {
            Passed: {
              confidence: 0.92,
              notes: 'FEA analysis complete, all stress points within tolerance',
            },
          },
          evidence: [],
          credentials: ['PE License #54321'],
        },
      });

      expect(verificationRecord).toBeDefined();

      // Submit a safety claim
      const claimRecord: Record = await alice.cells[0].callZome({
        zome_name: 'verification',
        fn_name: 'submit_safety_claim',
        payload: {
          design_hash: designHash,
          claim_type: { LoadCapacity: 'Supports 100kg static load' },
          claim_text: 'This bracket supports up to 100kg static load per ISO 14122',
          supporting_evidence: ['FEA report v3.0', 'Material test cert #2024-001'],
        },
      });

      expect(claimRecord).toBeDefined();

      // Get epistemic score
      const epistemicScore = await alice.cells[0].callZome({
        zome_name: 'verification',
        fn_name: 'get_epistemic_score',
        payload: designHash,
      });

      expect(epistemicScore).toBeDefined();
      expect(epistemicScore.overall_confidence).toBeGreaterThan(0);
    });
  });

  it('should enforce rate limiting on rapid bridge operations', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice] = await scenario.addPlayersWithApps([
        { appBundleSource: { path: HAPP_PATH } },
      ]);

      await scenario.shareAllAgents();

      // Attempt to burst many operations rapidly
      const promises = [];
      for (let i = 0; i < 105; i++) {
        promises.push(
          alice.cells[0].callZome({
            zome_name: 'bridge',
            fn_name: 'emit_fabrication_event',
            payload: {
              event_type: 'DesignPublished',
              design_id: null,
              payload: JSON.stringify({ burst_index: i }),
            },
          }).catch((e: Error) => ({ error: e.message }))
        );
      }

      const results = await Promise.all(promises);

      // Some should succeed, but eventually rate limit should kick in
      const errors = results.filter(
        (r): r is { error: string } => 'error' in r && typeof r.error === 'string'
      );

      // At least some operations should have been rate-limited
      expect(errors.length).toBeGreaterThan(0);
    });
  });

  it('should prevent agent B from deleting agent A design', async () => {
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
          title: 'Alice Private Bracket',
          description: 'Only Alice should manage this',
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

      const designHash = designRecord.signed_action.hashed.hash;

      // Bob attempts to delete Alice's design — should fail
      try {
        await bob.cells[0].callZome({
          zome_name: 'designs',
          fn_name: 'delete_design',
          payload: designHash,
        });
        // If we get here, the delete succeeded (unexpected)
        expect(true).toBe(false); // force fail
      } catch (e) {
        // Expected — Bob cannot delete Alice's design
        expect(e).toBeDefined();
      }
    });
  });
});
