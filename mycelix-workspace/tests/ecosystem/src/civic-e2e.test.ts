/**
 * Civic Cluster E2E Tests
 *
 * Tests justice, emergency, and media workflows via Tryorama.
 * Requires: mycelix-civic.happ bundle built via `just build-civic`
 */

import { describe, it, expect } from 'vitest';
import { Scenario, runScenario, dhtSync, type PlayerApp } from '@holochain/tryorama';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __dirname = dirname(fileURLToPath(import.meta.url));

const CIVIC_HAPP_PATH = join(__dirname, '../../../../mycelix-civic/mycelix-civic.happ');

function skipIfNoBuild() {
  if (!existsSync(CIVIC_HAPP_PATH)) {
    console.warn(`Skipping: ${CIVIC_HAPP_PATH} not found. Run 'just build-civic' first.`);
    return true;
  }
  return false;
}

async function callZome<T>(
  player: PlayerApp,
  zome: string,
  fn: string,
  payload: unknown = null,
): Promise<T> {
  return await player.appWs.callZome({
    role_name: 'civic',
    zome_name: zome,
    fn_name: fn,
    payload,
  }) as T;
}

describe('Civic Cluster E2E', () => {
  if (skipIfNoBuild()) return;

  describe('Justice Case Flow', () => {
    it('should file a case and submit evidence', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
        ]);

        // Alice files a case via file_case(Case)
        const caseRecord = await callZome(alice, 'justice_cases', 'file_case', {
          id: 'case-e2e-001',
          title: 'Property boundary dispute',
          description: 'Disagreement about community garden boundaries',
          case_type: 'PropertyDispute',
          complainant: 'did:example:alice',
          respondent: 'did:example:bob',
          parties: [],
          phase: 'Filed',
          status: 'Active',
          severity: 'Moderate',
          context: {
            happ: 'mycelix-civic',
            reference_id: null,
            community: null,
            jurisdiction: null,
          },
          created_at: 0,
          updated_at: 0,
          phase_deadline: null,
        });
        expect(caseRecord).toBeDefined();

        // Alice submits evidence via submit_evidence(Evidence)
        // Note: justice_evidence zome has its own submit_evidence that takes Evidence struct
        const evidence = await callZome(alice, 'justice_evidence', 'submit_evidence', {
          id: 'ev-e2e-001',
          complaint_id: 'case-e2e-001',
          submitter: 'did:example:alice',
          evidence_type: 'Document',
          title: 'Survey map',
          description: 'Official property survey from 2025',
          content_hash: 'sha256:survey12345',
          encrypted_content: null,
          submitted: 0,
        });
        expect(evidence).toBeDefined();

        await dhtSync([alice, bob], alice.cells[0].cell_id[0]);
      });
    });
  });

  describe('Emergency Incident Flow', () => {
    it('should declare disaster and register resources', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
        ]);

        // Alice declares a disaster via declare_disaster(DeclareDisasterInput)
        const incident = await callZome(alice, 'emergency_incidents', 'declare_disaster', {
          id: 'disaster-e2e-001',
          disaster_type: 'Infrastructure',
          title: 'Water main break',
          description: 'Major water leak at intersection of Main and 3rd',
          severity: 'Level3',
          affected_area: {
            center_lat: 32.95,
            center_lon: -96.73,
            radius_km: 5.0,
            boundary: null,
            zones: [],
          },
          estimated_affected: 500,
          coordination_lead: null,
        });
        expect(incident).toBeDefined();

        // Bob registers an emergency resource via register_resource(RegisterResourceInput)
        const resource = await callZome(bob, 'emergency_resources', 'register_resource', {
          id: 'res-e2e-001',
          resource_type: 'Equipment',
          name: 'Plumbing repair kit',
          quantity: 1,
          unit: 'kit',
          location: 'Warehouse A',
        });
        expect(resource).toBeDefined();

        await dhtSync([alice, bob], alice.cells[0].cell_id[0]);
      });
    });
  });

  describe('Media Publication Flow', () => {
    it('should publish article and add fact check', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice, bob] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
        ]);

        // Alice publishes an article via publish(PublishInput)
        const article = await callZome(alice, 'media_publication', 'publish', {
          title: 'Community Garden Expansion Approved',
          content_hash: 'sha256:garden-article-001',
          content_type: 'Article',
          author_did: 'did:example:alice',
          co_authors: [],
          language: 'en',
          tags: ['local-news', 'community'],
          license: {
            license_type: 'CCBY',
            attribution_required: true,
            commercial_use: true,
            derivative_works: true,
          },
          encrypted: false,
        });
        expect(article).toBeDefined();

        await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

        // Bob adds a fact check via submit_fact_check(SubmitFactCheckInput)
        const factcheck = await callZome(bob, 'media_factcheck', 'submit_fact_check', {
          publication_id: 'pub-e2e-001',
          claim_text: 'Garden expansion is 500 sqm',
          claim_location: 'paragraph 2',
          epistemic_position: {
            empirical: 0.9,
            normative: 0.1,
            mythic: 0.0,
          },
          verdict: 'True',
          evidence: [
            {
              source_type: 'OfficialDocument',
              source_url: null,
              source_did: null,
              description: 'Board meeting minutes from March 5, 2026',
              supports_claim: true,
            },
          ],
          checker_did: 'did:example:bob',
        });
        expect(factcheck).toBeDefined();
      });
    });
  });

  describe('Restorative Justice Flow', () => {
    it('should initiate restorative circle', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
        ]);

        // Create a restorative circle via create_circle(RestorativeCircle)
        const process = await callZome(alice, 'justice_restorative', 'create_circle', {
          id: 'circle-e2e-001',
          case_id: 'case-e2e-001',
          facilitator: 'did:example:alice',
          participants: [
            {
              did: 'did:example:alice',
              role: 'HarmReceiver',
              consented: false,
              attended_sessions: [],
            },
            {
              did: 'did:example:bob',
              role: 'HarmDoer',
              consented: false,
              attended_sessions: [],
            },
          ],
          status: 'Forming',
          sessions: [],
          agreements: [],
          created_at: 0,
        });
        expect(process).toBeDefined();
      });
    });
  });

  describe('Civic Bridge Health', () => {
    it('should return bridge health status', async () => {
      await runScenario(async (scenario: Scenario) => {
        const [alice] = await scenario.addPlayersWithApps([
          { appBundleSource: { type: 'path', value: CIVIC_HAPP_PATH } },
        ]);

        const health = await callZome(alice, 'civic_bridge', 'health_check', null);
        expect(health).toBeDefined();
      });
    });
  });
});
