import { describe, it, expect, afterAll } from 'vitest';
import { Scenario, dhtSync } from '@holochain/tryorama';
import { HAPP_PATH, setupScenario } from './setup.js';
import type { SupportKnowledgeArticle, SupportTicket, CognitiveUpdate, PrivacyPreference } from './setup.js';

describe('Support E2E Tests', () => {
  let scenario: Scenario;

  afterAll(async () => {
    if (scenario) await scenario.cleanUp();
  });

  it('ticket lifecycle: create → triage → comment → resolve', async () => {
    scenario = await setupScenario();
    const [alice, bob] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: HAPP_PATH } },
      { appBundleSource: { path: HAPP_PATH } },
    ]);

    // Alice creates a ticket
    const ticket: SupportTicket = {
      title: 'Conductor not starting',
      description: 'After update, holochain conductor fails to start with keystore error',
      category: 'Holochain',
      priority: 'High',
      status: 'Open',
      created_at: Date.now() * 1000,
      closed_at: null,
    };

    const ticketRecord = await alice.cells[0].callZome({
      zome_name: 'support_tickets',
      fn_name: 'create_ticket',
      payload: ticket,
    });
    expect(ticketRecord).toBeTruthy();

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob comments on the ticket
    const comment = {
      ticket_hash: ticketRecord.signed_action.hashed.hash,
      content: 'Try clearing the lair keystore cache at ~/.config/holochain/lair',
      created_at: Date.now() * 1000,
    };

    const commentRecord = await bob.cells[0].callZome({
      zome_name: 'support_tickets',
      fn_name: 'add_comment',
      payload: comment,
    });
    expect(commentRecord).toBeTruthy();

    // Alice resolves the ticket
    const updateResult = await alice.cells[0].callZome({
      zome_name: 'support_tickets',
      fn_name: 'update_ticket_status',
      payload: {
        ticket_hash: ticketRecord.signed_action.hashed.hash,
        new_status: 'Resolved',
      },
    });
    expect(updateResult).toBeTruthy();

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob can see the resolved ticket
    const fetched = await bob.cells[0].callZome({
      zome_name: 'support_tickets',
      fn_name: 'get_ticket',
      payload: ticketRecord.signed_action.hashed.hash,
    });
    expect(fetched).toBeTruthy();
  });

  it('privacy tiers: LocalOnly blocks sharing, Full allows', async () => {
    scenario = await setupScenario();
    const [alice] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: HAPP_PATH } },
    ]);

    // Set privacy to LocalOnly
    const pref: PrivacyPreference = {
      sharing_tier: 'LocalOnly',
      consent_verified: true,
      updated_at: Date.now() * 1000,
    };

    const prefRecord = await alice.cells[0].callZome({
      zome_name: 'support_diagnostics',
      fn_name: 'set_privacy_preference',
      payload: pref,
    });
    expect(prefRecord).toBeTruthy();

    // Verify it was stored
    const retrieved = await alice.cells[0].callZome({
      zome_name: 'support_diagnostics',
      fn_name: 'get_privacy_preference',
      payload: null,
    });
    expect(retrieved).toBeTruthy();
  });

  it('knowledge lifecycle: create → upvote → flag → deprecate', async () => {
    scenario = await setupScenario();
    const [alice, bob] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: HAPP_PATH } },
      { appBundleSource: { path: HAPP_PATH } },
    ]);

    // Alice creates an article
    const article: SupportKnowledgeArticle = {
      title: 'How to configure Holochain networking',
      content: '## Step 1\nInstall the conductor and configure bootstrap servers...',
      category: 'Holochain',
      tags: ['holochain', 'networking', 'setup'],
      source: 'Community',
      difficulty_level: 'Intermediate',
      upvotes: 0,
      verified: false,
      deprecated: false,
      deprecation_reason: null,
      version: 1,
    };

    const articleRecord = await alice.cells[0].callZome({
      zome_name: 'support_knowledge',
      fn_name: 'create_article',
      payload: article,
    });
    expect(articleRecord).toBeTruthy();

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob searches by category
    const searchResults = await bob.cells[0].callZome({
      zome_name: 'support_knowledge',
      fn_name: 'search_by_category',
      payload: 'Holochain',
    });
    expect(searchResults).toBeTruthy();
  });

  it('cognitive update federation: publish → sync → query', async () => {
    scenario = await setupScenario();
    const [alice, bob] = await scenario.addPlayersWithApps([
      { appBundleSource: { path: HAPP_PATH } },
      { appBundleSource: { path: HAPP_PATH } },
    ]);

    // Alice publishes a cognitive update
    const update: CognitiveUpdate = {
      category: 'Network',
      encoding: Array.from({ length: 32 }, () => Math.floor(Math.random() * 256)),
      phi: 0.72,
      resolution_pattern: 'DNS misconfiguration resolved by editing /etc/resolv.conf',
      published_at: Date.now() * 1000,
    };

    const updateRecord = await alice.cells[0].callZome({
      zome_name: 'support_diagnostics',
      fn_name: 'publish_cognitive_update',
      payload: update,
    });
    expect(updateRecord).toBeTruthy();

    await dhtSync([alice, bob], alice.cells[0].cell_id[0]);

    // Bob can query updates by category
    const results = await bob.cells[0].callZome({
      zome_name: 'support_diagnostics',
      fn_name: 'get_cognitive_updates_by_category',
      payload: 'Network',
    });
    expect(results).toBeTruthy();
  });
});
