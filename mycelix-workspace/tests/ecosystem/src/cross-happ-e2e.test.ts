/**
 * Cross-hApp E2E Tests
 *
 * Tests multi-hApp interactions using the EcosystemTestHarness.
 * These tests verify that identity, finance, and commons hApps
 * can operate together in a unified ecosystem scenario.
 *
 * Requires: Built hApp bundles for each cluster being tested.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  EcosystemTestHarness,
  createFinanceScenario,
  createCoreScenario,
  happBundleExists,
  resolveHappBundlePath,
} from './harness';

describe('Cross-hApp Identity + Finance Flow', () => {
  const requiredHapps = ['identity', 'finance'] as const;

  const missing = requiredHapps.filter((h) => !happBundleExists(h));
  if (missing.length > 0) {
    it.skip(`Skipping: missing bundles: ${missing.join(', ')}`, () => {});
    return;
  }

  let harness: EcosystemTestHarness;

  beforeAll(async () => {
    harness = createFinanceScenario(2);
    await harness.setup();
  }, 120_000);

  afterAll(async () => {
    await harness?.teardown();
  });

  it('should register identity then initialize SAP balance', async () => {
    const aliceDid = harness.generateTestDid(0);

    // Step 1: Register identity via identity hApp
    const identityRecord = await harness.callZome(
      0,
      'identity',
      'did_registry',
      'register_did',
      {
        did: aliceDid,
        verification_methods: [],
        services: [],
      },
    );
    expect(identityRecord).toBeDefined();

    // Step 2: Use that DID to initialize a SAP balance in finance hApp
    const sapRecord = await harness.callZome(
      0,
      'finance',
      'payments',
      'initialize_sap_balance',
      aliceDid,
    );
    expect(sapRecord).toBeDefined();

    // Step 3: Query the balance back
    const balance = await harness.callZome(
      0,
      'finance',
      'payments',
      'get_sap_balance',
      aliceDid,
    );
    expect(balance).toBeDefined();
  });

  it('should allow cross-agent TEND exchange with identity-backed DIDs', async () => {
    const aliceDid = harness.generateTestDid(0);
    const bobDid = harness.generateTestDid(1);

    // Both agents register identity
    await harness.callZome(0, 'identity', 'did_registry', 'register_did', {
      did: aliceDid,
      verification_methods: [],
      services: [],
    });
    await harness.callZome(1, 'identity', 'did_registry', 'register_did', {
      did: bobDid,
      verification_methods: [],
      services: [],
    });

    // Alice records time exchange to Bob
    const exchange = await harness.callZome(
      0,
      'finance',
      'tend',
      'record_exchange',
      {
        receiver_did: bobDid,
        hours: 2.0,
        service_description: 'Tutoring session',
        service_category: 'Education',
        cultural_alias: null,
        dao_did: 'did:mycelix:test-dao',
        service_date: null,
      },
    );
    expect(exchange).toBeDefined();

    await harness.waitForSync(3000);
  });
});

describe('Cross-hApp Identity + Governance Flow', () => {
  const requiredHapps = ['identity', 'governance'] as const;

  const missing = requiredHapps.filter((h) => !happBundleExists(h));
  if (missing.length > 0) {
    it.skip(`Skipping: missing bundles: ${missing.join(', ')}`, () => {});
    return;
  }

  let harness: EcosystemTestHarness;

  beforeAll(async () => {
    harness = createCoreScenario(3);
    await harness.setup();
  }, 120_000);

  afterAll(async () => {
    await harness?.teardown();
  });

  it('should register identities then create a governance proposal', async () => {
    // All 3 agents register identity
    for (let i = 0; i < 3; i++) {
      const did = harness.generateTestDid(i);
      await harness.callZome(i, 'identity', 'did_registry', 'register_did', {
        did,
        verification_methods: [],
        services: [],
      });
    }

    const proposerDid = harness.generateTestDid(0);

    // Agent 0 creates a governance proposal
    const proposal = await harness.callZome(
      0,
      'governance',
      'proposals',
      'create_proposal',
      {
        title: 'Increase community garden allocation',
        description: 'Allocate 200 sqm additional land for community gardens',
        proposal_type: 'ResourceAllocation',
        proposer_did: proposerDid,
        discussion_period_ms: 86_400_000,
        voting_period_ms: 172_800_000,
      },
    );
    expect(proposal).toBeDefined();
  });

  it('should query reputation across identity bridge', async () => {
    const agent0 = harness.getAgent(0);
    const agent1 = harness.getAgent(1);

    // Query reputation — may return null if no interactions yet
    const reputation = await harness.queryAggregateReputation(
      0,
      agent1.pubKey,
    );
    // Reputation may be null for new agents; the important thing is no error
    // (the identity bridge should handle the "not found" case gracefully)
    expect(reputation === null || typeof reputation === 'object').toBe(true);
  });
});

describe('Ecosystem Health Check', () => {
  it('should resolve all known hApp bundle paths', () => {
    const happs = [
      'identity',
      'governance',
      'finance',
      'commons',
      'civic',
    ] as const;

    for (const happ of happs) {
      const bundlePath = resolveHappBundlePath(happ);
      expect(bundlePath).toBeTruthy();
      // Path should be absolute
      expect(bundlePath.startsWith('/')).toBe(true);
    }
  });

  it('should report which bundles exist', () => {
    const happs = [
      'identity',
      'governance',
      'finance',
      'commons',
      'civic',
      'core',
      'marketplace',
      'edunet',
      'epistemicMarkets',
      'fabrication',
    ] as const;

    const status = happs.map((h) => ({
      name: h,
      exists: happBundleExists(h),
      path: resolveHappBundlePath(h),
    }));

    // Log status for visibility
    console.table(status);

    // At minimum, paths should be resolvable (even if not built)
    for (const s of status) {
      expect(s.path).toBeTruthy();
    }
  });
});
