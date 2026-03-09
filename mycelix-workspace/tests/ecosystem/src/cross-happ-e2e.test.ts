/**
 * Cross-hApp E2E Tests
 *
 * Tests multi-hApp interactions using Tryorama Scenario API.
 * These tests verify that identity, finance, and commons hApps
 * can operate together in a unified ecosystem scenario.
 *
 * Requires: Built hApp bundles for each cluster being tested.
 */

import { describe, it, expect } from 'vitest';
import {
  happBundleExists,
  resolveHappBundlePath,
  appWithOptions,
  generateTestDid,
  waitForSync,
  callZome,
  runScenario,
} from './harness';
import type { Scenario } from './harness';

describe('Cross-hApp Identity + Finance Flow', () => {
  const requiredHapps = ['identity', 'finance'] as const;

  const missing = requiredHapps.filter((h) => !happBundleExists(h));
  if (missing.length > 0) {
    it.skip(`Skipping: missing bundles: ${missing.join(', ')}`, () => {});
    return;
  }

  it('should register identity then initialize SAP balance', async () => {
    await runScenario(async (scenario: Scenario) => {
      // Install identity hApp for alice
      const [aliceIdentity] = await scenario.addPlayersWithApps([
        appWithOptions('identity'),
      ]);

      const aliceDid = generateTestDid(aliceIdentity.agentPubKey);

      // Register identity
      const identityRecord = await callZome(
        aliceIdentity,
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

      // Install finance hApp for alice (same conductor, different hApp)
      const [aliceFinance] = await scenario.addPlayersWithApps([
        appWithOptions('finance'),
      ]);

      // Initialize SAP balance
      const sapRecord = await callZome(
        aliceFinance,
        'finance',
        'payments',
        'initialize_sap_balance',
        aliceDid,
      );
      expect(sapRecord).toBeDefined();

      // Query the balance back
      const balance = await callZome(
        aliceFinance,
        'finance',
        'payments',
        'get_sap_balance',
        aliceDid,
      );
      expect(balance).toBeDefined();
    });
  });

  it('should allow cross-agent TEND exchange with identity-backed DIDs', async () => {
    await runScenario(async (scenario: Scenario) => {
      const [alice, bob] = await scenario.addPlayersWithApps([
        appWithOptions('identity'),
        appWithOptions('identity'),
      ]);

      const aliceDid = generateTestDid(alice.agentPubKey);
      const bobDid = generateTestDid(bob.agentPubKey);

      // Both register identity
      await callZome(alice, 'identity', 'did_registry', 'register_did', {
        did: aliceDid,
        verification_methods: [],
        services: [],
      });
      await callZome(bob, 'identity', 'did_registry', 'register_did', {
        did: bobDid,
        verification_methods: [],
        services: [],
      });

      // Install finance for alice
      const [aliceFinance] = await scenario.addPlayersWithApps([
        appWithOptions('finance'),
      ]);

      // Alice records time exchange to Bob
      const exchange = await callZome(
        aliceFinance,
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

      await waitForSync(3000);
    });
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

    console.table(status);

    for (const s of status) {
      expect(s.path).toBeTruthy();
    }
  });
});
