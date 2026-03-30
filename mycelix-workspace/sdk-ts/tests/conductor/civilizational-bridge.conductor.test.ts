// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civilizational OS Bridge Conductor Tests
 *
 * Comprehensive integration tests for all 8 Bridge clients against a real Holochain conductor.
 * Tests actual zome calls and cross-hApp coordination.
 *
 * Run with: CONDUCTOR_AVAILABLE=true npm run test:conductor
 *
 * Requirements:
 * - Running Holochain conductor with Mycelix ecosystem hApp installed
 * - All 8 DNA roles: identity, finance, property, energy, media, governance, justice, knowledge
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import {
  CONDUCTOR_ENABLED,
  getConductorConfig,
  setupTestContext,
  teardownTestContext,
  retry,
  generateTestAgentId,
  waitForSync,
  type TestContext,
} from './conductor-harness.js';

// describeIf runs tests only when conductor is available
const describeIf = CONDUCTOR_ENABLED ? describe : describe.skip;

// ============================================================================
// Test Data Generators
// ============================================================================

function generateTestDID(): string {
  return `did:mycelix:test-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function generateTestPublicKey(): string {
  return `pubkey-${Date.now()}-${Math.random().toString(36).slice(2, 20)}`.padEnd(32, 'x');
}

function generateTestHash(): string {
  return `Qm${Math.random().toString(36).slice(2, 46)}`.padEnd(46, 'a');
}

// ============================================================================
// Identity Bridge Tests
// ============================================================================

describeIf('Identity Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let identityClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    // Dynamic import to avoid loading unless conductor available
    const { getIdentityBridgeClient } = await import('../../src/integrations/identity/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    identityClient = getIdentityBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should register a hApp with identity bridge', async () => {
    const result = await retry(() =>
      identityClient.registerHapp({
        happ_name: 'test-marketplace',
        capabilities: ['identity_query', 'reputation_report'],
      })
    );

    expect(result).toBeDefined();
    expect(result.happ_name).toBe('test-marketplace');
  });

  it('should query identity information', async () => {
    const testDid = generateTestDID();

    const result = await retry(() =>
      identityClient.queryIdentity({
        did: testDid,
        source_happ: 'test-marketplace',
        requested_fields: ['matl_score', 'credential_count'],
      })
    );

    expect(result).toBeDefined();
    // New DIDs should have default/zero values
    expect(typeof result.matl_score).toBe('number');
  });

  it('should report reputation changes', async () => {
    const testDid = generateTestDID();

    const result = await retry(() =>
      identityClient.reportReputation({
        target_did: testDid,
        delta: 0.1,
        reason: 'Successful transaction',
        evidence: { transaction_id: 'tx-123' },
      })
    );

    expect(result).toBeDefined();
    expect(result.success).toBe(true);
  });

  it('should broadcast identity events', async () => {
    const result = await retry(() =>
      identityClient.broadcastIdentityEvent({
        event_type: 'IdentityCreated',
        did: generateTestDID(),
        payload: JSON.stringify({ source: 'test' }),
      })
    );

    expect(result).toBeDefined();
    expect(result.event_type).toBe('IdentityCreated');
  });

  it('should get recent identity events', async () => {
    const result = await retry(() => identityClient.getRecentEvents(10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Finance Bridge Tests
// ============================================================================

describeIf('Finance Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let financeClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getFinanceBridgeClient } = await import('../../src/integrations/finance/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    financeClient = getFinanceBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should query credit score', async () => {
    const testDid = generateTestDID();

    const result = await retry(() =>
      financeClient.queryCredit({
        did: testDid,
        purpose: 'LoanApplication',
        amount_requested: 10000,
      })
    );

    expect(result).toBeDefined();
    expect(typeof result.credit_score).toBe('number');
  });

  it('should process payments', async () => {
    const result = await retry(() =>
      financeClient.processPayment({
        payee_did: generateTestDID(),
        amount: 100,
        currency: 'MCX',
        target_happ: 'test-marketplace',
        reference_id: `payment-${Date.now()}`,
      })
    );

    expect(result).toBeDefined();
    expect(result.success).toBe(true);
  });

  it('should register collateral', async () => {
    const result = await retry(() =>
      financeClient.registerCollateral({
        asset_type: 'RealEstate',
        asset_id: `property-${Date.now()}`,
        valuation: 500000,
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should get payment history', async () => {
    const testDid = generateTestDID();

    const result = await retry(() => financeClient.getPaymentHistory(testDid, 10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Property Bridge Tests
// ============================================================================

describeIf('Property Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let propertyClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getPropertyBridgeClient } = await import('../../src/integrations/property/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    propertyClient = getPropertyBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should verify ownership', async () => {
    const result = await retry(() =>
      propertyClient.verifyOwnership({
        asset_id: `asset-${Date.now()}`,
        verification_type: 'Current',
      })
    );

    expect(result).toBeDefined();
    expect(typeof result.verified).toBe('boolean');
  });

  it('should pledge collateral', async () => {
    const result = await retry(() =>
      propertyClient.pledgeCollateral({
        asset_id: `asset-${Date.now()}`,
        pledge_to_happ: 'mycelix-finance',
        pledge_amount: 100000,
        loan_reference: `loan-${Date.now()}`,
      })
    );

    expect(result).toBeDefined();
    expect(result.status).toBe('Active');
  });

  it('should get properties by owner', async () => {
    const testDid = generateTestDID();

    const result = await retry(() => propertyClient.getPropertiesByOwner(testDid));

    expect(Array.isArray(result)).toBe(true);
  });

  it('should broadcast property events', async () => {
    const result = await retry(() =>
      propertyClient.broadcastPropertyEvent(
        'PropertyRegistered',
        `asset-${Date.now()}`,
        generateTestDID(),
        '{}'
      )
    );

    expect(result).toBeDefined();
    expect(result.event_type).toBe('PropertyRegistered');
  });
});

// ============================================================================
// Energy Bridge Tests
// ============================================================================

describeIf('Energy Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let energyClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getEnergyBridgeClient } = await import('../../src/integrations/energy/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    energyClient = getEnergyBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should query available energy', async () => {
    const result = await retry(() =>
      energyClient.queryAvailableEnergy({
        source_happ: 'energy-grid',
        location: { lat: 37.7749, lng: -122.4194, radius_km: 50 },
        energy_sources: ['solar', 'wind'],
      })
    );

    expect(result).toBeDefined();
    expect(typeof result.total_available).toBe('number');
  });

  it('should request energy purchase', async () => {
    const result = await retry(() =>
      energyClient.requestEnergyPurchase({
        provider_id: `provider-${Date.now()}`,
        amount_kwh: 100,
        max_price_per_kwh: 0.15,
        delivery_location: { lat: 37.7749, lng: -122.4194 },
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should report production', async () => {
    const result = await retry(() =>
      energyClient.reportProduction({
        participant_id: `producer-${Date.now()}`,
        production_kwh: 500,
        consumption_kwh: 100,
        energy_source: 'solar',
        timestamp: Date.now() * 1000,
      })
    );

    expect(result).toBeDefined();
    expect(result.success).toBe(true);
  });

  it('should get energy events', async () => {
    const result = await retry(() => energyClient.getRecentEvents(10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Media Bridge Tests
// ============================================================================

describeIf('Media Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let mediaClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getMediaBridgeClient } = await import('../../src/integrations/media/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    mediaClient = getMediaBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should query content', async () => {
    const result = await retry(() =>
      mediaClient.queryContent({
        query_type: 'ByAuthor',
        query_params: JSON.stringify({ author_did: generateTestDID() }),
        source_happ: 'news-platform',
      })
    );

    expect(result).toBeDefined();
    expect(Array.isArray(result.results)).toBe(true);
  });

  it('should request license', async () => {
    const result = await retry(() =>
      mediaClient.requestLicense({
        content_id: `content-${Date.now()}`,
        licensee_did: generateTestDID(),
        license_type: 'CCBY',
        purpose: 'Educational use',
        duration_days: 365,
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should verify license', async () => {
    const result = await retry(() =>
      mediaClient.verifyLicense({
        content_id: `content-${Date.now()}`,
        licensee_did: generateTestDID(),
      })
    );

    expect(result).toBeDefined();
    expect(typeof result.valid).toBe('boolean');
  });

  it('should distribute royalties', async () => {
    const result = await retry(() =>
      mediaClient.distributeRoyalties({
        content_id: `content-${Date.now()}`,
        total_amount: 1000,
        currency: 'MCX',
        reason: 'streaming',
      })
    );

    expect(result).toBeDefined();
    expect(result.success).toBe(true);
  });
});

// ============================================================================
// Governance Bridge Tests
// ============================================================================

describeIf('Governance Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let governanceClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getGovernanceBridgeClient } =
      await import('../../src/integrations/governance/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    governanceClient = getGovernanceBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should query governance status', async () => {
    const result = await retry(() =>
      governanceClient.queryGovernance({
        query_type: 'ProposalStatus',
        query_params: JSON.stringify({ proposal_id: 'test-proposal' }),
        source_happ: 'dao-app',
      })
    );

    expect(result).toBeDefined();
  });

  it('should request execution', async () => {
    const result = await retry(() =>
      governanceClient.requestExecution({
        proposal_hash: generateTestHash(),
        target_happ: 'mycelix-finance',
        action: 'TransferFunds',
        payload: JSON.stringify({ amount: 1000, recipient: generateTestDID() }),
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should broadcast governance events', async () => {
    const result = await retry(() =>
      governanceClient.broadcastGovernanceEvent({
        event_type: 'ProposalCreated',
        proposal_hash: generateTestHash(),
        payload: JSON.stringify({ title: 'Test Proposal' }),
      })
    );

    expect(result).toBeDefined();
    expect(result.event_type).toBe('ProposalCreated');
  });

  it('should get governance events', async () => {
    const result = await retry(() => governanceClient.getRecentEvents(10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Justice Bridge Tests
// ============================================================================

describeIf('Justice Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let justiceClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getJusticeBridgeClient } = await import('../../src/integrations/justice/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    justiceClient = getJusticeBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should file cross-hApp dispute', async () => {
    const result = await retry(() =>
      justiceClient.fileCrossHappDispute({
        dispute_type: 'ContractBreach',
        respondent_did: generateTestDID(),
        related_happs: ['marketplace', 'finance'],
        title: 'Test Dispute',
        description: 'Testing dispute filing',
        evidence_hashes: [generateTestHash()],
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should request enforcement', async () => {
    const result = await retry(() =>
      justiceClient.requestEnforcement({
        dispute_id: `dispute-${Date.now()}`,
        target_happ: 'marketplace',
        target_did: generateTestDID(),
        action_type: 'ReputationPenalty',
        details: 'Reduce reputation for contract breach',
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should get dispute history', async () => {
    const testDid = generateTestDID();

    const result = await retry(() =>
      justiceClient.getDisputeHistory({
        did: testDid,
        role: 'both',
        limit: 10,
      })
    );

    expect(result).toBeDefined();
    expect(Array.isArray(result.disputes)).toBe(true);
  });

  it('should get justice events', async () => {
    const result = await retry(() => justiceClient.getRecentEvents(10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Knowledge Bridge Tests
// ============================================================================

describeIf('Knowledge Bridge Conductor Tests', () => {
  let ctx: TestContext;
  let knowledgeClient: any;

  beforeAll(async () => {
    ctx = await setupTestContext();
    const { getKnowledgeBridgeClient } = await import('../../src/integrations/knowledge/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();
    knowledgeClient = getKnowledgeBridgeClient(client);
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should query knowledge', async () => {
    const result = await retry(() =>
      knowledgeClient.queryKnowledge({
        query_type: 'VerifyClaim',
        query_params: JSON.stringify({ claim_hash: generateTestHash() }),
        source_happ: 'fact-checker',
      })
    );

    expect(result).toBeDefined();
  });

  it('should perform fact check', async () => {
    const result = await retry(() =>
      knowledgeClient.factCheck({
        claim_text: 'The Earth orbits the Sun.',
        source_happ: 'verification-app',
      })
    );

    expect(result).toBeDefined();
    expect(typeof result.verified).toBe('boolean');
    expect(typeof result.confidence).toBe('number');
  });

  it('should register external claim', async () => {
    const result = await retry(() =>
      knowledgeClient.registerExternalClaim({
        claim_hash: generateTestHash(),
        source_happ: 'research-platform',
        title: 'Climate Impact Study 2024',
        empirical: 0.9,
        normative: 0.3,
        mythic: 0.1,
        author_did: generateTestDID(),
      })
    );

    expect(result).toBeDefined();
    expect(result.id).toBeDefined();
  });

  it('should get claims by subject', async () => {
    const result = await retry(() => knowledgeClient.getClaimsBySubject('climate-change'));

    expect(Array.isArray(result)).toBe(true);
  });

  it('should get knowledge events', async () => {
    const result = await retry(() => knowledgeClient.getRecentEvents(10));

    expect(Array.isArray(result)).toBe(true);
  });
});

// ============================================================================
// Cross-hApp Integration Tests
// ============================================================================

describeIf('Cross-hApp Integration Conductor Tests', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await setupTestContext();
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should coordinate identity → finance flow', async () => {
    const { getIdentityBridgeClient } = await import('../../src/integrations/identity/index.js');
    const { getFinanceBridgeClient } = await import('../../src/integrations/finance/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const identityClient = getIdentityBridgeClient(client);
    const financeClient = getFinanceBridgeClient(client);

    const testDid = generateTestDID();

    // 1. Query identity
    const identity = await retry(() =>
      identityClient.queryIdentity({
        did: testDid,
        source_happ: 'lending-platform',
        requested_fields: ['matl_score'],
      })
    );

    // 2. Use MATL score for credit decision
    const credit = await retry(() =>
      financeClient.queryCredit({
        did: testDid,
        purpose: 'LoanApplication',
        amount_requested: 5000,
      })
    );

    expect(identity).toBeDefined();
    expect(credit).toBeDefined();
    expect(typeof credit.credit_score).toBe('number');

    await client.disconnect();
  });

  it('should coordinate property → finance collateral flow', async () => {
    const { getPropertyBridgeClient } = await import('../../src/integrations/property/index.js');
    const { getFinanceBridgeClient } = await import('../../src/integrations/finance/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const propertyClient = getPropertyBridgeClient(client);
    const financeClient = getFinanceBridgeClient(client);

    const assetId = `property-${Date.now()}`;

    // 1. Verify property ownership
    const ownership = await retry(() =>
      propertyClient.verifyOwnership({
        asset_id: assetId,
        verification_type: 'Current',
      })
    );

    // 2. Register as collateral
    const collateral = await retry(() =>
      financeClient.registerCollateral({
        asset_type: 'RealEstate',
        asset_id: assetId,
        valuation: 250000,
      })
    );

    // 3. Pledge for loan
    const pledge = await retry(() =>
      propertyClient.pledgeCollateral({
        asset_id: assetId,
        pledge_to_happ: 'mycelix-finance',
        pledge_amount: 200000,
        loan_reference: `loan-${Date.now()}`,
      })
    );

    expect(ownership).toBeDefined();
    expect(collateral).toBeDefined();
    expect(pledge.status).toBe('Active');

    await client.disconnect();
  });

  it('should coordinate knowledge → governance decision flow', async () => {
    const { getKnowledgeBridgeClient } = await import('../../src/integrations/knowledge/index.js');
    const { getGovernanceBridgeClient } =
      await import('../../src/integrations/governance/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const knowledgeClient = getKnowledgeBridgeClient(client);
    const governanceClient = getGovernanceBridgeClient(client);

    // 1. Register verified claim
    const claim = await retry(() =>
      knowledgeClient.registerExternalClaim({
        claim_hash: generateTestHash(),
        source_happ: 'research-institute',
        title: 'Energy Policy Impact Assessment',
        empirical: 0.95,
        normative: 0.4,
        mythic: 0.1,
        author_did: generateTestDID(),
      })
    );

    // 2. Use in governance proposal
    const proposal = await retry(() =>
      governanceClient.broadcastGovernanceEvent({
        event_type: 'ProposalCreated',
        proposal_hash: generateTestHash(),
        payload: JSON.stringify({
          title: 'Energy Transition Policy',
          supporting_evidence: [claim.id],
        }),
      })
    );

    expect(claim).toBeDefined();
    expect(proposal.event_type).toBe('ProposalCreated');

    await client.disconnect();
  });

  it('should coordinate media → justice dispute flow', async () => {
    const { getMediaBridgeClient } = await import('../../src/integrations/media/index.js');
    const { getJusticeBridgeClient } = await import('../../src/integrations/justice/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const mediaClient = getMediaBridgeClient(client);
    const justiceClient = getJusticeBridgeClient(client);

    const contentId = `content-${Date.now()}`;
    const contentOwner = generateTestDID();
    const infringer = generateTestDID();

    // 1. Verify license (should fail for unlicensed use)
    const licenseCheck = await retry(() =>
      mediaClient.verifyLicense({
        content_id: contentId,
        licensee_did: infringer,
      })
    );

    // 2. File copyright dispute if unlicensed
    if (!licenseCheck.valid) {
      const dispute = await retry(() =>
        justiceClient.fileCrossHappDispute({
          dispute_type: 'CopyrightInfringement',
          respondent_did: infringer,
          related_happs: ['media-platform'],
          title: 'Unlicensed Content Use',
          description: 'Content used without proper license',
          evidence_hashes: [generateTestHash()],
        })
      );

      expect(dispute.id).toBeDefined();
    }

    expect(licenseCheck).toBeDefined();

    await client.disconnect();
  });
});

// ============================================================================
// Stress Tests
// ============================================================================

describeIf('Conductor Stress Tests', () => {
  let ctx: TestContext;

  beforeAll(async () => {
    ctx = await setupTestContext();
  }, 30000);

  afterAll(async () => {
    await teardownTestContext(ctx);
  });

  it('should handle concurrent zome calls', async () => {
    const { getIdentityBridgeClient } = await import('../../src/integrations/identity/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const identityClient = getIdentityBridgeClient(client);

    // Fire 20 concurrent queries
    const queries = Array.from({ length: 20 }, (_, i) =>
      identityClient.queryIdentity({
        did: generateTestDID(),
        source_happ: 'stress-test',
        requested_fields: ['matl_score'],
      })
    );

    const start = performance.now();
    const results = await Promise.allSettled(queries);
    const duration = performance.now() - start;

    const successful = results.filter((r) => r.status === 'fulfilled').length;

    console.log(
      `Concurrent queries: ${successful}/${queries.length} succeeded in ${duration.toFixed(0)}ms`
    );

    expect(successful).toBeGreaterThan(15); // At least 75% success rate

    await client.disconnect();
  }, 60000);

  it('should handle rapid sequential operations', async () => {
    const { getIdentityBridgeClient } = await import('../../src/integrations/identity/index.js');
    const { MycelixClient } = await import('../../src/client/index.js');

    const client = new MycelixClient({
      appUrl: ctx.config.appUrl,
      adminUrl: ctx.config.adminUrl,
      installedAppId: ctx.config.appId,
    });
    await client.connect();

    const identityClient = getIdentityBridgeClient(client);

    const iterations = 50;
    let successful = 0;

    const start = performance.now();

    for (let i = 0; i < iterations; i++) {
      try {
        await identityClient.reportReputation({
          target_did: generateTestDID(),
          delta: Math.random() * 0.2 - 0.1,
          reason: 'Stress test iteration',
          evidence: { iteration: i },
        });
        successful++;
      } catch {
        // Continue on error
      }
    }

    const duration = performance.now() - start;
    const opsPerSecond = (successful / duration) * 1000;

    console.log(
      `Sequential operations: ${successful}/${iterations} at ${opsPerSecond.toFixed(1)} ops/sec`
    );

    expect(successful).toBeGreaterThan(40); // At least 80% success rate

    await client.disconnect();
  }, 120000);
});
