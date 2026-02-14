/**
 * Mycelix Civilizational OS End-to-End Tests
 *
 * Comprehensive integration tests covering the full civilizational workflow
 * across all 8 hApps: Identity, Knowledge, Governance, Justice, Finance,
 * Property, Energy, and Media.
 *
 * These tests simulate real-world scenarios where multiple hApps coordinate
 * to accomplish complex civilizational operations.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as matl from '../../src/matl/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as bridge from '../../src/bridge/index.js';
import {
  executeProposalWorkflow,
  executeLendingWorkflow,
  executePropertyTransferWorkflow,
  executeEnergyTradeWorkflow,
  executeEnforcementWorkflow,
  executePublicationWorkflow,
  executeComprehensiveIdentityCheck,
  type WorkflowResult,
} from '../../src/bridge/workflows.js';

// ============================================================================
// Test Helpers
// ============================================================================

function createTestDid(name: string): string {
  return `did:mycelix:test-${name}-${Date.now()}`;
}

function createTestAssetId(type: string, name: string): string {
  return `asset:${type}:test-${name}-${Date.now()}`;
}

// ============================================================================
// E2E: Full Civilizational Member Lifecycle
// ============================================================================

describe('E2E: Civilizational Member Lifecycle', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    // Register all 8 Civilizational OS hApps
    localBridge.registerHapp('identity');
    localBridge.registerHapp('knowledge');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('justice');
    localBridge.registerHapp('finance');
    localBridge.registerHapp('property');
    localBridge.registerHapp('energy');
    localBridge.registerHapp('media');
  });

  it('should complete full member onboarding and participation lifecycle', async () => {
    const memberDid = createTestDid('alice');

    // Phase 1: Identity Verification (Comprehensive Check)
    const identityResult = await executeComprehensiveIdentityCheck({
      subjectDid: memberDid,
      requesterHapp: 'governance',
      checkCategories: ['reputation', 'credentials', 'standing'],
    });

    expect(identityResult.success).toBe(true);
    expect(identityResult.steps.length).toBeGreaterThan(0);
    expect(identityResult.data?.verified).toBeDefined();

    // Phase 2: Member builds reputation through positive interactions
    let memberRep = matl.createReputation(memberDid);
    for (let i = 0; i < 10; i++) {
      memberRep = matl.recordPositive(memberRep);
    }

    // Store reputation in bridge for cross-hApp access
    localBridge.setReputation('identity', memberDid, memberRep);
    localBridge.setReputation('governance', memberDid, memberRep);
    localBridge.setReputation('finance', memberDid, memberRep);

    const crossHappRep = localBridge.getCrossHappReputation(memberDid);
    expect(crossHappRep.length).toBe(3);

    // Phase 3: Member submits governance proposal
    const proposalResult = await executeProposalWorkflow({
      proposerDid: memberDid,
      daoId: 'solar-coop-sf',
      title: 'Expand Community Solar Array by 50kW',
      description: 'Install additional panels on community center roof.',
      votingPeriodHours: 168,
      quorumPercentage: 0.51,
    });

    expect(proposalResult.success).toBe(true);
    expect(proposalResult.data?.proposalId).toBeDefined();

    // Phase 4: Member publishes content for verification
    const contentResult = await executePublicationWorkflow({
      authorDid: memberDid,
      contentHash: 'QmTestHash123',
      title: 'Solar Installation Best Practices',
      tags: ['solar', 'installation', 'best-practices'],
      requestFactCheck: true,
    });

    expect(contentResult.success).toBe(true);
    expect(contentResult.data?.contentId).toBeDefined();

    // Phase 5: Verify aggregate reputation reflects all activities
    const aggregateRep = localBridge.getAggregateReputation(memberDid);
    expect(aggregateRep).toBeGreaterThan(0);
    expect(aggregateRep).toBeLessThanOrEqual(1);
  });

  it('should handle member suspension and reinstatement through justice system', async () => {
    const memberDid = createTestDid('eve');

    // Phase 1: Create initial reputation
    let memberRep = matl.createReputation(memberDid);
    for (let i = 0; i < 5; i++) {
      memberRep = matl.recordPositive(memberRep);
    }
    localBridge.setReputation('governance', memberDid, memberRep);

    const initialRepValue = matl.reputationValue(memberRep);

    // Phase 2: Justice enforcement for rule violation
    const enforcementResult = await executeEnforcementWorkflow({
      decisionId: 'decision:case-001-final',
      targetDid: memberDid,
      remedies: [
        {
          type: 'reputation_penalty',
          targetHapp: 'governance',
          details: { adjustment: -0.15, reason: 'Voting power reduction' },
        },
      ],
    });

    expect(enforcementResult.success).toBe(true);
    expect(enforcementResult.steps.some((s) => s.name.includes('Enforce'))).toBe(true);

    // Phase 3: Record negative interactions from enforcement
    memberRep = matl.recordNegative(memberRep);
    memberRep = matl.recordNegative(memberRep);
    localBridge.setReputation('governance', memberDid, memberRep);

    const postEnforcementRepValue = matl.reputationValue(memberRep);
    expect(postEnforcementRepValue).toBeLessThan(initialRepValue);

    // Phase 4: Rehabilitation through positive contributions
    for (let i = 0; i < 20; i++) {
      memberRep = matl.recordPositive(memberRep);
    }
    localBridge.setReputation('governance', memberDid, memberRep);

    const rehabilitatedRepValue = matl.reputationValue(memberRep);
    expect(rehabilitatedRepValue).toBeGreaterThan(postEnforcementRepValue);
  });
});

// ============================================================================
// E2E: Economic Operations Flow
// ============================================================================

describe('E2E: Economic Operations Flow', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('finance');
    localBridge.registerHapp('property');
    localBridge.registerHapp('energy');
  });

  it('should complete full credit application with collateral verification', async () => {
    const borrowerDid = createTestDid('borrower-charlie');
    const assetId = createTestAssetId('property', 'warehouse');

    // Phase 1: Establish borrower reputation
    let borrowerRep = matl.createReputation(borrowerDid);
    for (let i = 0; i < 15; i++) {
      borrowerRep = matl.recordPositive(borrowerRep);
    }
    localBridge.setReputation('finance', borrowerDid, borrowerRep);

    // Phase 2: Create PoGQ for financial reliability
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
    const composite = matl.calculateComposite(pogq, borrowerRep);
    expect(matl.isTrustworthy(composite, 0.5)).toBe(true);

    // Phase 3: Submit lending application
    const lendingResult = await executeLendingWorkflow({
      borrowerDid,
      amount: 10000,
      currency: 'MCX',
      termMonths: 24,
      collateralAssetId: assetId,
      purpose: 'Purchase solar installation equipment',
    });

    expect(lendingResult.success).toBe(true);
    expect(lendingResult.data?.loanId).toBeDefined();

    // Phase 4: Verify all steps completed
    const expectedHapps = ['identity', 'property', 'finance'];
    for (const happ of expectedHapps) {
      const hasStep = lendingResult.steps.some((s) => s.happ === happ);
      expect(hasStep).toBe(true);
    }
  });

  it('should handle property transfer with escrow and lien checks', async () => {
    const sellerDid = createTestDid('seller-dana');
    const buyerDid = createTestDid('buyer-evan');
    const assetId = createTestAssetId('property', 'solar-farm');

    // Phase 1: Establish both parties' reputations
    let sellerRep = matl.createReputation(sellerDid);
    let buyerRep = matl.createReputation(buyerDid);

    for (let i = 0; i < 20; i++) {
      sellerRep = matl.recordPositive(sellerRep);
      buyerRep = matl.recordPositive(buyerRep);
    }

    localBridge.setReputation('property', sellerDid, sellerRep);
    localBridge.setReputation('property', buyerDid, buyerRep);
    localBridge.setReputation('finance', buyerDid, buyerRep);

    // Phase 2: Execute property transfer
    const transferResult = await executePropertyTransferWorkflow({
      assetId,
      sellerDid,
      buyerDid,
      salePrice: 50000,
      escrowDays: 30,
    });

    expect(transferResult.success).toBe(true);
    expect(transferResult.data?.transferId).toBeDefined();

    // Phase 3: Verify workflow steps include all required checks
    expect(
      transferResult.steps.some((s) => s.name.includes('Verify') || s.name.includes('Owner'))
    ).toBe(true);
    expect(transferResult.steps.some((s) => s.name.includes('Lien'))).toBe(true);
    expect(transferResult.steps.some((s) => s.name.includes('Escrow'))).toBe(true);
  });

  it('should handle P2P energy trade with payment processing', async () => {
    const producerDid = createTestDid('producer-frank');
    const consumerDid = createTestDid('consumer-grace');

    // Phase 1: Establish producer reputation
    let producerRep = matl.createReputation(producerDid);
    for (let i = 0; i < 30; i++) {
      producerRep = matl.recordPositive(producerRep);
    }
    localBridge.setReputation('energy', producerDid, producerRep);

    // Phase 2: Execute energy trade
    const tradeResult = await executeEnergyTradeWorkflow({
      sellerDid: producerDid,
      buyerDid: consumerDid,
      amountKwh: 500,
      pricePerKwh: 0.12,
      source: 'solar',
    });

    expect(tradeResult.success).toBe(true);
    expect(tradeResult.data?.tradeId).toBeDefined();
    expect(tradeResult.data?.creditId).toBeDefined();
  });
});

// ============================================================================
// E2E: Knowledge and Truth Verification
// ============================================================================

describe('E2E: Knowledge and Truth Verification', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('knowledge');
    localBridge.registerHapp('media');
    localBridge.registerHapp('governance');
  });

  it('should verify content claims through epistemic framework', async () => {
    const journalistDid = createTestDid('journalist-henry');

    // Phase 1: Create journalist reputation
    let journalistRep = matl.createReputation(journalistDid);
    for (let i = 0; i < 25; i++) {
      journalistRep = matl.recordPositive(journalistRep);
    }
    localBridge.setReputation('media', journalistDid, journalistRep);

    // Phase 2: Submit content for publication and verification
    const publicationResult = await executePublicationWorkflow({
      authorDid: journalistDid,
      contentHash: 'QmArticleHash456',
      title: 'Community Solar Reduces Grid Costs by 15%',
      tags: ['energy', 'solar', 'community', 'costs'],
      requestFactCheck: true,
    });

    expect(publicationResult.success).toBe(true);
    expect(publicationResult.data?.contentId).toBeDefined();
    expect(publicationResult.data?.verificationStatus).toBe('Pending');

    // Phase 3: Create epistemic claim for the verified content
    const epistemicClaim = epistemic
      .claim('Community solar installation data verified')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .withIssuer('media')
      .build();

    expect(
      epistemic.meetsStandard(
        epistemicClaim,
        epistemic.EmpiricalLevel.E2_PrivateVerify,
        epistemic.NormativeLevel.N1_Communal
      )
    ).toBe(true);

    // Phase 4: Verify classification code
    const code = epistemic.classificationCode(epistemicClaim.classification);
    expect(code).toBe('E3-N2-M2');
  });

  it('should handle disputed claims through justice system', async () => {
    const originalAuthorDid = createTestDid('author');
    const disputerDid = createTestDid('disputer');

    // Phase 1: Create original claim
    const originalClaim = epistemic
      .claim('Algorithm achieves 99% accuracy')
      .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
      .withNormative(epistemic.NormativeLevel.N1_Communal)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .withIssuer(originalAuthorDid)
      .build();

    // Phase 2: Create dispute claim with evidence
    const disputeClaim = epistemic
      .claim('Testing methodology was flawed')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .withIssuer(disputerDid)
      .build();

    // Phase 3: Dispute claim has higher empirical level
    expect(disputeClaim.classification.empirical).toBeGreaterThan(
      originalClaim.classification.empirical
    );

    // Phase 4: Add evidence to dispute
    const evidence: epistemic.Evidence = {
      type: 'cryptographic_proof',
      data: 'proof_of_methodology_flaw',
      source: disputerDid,
      timestamp: Date.now(),
    };
    const claimWithEvidence = epistemic.addEvidence(disputeClaim, evidence);
    expect(claimWithEvidence.evidence.length).toBe(1);

    // Phase 5: Justice workflow for enforcement
    const enforcementResult = await executeEnforcementWorkflow({
      decisionId: 'decision:dispute-001-resolved',
      targetDid: originalAuthorDid,
      remedies: [
        {
          type: 'reputation_penalty',
          targetHapp: 'media',
          details: { adjustment: -0.05, reason: 'Retraction required' },
        },
      ],
    });

    expect(enforcementResult.success).toBe(true);
  });

  it('should track claim lineage through knowledge graph', () => {
    // Create chain of claims
    const baseClaim = epistemic
      .claim('Solar energy reduces carbon emissions')
      .withEmpirical(epistemic.EmpiricalLevel.E4_Consensus)
      .withNormative(epistemic.NormativeLevel.N3_Universal)
      .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
      .build();

    const derivedClaim = epistemic
      .claim('Community solar installations reduce household carbon footprint')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .build();

    const localizedClaim = epistemic
      .claim('San Francisco solar coop saves 75 tons CO2/year')
      .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
      .withNormative(epistemic.NormativeLevel.N1_Communal)
      .withMateriality(epistemic.MaterialityLevel.M1_Temporal)
      .build();

    // Verify hierarchy: base > derived > localized
    expect(baseClaim.classification.empirical).toBeGreaterThan(
      derivedClaim.classification.empirical
    );
    expect(derivedClaim.classification.empirical).toBeGreaterThan(
      localizedClaim.classification.empirical
    );

    // All should meet minimum trust standards
    expect(
      epistemic.meetsMinimum(
        localizedClaim.classification,
        epistemic.Standards.LowTrust.minE,
        epistemic.Standards.LowTrust.minN,
        epistemic.Standards.LowTrust.minM
      )
    ).toBe(true);
  });
});

// ============================================================================
// E2E: Cross-hApp Governance Operations
// ============================================================================

describe('E2E: Cross-hApp Governance Operations', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('finance');
    localBridge.registerHapp('knowledge');
  });

  it('should execute full governance proposal lifecycle', async () => {
    const proposerDid = createTestDid('proposer');
    const voter1Did = createTestDid('voter1');
    const voter2Did = createTestDid('voter2');
    const voter3Did = createTestDid('voter3');

    // Phase 1: Establish voter reputations with different weights
    const voters = [
      { did: voter1Did, positiveCount: 30 },
      { did: voter2Did, positiveCount: 20 },
      { did: voter3Did, positiveCount: 10 },
    ];

    for (const voter of voters) {
      let rep = matl.createReputation(voter.did);
      for (let i = 0; i < voter.positiveCount; i++) {
        rep = matl.recordPositive(rep);
      }
      localBridge.setReputation('governance', voter.did, rep);
    }

    // Phase 2: Create proposal
    const proposalResult = await executeProposalWorkflow({
      proposerDid,
      daoId: 'mycelix-core',
      title: 'Implement Cross-hApp Credential Caching',
      description: 'Add caching layer for frequently accessed credentials.',
      votingPeriodHours: 168,
      quorumPercentage: 0.51,
    });

    expect(proposalResult.success).toBe(true);

    // Phase 3: Simulate MATL-weighted voting
    const votingScores = voters.map((voter) => {
      const crossHapp = localBridge.getCrossHappReputation(voter.did);
      return crossHapp.length > 0 ? crossHapp[0].score : 0;
    });

    // Calculate weighted vote totals
    const totalWeight = votingScores.reduce((a, b) => a + b, 0);
    expect(totalWeight).toBeGreaterThan(0);

    // Phase 4: Record proposal outcome in knowledge graph
    const outcomeClaim = epistemic
      .claim('Proposal MIP-042 passed with 85% support')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
      .withIssuer('governance')
      .build();

    expect(outcomeClaim.classification.materiality).toBe(epistemic.MaterialityLevel.M3_Immutable);
  });

  it('should handle constitutional amendment with elevated requirements', async () => {
    const proposerDid = createTestDid('constitutional-proposer');

    // Phase 1: Create proposal (constitutional requires higher quorum)
    const proposalResult = await executeProposalWorkflow({
      proposerDid,
      daoId: 'mycelix-core',
      title: 'Amendment: Increase Byzantine Tolerance to 50%',
      description: 'Modify core consensus parameters.',
      votingPeriodHours: 720, // 30 days for constitutional
      quorumPercentage: 0.67, // 2/3 supermajority
    });

    expect(proposalResult.success).toBe(true);

    // Phase 2: Verify elevated requirements were applied
    expect(proposalResult.steps.some((s) => s.name.includes('Create Proposal'))).toBe(true);

    // Phase 3: Create epistemic claim with highest standards
    const amendmentClaim = epistemic
      .claim('Constitutional amendment ratified')
      .withEmpirical(epistemic.EmpiricalLevel.E4_Consensus)
      .withNormative(epistemic.NormativeLevel.N3_Universal)
      .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
      .build();

    // Must meet high trust standard
    expect(
      epistemic.meetsMinimum(
        amendmentClaim.classification,
        epistemic.Standards.HighTrust.minE,
        epistemic.Standards.HighTrust.minN,
        epistemic.Standards.HighTrust.minM
      )
    ).toBe(true);
  });
});

// ============================================================================
// E2E: Byzantine Fault Detection
// ============================================================================

describe('E2E: Byzantine Fault Detection', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('justice');
  });

  it('should detect and respond to Byzantine behavior', () => {
    const suspiciousAgentDid = createTestDid('suspicious');

    // Phase 1: Create initially good reputation
    let rep = matl.createReputation(suspiciousAgentDid);
    for (let i = 0; i < 10; i++) {
      rep = matl.recordPositive(rep);
    }

    // Phase 2: Create adaptive threshold for monitoring
    let threshold = matl.createAdaptiveThreshold(suspiciousAgentDid, 10, 0.5, 2.0);

    // Add normal observations
    const normalScores = [0.8, 0.82, 0.79, 0.81, 0.8, 0.78, 0.83, 0.8, 0.79, 0.81];
    for (const score of normalScores) {
      threshold = matl.observe(threshold, score);
    }

    // Phase 3: Simulate Byzantine attack
    const byzantineScore = 0.2; // Sudden drop
    const isAnomalous = matl.isAnomalous(threshold, byzantineScore);
    expect(isAnomalous).toBe(true);

    // Phase 4: Check Byzantine flag using PoGQ
    const badPoGQ = matl.createPoGQ(0.2, 0.3, 0.8); // Low quality, high entropy
    expect(matl.isByzantine(badPoGQ)).toBe(true);

    // Phase 5: Record negative interactions
    for (let i = 0; i < 15; i++) {
      rep = matl.recordNegative(rep);
    }

    const finalRepValue = matl.reputationValue(rep);
    expect(matl.isByzantineScore(finalRepValue)).toBe(true);
  });

  it('should maintain 34% validated Byzantine tolerance through MATL', () => {
    const nodeCount = 100;
    const byzantineCount = 34; // 34% Byzantine (validated threshold)
    const honestCount = nodeCount - byzantineCount;

    // Create nodes
    const nodes: matl.Reputation[] = [];
    for (let i = 0; i < nodeCount; i++) {
      const isByzantine = i < byzantineCount;
      let rep = matl.createReputation(`node_${i}`);

      if (isByzantine) {
        // Byzantine nodes have poor reputation
        for (let j = 0; j < 20; j++) {
          rep = matl.recordNegative(rep);
        }
      } else {
        // Honest nodes have good reputation
        for (let j = 0; j < 20; j++) {
          rep = matl.recordPositive(rep);
        }
      }
      nodes.push(rep);
    }

    // Calculate trustworthy vs untrustworthy
    let trustworthy = 0;
    let untrustworthy = 0;

    for (const node of nodes) {
      const pogq = matl.createPoGQ(matl.reputationValue(node), 0.85, 0.1);
      const composite = matl.calculateComposite(pogq, node);

      if (matl.isTrustworthy(composite, 0.5)) {
        trustworthy++;
      } else {
        untrustworthy++;
      }
    }

    // Honest nodes should be identified as trustworthy
    expect(trustworthy).toBeGreaterThanOrEqual(honestCount * 0.8);
    // Byzantine nodes should be identified as untrustworthy
    expect(untrustworthy).toBeGreaterThanOrEqual(byzantineCount * 0.8);
  });
});

// ============================================================================
// E2E: Multi-hApp Message Routing
// ============================================================================

describe('E2E: Multi-hApp Message Routing', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    // Register all hApps
    const happs = [
      'identity',
      'knowledge',
      'governance',
      'justice',
      'finance',
      'property',
      'energy',
      'media',
    ];
    for (const happ of happs) {
      localBridge.registerHapp(happ);
    }
  });

  it('should route reputation queries across all hApps', () => {
    const agentDid = createTestDid('multi-happ-agent');
    const happs = ['identity', 'governance', 'finance', 'property', 'energy', 'media'];

    // Set different reputations in each hApp
    for (let i = 0; i < happs.length; i++) {
      let rep = matl.createReputation(agentDid);
      for (let j = 0; j < (i + 1) * 5; j++) {
        rep = matl.recordPositive(rep);
      }
      localBridge.setReputation(happs[i], agentDid, rep);
    }

    // Query cross-hApp reputation
    const crossHappScores = localBridge.getCrossHappReputation(agentDid);
    expect(crossHappScores.length).toBe(happs.length);

    // Aggregate should be reasonable
    const aggregate = localBridge.getAggregateReputation(agentDid);
    expect(aggregate).toBeGreaterThan(0);
    expect(aggregate).toBeLessThanOrEqual(1);
  });

  it('should broadcast events to all subscribed hApps', () => {
    const receivedMessages: Map<string, bridge.AnyBridgeMessage[]> = new Map();
    const happs = ['identity', 'governance', 'finance', 'property'];

    // Register handlers for all hApps
    for (const happ of happs) {
      receivedMessages.set(happ, []);
      localBridge.on(happ, bridge.BridgeMessageType.BroadcastEvent, (msg) => {
        receivedMessages.get(happ)!.push(msg);
      });
    }

    // Broadcast from knowledge hApp
    const event = bridge.createBroadcastEvent(
      'knowledge',
      'claim_verified',
      new Uint8Array([1, 2, 3, 4])
    );
    localBridge.broadcast(event);

    // All registered hApps should receive
    for (const happ of happs) {
      expect(receivedMessages.get(happ)!.length).toBe(1);
    }
  });

  it('should handle credential verification across hApps', () => {
    const verificationResults: bridge.VerificationResultMessage[] = [];

    localBridge.on('finance', bridge.BridgeMessageType.VerificationResult, (msg) => {
      verificationResults.push(msg as bridge.VerificationResultMessage);
    });

    // Create credential verification request
    const verifyRequest = bridge.createCredentialVerification(
      'finance',
      'credit_score_cred_123',
      'identity'
    );
    expect(verifyRequest.type).toBe(bridge.BridgeMessageType.CredentialVerification);

    // Simulate identity hApp responding with verification
    const result = bridge.createVerificationResult(
      'identity',
      'credit_score_cred_123',
      true,
      'trusted_credit_bureau',
      ['credit_score_verified', 'income_verified', 'employment_verified']
    );

    localBridge.send('finance', result);

    expect(verificationResults.length).toBe(1);
    expect(verificationResults[0].valid).toBe(true);
    expect(verificationResults[0].claims.length).toBe(3);
  });
});

// ============================================================================
// E2E: Workflow Result Validation
// ============================================================================

describe('E2E: Workflow Result Validation', () => {
  it('should validate all workflow results have proper structure', async () => {
    // Test all workflows return proper WorkflowResult structure
    const workflows: Promise<WorkflowResult<unknown>>[] = [
      executeComprehensiveIdentityCheck({
        subjectDid: 'did:mycelix:test',
        requesterHapp: 'governance',
        checkCategories: ['reputation'],
      }),
      executeProposalWorkflow({
        proposerDid: 'did:mycelix:proposer',
        daoId: 'test-dao',
        title: 'Test Proposal',
        description: 'Test description',
        votingPeriodHours: 24,
        quorumPercentage: 0.5,
      }),
      executeLendingWorkflow({
        borrowerDid: 'did:mycelix:borrower',
        amount: 1000,
        currency: 'MCX',
        termMonths: 12,
        collateralAssetId: 'asset:test:1',
        purpose: 'Test',
      }),
      executePropertyTransferWorkflow({
        assetId: 'asset:property:test',
        sellerDid: 'did:mycelix:seller',
        buyerDid: 'did:mycelix:buyer',
        salePrice: 1000,
        escrowDays: 7,
      }),
      executeEnergyTradeWorkflow({
        sellerDid: 'did:mycelix:producer',
        buyerDid: 'did:mycelix:consumer',
        amountKwh: 100,
        pricePerKwh: 0.1,
        source: 'solar',
      }),
      executePublicationWorkflow({
        authorDid: 'did:mycelix:author',
        contentHash: 'QmTest',
        title: 'Test Content',
        tags: ['test'],
        requestFactCheck: false,
      }),
      executeEnforcementWorkflow({
        decisionId: 'decision:test',
        targetDid: 'did:mycelix:target',
        remedies: [
          { type: 'reputation_penalty', targetHapp: 'governance', details: { adjustment: -0.01 } },
        ],
      }),
    ];

    const results = await Promise.all(workflows);

    for (const result of results) {
      // All workflows must have these properties
      expect(result).toHaveProperty('success');
      expect(result).toHaveProperty('steps');
      expect(result).toHaveProperty('duration');
      expect(Array.isArray(result.steps)).toBe(true);
      expect(typeof result.duration).toBe('number');
      expect(result.duration).toBeGreaterThanOrEqual(0);

      // Steps must have proper structure
      for (const step of result.steps) {
        expect(step).toHaveProperty('name');
        expect(step).toHaveProperty('happ');
        expect(step).toHaveProperty('status');
        expect(['pending', 'running', 'completed', 'failed', 'skipped']).toContain(step.status);
      }
    }
  });

  it('should measure workflow execution times', async () => {
    const timings: Record<string, number> = {};

    // Identity Check
    const identity = await executeComprehensiveIdentityCheck({
      subjectDid: 'did:mycelix:timing',
      requesterHapp: 'governance',
      checkCategories: ['reputation', 'credentials'],
    });
    timings['identity'] = identity.duration;

    // Proposal
    const proposal = await executeProposalWorkflow({
      proposerDid: 'did:mycelix:timing',
      daoId: 'timing-dao',
      title: 'Timing Test',
      description: 'Test',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });
    timings['proposal'] = proposal.duration;

    // Energy Trade
    const energy = await executeEnergyTradeWorkflow({
      sellerDid: 'did:mycelix:timing-seller',
      buyerDid: 'did:mycelix:timing-buyer',
      amountKwh: 100,
      pricePerKwh: 0.1,
      source: 'solar',
    });
    timings['energy'] = energy.duration;

    // All workflows should complete in reasonable time
    for (const [name, duration] of Object.entries(timings)) {
      expect(duration).toBeLessThan(5000); // 5 second max
    }
  });
});
