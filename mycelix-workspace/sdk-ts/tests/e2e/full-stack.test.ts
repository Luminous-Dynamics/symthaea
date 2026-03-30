// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix End-to-End Integration Tests
 *
 * Tests the complete flow from TypeScript SDK through to zome interactions.
 * Uses actual SDK modules with realistic test scenarios.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as matl from '../../src/matl/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as bridge from '../../src/bridge/index.js';

describe('E2E: Complete Trust Flow', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('marketplace');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('trust_layer');
  });

  it('should complete full trust verification workflow', () => {
    const agentId = 'agent_001';

    // Step 1: Create PoGQ measurement
    const pogq = matl.createPoGQ(0.85, 0.9, 0.1);
    expect(pogq.quality).toBe(0.85);
    expect(pogq.consistency).toBe(0.9);

    // Step 2: Create reputation
    const reputation = matl.createReputation(agentId);
    expect(reputation.agentId).toBe(agentId);

    // Step 3: Record positive interactions
    let updatedRep = matl.recordPositive(reputation);
    updatedRep = matl.recordPositive(updatedRep);
    updatedRep = matl.recordPositive(updatedRep);
    expect(updatedRep.positiveCount).toBe(4); // 1 prior + 3

    // Step 4: Calculate composite score
    const composite = matl.calculateComposite(pogq, updatedRep);
    expect(composite.finalScore).toBeGreaterThan(0);
    expect(composite.confidence).toBeGreaterThan(0);

    // Step 5: Check trustworthiness
    const trusted = matl.isTrustworthy(composite, 0.5);
    expect(trusted).toBe(true);

    // Step 6: Create epistemic claim about verification
    const claim = epistemic.claim('Agent passed trust verification')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .withIssuer('trust_layer')
      .build();

    expect(claim.classification.empirical).toBe(epistemic.EmpiricalLevel.E3_Cryptographic);
    expect(claim.classification.normative).toBe(epistemic.NormativeLevel.N2_Network);

    // Step 7: Verify claim meets standard
    const meetsStandard = epistemic.meetsStandard(
      claim,
      epistemic.EmpiricalLevel.E2_PrivateVerify,
      epistemic.NormativeLevel.N1_Communal
    );
    expect(meetsStandard).toBe(true);
  });

  it('should handle reputation updates across multiple hApps', () => {
    const agentId = 'multi_happ_agent';

    // Create reputations in different hApps
    const marketplaceRep = matl.createReputation(agentId);
    const governanceRep = matl.createReputation(agentId);
    const socialRep = matl.createReputation(agentId);

    // Simulate interactions
    let mpRep = marketplaceRep;
    for (let i = 0; i < 10; i++) {
      mpRep = matl.recordPositive(mpRep);
    }

    let govRep = governanceRep;
    for (let i = 0; i < 5; i++) {
      govRep = matl.recordPositive(govRep);
      govRep = matl.recordNegative(govRep);
    }

    // Store in bridge
    localBridge.setReputation('marketplace', agentId, mpRep);
    localBridge.setReputation('governance', agentId, govRep);
    localBridge.setReputation('social', agentId, socialRep);

    // Get cross-hApp reputation
    const crossHappScores = localBridge.getCrossHappReputation(agentId);
    expect(crossHappScores.length).toBe(3);

    // Calculate aggregate
    const aggregate = localBridge.getAggregateReputation(agentId);
    expect(aggregate).toBeGreaterThan(0);
    expect(aggregate).toBeLessThanOrEqual(1);
  });

  it('should detect anomalous reputation changes', () => {
    const nodeId = 'monitored_node';

    // Create adaptive threshold
    let threshold = matl.createAdaptiveThreshold(nodeId, 10, 0.5, 2.0);

    // Add normal observations
    const normalScores = [0.8, 0.82, 0.79, 0.81, 0.8, 0.78, 0.83, 0.8, 0.79, 0.81];
    for (const score of normalScores) {
      threshold = matl.observe(threshold, score);
    }

    // Get threshold value
    const currentThreshold = matl.getThreshold(threshold);
    expect(currentThreshold).toBeGreaterThan(0);

    // Normal value should not be anomalous
    expect(matl.isAnomalous(threshold, 0.8)).toBe(false);
    expect(matl.isAnomalous(threshold, 0.78)).toBe(false);

    // Sudden drop should be anomalous
    expect(matl.isAnomalous(threshold, 0.2)).toBe(true);
  });
});

describe('E2E: Epistemic Verification Flow', () => {
  it('should track claim verification through lifecycle', () => {
    // Step 1: Create initial claim
    const claim = epistemic.claim('Smart contract audited by 3 independent parties')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
      .withIssuer('audit_firm')
      .build();

    expect(claim.classification.empirical).toBe(epistemic.EmpiricalLevel.E3_Cryptographic);

    // Step 2: Get classification code
    const code = epistemic.classificationCode(claim.classification);
    expect(code).toBe('E3-N2-M3');

    // Step 3: Check against standards
    expect(epistemic.meetsMinimum(
      claim.classification,
      epistemic.EmpiricalLevel.E2_PrivateVerify,
      epistemic.NormativeLevel.N1_Communal,
      epistemic.MaterialityLevel.M2_Persistent
    )).toBe(true);

    // Step 4: Add evidence
    const evidence: epistemic.Evidence = {
      type: 'cryptographic_signature',
      data: 'sig_abc123...',
      source: 'auditor_1',
      timestamp: Date.now(),
    };
    const claimWithEvidence = epistemic.addEvidence(claim, evidence);
    expect(claimWithEvidence.evidence.length).toBe(1);
  });

  it('should handle claim disputes and resolution', () => {
    // Original claim
    const originalClaim = epistemic.claim('Transaction completed successfully')
      .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
      .withNormative(epistemic.NormativeLevel.N1_Communal)
      .build();

    // Dispute claim (lower empirical level)
    const disputeClaim = epistemic.claim('Transaction shows inconsistencies')
      .withEmpirical(epistemic.EmpiricalLevel.E1_Testimonial)
      .withNormative(epistemic.NormativeLevel.N1_Communal)
      .build();

    // Resolution claim (highest empirical level - cryptographic proof)
    const resolutionClaim = epistemic.claim('Blockchain confirms transaction valid')
      .withEmpirical(epistemic.EmpiricalLevel.E4_Consensus)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .build();

    // Resolution should have highest empirical level
    expect(resolutionClaim.classification.empirical).toBeGreaterThan(
      originalClaim.classification.empirical
    );
    expect(resolutionClaim.classification.empirical).toBeGreaterThan(
      disputeClaim.classification.empirical
    );

    // Resolution meets high trust standard
    expect(epistemic.meetsMinimum(
      resolutionClaim.classification,
      epistemic.EmpiricalLevel.E3_Cryptographic,
      epistemic.NormativeLevel.N2_Network
    )).toBe(true);
  });

  it('should parse and validate classification codes', () => {
    const codes = ['E0-N0-M0', 'E3-N2-M2', 'E4-N3-M3'];

    for (const code of codes) {
      const parsed = epistemic.parseClassificationCode(code);
      expect(parsed).not.toBeNull();
      expect(epistemic.classificationCode(parsed!)).toBe(code);
    }

    // Invalid code
    expect(epistemic.parseClassificationCode('invalid')).toBeNull();
  });
});

describe('E2E: Bridge Message Routing', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('marketplace');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('trust_layer');
  });

  it('should route messages between hApps', () => {
    const receivedMessages: bridge.AnyBridgeMessage[] = [];

    // Register handler for reputation queries
    localBridge.on('marketplace', bridge.BridgeMessageType.ReputationQuery, (msg) => {
      receivedMessages.push(msg);
    });

    // Send reputation query
    const query = bridge.createReputationQuery('trust_layer', 'agent_001');
    localBridge.send('marketplace', query);

    expect(receivedMessages.length).toBe(1);
    expect(receivedMessages[0].type).toBe(bridge.BridgeMessageType.ReputationQuery);
  });

  it('should calculate aggregate reputation from multiple sources', () => {
    const scores: bridge.HappReputationScore[] = [
      { happ: 'marketplace', score: 0.9, weight: 2.0, lastUpdate: Date.now() },
      { happ: 'social', score: 0.7, weight: 1.0, lastUpdate: Date.now() },
      { happ: 'governance', score: 0.95, weight: 0.5, lastUpdate: Date.now() },
    ];

    const aggregate = bridge.calculateAggregateReputation(scores);

    // Weighted average: (0.9*2 + 0.7*1 + 0.95*0.5) / (2+1+0.5)
    // = (1.8 + 0.7 + 0.475) / 3.5 = 2.975 / 3.5 ≈ 0.85
    expect(aggregate).toBeCloseTo(0.85, 2);
  });

  it('should broadcast events to all hApps', () => {
    const receivedByMarketplace: bridge.AnyBridgeMessage[] = [];
    const receivedByGovernance: bridge.AnyBridgeMessage[] = [];

    localBridge.on('marketplace', bridge.BridgeMessageType.BroadcastEvent, (msg) => {
      receivedByMarketplace.push(msg);
    });

    localBridge.on('governance', bridge.BridgeMessageType.BroadcastEvent, (msg) => {
      receivedByGovernance.push(msg);
    });

    // Broadcast from trust_layer
    const event = bridge.createBroadcastEvent(
      'trust_layer',
      'reputation_updated',
      new Uint8Array([1, 2, 3])
    );
    localBridge.broadcast(event);

    // Both should receive (but not the sender)
    expect(receivedByMarketplace.length).toBe(1);
    expect(receivedByGovernance.length).toBe(1);
  });

  it('should handle cross-hApp credential verification', () => {
    const verificationResults: bridge.VerificationResultMessage[] = [];

    localBridge.on('marketplace', bridge.BridgeMessageType.VerificationResult, (msg) => {
      verificationResults.push(msg as bridge.VerificationResultMessage);
    });

    // Create verification request
    const verifyRequest = bridge.createCredentialVerification(
      'marketplace',
      'cred_hash_123',
      'identity_happ'
    );
    expect(verifyRequest.type).toBe(bridge.BridgeMessageType.CredentialVerification);

    // Create verification result
    const result = bridge.createVerificationResult(
      'identity_happ',
      'cred_hash_123',
      true,
      'trusted_issuer',
      ['email_verified', 'kyc_passed']
    );

    localBridge.send('marketplace', result);

    expect(verificationResults.length).toBe(1);
    expect(verificationResults[0].valid).toBe(true);
    expect(verificationResults[0].claims).toContain('email_verified');
  });
});

describe('E2E: PoGQ Trust Verification', () => {
  it('should perform complete Proof of Gradient Quality verification', () => {
    // Step 1: Create PoGQ for content
    const pogq = matl.createPoGQ(0.92, 0.88, 0.05);

    expect(pogq.quality).toBe(0.92);
    expect(pogq.consistency).toBe(0.88);
    expect(pogq.entropy).toBe(0.05);

    // Step 2: Create reputation for verifier
    const reputation = matl.createReputation('verifier_001');

    // Step 3: Calculate composite score
    const composite = matl.calculateComposite(pogq, reputation);
    expect(composite.finalScore).toBeGreaterThan(0.5);

    // Step 4: Check if Byzantine behavior using PoGQ
    const isByzantineFromPoGQ = matl.isByzantine(pogq);
    expect(isByzantineFromPoGQ).toBe(false);

    // Also check using score directly
    const isByzantineFromScore = matl.isByzantineScore(composite.finalScore);
    expect(isByzantineFromScore).toBe(false);
  });

  it('should detect Byzantine behavior', () => {
    // Create low quality PoGQ
    const badPogq = matl.createPoGQ(0.2, 0.3, 0.8); // Low quality, high entropy

    const reputation = matl.createReputation('suspicious_agent');

    const composite = matl.calculateComposite(badPogq, reputation);

    // Should be detected as Byzantine using PoGQ (quality=0.2 * (1 - entropy=0.8) = 0.04 < 0.5)
    const isByzantineFromPoGQ = matl.isByzantine(badPogq);
    expect(isByzantineFromPoGQ).toBe(true);
  });

  it('should handle reputation decay through negative interactions', () => {
    let reputation = matl.createReputation('agent_with_issues');

    // Add some positive history
    for (let i = 0; i < 5; i++) {
      reputation = matl.recordPositive(reputation);
    }

    const initialValue = matl.reputationValue(reputation);
    expect(initialValue).toBeGreaterThan(0.5);

    // Add negative interactions
    for (let i = 0; i < 10; i++) {
      reputation = matl.recordNegative(reputation);
    }

    const finalValue = matl.reputationValue(reputation);
    expect(finalValue).toBeLessThan(initialValue);
  });
});

describe('E2E: Standards Compliance', () => {
  it('should verify claims against standard requirements', () => {
    // High trust claim
    const highTrustClaim = epistemic.claim('Critical system audit')
      .withClassification(
        epistemic.EmpiricalLevel.E3_Cryptographic,
        epistemic.NormativeLevel.N2_Network,
        epistemic.MaterialityLevel.M2_Persistent
      )
      .build();

    expect(epistemic.meetsMinimum(
      highTrustClaim.classification,
      epistemic.Standards.HighTrust.minE,
      epistemic.Standards.HighTrust.minN,
      epistemic.Standards.HighTrust.minM
    )).toBe(true);

    // Low trust claim should not meet high trust standard
    const lowTrustClaim = epistemic.claim('User preference')
      .withClassification(
        epistemic.EmpiricalLevel.E1_Testimonial,
        epistemic.NormativeLevel.N0_Personal,
        epistemic.MaterialityLevel.M0_Ephemeral
      )
      .build();

    expect(epistemic.meetsMinimum(
      lowTrustClaim.classification,
      epistemic.Standards.HighTrust.minE,
      epistemic.Standards.HighTrust.minN,
      epistemic.Standards.HighTrust.minM
    )).toBe(false);

    // But should meet low trust standard
    expect(epistemic.meetsMinimum(
      lowTrustClaim.classification,
      epistemic.Standards.LowTrust.minE,
      epistemic.Standards.LowTrust.minN,
      epistemic.Standards.LowTrust.minM
    )).toBe(true);
  });
});
