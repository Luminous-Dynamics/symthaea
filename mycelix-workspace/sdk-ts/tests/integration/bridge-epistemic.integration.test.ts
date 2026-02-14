/**
 * Bridge + Epistemic Integration Tests
 *
 * Tests the integration between the Bridge protocol and
 * Epistemic Framework for verified claims across hApps.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as bridge from '../../src/bridge/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as matl from '../../src/matl/index.js';

describe('Integration: Bridge + Epistemic', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('marketplace');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('social');
  });

  describe('Cross-hApp Credential Verification', () => {
    it('should verify credentials with epistemic classification', () => {
      // Create a high-trust credential claim
      const credentialClaim = epistemic
        .claim('User identity verified via government ID')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
        .withIssuer('identity_happ')
        .build();

      // Verify it meets high trust standard
      expect(
        epistemic.meetsStandard(
          credentialClaim,
          epistemic.EmpiricalLevel.E3_Cryptographic,
          epistemic.NormativeLevel.N2_Network
        )
      ).toBe(true);

      // Classification code for reference
      const code = epistemic.classificationCode(credentialClaim.classification);
      expect(code).toBe('E3-N2-M3');

      // Create verification result message
      const verificationResult = bridge.createVerificationResult(
        'identity',
        'cred_' + credentialClaim.id,
        true,
        'identity_happ',
        ['government_id_verified', 'biometric_match']
      );

      // Send to marketplace
      const receivedMessages: bridge.AnyBridgeMessage[] = [];
      localBridge.on(
        'marketplace',
        bridge.BridgeMessageType.VerificationResult,
        (msg) => receivedMessages.push(msg)
      );

      localBridge.send('marketplace', verificationResult);

      expect(receivedMessages.length).toBe(1);
      const result = receivedMessages[0] as bridge.VerificationResultMessage;
      expect(result.valid).toBe(true);
      expect(result.claims).toContain('government_id_verified');
    });

    it('should reject low-trust credentials for high-trust requirements', () => {
      // Create a low-trust claim (self-attestation)
      const lowTrustClaim = epistemic
        .claim('User claims to be over 18')
        .withEmpirical(epistemic.EmpiricalLevel.E1_Testimonial)
        .withNormative(epistemic.NormativeLevel.N0_Personal)
        .withMateriality(epistemic.MaterialityLevel.M0_Ephemeral)
        .build();

      // Should NOT meet high trust standard
      expect(
        epistemic.meetsStandard(
          lowTrustClaim,
          epistemic.EmpiricalLevel.E3_Cryptographic,
          epistemic.NormativeLevel.N2_Network
        )
      ).toBe(false);

      // But should meet low trust standard
      expect(
        epistemic.meetsMinimum(
          lowTrustClaim.classification,
          epistemic.Standards.LowTrust.minE,
          epistemic.Standards.LowTrust.minN,
          epistemic.Standards.LowTrust.minM
        )
      ).toBe(true);
    });

    it('should propagate trust levels through claim chain', () => {
      // Original claim from identity hApp
      const identityClaim = epistemic
        .claim('KYC verification complete')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('identity')
        .build();

      // Derived claim in marketplace
      const marketplaceClaim = epistemic
        .claim('Verified seller status')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify) // Derived, so one level lower
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('marketplace')
        .build();

      // Original should meet higher standard
      expect(
        epistemic.meetsMinimum(
          identityClaim.classification,
          epistemic.EmpiricalLevel.E3_Cryptographic,
          epistemic.NormativeLevel.N2_Network
        )
      ).toBe(true);

      // Derived claim should be slightly lower
      expect(
        identityClaim.classification.empirical
      ).toBeGreaterThan(marketplaceClaim.classification.empirical);
    });
  });

  describe('Reputation Bridging with Epistemic Evidence', () => {
    it('should create claims about reputation transfers', () => {
      const agentId = 'cross_happ_agent';

      // Build reputation in marketplace
      let marketplaceRep = matl.createReputation(agentId);
      for (let i = 0; i < 50; i++) {
        marketplaceRep = matl.recordPositive(marketplaceRep);
      }
      for (let i = 0; i < 5; i++) {
        marketplaceRep = matl.recordNegative(marketplaceRep);
      }

      localBridge.setReputation('marketplace', agentId, marketplaceRep);

      // Create epistemic claim about reputation
      const repValue = matl.reputationValue(marketplaceRep);
      const reputationClaim = epistemic
        .claim(`Agent has ${repValue.toFixed(2)} marketplace reputation`)
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify) // Verifiable within network
        .withNormative(epistemic.NormativeLevel.N2_Network) // Network consensus
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent) // Stored
        .withIssuer('marketplace')
        .build();

      // Add evidence
      const evidence: epistemic.Evidence = {
        type: 'reputation_snapshot',
        data: JSON.stringify({
          positiveCount: marketplaceRep.positiveCount,
          negativeCount: marketplaceRep.negativeCount,
          totalInteractions: marketplaceRep.totalInteractions,
          calculatedValue: repValue,
        }),
        source: 'marketplace_reputation_service',
        timestamp: Date.now(),
      };

      const claimWithEvidence = epistemic.addEvidence(reputationClaim, evidence);
      expect(claimWithEvidence.evidence.length).toBe(1);

      // Now another hApp can verify this claim
      expect(
        epistemic.meetsMinimum(
          claimWithEvidence.classification,
          epistemic.EmpiricalLevel.E2_PrivateVerify,
          epistemic.NormativeLevel.N1_Communal
        )
      ).toBe(true);
    });

    it('should aggregate reputation claims from multiple hApps', () => {
      const agentId = 'multi_happ_trusted_agent';

      // Setup reputation in multiple hApps
      const happs = ['marketplace', 'governance', 'social'];
      const claims: epistemic.EpistemicClaim[] = [];

      for (const happ of happs) {
        let rep = matl.createReputation(agentId);
        const interactions = Math.floor(Math.random() * 50) + 10;
        for (let i = 0; i < interactions; i++) {
          rep = Math.random() > 0.1
            ? matl.recordPositive(rep)
            : matl.recordNegative(rep);
        }

        localBridge.setReputation(happ, agentId, rep);

        // Create claim
        const claim = epistemic
          .claim(`${happ} reputation: ${matl.reputationValue(rep).toFixed(2)}`)
          .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
          .withNormative(epistemic.NormativeLevel.N2_Network)
          .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
          .withIssuer(happ)
          .build();

        claims.push(claim);
      }

      // All claims should be verifiable
      for (const claim of claims) {
        expect(
          epistemic.meetsMinimum(
            claim.classification,
            epistemic.EmpiricalLevel.E1_Testimonial,
            epistemic.NormativeLevel.N1_Communal
          )
        ).toBe(true);
      }

      // Get aggregate reputation
      const aggregate = localBridge.getAggregateReputation(agentId);
      expect(aggregate).toBeGreaterThan(0);
      expect(aggregate).toBeLessThanOrEqual(1);
    });
  });

  describe('Event Broadcasting with Claims', () => {
    it('should broadcast reputation updates with epistemic metadata', () => {
      const receivedByGovernance: bridge.AnyBridgeMessage[] = [];
      const receivedBySocial: bridge.AnyBridgeMessage[] = [];

      localBridge.on(
        'governance',
        bridge.BridgeMessageType.BroadcastEvent,
        (msg) => receivedByGovernance.push(msg)
      );

      localBridge.on(
        'social',
        bridge.BridgeMessageType.BroadcastEvent,
        (msg) => receivedBySocial.push(msg)
      );

      // Create a claim for the event with explicit materiality
      const eventClaim = epistemic
        .claim('Reputation milestone achieved')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .build();

      // Encode claim info in event payload
      const payload = new TextEncoder().encode(
        JSON.stringify({
          agentId: 'milestone_agent',
          milestone: 'trusted_seller',
          claimCode: epistemic.classificationCode(eventClaim.classification),
          claimId: eventClaim.id,
        })
      );

      const event = bridge.createBroadcastEvent(
        'marketplace',
        'reputation_milestone',
        payload
      );

      localBridge.broadcast(event);

      // Both governance and social should receive
      expect(receivedByGovernance.length).toBe(1);
      expect(receivedBySocial.length).toBe(1);

      // Verify payload
      const receivedEvent = receivedByGovernance[0] as bridge.BroadcastEventMessage;
      const decodedPayload = JSON.parse(
        new TextDecoder().decode(receivedEvent.payload)
      );
      expect(decodedPayload.milestone).toBe('trusted_seller');
      expect(decodedPayload.claimCode).toBe('E2-N2-M2');
    });
  });

  describe('Dispute Resolution Flow', () => {
    it('should handle claim disputes across hApps', () => {
      // Original claim
      const originalClaim = epistemic
        .claim('Transaction completed successfully')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .withIssuer('marketplace')
        .build();

      // Dispute from buyer
      const disputeClaim = epistemic
        .claim('Item not received as described')
        .withEmpirical(epistemic.EmpiricalLevel.E1_Testimonial)
        .withNormative(epistemic.NormativeLevel.N0_Personal)
        .withMateriality(epistemic.MaterialityLevel.M1_Versioned)
        .withIssuer('buyer_agent')
        .build();

      // Arbitration claim (higher authority)
      const arbitrationClaim = epistemic
        .claim('Dispute resolved in favor of buyer')
        .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
        .withNormative(epistemic.NormativeLevel.N3_Universal)
        .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
        .withIssuer('governance')
        .build();

      // Arbitration should have highest empirical level
      expect(arbitrationClaim.classification.empirical).toBeGreaterThan(
        originalClaim.classification.empirical
      );
      expect(arbitrationClaim.classification.empirical).toBeGreaterThan(
        disputeClaim.classification.empirical
      );

      // Arbitration claim meets global consensus standard
      expect(
        epistemic.meetsMinimum(
          arbitrationClaim.classification,
          epistemic.EmpiricalLevel.E3_Cryptographic,
          epistemic.NormativeLevel.N3_Universal,
          epistemic.MaterialityLevel.M3_Immutable
        )
      ).toBe(true);
    });
  });

  describe('Standards-Based Access Control', () => {
    it('should enforce trust standards for hApp access', () => {
      const accessLevels = [
        { name: 'public_read', minE: epistemic.EmpiricalLevel.E0_Unverified, minN: epistemic.NormativeLevel.N0_Personal },
        { name: 'community_post', minE: epistemic.EmpiricalLevel.E1_Testimonial, minN: epistemic.NormativeLevel.N1_Communal },
        { name: 'verified_trade', minE: epistemic.EmpiricalLevel.E2_PrivateVerify, minN: epistemic.NormativeLevel.N2_Network },
        { name: 'governance_vote', minE: epistemic.EmpiricalLevel.E3_Cryptographic, minN: epistemic.NormativeLevel.N2_Network },
        { name: 'admin_action', minE: epistemic.EmpiricalLevel.E4_Consensus, minN: epistemic.NormativeLevel.N3_Universal },
      ];

      // User with medium verification (explicit materiality to avoid default M0)
      const userClaim = epistemic
        .claim('User verified via email')
        .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
        .withNormative(epistemic.NormativeLevel.N2_Network)
        .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
        .build();

      // Check which access levels user qualifies for
      const allowedLevels = accessLevels.filter((level) =>
        epistemic.meetsMinimum(
          userClaim.classification,
          level.minE,
          level.minN
        )
      );

      expect(allowedLevels.map((l) => l.name)).toContain('public_read');
      expect(allowedLevels.map((l) => l.name)).toContain('community_post');
      expect(allowedLevels.map((l) => l.name)).toContain('verified_trade');
      expect(allowedLevels.map((l) => l.name)).not.toContain('admin_action');
    });
  });
});
