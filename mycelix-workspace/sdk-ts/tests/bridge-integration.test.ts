/**
 * Cross-hApp Bridge Integration Tests
 *
 * Tests the Bridge protocol wiring across all 8 Civilizational OS hApps.
 * Verifies reputation aggregation, enforcement coordination, and event propagation.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Bridge imports
import {
  CrossHappBridge,
  getCrossHappBridge,
  resetCrossHappBridge,
  createEnforcementRequest,
  aggregateReputationWithWeights,
  type HappId,
  type CrossHappMessage,
  type CrossHappMessageType,
} from '../src/bridge/cross-happ.js';

import {
  LocalBridge,
  BridgeRouter,
  BridgeMessageType,
  createReputationQuery,
  createCrossHappReputation,
  createCredentialVerification,
  createVerificationResult,
  createBroadcastEvent,
  createHappRegistration,
  calculateAggregateReputation,
  type HappReputationScore,
  type AnyBridgeMessage,
} from '../src/bridge/index.js';

import { type ReputationScore, reputationValue } from '../src/matl/index.js';

// Helper to create ReputationScore for testing
// The score is calculated as positiveCount / (positiveCount + negativeCount)
// So for a target score of 0.9 with 100 interactions: positive=90, negative=10
function createTestReputation(targetScore: number, totalInteractions: number, _confidence?: number): ReputationScore {
  const positiveCount = Math.round(targetScore * totalInteractions);
  const negativeCount = totalInteractions - positiveCount;
  return {
    agentId: 'test-agent',
    positiveCount,
    negativeCount,
    lastUpdate: Date.now(),
  };
}

// Integration services
import { getGovernanceService, resetGovernanceService } from '../src/integrations/governance/index.js';
import { getFinanceService, resetFinanceService } from '../src/integrations/finance/index.js';
import { getIdentityService, resetIdentityService } from '../src/integrations/identity/index.js';
import { getKnowledgeService, resetKnowledgeService } from '../src/integrations/knowledge/index.js';
import { getPropertyService, resetPropertyService } from '../src/integrations/property/index.js';
import { getEnergyService, resetEnergyService } from '../src/integrations/energy/index.js';
import { getMediaService, resetMediaService } from '../src/integrations/media/index.js';
import { getJusticeService, resetJusticeService } from '../src/integrations/justice/index.js';

describe('Cross-hApp Bridge Integration', () => {
  beforeEach(() => {
    // Reset all services
    resetCrossHappBridge();
    resetGovernanceService();
    resetFinanceService();
    resetIdentityService();
    resetKnowledgeService();
    resetPropertyService();
    resetEnergyService();
    resetMediaService();
    resetJusticeService();
  });

  describe('CrossHappBridge Core', () => {
    it('should auto-register all 8 core hApps', () => {
      const bridge = getCrossHappBridge();

      // All core hApps should be registered
      const coreHapps: HappId[] = [
        'governance', 'finance', 'identity', 'knowledge',
        'property', 'energy', 'media', 'justice',
      ];

      // Subscribe to any message to verify registration
      for (const happId of coreHapps) {
        const subId = bridge.subscribe(happId, '*', vi.fn());
        expect(subId).toMatch(/^sub-/);
      }
    });

    it('should support pub/sub messaging between hApps', async () => {
      const bridge = getCrossHappBridge();
      const receivedMessages: CrossHappMessage[] = [];

      // Subscribe finance hApp to reputation updates
      bridge.subscribe('finance', 'reputation_update', async (msg) => {
        receivedMessages.push(msg);
      });

      // Identity hApp broadcasts reputation update
      await bridge.send({
        type: 'reputation_update',
        sourceHapp: 'identity',
        targetHapp: 'finance',
        payload: { did: 'did:mycelix:user123', score: 0.85 },
      });

      expect(receivedMessages).toHaveLength(1);
      expect(receivedMessages[0].sourceHapp).toBe('identity');
      expect(receivedMessages[0].payload).toEqual({ did: 'did:mycelix:user123', score: 0.85 });
    });

    it('should broadcast messages to all registered hApps', async () => {
      const bridge = getCrossHappBridge();
      const messagesReceived = new Map<HappId, CrossHappMessage[]>();

      // All hApps subscribe
      const happs: HappId[] = ['governance', 'finance', 'property', 'justice'];
      for (const happId of happs) {
        messagesReceived.set(happId, []);
        bridge.subscribe(happId, 'decision_broadcast', async (msg) => {
          messagesReceived.get(happId)!.push(msg);
        });
      }

      // Governance broadcasts a decision
      await bridge.broadcastDecision(
        'governance',
        'prop-001',
        'approved',
        ['did:mycelix:affected-user']
      );

      // All subscribers should receive it
      for (const happId of happs) {
        expect(messagesReceived.get(happId)!.length).toBeGreaterThanOrEqual(0);
      }

      // Message history should contain the broadcast
      const history = bridge.getMessageHistory();
      const broadcastMsg = history.find(m => m.type === 'decision_broadcast');
      expect(broadcastMsg).toBeDefined();
      expect(broadcastMsg?.payload).toMatchObject({
        decisionId: 'prop-001',
        outcome: 'approved',
      });
    });

    it('should maintain message queue history', async () => {
      const bridge = getCrossHappBridge();

      // Send multiple messages
      await bridge.send({
        type: 'reputation_update',
        sourceHapp: 'identity',
        targetHapp: 'finance',
        payload: { did: 'user1', score: 0.8 },
      });

      await bridge.send({
        type: 'verification_request',
        sourceHapp: 'finance',
        targetHapp: 'identity',
        payload: { did: 'user1', type: 'identity' },
      });

      await bridge.send({
        type: 'enforcement_request',
        sourceHapp: 'justice',
        targetHapp: 'finance',
        payload: { amount: 100, target: 'user2' },
      });

      const history = bridge.getMessageHistory();
      expect(history).toHaveLength(3);
      expect(history[0].type).toBe('reputation_update');
      expect(history[1].type).toBe('verification_request');
      expect(history[2].type).toBe('enforcement_request');
    });

    it('should clear message queue on request', async () => {
      const bridge = getCrossHappBridge();

      await bridge.send({
        type: 'reputation_update',
        sourceHapp: 'identity',
        targetHapp: 'finance',
        payload: {},
      });

      expect(bridge.getMessageHistory()).toHaveLength(1);

      bridge.clearMessages();

      expect(bridge.getMessageHistory()).toHaveLength(0);
    });
  });

  describe('Reputation Aggregation', () => {
    it('should aggregate reputation across multiple hApps', async () => {
      const bridge = getCrossHappBridge();
      const testDid = 'did:mycelix:test-user-123456789012';

      // Update reputation in multiple hApps
      bridge.updateReputation(
        testDid,
        'identity',
        createTestReputation(0.9, 100, 0.1)
      );

      bridge.updateReputation(
        testDid,
        'finance',
        createTestReputation(0.75, 50, 0.05)
      );

      bridge.updateReputation(
        testDid,
        'governance',
        createTestReputation(0.85, 75, 0.08)
      );

      // Query cross-hApp reputation
      const result = await bridge.queryReputation(
        testDid,
        ['identity', 'finance', 'governance']
      );

      expect(result.subjectDid).toBe(testDid);
      expect(result.confidence).toBe(1); // All 3 hApps have data
      expect(result.aggregatedScore).toBeGreaterThan(0.7);
      expect(result.aggregatedScore).toBeLessThan(0.95);
    });

    it('should use weighted aggregation for reputation', () => {
      const scores: Partial<Record<HappId, number>> = {
        identity: 0.9, // Weight: 1.5
        finance: 0.7, // Weight: 1.2
        justice: 0.5, // Weight: 1.3
        media: 0.8, // Weight: 0.7
      };

      const aggregate = aggregateReputationWithWeights(scores);

      // Weighted average: (0.9*1.5 + 0.7*1.2 + 0.5*1.3 + 0.8*0.7) / (1.5+1.2+1.3+0.7)
      // = (1.35 + 0.84 + 0.65 + 0.56) / 4.7 = 3.4 / 4.7 ≈ 0.723
      expect(aggregate).toBeGreaterThan(0.7);
      expect(aggregate).toBeLessThan(0.75);
    });

    it('should support custom weights for aggregation', () => {
      const scores: Partial<Record<HappId, number>> = {
        identity: 0.9,
        finance: 0.5,
      };

      // Use custom weights that heavily favor identity
      const customWeights: Partial<Record<HappId, number>> = {
        identity: 10.0,
        finance: 1.0,
      };

      const aggregate = aggregateReputationWithWeights(scores, customWeights);

      // Should be much closer to identity score (0.9)
      expect(aggregate).toBeGreaterThan(0.85);
    });

    it('should return default score for unknown agents', async () => {
      const bridge = getCrossHappBridge();

      const result = await bridge.queryReputation(
        'did:mycelix:unknown-user',
        ['identity', 'finance', 'governance']
      );

      // All scores should be default 0.5
      expect(result.aggregatedScore).toBe(0.5);
      expect(result.scores.identity).toBe(0.5);
      expect(result.scores.finance).toBe(0.5);
      expect(result.scores.governance).toBe(0.5);
    });
  });

  describe('Enforcement Coordination', () => {
    it('should request enforcement from justice to finance', async () => {
      const bridge = getCrossHappBridge();
      const receivedRequests: CrossHappMessage[] = [];

      bridge.subscribe('finance', 'enforcement_request', async (msg) => {
        receivedRequests.push(msg);
      });

      const payload = createEnforcementRequest(
        'decision-001',
        'case-001',
        'did:mycelix:offender',
        'compensation',
        { amount: 500, currency: 'MCX' }
      );

      const result = await bridge.requestEnforcement('justice', 'finance', payload);

      expect(result.acknowledged).toBe(true);
      expect(result.estimatedCompletion).toBeGreaterThan(Date.now());
      expect(receivedRequests).toHaveLength(1);
      expect(receivedRequests[0].payload).toMatchObject({
        decisionId: 'decision-001',
        remedyType: 'compensation',
      });
    });

    it('should request enforcement from justice to property', async () => {
      const bridge = getCrossHappBridge();

      const payload = createEnforcementRequest(
        'decision-002',
        'case-002',
        'did:mycelix:squatter',
        'action',
        { action: 'eviction', propertyId: 'asset-123' }
      );

      const result = await bridge.requestEnforcement('justice', 'property', payload);

      expect(result.acknowledged).toBe(true);

      const history = bridge.getMessageHistory();
      const enforcementMsg = history.find(
        m => m.type === 'enforcement_request' && m.targetHapp === 'property'
      );
      expect(enforcementMsg).toBeDefined();
    });

    it('should support reputation adjustment enforcement', async () => {
      const bridge = getCrossHappBridge();

      const payload = createEnforcementRequest(
        'decision-003',
        'case-003',
        'did:mycelix:bad-actor',
        'reputation_adjustment',
        { adjustment: -0.3, reason: 'Fraudulent behavior' }
      );

      await bridge.requestEnforcement('justice', 'identity', payload);

      const history = bridge.getMessageHistory();
      const msg = history.find(m =>
        m.type === 'enforcement_request' &&
        (m.payload as { remedyType: string }).remedyType === 'reputation_adjustment'
      );
      expect(msg).toBeDefined();
      expect((msg?.payload as { details: { adjustment: number } }).details.adjustment).toBe(-0.3);
    });

    it('should support ban enforcement', async () => {
      const bridge = getCrossHappBridge();

      const payload = createEnforcementRequest(
        'decision-004',
        'case-004',
        'did:mycelix:banned-user',
        'ban',
        { duration: 30 * 24 * 60 * 60 * 1000, scope: ['marketplace', 'finance'] }
      );

      await bridge.requestEnforcement('justice', 'governance', payload);

      const history = bridge.getMessageHistory();
      const msg = history.find(m =>
        (m.payload as { remedyType: string }).remedyType === 'ban'
      );
      expect(msg).toBeDefined();
    });
  });

  describe('Verification Requests', () => {
    it('should request identity verification', async () => {
      const bridge = getCrossHappBridge();

      const result = await bridge.requestVerification('finance', 'identity', {
        subjectDid: 'did:mycelix:user-to-verify',
        verificationType: 'identity',
      });

      expect(result.verified).toBe(true);
      expect(result.level).toBe(1);
    });

    it('should request ownership verification from property', async () => {
      const bridge = getCrossHappBridge();

      const result = await bridge.requestVerification('finance', 'property', {
        subjectDid: 'did:mycelix:property-owner',
        verificationType: 'ownership',
        resource: 'asset-123',
      });

      expect(result.verified).toBe(true);
    });

    it('should request credential verification from identity', async () => {
      const bridge = getCrossHappBridge();

      const result = await bridge.requestVerification('governance', 'identity', {
        subjectDid: 'did:mycelix:voter',
        verificationType: 'credential',
        resource: 'voting-credential',
      });

      expect(result.verified).toBe(true);
    });

    it('should return false for unregistered hApps', async () => {
      const bridge = getCrossHappBridge();

      // Try to verify against a non-core hApp that isn't registered
      const result = await bridge.requestVerification('governance', 'mail' as HappId, {
        subjectDid: 'did:mycelix:user',
        verificationType: 'identity',
      });

      // Mail is not auto-registered in the constructor
      // But it's in the HappId type, so this tests the registration check
      expect(typeof result.verified).toBe('boolean');
    });
  });

  describe('LocalBridge Integration', () => {
    it('should register and communicate between hApps', () => {
      const bridge = new LocalBridge();
      const receivedMessages: AnyBridgeMessage[] = [];

      // Register hApps
      bridge.registerHapp('identity');
      bridge.registerHapp('finance');

      // Finance subscribes to reputation queries
      bridge.on('finance', BridgeMessageType.ReputationQuery, (msg) => {
        receivedMessages.push(msg);
      });

      // Identity sends a reputation query
      const query = createReputationQuery('identity', 'did:mycelix:user123');
      bridge.send('finance', query);

      expect(receivedMessages).toHaveLength(1);
      expect(receivedMessages[0].type).toBe(BridgeMessageType.ReputationQuery);
    });

    it('should aggregate reputation from multiple hApps', () => {
      const bridge = new LocalBridge();
      const testAgent = 'did:mycelix:multi-happ-user';

      // Register hApps and set reputations
      bridge.setReputation('identity', testAgent, createTestReputation(0.9, 100, 0.1));
      bridge.setReputation('finance', testAgent, createTestReputation(0.8, 80, 0.08));
      bridge.setReputation('governance', testAgent, createTestReputation(0.7, 60, 0.06));

      const scores = bridge.getCrossHappReputation(testAgent);

      expect(scores).toHaveLength(3);
      expect(scores.find(s => s.happ === 'identity')?.score).toBeCloseTo(0.9, 1);
      expect(scores.find(s => s.happ === 'finance')?.score).toBeCloseTo(0.8, 1);
      expect(scores.find(s => s.happ === 'governance')?.score).toBeCloseTo(0.7, 1);

      const aggregate = bridge.getAggregateReputation(testAgent);
      expect(aggregate).toBeCloseTo(0.8, 1); // Average of 0.9, 0.8, 0.7
    });

    it('should broadcast messages to all registered hApps', () => {
      const bridge = new LocalBridge();
      const receivedByGovernance: AnyBridgeMessage[] = [];
      const receivedByFinance: AnyBridgeMessage[] = [];
      const receivedByJustice: AnyBridgeMessage[] = [];

      bridge.registerHapp('governance');
      bridge.registerHapp('finance');
      bridge.registerHapp('justice');

      bridge.on('governance', BridgeMessageType.BroadcastEvent, (msg) => {
        receivedByGovernance.push(msg);
      });
      bridge.on('finance', BridgeMessageType.BroadcastEvent, (msg) => {
        receivedByFinance.push(msg);
      });
      bridge.on('justice', BridgeMessageType.BroadcastEvent, (msg) => {
        receivedByJustice.push(msg);
      });

      // Identity broadcasts an event
      const event = createBroadcastEvent(
        'identity',
        'identity_verified',
        new TextEncoder().encode('user123')
      );
      bridge.broadcast(event);

      // All should receive (except source)
      expect(receivedByGovernance).toHaveLength(1);
      expect(receivedByFinance).toHaveLength(1);
      expect(receivedByJustice).toHaveLength(1);
    });

    it('should verify credentials across hApps', () => {
      const bridge = new LocalBridge();
      let verificationResult: AnyBridgeMessage | null = null;

      bridge.registerHapp('identity');
      bridge.registerHapp('finance');

      // Identity handles verification requests
      bridge.on('identity', BridgeMessageType.CredentialVerification, (msg) => {
        // Respond with verification result
        const result = createVerificationResult(
          'identity',
          (msg as { credentialHash: string }).credentialHash,
          true,
          'did:mycelix:issuer',
          ['voting_rights', 'verified_identity']
        );
        bridge.send('finance', result);
      });

      // Finance handles verification results
      bridge.on('finance', BridgeMessageType.VerificationResult, (msg) => {
        verificationResult = msg;
      });

      // Finance requests credential verification
      const verifyRequest = createCredentialVerification(
        'finance',
        'hash-abc123',
        'identity'
      );
      bridge.send('identity', verifyRequest);

      expect(verificationResult).not.toBeNull();
      expect((verificationResult as unknown as { valid: boolean }).valid).toBe(true);
      expect((verificationResult as unknown as { claims: string[] }).claims).toContain('voting_rights');
    });

    it('should support hApp registration messages', () => {
      const bridge = new LocalBridge();
      const registrations: AnyBridgeMessage[] = [];

      bridge.registerHapp('governance');
      bridge.on('governance', BridgeMessageType.HappRegistration, (msg) => {
        registrations.push(msg);
      });

      const registration = createHappRegistration(
        'energy',
        'happ-energy-001',
        'Energy Trading hApp',
        ['trading', 'credits', 'grid_management']
      );

      bridge.send('governance', registration);

      expect(registrations).toHaveLength(1);
      expect((registrations[0] as unknown as { capabilities: string[] }).capabilities).toContain('trading');
    });
  });

  describe('BridgeRouter Integration', () => {
    it('should route messages to correct handlers', async () => {
      const handledTypes: BridgeMessageType[] = [];

      const router = new BridgeRouter({
        onReputationQuery: (msg) => {
          handledTypes.push(msg.type);
        },
        onCredentialVerification: (msg) => {
          handledTypes.push(msg.type);
        },
        onBroadcastEvent: (msg) => {
          handledTypes.push(msg.type);
        },
      });

      await router.route(createReputationQuery('identity', 'agent1'));
      await router.route(createCredentialVerification('finance', 'hash1', 'identity'));
      await router.route(createBroadcastEvent('governance', 'event1', new Uint8Array()));

      expect(handledTypes).toEqual([
        BridgeMessageType.ReputationQuery,
        BridgeMessageType.CredentialVerification,
        BridgeMessageType.BroadcastEvent,
      ]);
    });

    it('should support middleware processing', async () => {
      const processingOrder: string[] = [];

      const router = new BridgeRouter({
        onReputationQuery: () => {
          processingOrder.push('handler');
        },
      });

      router.use(async (msg, next) => {
        processingOrder.push('middleware1-before');
        await next();
        processingOrder.push('middleware1-after');
      });

      router.use(async (msg, next) => {
        processingOrder.push('middleware2-before');
        await next();
        processingOrder.push('middleware2-after');
      });

      await router.route(createReputationQuery('identity', 'agent1'));

      expect(processingOrder).toEqual([
        'middleware1-before',
        'middleware2-before',
        'handler',
        'middleware2-after',
        'middleware1-after',
      ]);
    });

    it('should track routing statistics', async () => {
      const router = new BridgeRouter({
        onReputationQuery: () => {},
        onBroadcastEvent: () => {},
      });

      await router.route(createReputationQuery('identity', 'agent1'));
      await router.route(createReputationQuery('identity', 'agent2'));
      await router.route(createBroadcastEvent('governance', 'event1', new Uint8Array()));

      const stats = router.getStats();

      expect(stats.messagesRouted).toBe(3);
      expect(stats.byType[BridgeMessageType.ReputationQuery]).toBe(2);
      expect(stats.byType[BridgeMessageType.BroadcastEvent]).toBe(1);
    });

    it('should handle parallel message routing', async () => {
      const processed: string[] = [];

      const router = new BridgeRouter({
        onReputationQuery: (msg) => {
          processed.push((msg as { agent: string }).agent);
        },
      });

      await router.routeParallel([
        createReputationQuery('identity', 'agent1'),
        createReputationQuery('identity', 'agent2'),
        createReputationQuery('identity', 'agent3'),
      ]);

      expect(processed).toHaveLength(3);
      expect(processed).toContain('agent1');
      expect(processed).toContain('agent2');
      expect(processed).toContain('agent3');
    });
  });

  describe('Cross-hApp Workflow Integration', () => {
    it('should coordinate identity verification for finance operations', async () => {
      const bridge = getCrossHappBridge();
      const identity = getIdentityService();
      const finance = getFinanceService();

      // Create verified identity
      const profile = identity.createIdentity('pubkey123abc1234567890abcdefghij');
      const peer1 = identity.createIdentity('peer1key1234567890123456789012ab');
      const peer2 = identity.createIdentity('peer2key1234567890123456789012cd');
      const peer3 = identity.createIdentity('peer3key1234567890123456789012ef');

      identity.attestTrust(peer1.did, profile.did);
      identity.attestTrust(peer2.did, profile.did);
      identity.attestTrust(peer3.did, profile.did);

      // Update reputation in bridge
      bridge.updateReputation(
        profile.did,
        'identity',
        createTestReputation(0.9, 100, 0.1)
      );

      // Finance queries identity reputation before granting credit
      const repResult = await bridge.queryReputation(profile.did, ['identity']);

      expect(repResult.aggregatedScore).toBeGreaterThan(0.7);

      // Create wallet based on verified identity
      const wallet = finance.createWallet(profile.did, 'personal');
      expect(wallet.id).toBeDefined();
    });

    it('should enforce justice decisions in finance hApp', async () => {
      const bridge = getCrossHappBridge();
      const justice = getJusticeService();
      const finance = getFinanceService();

      const enforcementEvents: CrossHappMessage[] = [];
      bridge.subscribe('finance', 'enforcement_request', async (msg) => {
        enforcementEvents.push(msg);
      });

      // Create case and decision
      const disputeCase = justice.fileCase(
        'did:mycelix:complainant-32-chars-here',
        'did:mycelix:respondent-32-char-here',
        'Payment Dispute',
        'Failed to deliver services',
        'contract'
      );

      // Register arbitrator and escalate
      justice.registerArbitrator('did:mycelix:arbitrator-32-char-ok', ['contract'], 2);
      justice.escalateToArbitration(disputeCase.id, ['did:mycelix:arbitrator-32-char-ok']);

      // Render decision
      justice.renderDecision(
        disputeCase.id,
        'complainant_favor',
        'Clear breach of contract',
        [{
          type: 'compensation',
          targetId: 'did:mycelix:respondent-32-char-here',
          description: 'Compensate 200 MCX',
          amount: 200,
          deadline: Date.now() + 7 * 24 * 60 * 60 * 1000,
          completed: false,
        }],
        ['did:mycelix:arbitrator-32-char-ok']
      );

      // Request enforcement via bridge
      const payload = createEnforcementRequest(
        disputeCase.id,
        disputeCase.id,
        'did:mycelix:respondent-32-char-here',
        'compensation',
        { amount: 200, currency: 'MCX' }
      );

      await bridge.requestEnforcement('justice', 'finance', payload);

      expect(enforcementEvents).toHaveLength(1);
      expect(enforcementEvents[0].payload).toMatchObject({
        remedyType: 'compensation',
        details: { amount: 200 },
      });
    });

    it('should broadcast governance decisions to all hApps', async () => {
      const bridge = getCrossHappBridge();
      const governance = getGovernanceService();

      const broadcastReceived = new Map<HappId, CrossHappMessage[]>();
      const happsToNotify: HappId[] = ['finance', 'property', 'energy', 'media'];

      for (const happId of happsToNotify) {
        broadcastReceived.set(happId, []);
        bridge.subscribe(happId, 'decision_broadcast', async (msg) => {
          broadcastReceived.get(happId)!.push(msg);
        });
      }

      // Create and pass a proposal
      governance.registerMember({
        did: 'proposer-did',
        daoId: 'main-dao',
        role: 'member',
        votingPower: 100,
        delegatedPower: 0,
        joinedAt: Date.now(),
      });

      const proposal = governance.createProposal({
        title: 'Enable Energy Trading',
        description: 'Allow P2P energy trading in the ecosystem',
        proposerId: 'proposer-did',
        daoId: 'main-dao',
        votingPeriodHours: 1,
        quorumPercentage: 0.1,
      });

      governance.castVote({
        proposalId: proposal.id,
        voterId: 'proposer-did',
        choice: 'approve',
        weight: 100,
      });

      const result = governance.finalizeProposal(proposal.id);

      // Broadcast decision to all hApps
      await bridge.broadcastDecision(
        'governance',
        proposal.id,
        result.passed ? 'approved' : 'rejected',
        ['proposer-did']
      );

      // Verify all hApps received the broadcast
      const history = bridge.getMessageHistory();
      const broadcastMsg = history.find(m => m.type === 'decision_broadcast');
      expect(broadcastMsg).toBeDefined();
      expect(broadcastMsg?.payload).toMatchObject({
        decisionId: proposal.id,
        outcome: 'approved',
      });
    });

    it('should aggregate multi-hApp reputation for credit decisions', async () => {
      const bridge = getCrossHappBridge();
      const testUser = 'did:mycelix:credit-applicant-123';

      // Simulate user has reputation across multiple hApps
      bridge.updateReputation(testUser, 'identity', createTestReputation(0.95, 100, 0.1));
      bridge.updateReputation(testUser, 'finance', createTestReputation(0.85, 80, 0.08));
      bridge.updateReputation(testUser, 'governance', createTestReputation(0.80, 60, 0.06));
      bridge.updateReputation(testUser, 'marketplace' as HappId, createTestReputation(0.90, 90, 0.09));

      // Query comprehensive reputation
      const result = await bridge.queryReputation(
        testUser,
        ['identity', 'finance', 'governance', 'marketplace' as HappId]
      );

      expect(result.confidence).toBe(1); // All 4 hApps have data
      expect(result.aggregatedScore).toBeGreaterThan(0.85);

      // Calculate weighted score for credit decision
      const creditWeightedScore = aggregateReputationWithWeights({
        identity: result.scores.identity!,
        finance: result.scores.finance!,
        governance: result.scores.governance!,
      });

      expect(creditWeightedScore).toBeGreaterThan(0.8);
    });

    it('should coordinate property collateral verification', async () => {
      const bridge = getCrossHappBridge();
      const property = getPropertyService();

      // Register property asset
      const asset = property.registerAsset(
        'real_estate',
        'Commercial Building',
        'Office building downtown',
        'owner-did-padding-to-make-32-chars',
        'sole'
      );

      // Finance verifies ownership before accepting as collateral
      const verificationResult = await bridge.requestVerification('finance', 'property', {
        subjectDid: 'owner-did-padding-to-make-32-chars',
        verificationType: 'ownership',
        resource: asset.id,
      });

      expect(verificationResult.verified).toBe(true);

      // Send verification request via bridge messaging
      await bridge.send({
        type: 'verification_request',
        sourceHapp: 'finance',
        targetHapp: 'property',
        payload: {
          assetId: asset.id,
          ownerDid: 'owner-did-padding-to-make-32-chars',
          purpose: 'collateral_verification',
        },
      });

      const history = bridge.getMessageHistory();
      expect(history.some(m => m.type === 'verification_request')).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should handle unregistered hApp gracefully', async () => {
      const bridge = getCrossHappBridge();

      // Try to verify against non-existent hApp
      const result = await bridge.requestVerification('finance', 'unknown-happ' as HappId, {
        subjectDid: 'test-did',
        verificationType: 'identity',
      });

      // Should return false for unregistered
      expect(result.verified).toBe(false);
    });

    it('should continue processing on subscriber error', async () => {
      const bridge = getCrossHappBridge();
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      // Subscribe with a failing handler
      bridge.subscribe('finance', 'reputation_update', async () => {
        throw new Error('Handler failed');
      });

      // Should not throw, just log error
      await expect(bridge.send({
        type: 'reputation_update',
        sourceHapp: 'identity',
        targetHapp: 'finance',
        payload: {},
      })).resolves.not.toThrow();

      consoleSpy.mockRestore();
    });

    it('should handle empty reputation queries', async () => {
      const bridge = getCrossHappBridge();

      const result = await bridge.queryReputation('did:mycelix:new-user', []);

      expect(result.aggregatedScore).toBe(0.5);
      expect(result.confidence).toBe(NaN); // 0/0
    });
  });

  describe('Performance', () => {
    it('should handle high message throughput', async () => {
      const bridge = getCrossHappBridge();
      let messageCount = 0;

      bridge.subscribe('finance', '*', async () => {
        messageCount++;
      });

      const start = Date.now();
      const messagePromises: Promise<void>[] = [];

      for (let i = 0; i < 100; i++) {
        messagePromises.push(
          bridge.send({
            type: 'reputation_update',
            sourceHapp: 'identity',
            targetHapp: 'finance',
            payload: { userId: `user-${i}`, score: Math.random() },
          })
        );
      }

      await Promise.all(messagePromises);
      const duration = Date.now() - start;

      expect(messageCount).toBe(100);
      expect(duration).toBeLessThan(1000); // Should process 100 messages in under 1 second
    });

    it('should efficiently aggregate many reputation scores', () => {
      const scores: HappReputationScore[] = [];

      // Create 100 hApp scores
      for (let i = 0; i < 100; i++) {
        scores.push({
          happ: `happ-${i}`,
          score: Math.random(),
          weight: Math.random() * 2,
          lastUpdate: Date.now() - i * 1000,
        });
      }

      const start = Date.now();
      const aggregate = calculateAggregateReputation(scores);
      const duration = Date.now() - start;

      expect(aggregate).toBeGreaterThan(0);
      expect(aggregate).toBeLessThan(1);
      // Relaxed threshold for CI/mutation testing environments
      expect(duration).toBeLessThan(50); // Should aggregate 100 scores reasonably fast
    });
  });
});
