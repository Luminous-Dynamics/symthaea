// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Markets Bridge Integration Tests
 *
 * Tests the cross-hApp integration capabilities of the markets_bridge zome.
 * These tests validate that epistemic markets can be triggered and resolved
 * by events from other Mycelix ecosystem hApps.
 *
 * Test Categories:
 * 1. Governance Integration - Proposal predictions
 * 2. Knowledge Integration - Epistemic claim verification
 * 3. Supply Chain Integration - Provenance predictions
 * 4. Cross-hApp Resolution - Multi-source resolution
 * 5. Bridge Protocol - Message routing and handling
 */

import { test, describe, beforeEach, afterEach, expect } from 'vitest';
import {
  Conductor,
  Player,
  createConductor,
  addPlayerWithConfig,
} from '@holochain/tryorama';

import {
  EpistemicMarketsClient,
  PredictionRequest,
  ResolutionCriteria,
} from '../sdk-ts/src/index';

// ============================================================================
// Test Setup
// ============================================================================

let conductor: Conductor;
let players: Player[];
let clients: EpistemicMarketsClient[];

beforeEach(async () => {
  conductor = await createConductor();

  // Create players representing different hApp contexts
  players = await Promise.all([
    addPlayerWithConfig(conductor, 'alice'),   // Governance participant
    addPlayerWithConfig(conductor, 'bob'),     // Knowledge contributor
    addPlayerWithConfig(conductor, 'carol'),   // Supply chain verifier
    addPlayerWithConfig(conductor, 'dave'),    // Oracle
    addPlayerWithConfig(conductor, 'eve'),     // Market participant
  ]);

  clients = players.map(p => new EpistemicMarketsClient(p.cells[0]));
});

afterEach(async () => {
  await conductor.shutdown();
});

// ============================================================================
// Helper Functions
// ============================================================================

async function createBridgeRequest(
  client: EpistemicMarketsClient,
  sourceHapp: string,
  question: string,
  criteria: ResolutionCriteria,
): Promise<PredictionRequest> {
  return client.bridge.createPredictionRequest({
    source_happ: sourceHapp,
    question,
    context: { description: 'Test context' },
    resolution_criteria: criteria,
    urgency: 'medium',
    bounty: { amount: 100, currency: 'HAM' },
    expires_at: Date.now() + 86400000 * 7, // 7 days
  });
}

async function simulateGovernanceVote(
  marketId: string,
  proposalId: string,
  outcome: 'passed' | 'rejected',
): Promise<void> {
  // Simulate a governance vote completion event
  // In real integration, this would come from the governance hApp
  const mockEvent = {
    type: 'GovernanceVoteComplete',
    proposal_id: proposalId,
    outcome,
    vote_count: { yes: outcome === 'passed' ? 100 : 30, no: outcome === 'passed' ? 30 : 100 },
    timestamp: Date.now(),
  };
  // This would trigger auto-resolution
}

async function simulateKnowledgeClaim(
  claimId: string,
  newLevel: 'testimonial' | 'privateVerify' | 'cryptographic' | 'measurable',
): Promise<void> {
  // Simulate a knowledge claim reaching a new epistemic level
  const mockEvent = {
    type: 'KnowledgeClaimLevelChanged',
    claim_id: claimId,
    new_level: newLevel,
    verifications: 5,
    timestamp: Date.now(),
  };
}

// ============================================================================
// 1. Governance Integration Tests
// ============================================================================

describe('Governance hApp Integration', () => {
  test('governance proposals can trigger prediction markets', async () => {
    // Governance hApp requests a prediction market for a proposal
    const request = await createBridgeRequest(
      clients[0],
      'governance',
      'Will proposal GP-2026-042 pass with >66% approval?',
      {
        type: 'OnChainEvent',
        event_type: 'GovernanceVoteComplete',
        event_source: 'governance',
        expected_data_schema: {
          proposal_id: 'GP-2026-042',
          outcome_field: 'passed',
        },
      },
    );

    expect(request).toBeDefined();
    expect(request.source_happ).toBe('governance');
    expect(request.status).toBe('pending');

    // The request should create a market automatically
    const market = await clients[0].bridge.getMarketForRequest(request.id);
    expect(market).toBeDefined();
    expect(market?.title).toContain('GP-2026-042');

    // Teaching moment: Governance decisions can be predicted.
    // This creates a feedback loop where the community can surface
    // concerns before voting completes.
  });

  test('governance vote completion auto-resolves market', async () => {
    // Create governance-linked market
    const request = await createBridgeRequest(
      clients[0],
      'governance',
      'Will infrastructure proposal pass?',
      {
        type: 'OnChainEvent',
        event_type: 'GovernanceVoteComplete',
        event_source: 'governance',
        expected_data_schema: {
          proposal_id: 'INFRA-001',
          outcome_field: 'passed',
        },
      },
    );

    const market = await clients[0].bridge.getMarketForRequest(request.id);

    // Participants make predictions
    await clients[1].predictions.createPrediction({
      market_id: market!.id,
      outcome: 'Yes',
      probability: 0.7,
      reasoning: { summary: 'Strong community support observed' },
    });

    await clients[2].predictions.createPrediction({
      market_id: market!.id,
      outcome: 'No',
      probability: 0.6,
      reasoning: { summary: 'Budget concerns raised' },
    });

    // Governance vote completes (simulated bridge event)
    await clients[0].bridge.handleExternalEvent({
      source: 'governance',
      event_type: 'GovernanceVoteComplete',
      data: {
        proposal_id: 'INFRA-001',
        passed: true,
        final_approval: 0.72,
      },
    });

    // Market should be automatically resolved
    const resolvedMarket = await clients[0].markets.getMarket(market!.id);
    expect(resolvedMarket.status).toBe('resolved');
    expect(resolvedMarket.resolution?.outcome).toBe('Yes');
    expect(resolvedMarket.resolution?.mechanism).toBe('automated_bridge');
  });

  test('governance escalation creates prediction market', async () => {
    // When a resolution dispute escalates to governance,
    // a prediction market about the governance outcome can be created
    const disputedMarketId = 'market-with-dispute';

    const escalationRequest = await clients[0].bridge.createEscalationPrediction({
      disputed_market_id: disputedMarketId,
      governance_case_id: 'CASE-2026-007',
      question: 'Will governance rule in favor of original resolution?',
      stakes_at_risk: 5000,
    });

    expect(escalationRequest.urgency).toBe('high');
    expect(escalationRequest.resolution_criteria.type).toBe('OnChainEvent');
  });
});

// ============================================================================
// 2. Knowledge Integration Tests
// ============================================================================

describe('Knowledge hApp Integration', () => {
  test('knowledge claims can spawn epistemic level prediction markets', async () => {
    // Knowledge hApp: someone makes a claim, market predicts when it will be verified
    const request = await createBridgeRequest(
      clients[1],
      'knowledge',
      'Will claim "Mycelix improves cooperation" reach E3 (cryptographic) verification?',
      {
        type: 'KnowledgeClaim',
        claim_id: 'CLAIM-2026-123',
        target_level: 'cryptographic', // E3
        verification_criteria: 'Five independent replications with p < 0.01',
      },
    );

    expect(request).toBeDefined();
    const market = await clients[0].bridge.getMarketForRequest(request.id);
    expect(market?.epistemic_position.empirical).toBe('testimonial'); // Starts as E1

    // Teaching moment: Epistemic markets can be about epistemology itself.
    // "When will we know this for sure?" is a valid prediction target.
  });

  test('knowledge claim verification updates resolve markets', async () => {
    const request = await createBridgeRequest(
      clients[1],
      'knowledge',
      'Will quantum computing claim reach E4 verification by 2027?',
      {
        type: 'KnowledgeClaim',
        claim_id: 'QC-CLAIM-001',
        target_level: 'measurable', // E4
        verification_criteria: 'Peer-reviewed publication in Nature',
      },
    );

    const market = await clients[0].bridge.getMarketForRequest(request.id);

    // Prediction made
    await clients[2].predictions.createPrediction({
      market_id: market!.id,
      outcome: 'Yes',
      probability: 0.3,
      reasoning: { summary: 'Current progress suggests unlikely in timeframe' },
    });

    // Knowledge hApp reports claim reached target level
    await clients[0].bridge.handleExternalEvent({
      source: 'knowledge',
      event_type: 'ClaimLevelChanged',
      data: {
        claim_id: 'QC-CLAIM-001',
        new_level: 'measurable',
        verifications: 3,
        publication_doi: '10.1038/nature.2026.12345',
      },
    });

    const resolvedMarket = await clients[0].markets.getMarket(market!.id);
    expect(resolvedMarket.status).toBe('resolved');
    expect(resolvedMarket.resolution?.evidence_links).toContain('10.1038/nature.2026.12345');
  });

  test('dependent claims create linked prediction markets', async () => {
    // Claim B depends on Claim A
    const claimARequest = await createBridgeRequest(
      clients[1],
      'knowledge',
      'Will foundational claim A be verified?',
      { type: 'KnowledgeClaim', claim_id: 'CLAIM-A', target_level: 'cryptographic' },
    );

    const claimBRequest = await createBridgeRequest(
      clients[1],
      'knowledge',
      'Will dependent claim B be verified (requires A)?',
      {
        type: 'KnowledgeClaim',
        claim_id: 'CLAIM-B',
        target_level: 'cryptographic',
        dependencies: ['CLAIM-A'],
      },
    );

    const marketA = await clients[0].bridge.getMarketForRequest(claimARequest.id);
    const marketB = await clients[0].bridge.getMarketForRequest(claimBRequest.id);

    // Markets should be linked
    expect(marketB?.belief_dependencies).toContainEqual(
      expect.objectContaining({ market_id: marketA!.id })
    );
  });
});

// ============================================================================
// 3. Supply Chain Integration Tests
// ============================================================================

describe('Supply Chain hApp Integration', () => {
  test('product authenticity markets use expert consensus resolution', async () => {
    const request = await createBridgeRequest(
      clients[2],
      'supplychain',
      'Is batch BATCH-2026-Q1-789 authentic organic produce?',
      {
        type: 'ExpertConsensus',
        required_experts: 3,
        expert_domains: ['organic_certification', 'supply_chain_audit'],
        consensus_threshold: 0.8,
      },
    );

    const market = await clients[0].bridge.getMarketForRequest(request.id);

    // Experts register for resolution
    await clients[3].resolution.registerAsOracle({
      market_id: market!.id,
      domains: ['supply_chain_audit'],
      matl_score: { quality: 0.85, consistency: 0.9, reputation: 0.88 },
      reputation_stake: 50,
    });

    // Expert votes
    await clients[3].resolution.castOracleVote({
      market_id: market!.id,
      outcome: 'Yes',
      confidence: 0.9,
      evidence: ['Certificate CERT-2026-456', 'Audit report AR-2026-123'],
      reasoning: 'All documentation verified, on-site inspection confirmed',
    });

    // After consensus threshold reached
    const resolvedMarket = await clients[0].markets.getMarket(market!.id);
    // Depends on having enough experts vote - may still be pending
    expect(['pending', 'resolved']).toContain(resolvedMarket.status);
  });

  test('supply chain provenance tracking creates prediction chains', async () => {
    // Multiple prediction markets along the supply chain
    const originRequest = await createBridgeRequest(
      clients[2],
      'supplychain',
      'Is origin farm certified organic?',
      { type: 'ExpertConsensus', required_experts: 2, consensus_threshold: 0.9 },
    );

    const transportRequest = await createBridgeRequest(
      clients[2],
      'supplychain',
      'Was cold chain maintained during transport?',
      { type: 'ExpertConsensus', required_experts: 2, consensus_threshold: 0.9 },
    );

    const finalRequest = await createBridgeRequest(
      clients[2],
      'supplychain',
      'Is final product authentic?',
      {
        type: 'ExpertConsensus',
        required_experts: 3,
        consensus_threshold: 0.9,
        depends_on_markets: [originRequest.id, transportRequest.id],
      },
    );

    const finalMarket = await clients[0].bridge.getMarketForRequest(finalRequest.id);
    expect(finalMarket?.belief_dependencies?.length).toBe(2);
  });
});

// ============================================================================
// 4. Cross-hApp Resolution Tests
// ============================================================================

describe('Cross-hApp Resolution Mechanisms', () => {
  test('multi-source resolution combines evidence from multiple hApps', async () => {
    // A market that requires evidence from multiple hApps
    const request = await createBridgeRequest(
      clients[0],
      'governance',
      'Will the community energy initiative succeed within budget?',
      {
        type: 'MultiSource',
        required_sources: [
          { happ: 'governance', event_type: 'ProposalApproved' },
          { happ: 'finance', event_type: 'BudgetUnderLimit' },
          { happ: 'energy', event_type: 'InstallationComplete' },
        ],
        aggregation: 'all_required', // All must be true
      },
    );

    const market = await clients[0].bridge.getMarketForRequest(request.id);

    // Simulate events from multiple hApps
    await clients[0].bridge.handleExternalEvent({
      source: 'governance',
      event_type: 'ProposalApproved',
      data: { proposal_id: 'ENERGY-001', approved: true },
    });

    // Market shouldn't resolve yet - need all sources
    let currentMarket = await clients[0].markets.getMarket(market!.id);
    expect(currentMarket.status).toBe('open');
    expect(currentMarket.resolution_progress?.completed_sources).toEqual(['governance']);

    await clients[0].bridge.handleExternalEvent({
      source: 'finance',
      event_type: 'BudgetUnderLimit',
      data: { project_id: 'ENERGY-001', final_cost: 95000, budget: 100000 },
    });

    await clients[0].bridge.handleExternalEvent({
      source: 'energy',
      event_type: 'InstallationComplete',
      data: { project_id: 'ENERGY-001', capacity_kw: 50 },
    });

    // Now all sources confirmed - should resolve
    currentMarket = await clients[0].markets.getMarket(market!.id);
    expect(currentMarket.status).toBe('resolved');
    expect(currentMarket.resolution?.outcome).toBe('Yes');
  });

  test('bridge handles conflicting cross-hApp data gracefully', async () => {
    const request = await createBridgeRequest(
      clients[0],
      'governance',
      'Did project meet success criteria?',
      {
        type: 'MultiSource',
        required_sources: [
          { happ: 'governance', event_type: 'SuccessDeclared' },
          { happ: 'finance', event_type: 'ROIPositive' },
        ],
        aggregation: 'majority', // At least half must agree
      },
    );

    const market = await clients[0].bridge.getMarketForRequest(request.id);

    // Conflicting events
    await clients[0].bridge.handleExternalEvent({
      source: 'governance',
      event_type: 'SuccessDeclared',
      data: { project_id: 'TEST-001', success: true },
    });

    await clients[0].bridge.handleExternalEvent({
      source: 'finance',
      event_type: 'ROIPositive',
      data: { project_id: 'TEST-001', roi_positive: false }, // Conflict!
    });

    // Market should handle conflict - may escalate or use majority
    const currentMarket = await clients[0].markets.getMarket(market!.id);
    // With majority aggregation and 1 yes, 1 no, behavior depends on tie-breaker
    expect(['open', 'disputed', 'resolved']).toContain(currentMarket.status);
  });
});

// ============================================================================
// 5. Bridge Protocol Tests
// ============================================================================

describe('Bridge Protocol Mechanics', () => {
  test('bridge validates source hApp signatures', async () => {
    // Only legitimate hApps should be able to trigger events
    const validRequest = await createBridgeRequest(
      clients[0],
      'governance',
      'Test market',
      { type: 'OnChainEvent', event_type: 'Test' },
    );

    expect(validRequest).toBeDefined();

    // Attempt from unknown source should fail
    await expect(
      clients[0].bridge.handleExternalEvent({
        source: 'unknown_happ',
        event_type: 'FakeEvent',
        data: { fake: true },
      })
    ).rejects.toThrow(/unknown source/i);
  });

  test('bridge rate limits incoming requests', async () => {
    // Prevent spam from external hApps
    const promises = [];
    for (let i = 0; i < 20; i++) {
      promises.push(
        createBridgeRequest(
          clients[0],
          'governance',
          `Spam request ${i}`,
          { type: 'OnChainEvent', event_type: 'Test' },
        )
      );
    }

    // Some should succeed, later ones should be rate limited
    const results = await Promise.allSettled(promises);
    const successes = results.filter(r => r.status === 'fulfilled').length;
    const failures = results.filter(r => r.status === 'rejected').length;

    expect(successes).toBeGreaterThan(0);
    expect(failures).toBeGreaterThan(0); // Some should be rate limited
  });

  test('bridge tracks cross-hApp message statistics', async () => {
    await createBridgeRequest(
      clients[0],
      'governance',
      'Test 1',
      { type: 'OnChainEvent', event_type: 'Test' },
    );

    await createBridgeRequest(
      clients[0],
      'knowledge',
      'Test 2',
      { type: 'KnowledgeClaim', claim_id: 'TEST', target_level: 'cryptographic' },
    );

    const stats = await clients[0].bridge.getStats();

    expect(stats.total_requests).toBeGreaterThanOrEqual(2);
    expect(stats.by_source_happ.governance).toBeGreaterThanOrEqual(1);
    expect(stats.by_source_happ.knowledge).toBeGreaterThanOrEqual(1);
  });

  test('bridge supports priority queue for urgent requests', async () => {
    // Create low priority request first
    const lowPriority = await clients[0].bridge.createPredictionRequest({
      source_happ: 'governance',
      question: 'Low priority question',
      resolution_criteria: { type: 'OnChainEvent', event_type: 'Test' },
      urgency: 'low',
      bounty: { amount: 10, currency: 'HAM' },
      expires_at: Date.now() + 86400000 * 30,
    });

    // Create high priority request after
    const highPriority = await clients[0].bridge.createPredictionRequest({
      source_happ: 'governance',
      question: 'Critical infrastructure decision',
      resolution_criteria: { type: 'OnChainEvent', event_type: 'Critical' },
      urgency: 'critical',
      bounty: { amount: 1000, currency: 'HAM' },
      expires_at: Date.now() + 86400000,
    });

    // High priority should be processed first
    const queue = await clients[0].bridge.getProcessingQueue();
    expect(queue[0].id).toBe(highPriority.id);
  });
});

// ============================================================================
// 6. Error Handling and Edge Cases
// ============================================================================

describe('Bridge Error Handling', () => {
  test('handles malformed external events gracefully', async () => {
    // Missing required fields
    await expect(
      clients[0].bridge.handleExternalEvent({
        source: 'governance',
        event_type: 'GovernanceVoteComplete',
        data: {}, // Missing proposal_id and outcome
      })
    ).rejects.toThrow(/missing required field/i);
  });

  test('expired requests are cleaned up', async () => {
    const request = await clients[0].bridge.createPredictionRequest({
      source_happ: 'governance',
      question: 'Expiring question',
      resolution_criteria: { type: 'OnChainEvent', event_type: 'Test' },
      urgency: 'low',
      bounty: { amount: 10, currency: 'HAM' },
      expires_at: Date.now() - 1000, // Already expired
    });

    // Should be automatically marked as expired
    const status = await clients[0].bridge.getRequestStatus(request.id);
    expect(status).toBe('expired');
  });

  test('duplicate requests are deduplicated', async () => {
    const request1 = await createBridgeRequest(
      clients[0],
      'governance',
      'Unique question about GP-2026-100',
      { type: 'OnChainEvent', event_type: 'GovernanceVoteComplete', expected_data_schema: { proposal_id: 'GP-2026-100' } },
    );

    // Same question should return existing market
    const request2 = await createBridgeRequest(
      clients[0],
      'governance',
      'Unique question about GP-2026-100',
      { type: 'OnChainEvent', event_type: 'GovernanceVoteComplete', expected_data_schema: { proposal_id: 'GP-2026-100' } },
    );

    // Should point to same market
    const market1 = await clients[0].bridge.getMarketForRequest(request1.id);
    const market2 = await clients[0].bridge.getMarketForRequest(request2.id);

    expect(market1?.id).toBe(market2?.id);
  });
});
