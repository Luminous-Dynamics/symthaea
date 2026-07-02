// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Knowledge - Full Integration Test Suite
 *
 * Tests the complete flow from claim creation through fact-checking
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Scenario, Player, createConductor } from '@holochain/tryorama';
import path from 'path';
import { KnowledgeClient } from '../../client/src';

// ============================================================================
// Test Configuration
// ============================================================================

const DNA_PATH = path.join(__dirname, '../../dna/knowledge.dna');

describe('Mycelix Knowledge - Full Integration', () => {
  let scenario: Scenario;
  let alice: Player;
  let bob: Player;
  let aliceClient: KnowledgeClient;
  let bobClient: KnowledgeClient;

  beforeAll(async () => {
    scenario = new Scenario();

    // Set up two players
    alice = await scenario.addPlayerWithApp({ path: DNA_PATH });
    bob = await scenario.addPlayerWithApp({ path: DNA_PATH });

    // Initialize clients
    aliceClient = new KnowledgeClient({
      connection: alice.appWs,
      roleName: 'knowledge',
    });

    bobClient = new KnowledgeClient({
      connection: bob.appWs,
      roleName: 'knowledge',
    });
  });

  afterAll(async () => {
    await scenario.shutdown();
  });

  // ==========================================================================
  // Claims Module Tests
  // ==========================================================================

  describe('Claims', () => {
    let claimId: string;

    it('should create a claim with epistemic classification', async () => {
      const result = await aliceClient.claims.createClaim({
        content: 'The Earth orbits the Sun at an average distance of 93 million miles.',
        classification: {
          empirical: 0.95,
          normative: 0.02,
          mythic: 0.03,
        },
        tags: ['astronomy', 'science', 'solar-system'],
        sources: ['NASA', 'ESA'],
      });

      expect(result).toBeDefined();
      claimId = result;
    });

    it('should retrieve the created claim', async () => {
      const claim = await aliceClient.claims.getClaim(claimId);

      expect(claim).toBeDefined();
      expect(claim.content).toContain('Earth orbits the Sun');
      expect(claim.classification.empirical).toBeGreaterThan(0.9);
    });

    it('should list claims with filtering', async () => {
      const claims = await aliceClient.query.listClaims({
        tags: ['astronomy'],
        minCredibility: 0.5,
        limit: 10,
      });

      expect(claims.items.length).toBeGreaterThan(0);
    });

    it('should update a claim', async () => {
      await aliceClient.claims.updateClaim({
        originalHash: claimId,
        content: 'The Earth orbits the Sun at an average distance of approximately 93 million miles (150 million km).',
      });

      const updated = await aliceClient.claims.getClaim(claimId);
      expect(updated.content).toContain('150 million km');
    });
  });

  // ==========================================================================
  // Graph Module Tests
  // ==========================================================================

  describe('Knowledge Graph', () => {
    let claim1Id: string;
    let claim2Id: string;
    let relationshipId: string;

    beforeAll(async () => {
      // Create two related claims
      claim1Id = await aliceClient.claims.createClaim({
        content: 'Climate change is primarily caused by human activities.',
        classification: { empirical: 0.85, normative: 0.10, mythic: 0.05 },
        tags: ['climate', 'science'],
      });

      claim2Id = await aliceClient.claims.createClaim({
        content: 'Burning fossil fuels releases CO2 into the atmosphere.',
        classification: { empirical: 0.95, normative: 0.02, mythic: 0.03 },
        tags: ['climate', 'emissions'],
      });
    });

    it('should create a relationship between claims', async () => {
      relationshipId = await aliceClient.graph.createRelationship(
        claim2Id,  // source: fossil fuels release CO2
        claim1Id,  // target: climate change is human-caused
        'SUPPORTS',
        0.85
      );

      expect(relationshipId).toBeDefined();
    });

    it('should retrieve relationships for a claim', async () => {
      const relationships = await aliceClient.graph.getRelationships(claim1Id);

      expect(relationships.length).toBeGreaterThan(0);
      expect(relationships[0].type).toBe('SUPPORTS');
    });

    it('should get dependency tree', async () => {
      const tree = await aliceClient.graph.getDependencyTree(claim1Id, 3);

      expect(tree).toBeDefined();
      expect(tree.depth).toBeGreaterThanOrEqual(1);
    });

    it('should propagate belief updates', async () => {
      const result = await aliceClient.graph.propagateBelief(claim1Id);

      expect(result.converged).toBe(true);
      expect(result.iterations).toBeGreaterThan(0);
    });

    it('should rank claims by information value', async () => {
      const ranking = await aliceClient.graph.rankByInformationValue(10);

      expect(ranking.length).toBeGreaterThan(0);
      expect(ranking[0].informationValue).toBeDefined();
    });
  });

  // ==========================================================================
  // Inference Module Tests
  // ==========================================================================

  describe('Credibility Inference', () => {
    let testClaimId: string;

    beforeAll(async () => {
      testClaimId = await aliceClient.claims.createClaim({
        content: 'Water boils at 100 degrees Celsius at sea level.',
        classification: { empirical: 0.98, normative: 0.01, mythic: 0.01 },
        tags: ['physics', 'chemistry'],
        sources: ['textbook', 'experiment'],
      });
    });

    it('should calculate enhanced credibility score', async () => {
      const credibility = await aliceClient.inference.calculateEnhancedCredibility(
        testClaimId,
        'Claim'
      );

      expect(credibility.score).toBeGreaterThan(0);
      expect(credibility.score).toBeLessThanOrEqual(1);
      expect(credibility.factors).toBeDefined();
      expect(credibility.confidence).toBeGreaterThan(0);
    });

    it('should assess evidence strength', async () => {
      const evidence = await aliceClient.inference.assessEvidenceStrength(testClaimId);

      expect(evidence).toBeDefined();
      expect(evidence.empiricalEvidence).toBeDefined();
    });

    it('should track author reputation', async () => {
      const reputation = await aliceClient.inference.getAuthorReputation(
        alice.agentPubKey
      );

      expect(reputation).toBeDefined();
      expect(reputation.overallScore).toBeGreaterThanOrEqual(0);
    });
  });

  // ==========================================================================
  // Fact-Check Module Tests
  // ==========================================================================

  describe('Fact-Checking', () => {
    beforeAll(async () => {
      // Seed some claims for fact-checking
      await aliceClient.claims.createClaim({
        content: 'The speed of light in a vacuum is approximately 299,792 km/s.',
        classification: { empirical: 0.99, normative: 0.005, mythic: 0.005 },
        tags: ['physics', 'constants'],
      });

      await aliceClient.claims.createClaim({
        content: 'Light travels at about 300,000 kilometers per second.',
        classification: { empirical: 0.95, normative: 0.02, mythic: 0.03 },
        tags: ['physics'],
      });
    });

    it('should fact-check a statement against the knowledge graph', async () => {
      const result = await aliceClient.factcheck.factCheck({
        statement: 'Light travels at approximately 300,000 km/s in a vacuum.',
        context: 'physics discussion',
        minEpistemicLevel: 2,
        minConfidence: 0.6,
      });

      expect(result.verdict).toBeDefined();
      expect(['TRUE', 'MOSTLY_TRUE', 'MIXED', 'MOSTLY_FALSE', 'FALSE', 'UNVERIFIABLE', 'INSUFFICIENT_EVIDENCE']).toContain(result.verdict);
      expect(result.confidence).toBeGreaterThan(0);
      expect(result.explanation).toBeDefined();
    });

    it('should return supporting and contradicting claims', async () => {
      const result = await aliceClient.factcheck.factCheck({
        statement: 'The speed of light is constant.',
        minConfidence: 0.5,
      });

      expect(result.supportingClaims).toBeDefined();
      expect(result.contradictingClaims).toBeDefined();
    });

    it('should batch fact-check multiple statements', async () => {
      const results = await aliceClient.factcheck.batchFactCheck([
        { statement: 'Water is H2O.' },
        { statement: 'The sky is green.' },
      ]);

      expect(results.length).toBe(2);
    });
  });

  // ==========================================================================
  // Markets Integration Tests
  // ==========================================================================

  describe('Epistemic Markets Integration', () => {
    let claimForMarket: string;

    beforeAll(async () => {
      claimForMarket = await aliceClient.claims.createClaim({
        content: 'Quantum computers will achieve practical supremacy by 2030.',
        classification: { empirical: 0.4, normative: 0.1, mythic: 0.5 },
        tags: ['technology', 'prediction', 'quantum'],
      });
    });

    it('should spawn a verification market for a claim', async () => {
      const marketId = await aliceClient.claims.spawnVerificationMarket({
        claimId: claimForMarket,
        targetE: 4, // Target E4 epistemic level
        minConfidence: 0.7,
        closesAt: Date.now() + 30 * 24 * 60 * 60 * 1000, // 30 days
        initialSubsidy: 100,
        tags: ['quantum', 'prediction'],
      });

      expect(marketId).toBeDefined();
    });

    it('should get markets associated with a claim', async () => {
      const markets = await aliceClient.marketsIntegration.getClaimMarkets(claimForMarket);

      expect(markets.length).toBeGreaterThan(0);
      expect(markets[0].status).toBe('OPEN');
    });

    it('should calculate market value for a claim', async () => {
      const value = await aliceClient.marketsIntegration.calculateMarketValue(claimForMarket);

      expect(value).toBeDefined();
      expect(value.potentialValue).toBeGreaterThanOrEqual(0);
    });
  });

  // ==========================================================================
  // Search & Query Tests
  // ==========================================================================

  describe('Search & Query', () => {
    it('should search claims by text', async () => {
      const results = await aliceClient.query.search('climate change', {
        limit: 10,
      });

      expect(results.length).toBeGreaterThanOrEqual(0);
    });

    it('should search with epistemic type filter', async () => {
      const results = await aliceClient.query.search('science', {
        epistemicType: 'EMPIRICAL',
        minCredibility: 0.5,
        limit: 10,
      });

      results.forEach((claim) => {
        expect(claim.classification.empirical).toBeGreaterThan(0.5);
      });
    });

    it('should get claims by author', async () => {
      const claims = await aliceClient.query.getClaimsByAuthor(alice.agentPubKey, {
        limit: 10,
      });

      expect(claims.length).toBeGreaterThan(0);
    });

    it('should search with tag filtering', async () => {
      const results = await aliceClient.query.search('', {
        tags: ['physics'],
        limit: 10,
      });

      results.forEach((claim) => {
        expect(claim.tags).toContain('physics');
      });
    });
  });

  // ==========================================================================
  // Multi-Agent Tests
  // ==========================================================================

  describe('Multi-Agent Collaboration', () => {
    it('should allow Bob to see claims created by Alice', async () => {
      // Wait for gossip
      await new Promise((r) => setTimeout(r, 1000));

      const claims = await bobClient.query.listClaims({ limit: 20 });
      expect(claims.items.length).toBeGreaterThan(0);
    });

    it('should allow Bob to add supporting evidence to Alice\'s claim', async () => {
      const aliceClaims = await bobClient.query.getClaimsByAuthor(alice.agentPubKey, { limit: 1 });
      const aliceClaimId = aliceClaims[0].id;

      const bobClaim = await bobClient.claims.createClaim({
        content: 'Independent verification confirms the previous finding.',
        classification: { empirical: 0.9, normative: 0.05, mythic: 0.05 },
        tags: ['verification'],
      });

      const relationshipId = await bobClient.graph.createRelationship(
        bobClaim,
        aliceClaimId,
        'SUPPORTS',
        0.8
      );

      expect(relationshipId).toBeDefined();
    });

    it('should update credibility based on multi-agent evidence', async () => {
      const aliceClaims = await aliceClient.query.getClaimsByAuthor(alice.agentPubKey, { limit: 1 });
      const claimId = aliceClaims[0].id;

      // Propagate belief with new evidence
      await aliceClient.graph.propagateBelief(claimId);

      const credibility = await aliceClient.inference.calculateEnhancedCredibility(claimId, 'Claim');

      // Credibility should reflect multiple sources
      expect(credibility.factors.sourceDiversity).toBeGreaterThan(0);
    });
  });
});
