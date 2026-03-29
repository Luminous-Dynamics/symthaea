// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Knowledge Simulation Engine
 *
 * Simulates Knowledge hApp scenarios for testing and demonstration.
 * These simulations model the behavior of claims, verification markets,
 * belief propagation, and cross-hApp interactions.
 */

import { AppWebsocket } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

interface SimulationConfig {
  numAgents: number;
  numClaims: number;
  byzantineRatio: number; // 0.0 - 1.0
  networkDelay: number; // ms
  marketDuration: number; // ms
  seed?: number;
}

interface SimulationResult {
  scenario: string;
  success: boolean;
  duration: number;
  metrics: Record<string, number>;
  events: SimulationEvent[];
  errors: string[];
}

interface SimulationEvent {
  timestamp: number;
  type: string;
  data: Record<string, unknown>;
}

interface SimulatedAgent {
  id: string;
  pubkey: string;
  matl: number;
  isByzantine: boolean;
  claims: string[];
  votes: Map<string, boolean>;
}

interface SimulatedClaim {
  id: string;
  content: string;
  author: string;
  classification: { e: number; n: number; m: number };
  credibility: number;
  dependents: string[];
  dependencies: string[];
  marketId?: string;
  verified?: boolean;
}

interface SimulatedMarket {
  id: string;
  claimId: string;
  targetE: number;
  votes: Map<string, { vote: boolean; matl: number }>;
  resolved: boolean;
  outcome?: 'verified' | 'refuted' | 'inconclusive';
}

// ============================================================================
// Random Number Generator (Seeded)
// ============================================================================

class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  next(): number {
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }

  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  nextFloat(min: number, max: number): number {
    return this.next() * (max - min) + min;
  }

  shuffle<T>(array: T[]): T[] {
    const result = [...array];
    for (let i = result.length - 1; i > 0; i--) {
      const j = this.nextInt(0, i);
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}

// ============================================================================
// Simulation Engine
// ============================================================================

export class KnowledgeSimulation {
  private config: SimulationConfig;
  private random: SeededRandom;
  private agents: Map<string, SimulatedAgent> = new Map();
  private claims: Map<string, SimulatedClaim> = new Map();
  private markets: Map<string, SimulatedMarket> = new Map();
  private events: SimulationEvent[] = [];
  private startTime: number = 0;

  constructor(config: SimulationConfig) {
    this.config = config;
    this.random = new SeededRandom(config.seed || Date.now());
  }

  // --------------------------------------------------------------------------
  // Setup
  // --------------------------------------------------------------------------

  private setupAgents(): void {
    const numByzantine = Math.floor(this.config.numAgents * this.config.byzantineRatio);

    for (let i = 0; i < this.config.numAgents; i++) {
      const id = `agent-${i}`;
      const agent: SimulatedAgent = {
        id,
        pubkey: `uhCAk${id}`,
        matl: this.random.nextFloat(0.3, 1.0),
        isByzantine: i < numByzantine,
        claims: [],
        votes: new Map(),
      };
      this.agents.set(id, agent);
    }

    this.logEvent('setup', { agentsCreated: this.config.numAgents, byzantineAgents: numByzantine });
  }

  private setupClaims(): void {
    const agentIds = Array.from(this.agents.keys());

    for (let i = 0; i < this.config.numClaims; i++) {
      const id = `claim-${i}`;
      const authorId = agentIds[this.random.nextInt(0, agentIds.length - 1)];
      const claim: SimulatedClaim = {
        id,
        content: `Simulated claim ${i}`,
        author: authorId,
        classification: {
          e: this.random.nextFloat(0.3, 0.9),
          n: this.random.nextFloat(0.0, 0.5),
          m: this.random.nextFloat(0.2, 0.8),
        },
        credibility: this.random.nextFloat(0.4, 0.8),
        dependents: [],
        dependencies: [],
      };
      this.claims.set(id, claim);
      this.agents.get(authorId)!.claims.push(id);
    }

    // Create some dependencies (10% of claims depend on others)
    const claimIds = Array.from(this.claims.keys());
    for (const claim of this.claims.values()) {
      if (this.random.next() < 0.1) {
        const depId = claimIds[this.random.nextInt(0, claimIds.length - 1)];
        if (depId !== claim.id) {
          claim.dependencies.push(depId);
          this.claims.get(depId)!.dependents.push(claim.id);
        }
      }
    }

    this.logEvent('setup', { claimsCreated: this.config.numClaims });
  }

  // --------------------------------------------------------------------------
  // Market Simulation
  // --------------------------------------------------------------------------

  private createMarket(claimId: string, targetE: number): SimulatedMarket {
    const id = `market-${this.markets.size}`;
    const market: SimulatedMarket = {
      id,
      claimId,
      targetE,
      votes: new Map(),
      resolved: false,
    };
    this.markets.set(id, market);
    this.claims.get(claimId)!.marketId = id;
    this.logEvent('market_created', { marketId: id, claimId, targetE });
    return market;
  }

  private simulateVoting(market: SimulatedMarket): void {
    const claim = this.claims.get(market.claimId)!;

    // Each agent votes
    for (const agent of this.agents.values()) {
      // Byzantine agents vote against truth
      let vote: boolean;
      if (agent.isByzantine) {
        vote = claim.classification.e < market.targetE; // Vote wrong
      } else {
        // Honest agents vote based on evidence
        vote = claim.classification.e >= market.targetE;
      }

      market.votes.set(agent.id, { vote, matl: agent.matl });
      this.logEvent('vote', { marketId: market.id, agentId: agent.id, vote, matl: agent.matl });
    }
  }

  private resolveMarket(market: SimulatedMarket): void {
    // MATL² weighted voting
    let weightedYes = 0;
    let weightedNo = 0;
    let totalWeight = 0;

    for (const [agentId, voteData] of market.votes) {
      const weight = voteData.matl * voteData.matl; // MATL²
      totalWeight += weight;
      if (voteData.vote) {
        weightedYes += weight;
      } else {
        weightedNo += weight;
      }
    }

    const yesRatio = weightedYes / totalWeight;
    const noRatio = weightedNo / totalWeight;

    if (yesRatio > 0.6) {
      market.outcome = 'verified';
    } else if (noRatio > 0.6) {
      market.outcome = 'refuted';
    } else {
      market.outcome = 'inconclusive';
    }
    market.resolved = true;

    this.logEvent('market_resolved', {
      marketId: market.id,
      outcome: market.outcome,
      yesRatio,
      noRatio,
    });

    // Update claim if verified
    if (market.outcome === 'verified') {
      const claim = this.claims.get(market.claimId)!;
      claim.classification.e = market.targetE;
      claim.verified = true;
      this.logEvent('claim_verified', { claimId: claim.id, newE: market.targetE });
    }
  }

  // --------------------------------------------------------------------------
  // Belief Propagation
  // --------------------------------------------------------------------------

  private propagateBeliefs(startClaimId: string): number {
    const visited = new Set<string>();
    const queue = [startClaimId];
    let updatedCount = 0;

    while (queue.length > 0) {
      const claimId = queue.shift()!;
      if (visited.has(claimId)) continue;
      visited.add(claimId);

      const claim = this.claims.get(claimId)!;

      // Update dependents
      for (const dependentId of claim.dependents) {
        const dependent = this.claims.get(dependentId)!;

        // Simple propagation: dependent's E influenced by dependencies
        const oldE = dependent.classification.e;
        const avgDependencyE = dependent.dependencies
          .map((id) => this.claims.get(id)!.classification.e)
          .reduce((sum, e) => sum + e, 0) / dependent.dependencies.length;

        dependent.classification.e = (oldE + avgDependencyE) / 2;
        updatedCount++;

        this.logEvent('belief_propagated', {
          fromClaimId: claimId,
          toClaimId: dependentId,
          oldE,
          newE: dependent.classification.e,
        });

        queue.push(dependentId);
      }
    }

    return updatedCount;
  }

  // --------------------------------------------------------------------------
  // Scenarios
  // --------------------------------------------------------------------------

  /**
   * Scenario 1: Claim Verification Cascade
   * Market resolves → claim updates → dependents cascade
   */
  async runVerificationCascade(): Promise<SimulationResult> {
    this.reset();
    this.setupAgents();
    this.setupClaims();

    // Find a claim with dependents
    let rootClaim: SimulatedClaim | undefined;
    for (const claim of this.claims.values()) {
      if (claim.dependents.length >= 2) {
        rootClaim = claim;
        break;
      }
    }

    if (!rootClaim) {
      // Create dependencies if none exist
      const claimIds = Array.from(this.claims.keys());
      rootClaim = this.claims.get(claimIds[0])!;
      for (let i = 1; i < Math.min(4, claimIds.length); i++) {
        const dep = this.claims.get(claimIds[i])!;
        dep.dependencies.push(rootClaim.id);
        rootClaim.dependents.push(dep.id);
      }
    }

    // Create and resolve verification market
    const market = this.createMarket(rootClaim.id, 0.8);
    this.simulateVoting(market);
    this.resolveMarket(market);

    // Cascade update
    const cascadedCount = this.propagateBeliefs(rootClaim.id);

    return {
      scenario: 'verification_cascade',
      success: market.resolved && cascadedCount > 0,
      duration: Date.now() - this.startTime,
      metrics: {
        marketResolved: market.resolved ? 1 : 0,
        marketOutcome: market.outcome === 'verified' ? 1 : 0,
        claimsCascaded: cascadedCount,
        dependentsCount: rootClaim.dependents.length,
      },
      events: this.events,
      errors: [],
    };
  }

  /**
   * Scenario 2: Contradiction Detection
   * Two contradicting claims → detection → resolution
   */
  async runContradictionDetection(): Promise<SimulationResult> {
    this.reset();
    this.setupAgents();

    // Create two contradicting claims
    const claim1: SimulatedClaim = {
      id: 'claim-contradiction-1',
      content: 'Global temperature increased 1.2°C',
      author: 'agent-0',
      classification: { e: 0.8, n: 0.1, m: 0.3 },
      credibility: 0.7,
      dependents: [],
      dependencies: [],
    };

    const claim2: SimulatedClaim = {
      id: 'claim-contradiction-2',
      content: 'Global temperature decreased 0.5°C',
      author: 'agent-1',
      classification: { e: 0.6, n: 0.1, m: 0.3 },
      credibility: 0.5,
      dependents: [],
      dependencies: [],
    };

    this.claims.set(claim1.id, claim1);
    this.claims.set(claim2.id, claim2);
    this.logEvent('claims_created', { claim1: claim1.id, claim2: claim2.id });

    // Detect contradiction (simulated - in reality this would use NLP/semantic analysis)
    const contradictionDetected = true; // Simulated detection
    this.logEvent('contradiction_detected', { claim1: claim1.id, claim2: claim2.id });

    // Create verification market for the higher-E claim
    const marketClaim = claim1.classification.e > claim2.classification.e ? claim1 : claim2;
    const market = this.createMarket(marketClaim.id, 0.85);
    this.simulateVoting(market);
    this.resolveMarket(market);

    return {
      scenario: 'contradiction_detection',
      success: contradictionDetected && market.resolved,
      duration: Date.now() - this.startTime,
      metrics: {
        contradictionDetected: contradictionDetected ? 1 : 0,
        marketCreated: 1,
        marketOutcome: market.outcome === 'verified' ? 1 : market.outcome === 'refuted' ? -1 : 0,
        winningClaimE: marketClaim.classification.e,
      },
      events: this.events,
      errors: [],
    };
  }

  /**
   * Scenario 3: Cross-hApp Fact Check
   * Media hApp requests → Knowledge responds
   */
  async runCrossHappFactCheck(): Promise<SimulationResult> {
    this.reset();
    this.setupAgents();
    this.setupClaims();

    // Simulate external fact-check request
    const statement = 'The earth revolves around the sun';
    const sourceHapp = 'mycelix-media';
    this.logEvent('factcheck_request', { statement, sourceHapp });

    // Find relevant claims (simulated semantic search)
    const relevantClaims = Array.from(this.claims.values())
      .filter(() => this.random.next() < 0.3) // 30% chance of relevance
      .slice(0, 5);

    this.logEvent('relevant_claims_found', { count: relevantClaims.length });

    // Calculate verdict based on relevant claims
    const avgE =
      relevantClaims.reduce((sum, c) => sum + c.classification.e, 0) / relevantClaims.length || 0;
    const avgCredibility =
      relevantClaims.reduce((sum, c) => sum + c.credibility, 0) / relevantClaims.length || 0;

    let verdict: string;
    let confidence: number;

    if (relevantClaims.length === 0) {
      verdict = 'InsufficientEvidence';
      confidence = 0;
    } else if (avgE > 0.7 && avgCredibility > 0.6) {
      verdict = 'True';
      confidence = avgE * avgCredibility;
    } else if (avgE > 0.5) {
      verdict = 'MostlyTrue';
      confidence = avgE * avgCredibility * 0.8;
    } else {
      verdict = 'Mixed';
      confidence = 0.5;
    }

    this.logEvent('factcheck_result', { verdict, confidence, claimsAnalyzed: relevantClaims.length });

    return {
      scenario: 'cross_happ_factcheck',
      success: relevantClaims.length > 0,
      duration: Date.now() - this.startTime,
      metrics: {
        relevantClaimsFound: relevantClaims.length,
        averageE: avgE,
        averageCredibility: avgCredibility,
        confidence,
      },
      events: this.events,
      errors: [],
    };
  }

  /**
   * Scenario 4: Byzantine Resilience
   * 45% adversarial oracles → still correct resolution
   */
  async runByzantineResilience(): Promise<SimulationResult> {
    // Reset with high Byzantine ratio
    const originalConfig = { ...this.config };
    this.config.byzantineRatio = 0.45; // 45% adversarial
    this.config.numAgents = 100; // More agents for statistical significance

    this.reset();
    this.setupAgents();

    // Create a clearly verifiable claim
    const claim: SimulatedClaim = {
      id: 'claim-byzantine-test',
      content: 'Test claim for Byzantine resilience',
      author: 'agent-0',
      classification: { e: 0.85, n: 0.1, m: 0.3 }, // High E, should verify
      credibility: 0.8,
      dependents: [],
      dependencies: [],
    };
    this.claims.set(claim.id, claim);

    // Create market
    const market = this.createMarket(claim.id, 0.8);
    this.simulateVoting(market);
    this.resolveMarket(market);

    // Calculate metrics
    const byzantineCount = Array.from(this.agents.values()).filter((a) => a.isByzantine).length;
    const honestCount = this.agents.size - byzantineCount;

    // Verify that MATL² weighting overcame Byzantine actors
    const correctOutcome = market.outcome === 'verified'; // Should verify despite 45% adversarial

    this.config = originalConfig; // Restore

    return {
      scenario: 'byzantine_resilience',
      success: correctOutcome,
      duration: Date.now() - this.startTime,
      metrics: {
        totalAgents: this.agents.size,
        byzantineAgents: byzantineCount,
        honestAgents: honestCount,
        byzantineRatio: byzantineCount / this.agents.size,
        correctOutcome: correctOutcome ? 1 : 0,
        marketOutcome: market.outcome === 'verified' ? 1 : market.outcome === 'refuted' ? -1 : 0,
      },
      events: this.events,
      errors: correctOutcome ? [] : ['Byzantine attack succeeded - MATL² weighting failed'],
    };
  }

  /**
   * Scenario 5: Information Value Ranking
   * Rank claims by verification value
   */
  async runInformationValueRanking(): Promise<SimulationResult> {
    this.reset();
    this.setupAgents();
    this.setupClaims();

    // Calculate information value for each claim
    const rankedClaims: Array<{ claim: SimulatedClaim; iv: number }> = [];

    for (const claim of this.claims.values()) {
      // Information value formula:
      // IV = uncertainty * connectivity * age_factor
      const uncertainty = 1 - Math.abs(claim.classification.e - 0.5) * 2; // Max uncertainty at E=0.5
      const connectivity = 1 + claim.dependents.length * 0.2; // More dependents = higher value
      const ageFactor = 1; // Simulated as constant

      const iv = uncertainty * connectivity * ageFactor;
      rankedClaims.push({ claim, iv });
    }

    // Sort by IV descending
    rankedClaims.sort((a, b) => b.iv - a.iv);

    this.logEvent('claims_ranked', {
      totalClaims: rankedClaims.length,
      topClaimId: rankedClaims[0]?.claim.id,
      topClaimIV: rankedClaims[0]?.iv,
    });

    // Verify top claims for verification
    const topN = 5;
    const verificationTargets = rankedClaims.slice(0, topN);

    for (const { claim, iv } of verificationTargets) {
      this.logEvent('verification_recommended', {
        claimId: claim.id,
        informationValue: iv,
        currentE: claim.classification.e,
      });
    }

    return {
      scenario: 'information_value_ranking',
      success: rankedClaims.length > 0,
      duration: Date.now() - this.startTime,
      metrics: {
        totalClaims: rankedClaims.length,
        maxIV: rankedClaims[0]?.iv || 0,
        minIV: rankedClaims[rankedClaims.length - 1]?.iv || 0,
        avgIV: rankedClaims.reduce((sum, r) => sum + r.iv, 0) / rankedClaims.length || 0,
        recommendedForVerification: topN,
      },
      events: this.events,
      errors: [],
    };
  }

  // --------------------------------------------------------------------------
  // Utilities
  // --------------------------------------------------------------------------

  private reset(): void {
    this.agents.clear();
    this.claims.clear();
    this.markets.clear();
    this.events = [];
    this.startTime = Date.now();
  }

  private logEvent(type: string, data: Record<string, unknown>): void {
    this.events.push({
      timestamp: Date.now() - this.startTime,
      type,
      data,
    });
  }

  /**
   * Run all scenarios
   */
  async runAll(): Promise<SimulationResult[]> {
    const results: SimulationResult[] = [];

    results.push(await this.runVerificationCascade());
    results.push(await this.runContradictionDetection());
    results.push(await this.runCrossHappFactCheck());
    results.push(await this.runByzantineResilience());
    results.push(await this.runInformationValueRanking());

    return results;
  }
}

// ============================================================================
// CLI Runner
// ============================================================================

async function main() {
  console.log('Knowledge Simulation Engine');
  console.log('===========================\n');

  const simulation = new KnowledgeSimulation({
    numAgents: 50,
    numClaims: 100,
    byzantineRatio: 0.2,
    networkDelay: 100,
    marketDuration: 1000,
    seed: 12345,
  });

  const results = await simulation.runAll();

  for (const result of results) {
    console.log(`\n${result.scenario.toUpperCase()}`);
    console.log('-'.repeat(40));
    console.log(`Success: ${result.success}`);
    console.log(`Duration: ${result.duration}ms`);
    console.log('Metrics:', JSON.stringify(result.metrics, null, 2));
    if (result.errors.length > 0) {
      console.log('Errors:', result.errors);
    }
  }

  // Summary
  const passed = results.filter((r) => r.success).length;
  console.log(`\n${'='.repeat(40)}`);
  console.log(`SUMMARY: ${passed}/${results.length} scenarios passed`);
}

// Run if executed directly
if (require.main === module) {
  main().catch(console.error);
}

export { SimulationConfig, SimulationResult, SimulatedAgent, SimulatedClaim, SimulatedMarket };
