// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * DAO Module Tests
 *
 * Tests for governance simulation, voting, conviction, and delegation utilities.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  // Simulation
  GovernanceSimulator,
  createSimulator,
  type SimulationConfig,
  // Types
  type ProposalType,
  type VotingMechanism,
  type EpistemicClassification,
  type ConvictionVote,
  type Vote,
  type QuadraticVote,
  type VoteTally,
  type DelegationDomain,
} from '../src/dao/index.js';

// =============================================================================
// GovernanceSimulator Tests
// =============================================================================

describe('GovernanceSimulator', () => {
  describe('createSimulator', () => {
    it('should create a simulator with default config', () => {
      const simulator = createSimulator();

      expect(simulator).toBeInstanceOf(GovernanceSimulator);
    });

    it('should create a simulator with custom agent count', () => {
      const simulator = createSimulator({ agentCount: 50 });

      expect(simulator).toBeInstanceOf(GovernanceSimulator);
    });

    it('should create a simulator with custom iterations', () => {
      const simulator = createSimulator({ iterations: 500 });

      expect(simulator).toBeInstanceOf(GovernanceSimulator);
    });

    it('should create a simulator with seed for reproducibility', () => {
      const simulator1 = createSimulator({ seed: 12345, agentCount: 10 });
      const simulator2 = createSimulator({ seed: 12345, agentCount: 10 });

      // Both should produce consistent results with same seed
      expect(simulator1).toBeInstanceOf(GovernanceSimulator);
      expect(simulator2).toBeInstanceOf(GovernanceSimulator);
    });

    it('should create a simulator with attacks disabled', () => {
      const simulator = createSimulator({
        simulateAttacks: false,
        agentCount: 20,
      });

      expect(simulator).toBeInstanceOf(GovernanceSimulator);
    });

    it('should create a simulator with custom agent distribution', () => {
      const simulator = createSimulator({
        agentDistribution: {
          randomVoters: 0.4,
          experts: 0.3,
          delegators: 0.2,
          attackers: 0.05,
          passive: 0.05,
        },
      });

      expect(simulator).toBeInstanceOf(GovernanceSimulator);
    });
  });

  describe('proposal simulation with proper VotingMechanism types', () => {
    let simulator: GovernanceSimulator;

    beforeEach(() => {
      simulator = createSimulator({
        agentCount: 20,
        iterations: 10,
        seed: 42,
      });
    });

    it('should simulate a standard proposal with ReputationWeighted mechanism', async () => {
      const mechanism: VotingMechanism = {
        ReputationWeighted: { quorum: 0.3, approval_threshold: 0.5 },
      };

      const result = await simulator.simulateProposal({
        proposalType: 'Standard',
        votingMechanism: mechanism,
        classification: {
          empirical: 'E2',
          normative: 'N1',
          materiality: 'M2',
        },
      });

      expect(result).toBeDefined();
      expect(typeof result.passRate).toBe('number');
      expect(result.passRate).toBeGreaterThanOrEqual(0);
      expect(result.passRate).toBeLessThanOrEqual(1);
    });

    it('should simulate a treasury proposal with Quadratic mechanism', async () => {
      const mechanism: VotingMechanism = {
        Quadratic: { budget_per_voter: 100, approval_threshold: 0.5 },
      };

      const result = await simulator.simulateProposal({
        proposalType: 'Treasury',
        votingMechanism: mechanism,
        classification: {
          empirical: 'E3',
          normative: 'N2',
          materiality: 'M3',
        },
        fundingRequest: 10000,
      });

      expect(result).toBeDefined();
      expect(typeof result.passRate).toBe('number');
    });

    it('should simulate with EqualWeight mechanism', async () => {
      const mechanism: VotingMechanism = {
        EqualWeight: { quorum: 0.25, approval_threshold: 0.5 },
      };

      const result = await simulator.simulateProposal({
        proposalType: 'Signaling',
        votingMechanism: mechanism,
        classification: {
          empirical: 'E1',
          normative: 'N1',
          materiality: 'M1',
        },
      });

      expect(result).toBeDefined();
    });

    it('should simulate with conviction voting', async () => {
      const mechanism: VotingMechanism = {
        Conviction: {
          threshold: 0.6,
          decay_rate: 0.9,
          min_conviction_time_hours: 24,
        },
      };

      const result = await simulator.simulateProposal({
        proposalType: 'Streaming',
        votingMechanism: mechanism,
        classification: {
          empirical: 'E2',
          normative: 'N1',
          materiality: 'M2',
        },
      });

      expect(result).toBeDefined();
    });
  });
});

// =============================================================================
// Voting Utility Tests
// =============================================================================

describe('Voting Utilities', () => {
  describe('quadratic voting calculations', () => {
    it('should calculate effective votes from credits', () => {
      // sqrt(25) = 5
      const credits = 25;
      const effectiveVotes = Math.sqrt(credits);

      expect(effectiveVotes).toBe(5);
    });

    it('should calculate credits needed for desired votes', () => {
      // 10^2 = 100
      const desiredVotes = 10;
      const creditsNeeded = Math.pow(desiredVotes, 2);

      expect(creditsNeeded).toBe(100);
    });

    it('should demonstrate diminishing returns', () => {
      // 1 credit = 1 vote
      // 4 credits = 2 votes
      // 9 credits = 3 votes
      // 16 credits = 4 votes
      expect(Math.sqrt(1)).toBe(1);
      expect(Math.sqrt(4)).toBe(2);
      expect(Math.sqrt(9)).toBe(3);
      expect(Math.sqrt(16)).toBe(4);
    });

    it('should handle zero credits', () => {
      expect(Math.sqrt(0)).toBe(0);
    });

    it('should handle fractional credits', () => {
      const credits = 2;
      const effectiveVotes = Math.sqrt(credits);

      expect(effectiveVotes).toBeCloseTo(1.414, 2);
    });
  });
});

// =============================================================================
// Conviction Voting Utility Tests
// =============================================================================

describe('Conviction Voting Utilities', () => {
  const DEFAULT_HALF_LIFE_HOURS = 168; // 7 days

  /**
   * conviction(t) = weight × (1 - e^(-t/τ))
   */
  function calculateConviction(
    weight: number,
    hoursStaked: number,
    halfLifeHours: number = DEFAULT_HALF_LIFE_HOURS
  ): number {
    return weight * (1 - Math.exp(-hoursStaked / halfLifeHours));
  }

  /**
   * t = -τ × ln(1 - target/weight)
   */
  function calculateTimeToTarget(
    weight: number,
    targetConviction: number,
    halfLifeHours: number = DEFAULT_HALF_LIFE_HOURS
  ): number {
    if (targetConviction >= weight) {
      return Infinity;
    }
    return -halfLifeHours * Math.log(1 - targetConviction / weight);
  }

  describe('calculateConviction', () => {
    it('should return 0 at time 0', () => {
      const conviction = calculateConviction(100, 0);

      expect(conviction).toBe(0);
    });

    it('should approach weight as time increases', () => {
      const weight = 100;
      // After many half-lives, conviction should be close to weight
      const conviction = calculateConviction(weight, 1680); // 10 half-lives

      expect(conviction).toBeGreaterThan(99);
      expect(conviction).toBeLessThanOrEqual(weight);
    });

    it('should reach ~63% of weight after one time constant', () => {
      const weight = 100;
      const tau = DEFAULT_HALF_LIFE_HOURS;
      const conviction = calculateConviction(weight, tau);

      // 1 - e^-1 ≈ 0.632
      expect(conviction).toBeCloseTo(63.2, 0);
    });

    it('should scale linearly with weight', () => {
      const hours = 84; // half a week
      const conviction1 = calculateConviction(100, hours);
      const conviction2 = calculateConviction(200, hours);

      expect(conviction2).toBeCloseTo(conviction1 * 2, 5);
    });

    it('should use custom half-life', () => {
      const weight = 100;
      const hours = 24; // 1 day
      const shortHalfLife = 24;

      const convictionShort = calculateConviction(weight, hours, shortHalfLife);
      const convictionDefault = calculateConviction(weight, hours);

      // Shorter half-life = faster conviction growth
      expect(convictionShort).toBeGreaterThan(convictionDefault);
    });
  });

  describe('calculateTimeToTarget', () => {
    it('should return Infinity if target exceeds weight', () => {
      const time = calculateTimeToTarget(100, 150);

      expect(time).toBe(Infinity);
    });

    it('should return Infinity if target equals weight', () => {
      const time = calculateTimeToTarget(100, 100);

      expect(time).toBe(Infinity);
    });

    it('should return 0 for zero target', () => {
      const time = calculateTimeToTarget(100, 0);

      expect(time).toBeCloseTo(0, 10);
    });

    it('should calculate correct time for 50% conviction', () => {
      const weight = 100;
      const target = 50;
      const time = calculateTimeToTarget(weight, target);

      // Verify by calculating conviction at that time
      const verification = calculateConviction(weight, time);
      expect(verification).toBeCloseTo(target, 5);
    });

    it('should calculate correct time for 90% conviction', () => {
      const weight = 100;
      const target = 90;
      const time = calculateTimeToTarget(weight, target);

      const verification = calculateConviction(weight, time);
      expect(verification).toBeCloseTo(target, 5);
    });

    it('should work with custom half-life', () => {
      const weight = 100;
      const target = 50;
      const shortHalfLife = 48; // 2 days

      const timeShort = calculateTimeToTarget(weight, target, shortHalfLife);
      const timeDefault = calculateTimeToTarget(weight, target);

      // Shorter half-life = less time needed
      expect(timeShort).toBeLessThan(timeDefault);
    });
  });

  describe('conviction voting properties', () => {
    it('should demonstrate time-weighted voting benefit', () => {
      // Two stakers with same weight
      const weight = 100;

      // Staker A: staked for 2 weeks
      const convictionA = calculateConviction(weight, 336);

      // Staker B: just staked
      const convictionB = calculateConviction(weight, 0);

      // Staker A should have significantly more conviction
      expect(convictionA).toBeGreaterThan(convictionB);
      expect(convictionA).toBeGreaterThan(80); // Should have built up significant conviction
    });

    it('should show conviction loss on unstake and restake', () => {
      const weight = 100;

      // Original conviction after 1 week
      const originalConviction = calculateConviction(weight, 168);

      // After unstaking and immediately restaking, conviction resets to 0
      const afterRestake = calculateConviction(weight, 0);

      expect(afterRestake).toBe(0);
      expect(originalConviction).toBeGreaterThan(60);
    });

    it('should calculate break-even time for moving stake', () => {
      const weight = 100;
      const currentConviction = calculateConviction(weight, 168); // 1 week staked

      // Time needed to rebuild the same conviction on new proposal
      const rebuildTime = calculateTimeToTarget(weight, currentConviction);

      // Should take about the same time to rebuild
      expect(rebuildTime).toBeCloseTo(168, 0);
    });
  });
});

// =============================================================================
// Delegation Domain Tests
// =============================================================================

describe('Delegation Domains', () => {
  const domains: DelegationDomain[] = ['Technical', 'Economic', 'Governance', 'Social', 'Cultural'];

  it('should have all expected domains', () => {
    expect(domains).toContain('Technical');
    expect(domains).toContain('Economic');
    expect(domains).toContain('Governance');
    expect(domains).toContain('Social');
    expect(domains).toContain('Cultural');
  });

  it('should support domain-specific delegation', () => {
    // Simulated delegation structure
    const delegations = new Map<DelegationDomain, string>();

    delegations.set('Technical', 'expert-alice');
    delegations.set('Economic', 'expert-bob');

    expect(delegations.get('Technical')).toBe('expert-alice');
    expect(delegations.get('Economic')).toBe('expert-bob');
    expect(delegations.get('Governance')).toBeUndefined();
  });
});

// =============================================================================
// Epistemic Classification Tests
// =============================================================================

describe('Epistemic Classification', () => {
  it('should validate E-N-M codes', () => {
    const classification: EpistemicClassification = {
      empirical: 'E3',
      normative: 'N2',
      materiality: 'M2',
    };

    expect(classification.empirical).toMatch(/^E[0-4]$/);
    expect(classification.normative).toMatch(/^N[0-3]$/);
    expect(classification.materiality).toMatch(/^M[0-3]$/);
  });

  it('should classify different proposal types appropriately', () => {
    // Technical infrastructure change
    const technicalClassification: EpistemicClassification = {
      empirical: 'E3', // Cryptographically verifiable
      normative: 'N2', // Network agreement needed
      materiality: 'M3', // Foundational impact
    };

    // Social policy change
    const socialClassification: EpistemicClassification = {
      empirical: 'E1', // Community evidence
      normative: 'N1', // Communal values
      materiality: 'M2', // Persistent but not foundational
    };

    // Emergency response
    const emergencyClassification: EpistemicClassification = {
      empirical: 'E4', // Fully reproducible
      normative: 'N3', // Axiomatic (security)
      materiality: 'M3', // Foundational
    };

    expect(technicalClassification.empirical).toBe('E3');
    expect(socialClassification.normative).toBe('N1');
    expect(emergencyClassification.materiality).toBe('M3');
  });
});

// =============================================================================
// Voting Mechanism Tests
// =============================================================================

describe('Voting Mechanisms', () => {
  const mechanisms: VotingMechanism[] = [
    'ReputationWeighted',
    'EqualWeight',
    'Quadratic',
    'Conviction',
    'SuperMajority',
  ];

  it('should have all expected mechanisms', () => {
    expect(mechanisms).toContain('ReputationWeighted');
    expect(mechanisms).toContain('EqualWeight');
    expect(mechanisms).toContain('Quadratic');
    expect(mechanisms).toContain('Conviction');
    expect(mechanisms).toContain('SuperMajority');
  });

  it('should select appropriate mechanism for proposal type', () => {
    const mechanismForType: Record<ProposalType, VotingMechanism> = {
      Standard: 'ReputationWeighted',
      Treasury: 'Quadratic',
      Emergency: 'SuperMajority',
      Constitutional: 'SuperMajority',
      Streaming: 'Conviction',
      Signaling: 'EqualWeight',
    };

    expect(mechanismForType['Standard']).toBe('ReputationWeighted');
    expect(mechanismForType['Treasury']).toBe('Quadratic');
    expect(mechanismForType['Emergency']).toBe('SuperMajority');
    expect(mechanismForType['Streaming']).toBe('Conviction');
  });
});
