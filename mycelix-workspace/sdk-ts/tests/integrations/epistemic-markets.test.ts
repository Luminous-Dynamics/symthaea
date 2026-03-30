// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Markets Integration Tests
 *
 * Tests for EpistemicMarketsService -- the domain-specific SDK service for
 * prediction markets with 3D epistemic classification, multi-dimensional
 * stakes, reasoning traces, belief dependency graphs, and wisdom extraction.
 *
 * EpistemicMarketsService uses an in-memory data model with MATL reputation
 * scoring via LocalBridge, so tests verify local state management
 * rather than zome call dispatch.
 */

import { describe, it, expect, beforeEach } from 'vitest';

import {
  EpistemicMarketsService,
  getEpistemicMarketsService,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  ResolutionMechanism,
  MarketStatus,
  Visibility,
  getRecommendedResolution,
  getRecommendedDurationDays,
  toClassificationCode,
  calculateStakeValue,
  createReputationOnlyStake,
  createMonetaryOnlyStake,
  type EpistemicPosition,
  type Market,
  type CreateMarketInput,
  type Prediction,
  type SubmitPredictionInput,
  type MultiDimensionalStake,
  type ReasoningTrace,
  type ReasoningStep,
  type Assumption,
  type UpdateTrigger,
  type InformationSource,
  type ConfidenceBreakdown,
  type AlternativeFraming,
  type BeliefDependency,
  type TemporalCommitment,
  type WisdomSeed,
  type WisdomLesson,
  type ActivationCondition,
  type PredictionUpdate,
  type PredictionResolution,
  type Outcome,
  type Order,
  type Commitment,
  type StepSupport,
  type MonetaryStake,
  type ReputationStake,
  type SocialStake,
  type CommitmentStake,
  type TimeStake,
  type MarketMechanism,
  type LessonApplicability,
} from '../../src/integrations/epistemic-markets/index.js';

// ============================================================================
// Test helpers
// ============================================================================

function makeEpistemicPosition(overrides: Partial<EpistemicPosition> = {}): EpistemicPosition {
  return {
    empirical: overrides.empirical ?? EmpiricalLevel.Measurable,
    normative: overrides.normative ?? NormativeLevel.Universal,
    materiality: overrides.materiality ?? MaterialityLevel.Persistent,
  };
}

function makeMinimalReasoning(overrides: Partial<ReasoningTrace> = {}): ReasoningTrace {
  return {
    summary: overrides.summary ?? 'Based on current trends and analysis of historical data patterns, this outcome is likely.',
    steps: overrides.steps ?? [{
      stepNumber: 1,
      claim: 'Historical trend supports this conclusion',
      support: { type: 'Evidence', sources: ['dataset-A'], strength: 0.8 },
      confidenceContribution: 0.3,
    }],
    assumptions: overrides.assumptions ?? [{
      id: 'a1',
      statement: 'Current trends continue',
      confidence: 0.7,
      falsificationCriteria: 'Major disruption event',
      importance: 0.8,
    }],
    updateTriggers: overrides.updateTriggers ?? [{
      description: 'New data release',
      evidenceType: 'DataSet',
      updateMagnitude: 0.1,
      direction: 'Increase',
    }],
    acknowledgedWeaknesses: overrides.acknowledgedWeaknesses ?? ['Limited historical data'],
    alternativesConsidered: overrides.alternativesConsidered ?? [],
    sources: overrides.sources ?? [],
    confidenceBreakdown: overrides.confidenceBreakdown ?? {
      baseRate: 0.4,
      evidenceAdjustment: 0.25,
      modelUncertainty: 0.1,
      knownUnknowns: 0.1,
      unknownUnknownsAllowance: 0.15,
    },
  };
}

function makeMarketInput(overrides: Partial<CreateMarketInput> = {}): CreateMarketInput {
  return {
    question: overrides.question ?? 'Will global solar capacity exceed 2TW by 2028?',
    description: overrides.description ?? 'Based on IEA projections',
    outcomes: overrides.outcomes ?? ['Yes', 'No'],
    epistemicPosition: overrides.epistemicPosition ?? makeEpistemicPosition(),
    mechanism: overrides.mechanism ?? { type: 'Binary', yesShares: 1000, noShares: 1000 },
    closesAt: overrides.closesAt ?? Date.now() + 30 * 24 * 60 * 60 * 1000,
    resolutionDeadline: overrides.resolutionDeadline ?? Date.now() + 60 * 24 * 60 * 60 * 1000,
    tags: overrides.tags ?? ['energy', 'solar'],
    sourceHapp: overrides.sourceHapp,
    resolutionSource: overrides.resolutionSource,
  };
}

function makePredictionInput(marketId: string, overrides: Partial<SubmitPredictionInput> = {}): SubmitPredictionInput {
  return {
    marketId,
    outcome: overrides.outcome ?? 'outcome_0',
    confidence: overrides.confidence ?? 0.7,
    stake: overrides.stake ?? createReputationOnlyStake(['energy'], 0.05, 1.0),
    reasoning: overrides.reasoning ?? makeMinimalReasoning(),
    dependsOn: overrides.dependsOn,
    temporalCommitment: overrides.temporalCommitment,
    wisdomSeed: overrides.wisdomSeed,
    epistemicLineage: overrides.epistemicLineage,
  };
}

// ============================================================================
// Epistemic Classification
// ============================================================================

describe('Epistemic Classification', () => {
  describe('EmpiricalLevel enum', () => {
    it('should have all 5 levels', () => {
      const levels = [
        EmpiricalLevel.Subjective,
        EmpiricalLevel.Testimonial,
        EmpiricalLevel.PrivateVerify,
        EmpiricalLevel.Cryptographic,
        EmpiricalLevel.Measurable,
      ];
      expect(levels).toHaveLength(5);
    });
  });

  describe('NormativeLevel enum', () => {
    it('should have all 4 levels', () => {
      const levels = [
        NormativeLevel.Personal,
        NormativeLevel.Communal,
        NormativeLevel.Network,
        NormativeLevel.Universal,
      ];
      expect(levels).toHaveLength(4);
    });
  });

  describe('MaterialityLevel enum', () => {
    it('should have all 4 levels', () => {
      const levels = [
        MaterialityLevel.Ephemeral,
        MaterialityLevel.Temporal,
        MaterialityLevel.Persistent,
        MaterialityLevel.Foundational,
      ];
      expect(levels).toHaveLength(4);
    });
  });

  describe('MarketStatus enum', () => {
    it('should have all 6 statuses', () => {
      const statuses = [
        MarketStatus.Open,
        MarketStatus.Closed,
        MarketStatus.Resolving,
        MarketStatus.Resolved,
        MarketStatus.Disputed,
        MarketStatus.Cancelled,
      ];
      expect(statuses).toHaveLength(6);
    });
  });

  describe('ResolutionMechanism enum', () => {
    it('should have all 6 mechanisms', () => {
      const mechanisms = [
        ResolutionMechanism.Automated,
        ResolutionMechanism.ZkVerified,
        ResolutionMechanism.OracleConsensus,
        ResolutionMechanism.CommunityVote,
        ResolutionMechanism.ReputationStaked,
        ResolutionMechanism.GovernanceEscalation,
      ];
      expect(mechanisms).toHaveLength(6);
    });
  });

  describe('Visibility enum', () => {
    it('should have all 4 visibility levels', () => {
      const levels = [Visibility.Private, Visibility.Limited, Visibility.Community, Visibility.Public];
      expect(levels).toHaveLength(4);
    });
  });
});

// ============================================================================
// Utility functions
// ============================================================================

describe('Utility Functions', () => {
  describe('getRecommendedResolution', () => {
    it('should return Automated for Measurable empirical level', () => {
      const position = makeEpistemicPosition({ empirical: EmpiricalLevel.Measurable });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.Automated);
    });

    it('should return Automated for Cryptographic empirical level', () => {
      const position = makeEpistemicPosition({ empirical: EmpiricalLevel.Cryptographic });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.Automated);
    });

    it('should return ZkVerified for PrivateVerify empirical level', () => {
      const position = makeEpistemicPosition({ empirical: EmpiricalLevel.PrivateVerify });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.ZkVerified);
    });

    it('should return OracleConsensus for Testimonial + Universal', () => {
      const position = makeEpistemicPosition({
        empirical: EmpiricalLevel.Testimonial,
        normative: NormativeLevel.Universal,
      });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.OracleConsensus);
    });

    it('should return CommunityVote for Testimonial + Personal', () => {
      const position = makeEpistemicPosition({
        empirical: EmpiricalLevel.Testimonial,
        normative: NormativeLevel.Personal,
      });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.CommunityVote);
    });

    it('should return ReputationStaked for Subjective empirical level', () => {
      const position = makeEpistemicPosition({ empirical: EmpiricalLevel.Subjective });
      expect(getRecommendedResolution(position)).toBe(ResolutionMechanism.ReputationStaked);
    });
  });

  describe('getRecommendedDurationDays', () => {
    it('should return 1 for Ephemeral', () => {
      expect(getRecommendedDurationDays(MaterialityLevel.Ephemeral)).toBe(1);
    });

    it('should return 7 for Temporal', () => {
      expect(getRecommendedDurationDays(MaterialityLevel.Temporal)).toBe(7);
    });

    it('should return 30 for Persistent', () => {
      expect(getRecommendedDurationDays(MaterialityLevel.Persistent)).toBe(30);
    });

    it('should return 90 for Foundational', () => {
      expect(getRecommendedDurationDays(MaterialityLevel.Foundational)).toBe(90);
    });
  });

  describe('toClassificationCode', () => {
    it('should produce correct code for E4-N3-M2', () => {
      const position: EpistemicPosition = {
        empirical: EmpiricalLevel.Measurable,
        normative: NormativeLevel.Universal,
        materiality: MaterialityLevel.Persistent,
      };
      expect(toClassificationCode(position)).toBe('E4-N3-M2');
    });

    it('should produce correct code for E0-N0-M0', () => {
      const position: EpistemicPosition = {
        empirical: EmpiricalLevel.Subjective,
        normative: NormativeLevel.Personal,
        materiality: MaterialityLevel.Ephemeral,
      };
      expect(toClassificationCode(position)).toBe('E0-N0-M0');
    });

    it('should produce correct code for E2-N1-M3', () => {
      const position: EpistemicPosition = {
        empirical: EmpiricalLevel.PrivateVerify,
        normative: NormativeLevel.Communal,
        materiality: MaterialityLevel.Foundational,
      };
      expect(toClassificationCode(position)).toBe('E2-N1-M3');
    });
  });

  describe('calculateStakeValue', () => {
    it('should return 0 for empty stake', () => {
      const stake: MultiDimensionalStake = {};
      expect(calculateStakeValue(stake)).toBe(0);
    });

    it('should count monetary stake directly', () => {
      const stake = createMonetaryOnlyStake(100, 'USD');
      expect(calculateStakeValue(stake)).toBe(100);
    });

    it('should weight reputation stake with confidence multiplier', () => {
      const stake = createReputationOnlyStake(['energy'], 0.01, 1.0);
      const value = calculateStakeValue(stake);
      expect(value).toBe(0.01 * 10000 * 1.0);
    });

    it('should add social visibility value', () => {
      const stake: MultiDimensionalStake = {
        social: { visibility: Visibility.Public },
      };
      expect(calculateStakeValue(stake)).toBe(500);
    });

    it('should double value for identity-linked social stake', () => {
      const stake: MultiDimensionalStake = {
        social: { visibility: Visibility.Public, identityLink: 'did:key:123' },
      };
      expect(calculateStakeValue(stake)).toBe(1000);
    });

    it('should count time investment', () => {
      const stake: MultiDimensionalStake = {
        time: { researchHours: 10, evidenceSubmitted: ['doc1', 'doc2'] },
      };
      // 10 * 15 + 2 * 25 = 200
      expect(calculateStakeValue(stake)).toBe(200);
    });

    it('should aggregate across all dimensions', () => {
      const stake: MultiDimensionalStake = {
        monetary: { amount: 50, currency: 'USD' },
        reputation: { domains: ['tech'], stakePercentage: 0.01, confidenceMultiplier: 1.0 },
        time: { researchHours: 5, evidenceSubmitted: ['doc1'] },
      };
      // 50 + (0.01 * 10000 * 1.0) + (5 * 15 + 1 * 25) = 50 + 100 + 100 = 250
      expect(calculateStakeValue(stake)).toBe(250);
    });
  });

  describe('createReputationOnlyStake', () => {
    it('should create a stake with only reputation component', () => {
      const stake = createReputationOnlyStake(['energy', 'climate'], 0.05, 1.5);
      expect(stake.reputation).toBeDefined();
      expect(stake.reputation!.domains).toEqual(['energy', 'climate']);
      expect(stake.reputation!.stakePercentage).toBe(0.05);
      expect(stake.reputation!.confidenceMultiplier).toBe(1.5);
      expect(stake.monetary).toBeUndefined();
    });
  });

  describe('createMonetaryOnlyStake', () => {
    it('should create a stake with only monetary component', () => {
      const stake = createMonetaryOnlyStake(250, 'EUR');
      expect(stake.monetary).toBeDefined();
      expect(stake.monetary!.amount).toBe(250);
      expect(stake.monetary!.currency).toBe('EUR');
      expect(stake.reputation).toBeUndefined();
    });
  });
});

// ============================================================================
// Type construction tests
// ============================================================================

describe('Epistemic Markets Types', () => {
  describe('EpistemicPosition', () => {
    it('should construct with all three axes', () => {
      const position = makeEpistemicPosition();
      expect(position.empirical).toBe(EmpiricalLevel.Measurable);
      expect(position.normative).toBe(NormativeLevel.Universal);
      expect(position.materiality).toBe(MaterialityLevel.Persistent);
    });
  });

  describe('ReasoningStep', () => {
    it('should accept Evidence support type', () => {
      const step: ReasoningStep = {
        stepNumber: 1,
        claim: 'Solar deployment is accelerating',
        support: { type: 'Evidence', sources: ['IEA Report 2025'], strength: 0.9 },
        confidenceContribution: 0.3,
      };
      expect(step.support.type).toBe('Evidence');
    });

    it('should accept Inference support type', () => {
      const step: ReasoningStep = {
        stepNumber: 2,
        claim: 'Therefore growth will continue',
        support: { type: 'Inference', fromSteps: [1], rule: 'trend extrapolation' },
        confidenceContribution: 0.2,
      };
      expect(step.support.type).toBe('Inference');
    });

    it('should accept all support types', () => {
      const types: StepSupport[] = [
        { type: 'Evidence', sources: ['x'], strength: 0.5 },
        { type: 'Inference', fromSteps: [1], rule: 'modus ponens' },
        { type: 'Testimony', source: 'Expert A', credibility: 0.8 },
        { type: 'Prior', basis: 'historical frequency' },
        { type: 'Assumption', id: 'a1' },
      ];
      expect(types).toHaveLength(5);
    });
  });

  describe('BeliefDependency', () => {
    it('should construct with prediction dependency', () => {
      const dep: BeliefDependency = {
        dependsOn: { type: 'Prediction', id: 'pred-123' },
        dependencyType: 'PositiveEvidence',
        strength: 0.8,
        conditional: { ifTrue: 0.9, ifFalse: 0.3, dependencyProbability: 0.7 },
      };
      expect(dep.dependsOn.type).toBe('Prediction');
      expect(dep.dependencyType).toBe('PositiveEvidence');
    });

    it('should accept all dependency types', () => {
      const types: BeliefDependency['dependencyType'][] = [
        'PositiveEvidence', 'NegativeEvidence', 'Prerequisite',
        'Causal', 'Correlated', 'SpecializationOf',
      ];
      expect(types).toHaveLength(6);
    });
  });

  describe('TemporalCommitment', () => {
    it('should accept Standard type', () => {
      const tc: TemporalCommitment = { type: 'Standard' };
      expect(tc.type).toBe('Standard');
    });

    it('should accept Locked type', () => {
      const tc: TemporalCommitment = {
        type: 'Locked',
        lockUntil: Date.now() + 30 * 24 * 60 * 60 * 1000,
        earlyExitPenalty: 0.25,
      };
      expect(tc.type).toBe('Locked');
    });

    it('should accept Generational type', () => {
      const tc: TemporalCommitment = {
        type: 'Generational',
        generation: 1,
        cohortId: 'cohort-2025',
        expectedGenerations: 5,
      };
      expect(tc.type).toBe('Generational');
    });
  });

  describe('WisdomSeed', () => {
    it('should construct with correct fields', () => {
      const seed: WisdomSeed = {
        ifCorrect: {
          lesson: 'Exponential growth in renewables is real',
          confidence: 0.8,
          applicability: {
            domains: ['energy'],
            timeHorizons: ['5-year'],
            conditions: ['stable policy environment'],
            exceptions: ['major geopolitical disruption'],
          },
          caveats: ['Regional variation'],
        },
        ifIncorrect: {
          lesson: 'Policy barriers are more significant than expected',
          confidence: 0.6,
          applicability: {
            domains: ['energy', 'policy'],
            timeHorizons: ['10-year'],
            conditions: [],
            exceptions: [],
          },
          caveats: [],
        },
        metaLesson: 'Technology adoption is hard to predict',
        letterToFuture: 'May your energy be clean.',
        activationConditions: [{ type: 'AfterResolution' }],
      };
      expect(seed.ifCorrect.lesson).toContain('Exponential growth');
      expect(seed.ifIncorrect.lesson).toContain('Policy barriers');
      expect(seed.metaLesson).toBeDefined();
    });
  });

  describe('ActivationCondition', () => {
    it('should accept all variants', () => {
      const conditions: ActivationCondition[] = [
        { type: 'AfterResolution' },
        { type: 'AtTime', timestamp: Date.now() },
        { type: 'SimilarPrediction', similarityThreshold: 0.8 },
        { type: 'DomainActivity', domain: 'energy', threshold: 10 },
        { type: 'OnEvent', eventType: 'market_created' },
      ];
      expect(conditions).toHaveLength(5);
    });
  });

  describe('Commitment', () => {
    it('should accept all commitment types', () => {
      const commitments: Commitment[] = [
        { type: 'BeliefUpdate', topic: 'solar capacity' },
        { type: 'Investigation', hours: 10, topic: 'battery tech' },
        { type: 'Donation', amount: 50, recipient: 'solar-fund' },
        { type: 'Mentorship', hours: 5, domain: 'energy' },
        { type: 'Custom', description: 'Write blog post' },
      ];
      expect(commitments).toHaveLength(5);
    });
  });

  describe('MarketMechanism', () => {
    it('should accept Binary type', () => {
      const m: MarketMechanism = { type: 'Binary', yesShares: 1000, noShares: 1000 };
      expect(m.type).toBe('Binary');
    });

    it('should accept LMSR type', () => {
      const m: MarketMechanism = {
        type: 'LMSR',
        liquidityParameter: 100,
        subsidyPool: 1000,
        outcomeQuantities: [500, 500],
      };
      expect(m.type).toBe('LMSR');
    });

    it('should accept CDA type', () => {
      const m: MarketMechanism = { type: 'CDA', bids: [], asks: [] };
      expect(m.type).toBe('CDA');
    });

    it('should accept Parimutuel type', () => {
      const m: MarketMechanism = {
        type: 'Parimutuel',
        pools: [{ outcomeId: 'yes', amount: 100 }, { outcomeId: 'no', amount: 200 }],
      };
      expect(m.type).toBe('Parimutuel');
    });
  });
});

// ============================================================================
// Singleton
// ============================================================================

describe('EpistemicMarketsService Singleton', () => {
  it('should return same instance on repeated calls', () => {
    const a = getEpistemicMarketsService();
    const b = getEpistemicMarketsService();
    expect(a).toBe(b);
  });

  it('should return an instance of EpistemicMarketsService', () => {
    const service = getEpistemicMarketsService();
    expect(service).toBeInstanceOf(EpistemicMarketsService);
  });
});

// ============================================================================
// EpistemicMarketsService
// ============================================================================

describe('EpistemicMarketsService', () => {
  let service: EpistemicMarketsService;

  beforeEach(() => {
    service = new EpistemicMarketsService();
  });

  // ==========================================================================
  // Market Operations
  // ==========================================================================

  describe('Market Operations', () => {
    describe('createMarket', () => {
      it('should create a market with correct fields', () => {
        const market = service.createMarket(makeMarketInput());

        expect(market.id).toBeTruthy();
        expect(market.question).toContain('solar capacity');
        expect(market.description).toContain('IEA');
        expect(market.creator).toBe('local-agent');
        expect(market.status).toBe(MarketStatus.Open);
        expect(market.outcomes).toHaveLength(2);
        expect(market.outcomes[0].currentProbability).toBeCloseTo(0.5);
        expect(market.outcomes[1].currentProbability).toBeCloseTo(0.5);
        expect(market.tags).toContain('energy');
        expect(market.totalStakeValue).toBe(0);
        expect(market.predictorCount).toBe(0);
        expect(market.createdAt).toBeGreaterThan(0);
      });

      it('should set resolution mechanism based on epistemic position', () => {
        const market = service.createMarket(makeMarketInput({
          epistemicPosition: makeEpistemicPosition({ empirical: EmpiricalLevel.Measurable }),
        }));
        expect(market.resolution).toBe(ResolutionMechanism.Automated);
      });

      it('should distribute initial probabilities equally', () => {
        const market = service.createMarket(makeMarketInput({
          outcomes: ['Low', 'Medium', 'High'],
        }));
        expect(market.outcomes).toHaveLength(3);
        for (const outcome of market.outcomes) {
          expect(outcome.currentProbability).toBeCloseTo(1 / 3);
        }
      });

      it('should throw when close time is in the past', () => {
        expect(() => {
          service.createMarket(makeMarketInput({ closesAt: Date.now() - 1000 }));
        }).toThrow('Market close time must be in the future');
      });

      it('should throw when fewer than 2 outcomes', () => {
        expect(() => {
          service.createMarket(makeMarketInput({ outcomes: ['Only one'] }));
        }).toThrow('Market must have at least 2 outcomes');
      });

      it('should generate unique IDs', () => {
        const m1 = service.createMarket(makeMarketInput({ question: 'Q1?' }));
        const m2 = service.createMarket(makeMarketInput({ question: 'Q2?' }));
        expect(m1.id).not.toBe(m2.id);
      });

      it('should accept optional sourceHapp and resolutionSource', () => {
        const market = service.createMarket(makeMarketInput({
          sourceHapp: 'governance',
          resolutionSource: 'governance:proposal:42',
        }));
        expect(market.sourceHapp).toBe('governance');
        expect(market.resolutionSource).toBe('governance:proposal:42');
      });
    });

    describe('getMarket', () => {
      it('should return market by ID', () => {
        const created = service.createMarket(makeMarketInput());
        const fetched = service.getMarket(created.id);
        expect(fetched).toBeDefined();
        expect(fetched!.question).toBe(created.question);
      });

      it('should return undefined for non-existent market', () => {
        expect(service.getMarket('non-existent')).toBeUndefined();
      });
    });

    describe('listOpenMarkets', () => {
      it('should return only open markets sorted by creation time descending', () => {
        service.createMarket(makeMarketInput({ question: 'Q1?' }));
        const m2 = service.createMarket(makeMarketInput({ question: 'Q2?' }));
        service.closeMarket(m2.id);
        service.createMarket(makeMarketInput({ question: 'Q3?' }));

        const open = service.listOpenMarkets();
        expect(open).toHaveLength(2);
        expect(open.every((m) => m.status === MarketStatus.Open)).toBe(true);
        expect(open[0].createdAt).toBeGreaterThanOrEqual(open[1].createdAt);
      });

      it('should return empty array when no markets exist', () => {
        expect(service.listOpenMarkets()).toEqual([]);
      });
    });

    describe('searchMarketsByTag', () => {
      it('should find markets by tag (case-insensitive partial match)', () => {
        service.createMarket(makeMarketInput({ tags: ['solar', 'energy'] }));
        service.createMarket(makeMarketInput({ tags: ['AI', 'technology'] }));

        const results = service.searchMarketsByTag('SOLAR');
        expect(results).toHaveLength(1);
      });

      it('should return empty array for unmatched tag', () => {
        service.createMarket(makeMarketInput({ tags: ['climate'] }));
        expect(service.searchMarketsByTag('blockchain')).toEqual([]);
      });
    });

    describe('closeMarket', () => {
      it('should close an open market', () => {
        const market = service.createMarket(makeMarketInput());
        const closed = service.closeMarket(market.id);
        expect(closed.status).toBe(MarketStatus.Closed);
      });

      it('should throw for non-existent market', () => {
        expect(() => service.closeMarket('ghost')).toThrow('Market not found');
      });

      it('should throw if market is already closed', () => {
        const market = service.createMarket(makeMarketInput());
        service.closeMarket(market.id);
        expect(() => service.closeMarket(market.id)).toThrow('Market is not open');
      });
    });
  });

  // ==========================================================================
  // Prediction Operations
  // ==========================================================================

  describe('Prediction Operations', () => {
    let marketId: string;

    beforeEach(() => {
      const market = service.createMarket(makeMarketInput());
      marketId = market.id;
    });

    describe('submitPrediction', () => {
      it('should submit a prediction with correct fields', () => {
        const prediction = service.submitPrediction(makePredictionInput(marketId));

        expect(prediction.id).toBeTruthy();
        expect(prediction.marketId).toBe(marketId);
        expect(prediction.predictor).toBe('local-agent');
        expect(prediction.outcome).toBe('outcome_0');
        expect(prediction.confidence).toBe(0.7);
        expect(prediction.reasoning.summary).toContain('current trends');
        expect(prediction.dependsOn).toEqual([]);
        expect(prediction.temporalCommitment).toEqual({ type: 'Standard' });
        expect(prediction.updates).toEqual([]);
        expect(prediction.createdAt).toBeGreaterThan(0);
      });

      it('should update market stats', () => {
        const market = service.getMarket(marketId)!;
        const initialCount = market.predictorCount;
        const initialStake = market.totalStakeValue;

        service.submitPrediction(makePredictionInput(marketId));

        expect(market.predictorCount).toBe(initialCount + 1);
        expect(market.totalStakeValue).toBeGreaterThan(initialStake);
      });

      it('should throw for non-existent market', () => {
        expect(() => {
          service.submitPrediction(makePredictionInput('ghost'));
        }).toThrow('Market not found');
      });

      it('should throw for closed market', () => {
        service.closeMarket(marketId);
        expect(() => {
          service.submitPrediction(makePredictionInput(marketId));
        }).toThrow('Market is not open');
      });

      it('should throw for reasoning summary shorter than 50 chars', () => {
        expect(() => {
          service.submitPrediction(makePredictionInput(marketId, {
            reasoning: makeMinimalReasoning({ summary: 'Too short' }),
          }));
        }).toThrow('Reasoning summary must be at least 50 characters');
      });

      it('should throw for empty reasoning steps', () => {
        expect(() => {
          service.submitPrediction(makePredictionInput(marketId, {
            reasoning: makeMinimalReasoning({ steps: [] }),
          }));
        }).toThrow('At least one reasoning step required');
      });

      it('should throw for empty assumptions', () => {
        expect(() => {
          service.submitPrediction(makePredictionInput(marketId, {
            reasoning: makeMinimalReasoning({ assumptions: [] }),
          }));
        }).toThrow('At least one assumption must be made explicit');
      });

      it('should accept optional wisdom seed', () => {
        const wisdomSeed: WisdomSeed = {
          ifCorrect: {
            lesson: 'Solar is unstoppable',
            confidence: 0.8,
            applicability: { domains: ['energy'], timeHorizons: ['5y'], conditions: [], exceptions: [] },
            caveats: [],
          },
          ifIncorrect: {
            lesson: 'Never underestimate policy friction',
            confidence: 0.6,
            applicability: { domains: ['energy', 'policy'], timeHorizons: ['10y'], conditions: [], exceptions: [] },
            caveats: [],
          },
          activationConditions: [{ type: 'AfterResolution' }],
        };

        const prediction = service.submitPrediction(makePredictionInput(marketId, { wisdomSeed }));
        expect(prediction.wisdomSeed).toBeDefined();
        expect(prediction.wisdomSeed!.ifCorrect.lesson).toContain('unstoppable');
      });

      it('should generate unique IDs for predictions', () => {
        const p1 = service.submitPrediction(makePredictionInput(marketId, { confidence: 0.6 }));
        const p2 = service.submitPrediction(makePredictionInput(marketId, { confidence: 0.8 }));
        expect(p1.id).not.toBe(p2.id);
      });
    });

    describe('getPrediction', () => {
      it('should return prediction by ID', () => {
        const created = service.submitPrediction(makePredictionInput(marketId));
        const fetched = service.getPrediction(created.id);
        expect(fetched).toBeDefined();
        expect(fetched!.confidence).toBe(0.7);
      });

      it('should return undefined for non-existent prediction', () => {
        expect(service.getPrediction('ghost')).toBeUndefined();
      });
    });

    describe('getPredictionsForMarket', () => {
      it('should return all predictions for a market sorted by creation time descending', () => {
        service.submitPrediction(makePredictionInput(marketId, { confidence: 0.5 }));
        service.submitPrediction(makePredictionInput(marketId, { confidence: 0.8 }));

        const predictions = service.getPredictionsForMarket(marketId);
        expect(predictions).toHaveLength(2);
        expect(predictions[0].createdAt).toBeGreaterThanOrEqual(predictions[1].createdAt);
      });

      it('should return empty array for market with no predictions', () => {
        expect(service.getPredictionsForMarket(marketId)).toEqual([]);
      });

      it('should scope predictions to the specified market', () => {
        service.submitPrediction(makePredictionInput(marketId));
        const market2 = service.createMarket(makeMarketInput({ question: 'Other?' }));
        service.submitPrediction(makePredictionInput(market2.id));

        expect(service.getPredictionsForMarket(marketId)).toHaveLength(1);
        expect(service.getPredictionsForMarket(market2.id)).toHaveLength(1);
      });
    });

    describe('updatePrediction', () => {
      it('should update confidence and add update record', () => {
        const prediction = service.submitPrediction(makePredictionInput(marketId));

        const updated = service.updatePrediction(
          prediction.id,
          0.85,
          { type: 'NewEvidence', description: 'Q3 deployment data released', source: 'IEA' },
          'Increased confidence based on Q3 data'
        );

        expect(updated.confidence).toBe(0.85);
        expect(updated.updates).toHaveLength(1);
        expect(updated.updates[0].oldConfidence).toBe(0.7);
        expect(updated.updates[0].newConfidence).toBe(0.85);
        expect(updated.updates[0].reason.type).toBe('NewEvidence');
      });

      it('should throw for non-existent prediction', () => {
        expect(() => {
          service.updatePrediction('ghost', 0.5, { type: 'TimePassage' }, 'delta');
        }).toThrow('Prediction not found');
      });

      it('should support multiple updates', () => {
        const prediction = service.submitPrediction(makePredictionInput(marketId));

        service.updatePrediction(prediction.id, 0.8, { type: 'TimePassage' }, 'Time update');
        service.updatePrediction(prediction.id, 0.9, { type: 'TimePassage' }, 'Time update 2');

        const fetched = service.getPrediction(prediction.id)!;
        expect(fetched.updates).toHaveLength(2);
        expect(fetched.confidence).toBe(0.9);
      });
    });
  });

  // ==========================================================================
  // Reputation Operations
  // ==========================================================================

  describe('Reputation Operations', () => {
    it('should return default reputation for unknown predictor', () => {
      const rep = service.getPredictorReputation('unknown');
      expect(rep).toBeDefined();
    });

    it('should report unknown predictor as not trustworthy', () => {
      expect(service.isPredictorTrustworthy('unknown')).toBe(false);
    });

    it('should update reputation after submitting prediction', () => {
      const market = service.createMarket(makeMarketInput());
      service.submitPrediction(makePredictionInput(market.id));

      const rep = service.getPredictorReputation('local-agent');
      expect(rep).toBeDefined();
    });
  });

  // ==========================================================================
  // Wisdom Operations
  // ==========================================================================

  describe('Wisdom Operations', () => {
    it('should return wisdom seeds for a domain', () => {
      const market = service.createMarket(makeMarketInput());
      const wisdomSeed: WisdomSeed = {
        ifCorrect: {
          lesson: 'Solar growth is real',
          confidence: 0.8,
          applicability: { domains: ['energy'], timeHorizons: [], conditions: [], exceptions: [] },
          caveats: [],
        },
        ifIncorrect: {
          lesson: 'Be more cautious',
          confidence: 0.6,
          applicability: { domains: ['energy'], timeHorizons: [], conditions: [], exceptions: [] },
          caveats: [],
        },
        activationConditions: [],
      };

      service.submitPrediction(makePredictionInput(market.id, { wisdomSeed }));

      const seeds = service.getWisdomSeedsForDomain('energy');
      expect(seeds).toHaveLength(1);
      expect(seeds[0].wisdomSeed.ifCorrect.lesson).toContain('Solar growth');
      expect(seeds[0].resolved).toBe(false);
    });

    it('should return empty array for domain with no wisdom seeds', () => {
      expect(service.getWisdomSeedsForDomain('quantum-computing')).toEqual([]);
    });
  });

  // ==========================================================================
  // Belief Dependency Graph
  // ==========================================================================

  describe('Belief Dependency Graph', () => {
    it('should find predictions that depend on another prediction', () => {
      const market = service.createMarket(makeMarketInput());
      const p1 = service.submitPrediction(makePredictionInput(market.id));
      service.submitPrediction(makePredictionInput(market.id, {
        dependsOn: [{
          dependsOn: { type: 'Prediction', id: p1.id },
          dependencyType: 'PositiveEvidence',
          strength: 0.8,
          conditional: { ifTrue: 0.9, ifFalse: 0.3, dependencyProbability: 0.7 },
        }],
      }));

      const dependents = service.getBeliefDependents(p1.id);
      expect(dependents).toHaveLength(1);
    });

    it('should return empty array when no dependents exist', () => {
      const market = service.createMarket(makeMarketInput());
      const p1 = service.submitPrediction(makePredictionInput(market.id));
      expect(service.getBeliefDependents(p1.id)).toEqual([]);
    });
  });

  // ==========================================================================
  // Cross-hApp Operations
  // ==========================================================================

  describe('Cross-hApp Operations', () => {
    describe('queryExternalReputation', () => {
      it('should query without throwing', () => {
        expect(() => service.queryExternalReputation('agent-123')).not.toThrow();
      });
    });

    describe('createMarketFromGovernance', () => {
      it('should create a market linked to a governance proposal', () => {
        const market = service.createMarketFromGovernance(
          'prop-42',
          'Will proposal 42 pass?',
          Date.now() + 7 * 24 * 60 * 60 * 1000,
        );

        expect(market.question).toBe('Will proposal 42 pass?');
        expect(market.outcomes).toHaveLength(2);
        expect(market.outcomes[0].description).toBe('Proposal Passes');
        expect(market.outcomes[1].description).toBe('Proposal Fails');
        expect(market.sourceHapp).toBe('governance');
        expect(market.resolutionSource).toBe('governance:proposal:prop-42');
        expect(market.tags).toContain('governance');
        expect(market.tags).toContain('proposal');
        expect(market.epistemicPosition.empirical).toBe(EmpiricalLevel.Cryptographic);
      });
    });

    describe('getFLCoordinator', () => {
      it('should return the FL coordinator instance', () => {
        expect(service.getFLCoordinator()).toBeDefined();
      });
    });

    describe('getBridge', () => {
      it('should return the bridge instance', () => {
        expect(service.getBridge()).toBeDefined();
      });
    });
  });

  // ==========================================================================
  // Full Lifecycle
  // ==========================================================================

  describe('Full Lifecycle', () => {
    it('should support create market -> submit prediction -> update -> query workflow', () => {
      // Step 1: Create market
      const market = service.createMarket(makeMarketInput({
        question: 'Will battery prices fall below $50/kWh by 2030?',
        outcomes: ['Yes', 'No'],
        tags: ['battery', 'energy'],
      }));
      expect(market.status).toBe(MarketStatus.Open);
      expect(market.predictorCount).toBe(0);

      // Step 2: Submit prediction with wisdom seed
      const prediction = service.submitPrediction({
        marketId: market.id,
        outcome: 'outcome_0',
        confidence: 0.65,
        stake: createReputationOnlyStake(['energy', 'battery'], 0.03, 1.1),
        reasoning: makeMinimalReasoning(),
        wisdomSeed: {
          ifCorrect: {
            lesson: 'Wright law applies to batteries',
            confidence: 0.8,
            applicability: { domains: ['energy'], timeHorizons: ['5y'], conditions: [], exceptions: [] },
            caveats: [],
          },
          ifIncorrect: {
            lesson: 'Material constraints slow cost reduction',
            confidence: 0.6,
            applicability: { domains: ['energy'], timeHorizons: ['10y'], conditions: [], exceptions: [] },
            caveats: [],
          },
          activationConditions: [{ type: 'AfterResolution' }],
        },
      });
      expect(prediction.confidence).toBe(0.65);
      expect(market.predictorCount).toBe(1);

      // Step 3: Update prediction
      const updated = service.updatePrediction(
        prediction.id,
        0.75,
        { type: 'NewEvidence', description: 'CATL announced $45/kWh cells', source: 'Bloomberg' },
        'Increased based on CATL announcement'
      );
      expect(updated.confidence).toBe(0.75);
      expect(updated.updates).toHaveLength(1);

      // Step 4: Query wisdom seeds
      const seeds = service.getWisdomSeedsForDomain('energy');
      expect(seeds).toHaveLength(1);
      expect(seeds[0].resolved).toBe(false);

      // Step 5: Verify market stats
      const fetchedMarket = service.getMarket(market.id)!;
      expect(fetchedMarket.predictorCount).toBe(1);
      expect(fetchedMarket.totalStakeValue).toBeGreaterThan(0);

      // Step 6: Close market
      service.closeMarket(market.id);
      expect(fetchedMarket.status).toBe(MarketStatus.Closed);
    });
  });

  // ==========================================================================
  // Error handling
  // ==========================================================================

  describe('Error handling', () => {
    it('should throw when submitting to non-existent market', () => {
      expect(() => {
        service.submitPrediction(makePredictionInput('ghost'));
      }).toThrow('Market not found');
    });

    it('should throw when closing non-existent market', () => {
      expect(() => service.closeMarket('ghost')).toThrow('Market not found');
    });

    it('should throw when updating non-existent prediction', () => {
      expect(() => {
        service.updatePrediction('ghost', 0.5, { type: 'TimePassage' }, 'x');
      }).toThrow('Prediction not found');
    });

    it('should throw when submitting to closed market', () => {
      const market = service.createMarket(makeMarketInput());
      service.closeMarket(market.id);
      expect(() => {
        service.submitPrediction(makePredictionInput(market.id));
      }).toThrow('Market is not open');
    });

    it('should throw when market close time is in the past', () => {
      expect(() => {
        service.createMarket(makeMarketInput({ closesAt: Date.now() - 10000 }));
      }).toThrow('Market close time must be in the future');
    });

    it('should throw when market has fewer than 2 outcomes', () => {
      expect(() => {
        service.createMarket(makeMarketInput({ outcomes: ['Only'] }));
      }).toThrow('Market must have at least 2 outcomes');
    });
  });
});
