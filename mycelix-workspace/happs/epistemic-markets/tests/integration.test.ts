/**
 * Epistemic Markets Integration Tests
 *
 * These tests validate not just functionality, but embody the philosophical
 * principles of the system. Each test is a teaching moment.
 *
 * Test Categories:
 * 1. Core Mechanics - Basic market operations
 * 2. Epistemic Charter - E-N-M classification effects
 * 3. Multi-Dimensional Stakes - Beyond monetary value
 * 4. MATL Integration - Trust-weighted resolution
 * 5. Wisdom Flow - Intergenerational knowledge
 * 6. Collective Intelligence - Emergence properties
 * 7. Philosophical Principles - System values in action
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
  Market,
  Prediction,
  EpistemicPosition,
  MultiDimensionalStake,
  ReasoningTrace,
  WisdomSeed,
} from '../sdk-ts/src/index';

// ============================================================================
// Test Setup
// ============================================================================

let conductor: Conductor;
let players: Player[];
let clients: EpistemicMarketsClient[];

beforeEach(async () => {
  conductor = await createConductor();

  // Create a diverse community of predictors
  players = await Promise.all([
    addPlayerWithConfig(conductor, 'alice'),   // Domain expert
    addPlayerWithConfig(conductor, 'bob'),     // Generalist
    addPlayerWithConfig(conductor, 'carol'),   // Newcomer
    addPlayerWithConfig(conductor, 'dave'),    // Contrarian
    addPlayerWithConfig(conductor, 'eve'),     // Wisdom keeper
  ]);

  clients = players.map(p => new EpistemicMarketsClient(p.cells[0]));
});

afterEach(async () => {
  await conductor.shutdown();
});

// ============================================================================
// 1. Core Mechanics Tests
// ============================================================================

describe('Core Market Mechanics', () => {
  test('markets can be created with full epistemic context', async () => {
    const market = await clients[0].markets.createMarket({
      title: 'Will the community garden project succeed?',
      description: 'Success defined as: 50+ active participants by year end',
      outcomes: ['Yes', 'No'],
      epistemic_position: {
        empirical: 'testimonial',
        normative: 'communal',
        materiality: 'persistent',
      },
      resolution_criteria: {
        type: 'CommunityConsensus',
        threshold: 0.75,
      },
      closes_at: Date.now() + 86400000 * 30, // 30 days
    });

    expect(market).toBeDefined();
    expect(market.epistemic_position.empirical).toBe('testimonial');

    // Teaching moment: Markets aren't just about money, they're about
    // understanding reality together. The epistemic position tells us
    // HOW to evaluate truth.
  });

  test('predictions require reasoning traces for transparency', async () => {
    const market = await createTestMarket(clients[0]);

    // Prediction without reasoning should be rejected or flagged
    const prediction = await clients[1].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.7,
      stake: { monetary: { amount: 100, currency: 'HAM' } },
      reasoning: {
        summary: 'Based on historical success of similar projects',
        steps: [
          {
            claim: 'Similar projects in our area have 60% success rate',
            evidence: 'Local community records',
            confidence: 0.8,
          },
          {
            claim: 'This project has strong initial organizer commitment',
            evidence: 'Observable engagement levels',
            confidence: 0.85,
          },
        ],
        assumptions: [
          {
            statement: 'Economic conditions remain stable',
            sensitivity: 0.3, // How much outcome changes if wrong
          },
        ],
        update_triggers: [
          {
            condition: 'Organizer leaves project',
            new_probability: 0.3,
          },
        ],
        acknowledged_weaknesses: [
          'Limited data on urban garden projects specifically',
        ],
      },
    });

    expect(prediction.reasoning.steps.length).toBeGreaterThan(0);
    expect(prediction.reasoning.acknowledged_weaknesses.length).toBeGreaterThan(0);

    // Teaching moment: Transparent reasoning is an act of intellectual
    // generosity. It invites others to critique, learn, and improve.
  });

  test('predictions can be updated with new information', async () => {
    const market = await createTestMarket(clients[0]);
    const prediction = await createTestPrediction(clients[1], market.id, 0.7);

    // New information arrives
    const updatedPrediction = await clients[1].predictions.updatePrediction({
      prediction_id: prediction.id,
      new_probability: 0.5,
      update_reason: {
        trigger: 'Key organizer announced reduced involvement',
        information_source: 'Public announcement',
        belief_shift: -0.2,
      },
    });

    expect(updatedPrediction.probability).toBe(0.5);
    expect(updatedPrediction.update_history.length).toBe(1);

    // Teaching moment: Changing your mind in response to evidence
    // is a virtue, not a weakness. The system rewards intellectual
    // honesty over stubborn consistency.
  });
});

// ============================================================================
// 2. Epistemic Charter Tests
// ============================================================================

describe('Epistemic Charter Integration', () => {
  test('empirical level determines resolution mechanism', async () => {
    // Create market with MEASURABLE epistemic position
    const measurableMarket = await clients[0].markets.createMarket({
      title: 'Will temperature exceed 30C tomorrow?',
      description: 'As measured by official weather station',
      epistemic_position: {
        empirical: 'measurable',
        normative: 'universal',
        materiality: 'ephemeral',
      },
      resolution_criteria: { type: 'OnChainOracle', oracle_id: 'weather_oracle' },
    });

    // Create market with TESTIMONIAL epistemic position
    const testimonialMarket = await clients[0].markets.createMarket({
      title: 'Was the community event well-received?',
      description: 'Based on participant feedback',
      epistemic_position: {
        empirical: 'testimonial',
        normative: 'communal',
        materiality: 'persistent',
      },
      resolution_criteria: { type: 'CommunityConsensus', min_participants: 10 },
    });

    // Measurable should require fewer oracles but higher consensus
    expect(measurableMarket.resolution_config.min_oracles).toBeLessThan(
      testimonialMarket.resolution_config.min_oracles
    );
    expect(measurableMarket.resolution_config.consensus_threshold).toBeGreaterThan(
      testimonialMarket.resolution_config.consensus_threshold
    );

    // Teaching moment: Not all truth is the same. A thermometer reading
    // is different from community sentiment. Our systems should recognize
    // and respect these differences.
  });

  test('normative level affects whose voice matters in resolution', async () => {
    // Personal normative level - only the individual decides
    const personalMarket = await clients[0].markets.createMarket({
      title: "Will I feel satisfied with my career choice?",
      epistemic_position: {
        empirical: 'subjective',
        normative: 'personal',
        materiality: 'temporal',
      },
      resolution_criteria: { type: 'SelfReport', reporter: clients[0].myPubKey },
    });

    // Universal normative level - broader verification required
    const universalMarket = await clients[0].markets.createMarket({
      title: 'Is this mathematical theorem correct?',
      epistemic_position: {
        empirical: 'cryptographic',
        normative: 'universal',
        materiality: 'foundational',
      },
      resolution_criteria: { type: 'FormalVerification', verifier: 'proof_checker' },
    });

    // Personal resolution requires only self-report
    expect(personalMarket.resolution_config.allowed_resolvers).toContain(
      clients[0].myPubKey
    );

    // Universal requires formal verification
    expect(universalMarket.resolution_config.requires_formal_verification).toBe(true);

    // Teaching moment: Some truths are personal, some are communal,
    // some are universal. The epistemic charter helps us know which
    // is which and respond appropriately.
  });

  test('materiality level determines how long predictions matter', async () => {
    // Ephemeral - matters for a moment
    const ephemeralMarket = await clients[0].markets.createMarket({
      title: 'Will it rain in the next hour?',
      epistemic_position: {
        empirical: 'measurable',
        normative: 'universal',
        materiality: 'ephemeral',
      },
    });

    // Foundational - matters forever
    const foundationalMarket = await clients[0].markets.createMarket({
      title: 'Is this cryptographic primitive secure?',
      epistemic_position: {
        empirical: 'cryptographic',
        normative: 'universal',
        materiality: 'foundational',
      },
    });

    // Ephemeral markets have short wisdom seed dormancy
    expect(ephemeralMarket.wisdom_config.seed_dormancy_days).toBe(1);

    // Foundational markets have long wisdom dormancy
    expect(foundationalMarket.wisdom_config.seed_dormancy_days).toBeGreaterThan(365);

    // Teaching moment: Some lessons are immediate, some take
    // generations to fully understand. Materiality helps us
    // give truth the time it needs.
  });
});

// ============================================================================
// 3. Multi-Dimensional Stakes Tests
// ============================================================================

describe('Multi-Dimensional Stakes', () => {
  test('reputation stakes affect future market access', async () => {
    const market = await createTestMarket(clients[0]);

    // Bob stakes reputation
    const prediction = await clients[1].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.8,
      stake: {
        reputation: { amount: 50, domain: 'community_projects' },
      },
    });

    // Resolve market as NO - Bob was wrong
    await resolveMarket(market.id, 'No');

    // Bob's reputation in this domain should decrease
    const bobProfile = await clients[1].scoring.getCalibrationProfile();
    expect(bobProfile.domain_scores['community_projects'].reputation).toBeLessThan(50);

    // Teaching moment: Reputation stakes create accountability.
    // When we stake our reputation, we're saying "this matters to me,
    // and you can trust me less if I'm wrong."
  });

  test('social stakes create witnessing relationships', async () => {
    const market = await createTestMarket(clients[0]);

    // Carol stakes with social witness
    const prediction = await clients[2].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.75,
      stake: {
        social: {
          witnesses: [clients[3].myPubKey], // Dave witnesses
          commitment: 'I believe this project will succeed',
          visibility: 'public',
        },
      },
    });

    // Dave should see the witness request
    const witnessRequests = await clients[3].getWitnessRequests();
    expect(witnessRequests).toContainEqual(
      expect.objectContaining({ predictor: clients[2].myPubKey })
    );

    // Teaching moment: Social stakes embed predictions in relationships.
    // When someone witnesses your prediction, both of you become part
    // of a truth-seeking community.
  });

  test('commitment stakes require fulfilled promises', async () => {
    const market = await createTestMarket(clients[0]);

    // Eve stakes with commitment
    const prediction = await clients[4].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.9,
      stake: {
        commitment: {
          promise: 'I will volunteer 10 hours if this succeeds',
          verifiable_by: 'community_timebank',
          duration_days: 30,
        },
      },
    });

    // Market resolves YES - Eve must fulfill commitment
    await resolveMarket(market.id, 'Yes');

    // Eve's commitment should now be active
    const activeCommitments = await clients[4].getActiveCommitments();
    expect(activeCommitments).toContainEqual(
      expect.objectContaining({
        promise: 'I will volunteer 10 hours if this succeeds',
        status: 'active',
      })
    );

    // Teaching moment: Commitment stakes turn predictions into promises.
    // They bind our beliefs to our actions, creating integrity.
  });

  test('time stakes reward early accurate predictions', async () => {
    const market = await createTestMarket(clients[0]);
    const marketOpenTime = Date.now();

    // Alice predicts early
    const earlyPrediction = await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.8,
      stake: { time: { locked_until: marketOpenTime + 86400000 } },
    });

    // Wait and then Bob predicts late
    await delay(1000);
    const latePrediction = await clients[1].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.8,
      stake: { time: { locked_until: Date.now() + 86400000 } },
    });

    // Resolve as YES - both correct
    await resolveMarket(market.id, 'Yes');

    // Alice should get temporal bonus
    const aliceReward = await clients[0].getRewardForPrediction(earlyPrediction.id);
    const bobReward = await clients[1].getRewardForPrediction(latePrediction.id);

    expect(aliceReward.temporal_bonus).toBeGreaterThan(bobReward.temporal_bonus);

    // Teaching moment: Time stakes reward courage. It's harder to predict
    // early when uncertainty is high. Those who commit early with accuracy
    // deserve recognition.
  });
});

// ============================================================================
// 4. MATL Integration Tests
// ============================================================================

describe('MATL-Weighted Resolution', () => {
  test('oracle votes are weighted by MATL trust scores', async () => {
    const market = await createTestMarket(clients[0]);

    // Set up different MATL scores
    await setMatlScore(clients[0].myPubKey, { quality: 0.9, consistency: 0.85, reputation: 0.9 });
    await setMatlScore(clients[1].myPubKey, { quality: 0.5, consistency: 0.5, reputation: 0.5 });

    // Both vote on resolution
    await clients[0].resolution.castOracleVote({
      market_id: market.id,
      outcome: 'Yes',
      evidence: 'Direct observation of 60 participants',
    });

    await clients[1].resolution.castOracleVote({
      market_id: market.id,
      outcome: 'No',
      evidence: 'Counted fewer than 50',
    });

    // Alice's high-trust vote should dominate
    const resolution = await getResolutionState(market.id);
    expect(resolution.weighted_outcome).toBe('Yes');

    // Teaching moment: Not all voices are equal, but all can grow.
    // MATL weights aren't permanent - they reflect demonstrated
    // trustworthiness over time. Anyone can build trust.
  });

  test('Byzantine behavior is detected and handled', async () => {
    const market = await createTestMarket(clients[0]);

    // Set up multiple oracles with varying trust
    for (let i = 0; i < 5; i++) {
      await setMatlScore(players[i].pubKey, { quality: 0.8, consistency: 0.8, reputation: 0.8 });
    }

    // 4 oracles vote YES honestly
    for (let i = 0; i < 4; i++) {
      await clients[i].resolution.castOracleVote({
        market_id: market.id,
        outcome: 'Yes',
        evidence: 'Observed success',
      });
    }

    // 1 oracle votes NO maliciously (or mistakenly)
    await clients[4].resolution.castOracleVote({
      market_id: market.id,
      outcome: 'No',
      evidence: 'Claims project failed',
    });

    // Byzantine analysis should detect the outlier
    const analysis = await getByzantineAnalysis(market.id);
    expect(analysis.suspicious_oracles.length).toBe(1);
    expect(analysis.network_health).toBeGreaterThan(0.7);
    expect(analysis.recommendation).toBe('Proceed');

    // Teaching moment: Byzantine fault tolerance isn't just technical -
    // it's about building systems that can find truth even when some
    // participants are wrong or deceptive. The 45% threshold means
    // we can tolerate significant disagreement while still converging.
  });

  test('low trust situations trigger escalation', async () => {
    const market = await createTestMarket(clients[0]);

    // All oracles have low or conflicting trust
    for (let i = 0; i < 5; i++) {
      await setMatlScore(players[i].pubKey, { quality: 0.3, consistency: 0.3, reputation: 0.3 });
    }

    // Votes are split
    await clients[0].resolution.castOracleVote({ market_id: market.id, outcome: 'Yes' });
    await clients[1].resolution.castOracleVote({ market_id: market.id, outcome: 'Yes' });
    await clients[2].resolution.castOracleVote({ market_id: market.id, outcome: 'No' });
    await clients[3].resolution.castOracleVote({ market_id: market.id, outcome: 'No' });
    await clients[4].resolution.castOracleVote({ market_id: market.id, outcome: 'Yes' });

    // Should trigger escalation
    const resolution = await getResolutionState(market.id);
    expect(resolution.status).toBe('escalated');
    expect(resolution.escalation_reason).toContain('low_trust');

    // Teaching moment: Sometimes we don't know the truth yet, and
    // that's okay. The system recognizes when it needs more time,
    // more information, or different processes to find truth.
  });
});

// ============================================================================
// 5. Wisdom Flow Tests
// ============================================================================

describe('Intergenerational Wisdom', () => {
  test('wisdom seeds are planted during prediction', async () => {
    const market = await createTestMarket(clients[0]);

    const prediction = await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.75,
      reasoning: { summary: 'Strong community engagement observed' },
      wisdom_seed: {
        if_correct: {
          lesson: 'Community projects succeed when there is visible early enthusiasm',
          implication: 'Invest in building early momentum',
          domain_relevance: ['community_organizing', 'project_management'],
        },
        if_incorrect: {
          lesson: 'Visible enthusiasm may not translate to sustained commitment',
          implication: 'Look for commitment signals beyond enthusiasm',
          domain_relevance: ['community_organizing', 'psychology'],
        },
        meta_lesson: 'The relationship between enthusiasm and success is complex',
        letter_to_future: 'Dear future predictor: Pay attention to both energy and structure.',
      },
    });

    expect(prediction.wisdom_seed).toBeDefined();
    expect(prediction.wisdom_seed.letter_to_future).toContain('future predictor');

    // Teaching moment: Every prediction is an opportunity to teach.
    // Wisdom seeds plant knowledge for future generations, whether
    // we're right or wrong.
  });

  test('wisdom seeds activate after resolution dormancy', async () => {
    const market = await createTestMarket(clients[0], { dormancy_days: 1 });
    await createPredictionWithWisdomSeed(clients[0], market.id);

    // Resolve market
    await resolveMarket(market.id, 'Yes');

    // Wisdom should not be active immediately
    let activatedWisdom = await getActivatedWisdom(market.id);
    expect(activatedWisdom).toHaveLength(0);

    // Fast-forward past dormancy period (in tests, we can manipulate time)
    await advanceTime(86400000 * 2); // 2 days

    // Now wisdom should be active
    activatedWisdom = await getActivatedWisdom(market.id);
    expect(activatedWisdom.length).toBeGreaterThan(0);
    expect(activatedWisdom[0].lesson).toContain('succeed');

    // Teaching moment: Wisdom needs time to mature. We don't know
    // what a prediction teaches until we can see its consequences
    // unfold. The dormancy period prevents premature conclusions.
  });

  test('epistemic lineage tracks intellectual heritage', async () => {
    // First generation prediction
    const market1 = await createTestMarket(clients[0]);
    const gen1Prediction = await clients[0].predictions.createPrediction({
      market_id: market1.id,
      outcome: 'Yes',
      probability: 0.7,
      reasoning: { summary: 'Initial theory' },
    });

    // Resolve first market
    await resolveMarket(market1.id, 'Yes');

    // Second generation builds on first
    const market2 = await createTestMarket(clients[1]);
    const gen2Prediction = await clients[1].predictions.createPrediction({
      market_id: market2.id,
      outcome: 'Yes',
      probability: 0.8,
      reasoning: { summary: 'Building on previous insight' },
      epistemic_lineage: {
        parent_predictions: [gen1Prediction.id],
        relationship: 'builds_upon',
        attribution: 'This insight comes from prediction by Alice',
      },
    });

    // Check lineage
    const lineage = await getEpistemicLineage(gen2Prediction.id);
    expect(lineage.parent_predictions).toContain(gen1Prediction.id);
    expect(lineage.depth).toBe(1);

    // Original predictor should get lineage credit if gen2 succeeds
    await resolveMarket(market2.id, 'Yes');
    const aliceLineageCredit = await clients[0].getLineageCredit();
    expect(aliceLineageCredit).toBeGreaterThan(0);

    // Teaching moment: Ideas have ancestry. When we acknowledge our
    // intellectual heritage, we create a web of credit that rewards
    // foundational insights across time.
  });

  test('legacy stakes create intergenerational commitment', async () => {
    const market = await createTestMarket(clients[4], { closes_in_years: 10 });

    // Eve creates legacy stake
    const prediction = await clients[4].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.6,
      stake: {
        legacy: {
          amount: 1000,
          beneficiary: 'future_community_fund',
          matures_in_years: 10,
          letter: 'I believe our community will thrive. Use this to keep building.',
        },
      },
      wisdom_seed: {
        if_correct: { lesson: 'Long-term thinking pays off' },
        if_incorrect: { lesson: 'Even well-intentioned optimism can be misplaced' },
        activation_conditions: [{ type: 'time_elapsed', years: 10 }],
      },
    });

    expect(prediction.stake.legacy.matures_in_years).toBe(10);
    expect(prediction.wisdom_seed.activation_conditions[0].years).toBe(10);

    // Teaching moment: Legacy stakes are predictions about the world
    // our grandchildren will inherit. They bind us to the future
    // and give future generations both resources and wisdom.
  });
});

// ============================================================================
// 6. Collective Intelligence Tests
// ============================================================================

describe('Emergent Collective Intelligence', () => {
  test('cascade detection identifies herding behavior', async () => {
    const market = await createTestMarket(clients[0]);

    // Alice predicts first
    await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.7,
    });

    // Others follow quickly with similar predictions
    for (const client of clients.slice(1)) {
      await delay(100);
      await client.predictions.createPrediction({
        market_id: market.id,
        outcome: 'Yes',
        probability: 0.72 + Math.random() * 0.05, // Very similar
      });
    }

    // System should detect cascade
    const cascadeAnalysis = await clients[0].scoring.analyzeCascade(market.id);
    expect(cascadeAnalysis.is_cascade).toBe(true);
    expect(cascadeAnalysis.independence_score).toBeLessThan(0.5);

    // Predictions during cascade should be flagged
    for (const prediction of await getPredictions(market.id)) {
      if (prediction.predictor !== clients[0].myPubKey) {
        expect(prediction.cascade_flag).toBe(true);
      }
    }

    // Teaching moment: Collective wisdom requires independence.
    // When everyone copies everyone else, we lose the diversity
    // that makes crowds wise. The system detects and discounts
    // cascade-affected predictions.
  });

  test('disagreement mining finds productive conflict', async () => {
    const market = await createTestMarket(clients[0]);

    // Create genuinely divergent predictions
    await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.85,
      reasoning: {
        summary: 'Strong fundamentals',
        steps: [{ claim: 'High community engagement', confidence: 0.9 }],
      },
    });

    await clients[3].predictions.createPrediction({
      market_id: market.id,
      outcome: 'No',
      probability: 0.8,
      reasoning: {
        summary: 'Structural challenges',
        steps: [{ claim: 'Insufficient resources', confidence: 0.85 }],
      },
    });

    // System should identify the disagreement
    const disagreement = await clients[0].scoring.mineDisagreement(market.id);
    expect(disagreement.exists).toBe(true);
    expect(disagreement.magnitude).toBeGreaterThan(0.5);

    // System should find cruxes
    expect(disagreement.cruxes.length).toBeGreaterThan(0);
    expect(disagreement.cruxes[0].operationalization).toBeDefined();

    // Teaching moment: Disagreement isn't a bug, it's a feature.
    // When smart people disagree, there's information in that
    // disagreement. Crux identification helps us find what
    // would actually change minds.
  });

  test('productive disagreement is rewarded', async () => {
    const market = await createTestMarket(clients[0]);

    // Alice and Dave disagree productively
    const alicePrediction = await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.8,
      reasoning: { summary: 'Strong fundamentals' },
    });

    const davePrediction = await clients[3].predictions.createPrediction({
      market_id: market.id,
      outcome: 'No',
      probability: 0.75,
      reasoning: { summary: 'Hidden risks' },
    });

    // They engage in structured disagreement
    await clients[0].scoring.recordDisagreement({
      prediction_a: alicePrediction.id,
      prediction_b: davePrediction.id,
      cruxes: [
        {
          description: 'Resource availability',
          operationalization: 'Funding secured by month 3',
          if_true_shift: { alice: 0, dave: 0.3 },
          if_false_shift: { alice: -0.3, dave: 0 },
        },
      ],
      outcome: 'ongoing',
    });

    // Both should get disagreement mining rewards
    const aliceRewards = await clients[0].getRewards();
    const daveRewards = await clients[3].getRewards();

    expect(aliceRewards.disagreement_bonus).toBeGreaterThan(0);
    expect(daveRewards.disagreement_bonus).toBeGreaterThan(0);

    // Teaching moment: Productive disagreement is valuable.
    // When disagreers articulate their cruxes, everyone learns.
    // The system rewards this intellectual labor.
  });
});

// ============================================================================
// 7. Philosophical Principles Tests
// ============================================================================

describe('Philosophical Principles in Action', () => {
  test('uncertainty is celebrated, not hidden', async () => {
    const market = await createTestMarket(clients[0]);

    // Prediction with honest uncertainty
    const honestPrediction = await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.5, // Maximum uncertainty
      reasoning: {
        summary: 'Genuinely uncertain',
        confidence_breakdown: {
          model_uncertainty: 0.4,
          data_uncertainty: 0.3,
          meta_uncertainty: 0.3,
        },
        acknowledged_weaknesses: [
          'Limited data',
          'Novel situation',
          'Conflicting expert opinions',
        ],
      },
    });

    // Should not be penalized for uncertainty
    expect(honestPrediction.uncertainty_penalty).toBe(0);

    // Should get uncertainty celebration bonus
    expect(honestPrediction.uncertainty_bonus).toBeGreaterThan(0);

    // Teaching moment: "I don't know" is valuable information.
    // When calibrated predictors express uncertainty, it tells us
    // we're in uncertain territory. This prevents false confidence.
  });

  test('sacred pauses allow reflection before high-stakes predictions', async () => {
    const market = await createTestMarket(clients[0]);

    // High-stakes prediction
    const highStakesPrediction = await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.95, // Very confident
      stake: { monetary: { amount: 5000, currency: 'HAM' } }, // High stake
    });

    // System should have prompted for sacred pause
    expect(highStakesPrediction.sacred_pause_completed).toBe(true);
    expect(highStakesPrediction.reflection_prompts_shown).toContain(
      'Are you expressing genuine confidence, or avoiding the discomfort of uncertainty?'
    );

    // Teaching moment: Before major commitments, pause.
    // The sacred pause isn't friction - it's an invitation to
    // check whether we're acting from wisdom or reaction.
  });

  test('truth-seeking is valued over winning', async () => {
    const market = await createTestMarket(clients[0]);

    // Alice makes accurate prediction but doesn't update when she should
    await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.7,
    });

    // New information arrives that should shift probabilities
    await simulateNewInformation(market.id, { direction: 'negative', magnitude: 0.3 });

    // Bob updates appropriately
    await clients[1].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.4, // Adjusted for new info
    });

    // Market resolves - Alice happens to be right
    await resolveMarket(market.id, 'Yes');

    // Alice gets outcome reward but not process reward
    const aliceScores = await clients[0].scoring.getCalibrationProfile();
    const bobScores = await clients[1].scoring.getCalibrationProfile();

    // Bob should have higher process score for updating
    expect(bobScores.update_responsiveness).toBeGreaterThan(aliceScores.update_responsiveness);

    // Teaching moment: Being right by accident is less valuable than
    // being right for the right reasons. The system values good
    // epistemic process, not just correct outcomes.
  });

  test('healing is possible after epistemic trauma', async () => {
    // Carol makes a bad prediction that harms others
    const market1 = await createTestMarket(clients[0]);
    await clients[2].predictions.createPrediction({
      market_id: market1.id,
      outcome: 'Yes',
      probability: 0.95, // Overconfident
      stake: { social: { witnesses: [clients[1].myPubKey], visibility: 'public' } },
    });

    // It fails badly
    await resolveMarket(market1.id, 'No');

    // Carol's trust is damaged
    let carolProfile = await clients[2].scoring.getCalibrationProfile();
    expect(carolProfile.reliability_score).toBeLessThan(0.5);

    // Carol engages in healing process
    await clients[2].scoring.initiateHealing({
      acknowledgment: 'I was overconfident and it affected others who trusted me',
      learning: 'I need to be more careful about expressing certainty',
      commitment: 'I will express more uncertainty in novel domains',
    });

    // Carol makes humble, well-calibrated predictions
    for (let i = 0; i < 5; i++) {
      const market = await createTestMarket(clients[0]);
      await clients[2].predictions.createPrediction({
        market_id: market.id,
        outcome: 'Yes',
        probability: 0.6, // Humble
        reasoning: { acknowledged_weaknesses: ['Previous overconfidence in similar situations'] },
      });
      await resolveMarket(market.id, Math.random() > 0.4 ? 'Yes' : 'No');
    }

    // Carol's trust gradually recovers
    carolProfile = await clients[2].scoring.getCalibrationProfile();
    expect(carolProfile.reliability_score).toBeGreaterThan(0.5);
    expect(carolProfile.healing_journey?.status).toBe('progressing');

    // Teaching moment: We all make mistakes. The question is whether
    // we can learn from them and rebuild trust. The system provides
    // pathways for healing, not just punishment.
  });

  test('love and truth are inseparable', async () => {
    // This test validates that truth-seeking enhances relationships
    const market = await createTestMarket(clients[0]);

    // Alice and Bob are in social relationship
    await establishSocialBond(clients[0].myPubKey, clients[1].myPubKey);

    // They make predictions that turn out differently
    await clients[0].predictions.createPrediction({
      market_id: market.id,
      outcome: 'Yes',
      probability: 0.8,
      stake: { social: { witnesses: [clients[1].myPubKey] } },
    });

    await clients[1].predictions.createPrediction({
      market_id: market.id,
      outcome: 'No',
      probability: 0.7,
      stake: { social: { witnesses: [clients[0].myPubKey] } },
    });

    // They engage in respectful disagreement
    await clients[0].scoring.recordDisagreement({
      prediction_a: await getLatestPrediction(clients[0].myPubKey, market.id),
      prediction_b: await getLatestPrediction(clients[1].myPubKey, market.id),
      tone: 'respectful',
      learning_shared: true,
    });

    // Market resolves
    await resolveMarket(market.id, 'Yes');

    // Their relationship should be strengthened, not damaged
    const bondStrength = await getSocialBondStrength(clients[0].myPubKey, clients[1].myPubKey);
    expect(bondStrength.after).toBeGreaterThanOrEqual(bondStrength.before);

    // The loser should feel respected, not humiliated
    const bobExperience = await clients[1].getResolutionExperience(market.id);
    expect(bobExperience.felt_respected).toBe(true);
    expect(bobExperience.learning_occurred).toBe(true);

    // Teaching moment: Truth should bring us together, not apart.
    // When truth-seeking is done with love, even being wrong becomes
    // an opportunity for growth and connection.
  });
});

// ============================================================================
// Helper Functions
// ============================================================================

async function createTestMarket(
  client: EpistemicMarketsClient,
  options: Partial<{
    dormancy_days: number;
    closes_in_years: number;
  }> = {}
): Promise<Market> {
  return client.markets.createMarket({
    title: `Test Market ${Date.now()}`,
    description: 'A market for testing',
    outcomes: ['Yes', 'No'],
    epistemic_position: {
      empirical: 'testimonial',
      normative: 'communal',
      materiality: 'persistent',
    },
    closes_at: options.closes_in_years
      ? Date.now() + 86400000 * 365 * options.closes_in_years
      : Date.now() + 86400000 * 30,
    wisdom_config: options.dormancy_days
      ? { seed_dormancy_days: options.dormancy_days }
      : undefined,
  });
}

async function createTestPrediction(
  client: EpistemicMarketsClient,
  marketId: string,
  probability: number
): Promise<Prediction> {
  return client.predictions.createPrediction({
    market_id: marketId,
    outcome: 'Yes',
    probability,
    stake: { monetary: { amount: 100, currency: 'HAM' } },
    reasoning: { summary: 'Test prediction' },
  });
}

async function createPredictionWithWisdomSeed(
  client: EpistemicMarketsClient,
  marketId: string
): Promise<Prediction> {
  return client.predictions.createPrediction({
    market_id: marketId,
    outcome: 'Yes',
    probability: 0.7,
    wisdom_seed: {
      if_correct: { lesson: 'Projects can succeed with community support' },
      if_incorrect: { lesson: 'Good intentions are not enough' },
    },
  });
}

async function resolveMarket(marketId: string, outcome: string): Promise<void> {
  // Implementation would interact with resolution zome
}

async function setMatlScore(
  agentPubKey: string,
  scores: { quality: number; consistency: number; reputation: number }
): Promise<void> {
  // Implementation would set MATL scores for testing
}

async function getResolutionState(marketId: string): Promise<any> {
  // Implementation would get current resolution state
}

async function getByzantineAnalysis(marketId: string): Promise<any> {
  // Implementation would get Byzantine analysis
}

async function getActivatedWisdom(marketId: string): Promise<any[]> {
  // Implementation would get activated wisdom seeds
}

async function advanceTime(ms: number): Promise<void> {
  // Implementation would advance test time
}

async function getEpistemicLineage(predictionId: string): Promise<any> {
  // Implementation would get epistemic lineage
}

async function getPredictions(marketId: string): Promise<Prediction[]> {
  // Implementation would get all predictions for market
}

async function simulateNewInformation(marketId: string, info: any): Promise<void> {
  // Implementation would simulate new information arriving
}

async function establishSocialBond(agent1: string, agent2: string): Promise<void> {
  // Implementation would create social bond
}

async function getLatestPrediction(agentPubKey: string, marketId: string): Promise<string> {
  // Implementation would get latest prediction ID
}

async function getSocialBondStrength(agent1: string, agent2: string): Promise<any> {
  // Implementation would get social bond strength
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
