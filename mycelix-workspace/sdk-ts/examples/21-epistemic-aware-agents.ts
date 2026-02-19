/**
 * Epistemic-Aware AI Agents Example
 *
 * This example demonstrates the complete Epistemic-Aware AI Agency framework:
 * - Agent lifecycle with K-Vector trust profiles
 * - Moral uncertainty (GIS) integration
 * - FL participation and feedback
 * - KREDIT derivation from trust scores
 * - Coherence (Phi) gating
 *
 * Scenario: An AI trading assistant that learns from market data via FL,
 * with trust profiles evolving based on behavioral outcomes.
 *
 * Run with: npx ts-node examples/21-epistemic-aware-agents.ts
 */

import {
  // Agentic Framework
  AgentClass,
  AgentStatus,
  KVectorValues,
  getClassLimits,
  createDefaultConstraints,
  createDefaultCalibration,
  computeTrustScore,
  calculateKreditFromTrust,
  gateActionOnUncertainty,

  // GIS (Moral Uncertainty)
  createMoralUncertainty,
  totalMoralUncertainty,
  getMoralActionGuidance,
  getMoralRecommendations,
  MoralActionGuidance,

  // Epistemic Classification
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  claim,
  meetsStandard,
  Standards,
} from '../src/index.js';

// =============================================================================
// Type Definitions
// =============================================================================

interface SimulatedAgent {
  id: string;
  sponsorDid: string;
  agentClass: AgentClass;
  status: AgentStatus;
  kVector: KVectorValues;
  kreditBalance: number;
  kreditCap: number;
  totalActions: number;
  successfulActions: number;
  pendingEscalations: string[];
  calibration: ReturnType<typeof createDefaultCalibration>;
  createdAt: Date;
  lastActivity: Date;
}

interface AgentAction {
  type: 'market_analysis' | 'trade_recommendation' | 'portfolio_rebalance' | 'risk_alert';
  description: string;
  kreditCost: number;
  uncertainty: { epistemic: number; axiological: number; deontic: number };
  outcome?: 'success' | 'failure' | 'escalated';
}

interface FLRoundResult {
  roundId: number;
  gradientQuality: number; // 0-1, how good were the agent's gradients
  wasHonest: boolean;
  contributionScore: number;
}

// =============================================================================
// Helper Functions
// =============================================================================

function createInitialKVector(): KVectorValues {
  return {
    k_r: 0.5,     // Reputation: neutral starting point
    k_a: 0.3,     // Activity: low (new agent)
    k_i: 1.0,     // Integrity: perfect (no violations yet)
    k_p: 0.5,     // Performance: neutral
    k_m: 0.1,     // Membership: very low (just joined)
    k_s: 0.5,     // Stake: neutral
    k_h: 0.5,     // Historical: neutral
    k_topo: 0.2,  // Topology: few connections
    k_v: 0.8,     // Verification: sponsor verified
    k_coherence: 0.7,   // Coherence: acceptable
  };
}

function updateKVectorFromAction(
  kVector: KVectorValues,
  action: AgentAction,
  outcome: 'success' | 'failure' | 'escalated'
): KVectorValues {
  const updated = { ...kVector };

  // Activity always increases
  updated.k_a = Math.min(1.0, kVector.k_a + 0.02);

  switch (outcome) {
    case 'success':
      // Boost reputation and performance
      updated.k_r = Math.min(1.0, kVector.k_r + 0.05);
      updated.k_p = Math.min(1.0, kVector.k_p + 0.03);
      updated.k_h = Math.min(1.0, kVector.k_h + 0.02);
      break;

    case 'failure':
      // Penalize reputation and performance
      updated.k_r = Math.max(0.0, kVector.k_r - 0.08);
      updated.k_p = Math.max(0.0, kVector.k_p - 0.05);
      updated.k_h = Math.max(0.0, kVector.k_h - 0.03);
      break;

    case 'escalated':
      // Neutral - agent showed appropriate uncertainty
      // Small boost for being appropriately cautious
      updated.k_i = Math.min(1.0, kVector.k_i + 0.01);
      break;
  }

  // Membership increases with time (simulated)
  updated.k_m = Math.min(1.0, kVector.k_m + 0.01);

  return updated;
}

function updateKVectorFromFL(
  kVector: KVectorValues,
  flResult: FLRoundResult
): KVectorValues {
  const updated = { ...kVector };

  // FL participation affects multiple dimensions
  updated.k_a = Math.min(1.0, kVector.k_a + 0.03);

  if (flResult.wasHonest) {
    // Honest participation boosts integrity
    updated.k_i = Math.min(1.0, kVector.k_i + 0.02);

    // Good gradient quality boosts performance
    const qualityBoost = flResult.gradientQuality * 0.05;
    updated.k_p = Math.min(1.0, kVector.k_p + qualityBoost);

    // Contribution affects reputation
    const reputationBoost = flResult.contributionScore * 0.04;
    updated.k_r = Math.min(1.0, kVector.k_r + reputationBoost);
  } else {
    // Byzantine behavior severely penalizes integrity
    updated.k_i = Math.max(0.0, kVector.k_i - 0.15);
    updated.k_r = Math.max(0.0, kVector.k_r - 0.10);
  }

  // Coherence based on gradient consistency
  updated.k_coherence = flResult.gradientQuality * 0.3 + kVector.k_coherence * 0.7;

  return updated;
}

function formatKVector(kVector: KVectorValues): string {
  const dims = [
    `r=${kVector.k_r.toFixed(2)}`,
    `a=${kVector.k_a.toFixed(2)}`,
    `i=${kVector.k_i.toFixed(2)}`,
    `p=${kVector.k_p.toFixed(2)}`,
    `m=${kVector.k_m.toFixed(2)}`,
    `s=${kVector.k_s.toFixed(2)}`,
    `h=${kVector.k_h.toFixed(2)}`,
    `t=${kVector.k_topo.toFixed(2)}`,
    `v=${(kVector.k_v ?? 0).toFixed(2)}`,
    `phi=${(kVector.k_coherence ?? 0).toFixed(2)}`,
  ];
  return `[${dims.join(', ')}]`;
}

function printAgentSummary(agent: SimulatedAgent): void {
  const trustScore = computeTrustScore(agent.kVector);
  const successRate = agent.totalActions > 0
    ? (agent.successfulActions / agent.totalActions * 100).toFixed(1)
    : 'N/A';

  console.log(`  Agent ID: ${agent.id}`);
  console.log(`  Sponsor: ${agent.sponsorDid}`);
  console.log(`  Class: ${agent.agentClass}`);
  console.log(`  Status: ${agent.status}`);
  console.log(`  Trust Score: ${trustScore.toFixed(3)}`);
  console.log(`  KREDIT: ${agent.kreditBalance} / ${agent.kreditCap}`);
  console.log(`  Actions: ${agent.totalActions} (${successRate}% success)`);
  console.log(`  Escalations: ${agent.pendingEscalations.length} pending`);
  console.log(`  K-Vector: ${formatKVector(agent.kVector)}`);
}

// =============================================================================
// Main Demo
// =============================================================================

async function main() {
  console.log('=== Epistemic-Aware AI Agents Demo ===');
  console.log('Scenario: AI Trading Assistant with Trust Evolution\n');

  // -------------------------------------------------------------------------
  // Step 1: Create Agent with Initial K-Vector
  // -------------------------------------------------------------------------
  console.log('--- Step 1: Agent Creation ---');

  const agent: SimulatedAgent = {
    id: 'trading-bot-alpha',
    sponsorDid: 'did:mycelix:sponsor-hedge-fund-123',
    agentClass: AgentClass.Supervised,
    status: AgentStatus.Active,
    kVector: createInitialKVector(),
    kreditBalance: 5000,
    kreditCap: 10000,
    totalActions: 0,
    successfulActions: 0,
    pendingEscalations: [],
    calibration: createDefaultCalibration('trading-bot-alpha'),
    createdAt: new Date(),
    lastActivity: new Date(),
  };

  // Derive KREDIT cap from trust
  const initialTrust = computeTrustScore(agent.kVector);
  agent.kreditCap = calculateKreditFromTrust(initialTrust, 5000, 10);

  console.log('Created new agent with initial K-Vector:');
  printAgentSummary(agent);
  console.log();

  // Show class limits
  const classLimits = getClassLimits(agent.agentClass);
  console.log('Class limits (Supervised):');
  console.log(`  Max KREDIT/epoch: ${classLimits.max_kredit_per_epoch}`);
  console.log(`  Max TX/hour: ${classLimits.max_tx_per_hour}`);
  console.log(`  Approval required above: ${classLimits.requires_approval_above} SAP`);
  console.log();

  // -------------------------------------------------------------------------
  // Step 2: Agent Actions with Uncertainty Gating
  // -------------------------------------------------------------------------
  console.log('--- Step 2: Actions with Uncertainty Gating ---');

  const actions: AgentAction[] = [
    {
      type: 'market_analysis',
      description: 'Analyze tech sector trends',
      kreditCost: 10,
      uncertainty: { epistemic: 0.2, axiological: 0.1, deontic: 0.1 },
    },
    {
      type: 'trade_recommendation',
      description: 'Recommend buying AAPL',
      kreditCost: 50,
      uncertainty: { epistemic: 0.3, axiological: 0.2, deontic: 0.15 },
    },
    {
      type: 'portfolio_rebalance',
      description: 'Shift 20% from bonds to equities',
      kreditCost: 200,
      uncertainty: { epistemic: 0.5, axiological: 0.6, deontic: 0.4 },
    },
    {
      type: 'risk_alert',
      description: 'Emergency liquidation recommendation',
      kreditCost: 500,
      uncertainty: { epistemic: 0.8, axiological: 0.7, deontic: 0.9 },
    },
  ];

  for (const action of actions) {
    console.log(`\nAction: ${action.description}`);
    console.log(`  Type: ${action.type}, KREDIT cost: ${action.kreditCost}`);

    // Create moral uncertainty
    const uncertainty = createMoralUncertainty(
      action.uncertainty.epistemic,
      action.uncertainty.axiological,
      action.uncertainty.deontic
    );

    const total = totalMoralUncertainty(uncertainty);
    const guidance = getMoralActionGuidance(uncertainty);
    console.log(`  Uncertainty: total=${total.toFixed(2)}, guidance=${guidance}`);

    // Gate action based on uncertainty
    const gateResult = gateActionOnUncertainty(
      agent.id,
      uncertainty,
      action.type
    );

    console.log(`  Gate result: ${gateResult.type}`);

    // Simulate outcome based on gate result
    let outcome: 'success' | 'failure' | 'escalated';
    switch (gateResult.type) {
      case 'proceed':
        outcome = Math.random() > 0.1 ? 'success' : 'failure';
        agent.kreditBalance -= action.kreditCost;
        agent.totalActions++;
        if (outcome === 'success') agent.successfulActions++;
        console.log(`  Outcome: ${outcome}`);
        break;

      case 'proceed_with_monitoring':
        outcome = Math.random() > 0.2 ? 'success' : 'failure';
        agent.kreditBalance -= action.kreditCost;
        agent.totalActions++;
        if (outcome === 'success') agent.successfulActions++;
        console.log(`  Outcome: ${outcome} (monitored)`);
        break;

      case 'escalation_required':
        outcome = 'escalated';
        agent.pendingEscalations.push(action.type);
        console.log(`  Escalated to sponsor for approval`);
        if ('escalation' in gateResult) {
          const recs = getMoralRecommendations(uncertainty);
          console.log(`  Recommendations: ${recs.join('; ')}`);
        }
        break;

      case 'blocked':
        outcome = 'escalated';
        console.log(`  Action blocked: ${'reason' in gateResult ? gateResult.reason : 'uncertainty too high'}`);
        break;
    }

    // Update K-Vector based on outcome
    agent.kVector = updateKVectorFromAction(agent.kVector, action, outcome);
    agent.lastActivity = new Date();

    // Update KREDIT cap based on new trust
    const newTrust = computeTrustScore(agent.kVector);
    agent.kreditCap = calculateKreditFromTrust(newTrust, 5000, 10);
  }

  console.log('\nAgent state after actions:');
  printAgentSummary(agent);
  console.log();

  // -------------------------------------------------------------------------
  // Step 3: FL Participation and Feedback
  // -------------------------------------------------------------------------
  console.log('--- Step 3: Federated Learning Participation ---');

  // Simulate 5 FL rounds
  for (let round = 1; round <= 5; round++) {
    // Simulate gradient quality based on agent's current performance dimension
    const baseQuality = agent.kVector.k_p * 0.7 + Math.random() * 0.3;
    const gradientQuality = Math.min(1.0, Math.max(0.0, baseQuality));

    // Agent behaves honestly (integrity dimension influences this)
    const isHonest = agent.kVector.k_i > 0.3;

    const flResult: FLRoundResult = {
      roundId: round,
      gradientQuality,
      wasHonest: isHonest,
      contributionScore: gradientQuality * (isHonest ? 1.0 : 0.2),
    };

    // Update K-Vector from FL feedback
    agent.kVector = updateKVectorFromFL(agent.kVector, flResult);

    const trustAfter = computeTrustScore(agent.kVector);
    agent.kreditCap = calculateKreditFromTrust(trustAfter, 5000, 10);

    console.log(
      `  Round ${round}: quality=${gradientQuality.toFixed(2)}, ` +
      `honest=${isHonest}, trust=${trustAfter.toFixed(3)}, ` +
      `cap=${agent.kreditCap}`
    );
  }

  console.log('\nAgent state after FL rounds:');
  printAgentSummary(agent);
  console.log();

  // -------------------------------------------------------------------------
  // Step 4: Epistemic Classification of Outputs
  // -------------------------------------------------------------------------
  console.log('--- Step 4: Epistemic Classification of Outputs ---');

  // Agent produces a market analysis report
  const analysisReport = claim(
    'Market analysis: Tech sector expected to outperform in Q2 2026'
  )
    .withClassification(
      EmpiricalLevel.E2_PrivateVerify, // Agent's internal analysis
      NormativeLevel.N0_Personal,       // Personal recommendation
      MaterialityLevel.M1_Transient     // Short-term relevance
    )
    .withIssuer(agent.id)
    .withEvidence({
      type: 'model-output',
      description: 'Generated by trading-bot-alpha neural network',
    })
    .withExpiration(new Date(Date.now() + 90 * 24 * 60 * 60 * 1000)) // 90 days
    .build();

  console.log('Agent output classification:');
  console.log(`  Claim: ${analysisReport.claim}`);
  console.log(`  E-level: ${analysisReport.classification.empirical} (E2: Private verification)`);
  console.log(`  N-level: ${analysisReport.classification.normative} (N0: Personal scope)`);
  console.log(`  M-level: ${analysisReport.classification.materiality} (M1: Transient)`);
  console.log(`  Meets BASIC standard: ${meetsStandard(analysisReport, Standards.BASIC)}`);
  console.log(`  Meets NETWORK standard: ${meetsStandard(analysisReport, Standards.NETWORK)}`);
  console.log();

  // -------------------------------------------------------------------------
  // Step 5: Resolve Escalations (Sponsor Decision)
  // -------------------------------------------------------------------------
  console.log('--- Step 5: Escalation Resolution ---');

  if (agent.pendingEscalations.length > 0) {
    console.log(`Sponsor reviewing ${agent.pendingEscalations.length} pending escalations:`);

    for (const escalation of [...agent.pendingEscalations]) {
      // Simulate sponsor decision (approve 50% of the time)
      const approved = Math.random() > 0.5;
      console.log(`  ${escalation}: ${approved ? 'APPROVED' : 'REJECTED'}`);

      // Remove from pending
      agent.pendingEscalations = agent.pendingEscalations.filter(e => e !== escalation);

      // Update calibration
      if (approved) {
        // Agent was appropriately cautious
        agent.calibration.appropriate_uncertainty++;
        agent.kVector.k_i = Math.min(1.0, agent.kVector.k_i + 0.02);
      } else {
        // Agent was overcautious (could have proceeded)
        agent.calibration.overcautious++;
      }
      agent.calibration.total_events++;
    }

    // Recalculate calibration score
    agent.calibration.calibration_score =
      (agent.calibration.appropriate_uncertainty + agent.calibration.appropriate_confidence) /
      Math.max(1, agent.calibration.total_events);

    console.log(`\nCalibration updated:`);
    console.log(`  Score: ${agent.calibration.calibration_score.toFixed(2)}`);
    console.log(`  Appropriate uncertainty: ${agent.calibration.appropriate_uncertainty}`);
    console.log(`  Overcautious: ${agent.calibration.overcautious}`);
  } else {
    console.log('No pending escalations.');
  }
  console.log();

  // -------------------------------------------------------------------------
  // Step 6: Final Summary - Epistemic Fingerprint
  // -------------------------------------------------------------------------
  console.log('--- Step 6: Final Epistemic Fingerprint ---');

  const finalTrust = computeTrustScore(agent.kVector);
  const constraints = createDefaultConstraints();

  console.log('Agent Epistemic Fingerprint:');
  console.log('='.repeat(50));
  printAgentSummary(agent);
  console.log();

  console.log('Trust Breakdown:');
  console.log(`  Reputation (k_r):  ${agent.kVector.k_r.toFixed(3)} - Peer feedback & outcomes`);
  console.log(`  Activity (k_a):    ${agent.kVector.k_a.toFixed(3)} - Engagement level`);
  console.log(`  Integrity (k_i):   ${agent.kVector.k_i.toFixed(3)} - Constraint compliance`);
  console.log(`  Performance (k_p): ${agent.kVector.k_p.toFixed(3)} - Output quality`);
  console.log(`  Membership (k_m):  ${agent.kVector.k_m.toFixed(3)} - Time in network`);
  console.log(`  Stake (k_s):       ${agent.kVector.k_s.toFixed(3)} - KREDIT efficiency`);
  console.log(`  Historical (k_h):  ${agent.kVector.k_h.toFixed(3)} - Behavioral consistency`);
  console.log(`  Topology (k_topo): ${agent.kVector.k_topo.toFixed(3)} - Network connections`);
  console.log(`  Verification (k_v): ${(agent.kVector.k_v ?? 0).toFixed(3)} - Identity verification`);
  console.log(`  Coherence (k_coherence): ${(agent.kVector.k_coherence ?? 0).toFixed(3)} - Phi measurement`);
  console.log();

  console.log('Constitutional Constraints:');
  console.log(`  Can vote governance: ${constraints.can_vote_governance}`);
  console.log(`  Can become validator: ${constraints.can_become_validator}`);
  console.log(`  Can hold CIV: ${constraints.can_hold_civ}`);
  console.log(`  Can receive CGC: ${constraints.can_receive_cgc}`);
  console.log(`  Can send CGC: ${constraints.can_send_cgc}`);
  console.log();

  console.log('Key Insights:');
  console.log(`  - Trust evolved from ${initialTrust.toFixed(3)} to ${finalTrust.toFixed(3)}`);
  console.log(`  - KREDIT cap adjusted dynamically based on trust`);
  console.log(`  - High-uncertainty actions properly escalated to sponsor`);
  console.log(`  - FL participation improved coherence (k_coherence) dimension`);
  console.log(`  - Agent maintains full epistemic accountability`);
  console.log();

  console.log('Demo complete!');
}

main().catch(console.error);
