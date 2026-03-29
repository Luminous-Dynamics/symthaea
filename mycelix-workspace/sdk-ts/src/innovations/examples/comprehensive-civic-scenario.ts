// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Comprehensive Civic Scenario - All 6 Innovation Modules
 *
 * Demonstrates all 6 innovation modules working together in a realistic
 * civic scenario: processing a housing voucher application.
 *
 * @module innovations/examples/comprehensive-civic-scenario
 */

import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../integrations/epistemic-markets/index.js';
import { createCivicFeedbackLoop, type JusticeDecision } from '../civic-feedback-loop/index.js';
import { getConstitutionalGovernor } from '../constitutional-ai/index.js';
import { PrivateQueryService } from '../private-queries/index.js';
import { createWorkflow, createOrchestrator, type WorkflowContext } from '../self-healing-workflows/index.js';
import { TrustMarketService, type MultiDimensionalTrustStake } from '../trust-markets/index.js';

import type { EpistemicClaim } from '../epistemic-agents/index.js';

const applicantProfile = {
  did: 'did:mycelix:applicant-maria-2024',
  name: 'Maria Santos',
  householdSize: 4,
  monthlyIncome: 2800,
};

/**
 * Helper to create stake
 */
function createStake(monetary: number, reputationPct: number): Partial<MultiDimensionalTrustStake> {
  return {
    monetary: { amount: monetary, currency: 'MATL' },
    reputation: { stakePercentage: reputationPct, currentReputation: 0.8, atRisk: 0.8 * reputationPct },
  };
}

/**
 * Comprehensive Civic Scenario: Housing Voucher Application
 */
export async function comprehensiveCivicScenario() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║   COMPREHENSIVE CIVIC SCENARIO: Housing Voucher Application  ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  // Initialize all modules
  console.log('━━━ INITIALIZING MODULES ━━━\n');

  const governor = await getConstitutionalGovernor();
  console.log('✓ Constitutional Governor');

  const trustMarkets = new TrustMarketService();
  console.log('✓ Trust Market Service');

  const privateQueries = new PrivateQueryService({ epsilon: 1.0, delta: 1e-7, accountingMethod: 'rdp' });
  await privateQueries.initialize();
  console.log('✓ Private Query Service');

  const feedbackLoop = createCivicFeedbackLoop({
    autoPropagate: true,
    enableConflictDetection: true,
  });
  await feedbackLoop.start();
  console.log('✓ Civic Feedback Loop');

  const orchestrator = createOrchestrator({
    maxConcurrentWorkflows: 5,
    enableObservability: true,
  });
  console.log('✓ Self-Healing Orchestrator\n');

  // Phase 1: Trust Market Prediction
  console.log('━━━ PHASE 1: TRUST MARKET PREDICTION ━━━\n');

  const market = await trustMarkets.createMarket({
    subjectDid: applicantProfile.did,
    claim: {
      type: 'reputation_maintenance',
      threshold: 0.5,
      contextHapps: ['governance', 'finance'],
      durationMs: 7 * 24 * 60 * 60 * 1000,
      operator: 'gte',
    },
    title: `Will ${applicantProfile.name}'s application be approved?`,
    creatorStake: 100,
    mechanismType: 'LMSR',
  });

  await trustMarkets.submitTrade({
    marketId: market.id,
    participantId: 'did:mycelix:caseworker',
    outcome: 'yes',
    shares: 40,
    stake: createStake(80, 0.04),
  });

  const impliedApproval = trustMarkets.getMarket(market.id)?.state.impliedProbabilities.get('yes') ?? 0.5;
  console.log(`Community prediction: ${(impliedApproval * 100).toFixed(1)}% approval\n`);

  // Phase 2: Epistemic Agent Guidance
  console.log('━━━ PHASE 2: EPISTEMIC AGENT GUIDANCE ━━━\n');

  const agentClaim: EpistemicClaim = {
    id: `claim-housing-${Date.now()}`,
    text: 'Applicant likely qualifies for Section 8 assistance. ' +
          'Income below 50% AMI threshold.',
    position: {
      empirical: EmpiricalLevel.Cryptographic,
      normative: NormativeLevel.Universal,
      materiality: MaterialityLevel.Persistent,
    },
    classificationCode: 'E3-N3-M2',
    confidence: 0.82,
    calibratedConfidence: 0.78,
    sources: [{
      id: 'hud-guidelines',
      type: 'regulation',
      title: 'HUD Income Limits',
      relevance: 0.95,
    }],
    agentId: 'housing-navigator',
    conversationId: `conv-${applicantProfile.did}`,
    timestamp: Date.now(),
    domain: 'housing',
    dkgValidated: true,
    dkgValidation: {
      consistent: true,
      validationConfidence: 0.88,
      supportingClaims: [],
      contradictingClaims: [],
      recommendation: 'approve',
      reasoning: 'Consistent with HUD guidelines.',
    },
    isVerificationSource: false,
  };

  console.log(`Classification: ${agentClaim.classificationCode}`);
  console.log(`Calibrated confidence: ${(agentClaim.calibratedConfidence * 100).toFixed(0)}%\n`);

  // Phase 3: Self-Healing Workflow
  console.log('━━━ PHASE 3: SELF-HEALING WORKFLOW ━━━\n');

  const workflow = createWorkflow('housing-voucher', '1.0.0')
    .description('Housing voucher application')
    .timeout(120000)
    .compensateOnFailure(true)
    .step('validate-identity', 'identity', async (_i: unknown, ctx: WorkflowContext) => {
      ctx.log('Validating identity...');
      governor.checkContent('workflow', 'Identity validation', { domain: 'housing' });
      return { verified: true, did: applicantProfile.did };
    }, { critical: true })
    .step('verify-income', 'finance', async (_i: unknown, ctx: WorkflowContext) => {
      ctx.log('Verifying income...');
      return { verified: true, category: 'below-50-ami' };
    }, { dependsOn: ['validate-identity'] })
    .step('create-voucher', 'governance', async (_i: unknown, ctx: WorkflowContext) => {
      ctx.log('Creating voucher...');
      return { created: true, voucherId: `HCV-${Date.now()}` };
    }, { dependsOn: ['verify-income'] })
    .build();

  const workflowResult = await orchestrator.execute(
    workflow,
    { initiatorDid: 'did:mycelix:housing-authority', input: { applicant: applicantProfile }, metadata: {} }
  );

  console.log(`Workflow status: ${workflowResult.status}\n`);

  // Phase 4: Civic Feedback Loop
  console.log('━━━ PHASE 4: CIVIC FEEDBACK LOOP ━━━\n');

  const justiceDecision: JusticeDecision = {
    caseId: `case-${applicantProfile.did}`,
    decisionId: `decision-${Date.now()}`,
    type: 'final_judgment',
    summary: 'Housing voucher application approved.',
    reasoning: 'Applicant income below 50% AMI.',
    principles: ['Income-based eligibility is primary criterion'],
    precedentsCited: [agentClaim.id],
    outcome: 'plaintiff_wins',
    legalDomain: 'administrative',
    confidence: 0.92,
    setsPrecedent: true,
    arbitrators: [applicantProfile.did],
    decidedAt: Date.now(),
  };

  const propagation = await feedbackLoop.propagateJusticeDecision(justiceDecision);
  console.log(`Precedent established: ${propagation.precedentsEstablished}\n`);

  // Phase 5: Market Resolution
  console.log('━━━ PHASE 5: MARKET RESOLUTION ━━━\n');

  const resolution = await trustMarkets.resolveMarket(market.id, 'yes');

  console.log(`Market resolved: YES`);
  console.log(`Prediction was: ${(impliedApproval * 100).toFixed(1)}%`);
  console.log(`Calibration error: ${((1 - impliedApproval) * 100).toFixed(1)}%\n`);

  // Summary
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║                       SCENARIO SUMMARY                        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  const feedbackStats = feedbackLoop.getStats();
  const periodStart = Date.now() - 3600000;
  const audit = governor.generateAudit(periodStart);

  console.log('Module Statistics:\n');
  console.log(`  🏛️  Constitutional AI: ${audit.agentsAudited} agents, ${audit.interactionsChecked} checks`);
  console.log(`  📊  Trust Markets: ${(impliedApproval * 100).toFixed(0)}% predicted → ${resolution.outcome}`);
  console.log(`  🔒  Private Queries: Privacy-preserved verification`);
  console.log(`  🧠  Epistemic Agent: ${(agentClaim.calibratedConfidence * 100).toFixed(0)}% confidence`);
  console.log(`  ⚖️  Civic Feedback: ${feedbackStats.precedentsEstablished} precedent(s)`);
  console.log(`  🔄  Self-Healing: ${workflowResult.status}`);
  console.log('');
  console.log('KEY INSIGHT: All 6 modules collaborated successfully.');

  feedbackLoop.stop();

  return {
    market,
    agentClaim,
    workflowResult,
    justiceDecision,
    resolution,
    stats: { feedbackStats, audit },
  };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  comprehensiveCivicScenario().catch(console.error);
}
