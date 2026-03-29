// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Agents + Civic Feedback Loop Integration Example
 *
 * Demonstrates how epistemic agents create verified knowledge claims
 * that flow through the civic feedback loop, establishing precedents.
 *
 * @module innovations/examples/epistemic-civic-integration
 */

import { EmpiricalLevel, NormativeLevel, MaterialityLevel } from '../../integrations/epistemic-markets/index.js';
import { createCivicFeedbackLoop, type JusticeDecision } from '../civic-feedback-loop/index.js';

import type { EpistemicClaim } from '../epistemic-agents/index.js';

/**
 * Example: Epistemic Agent Response Becomes Legal Precedent
 */
export async function epistemicCivicIntegrationExample() {
  console.log('=== Epistemic Agents + Civic Feedback Loop ===\n');

  // Initialize civic feedback loop
  const feedbackLoop = createCivicFeedbackLoop({
    autoPropagate: true,
    enableConflictDetection: true,
  });
  await feedbackLoop.start();

  console.log('Step 1: Citizen asks about SNAP eligibility\n');

  // Agent formulates response with epistemic classification
  const agentClaim: EpistemicClaim = {
    id: `claim-snap-eligibility-${Date.now()}`,
    text: 'Employment status alone does not disqualify SNAP eligibility. ' +
          'Eligibility is based on gross monthly income and household size.',
    position: {
      empirical: EmpiricalLevel.Cryptographic,
      normative: NormativeLevel.Universal,
      materiality: MaterialityLevel.Persistent,
    },
    classificationCode: 'E3-N3-M2',
    confidence: 0.88,
    calibratedConfidence: 0.85,
    sources: [{
      id: 'usda-snap-2024',
      type: 'regulation',
      title: 'USDA SNAP Eligibility Guidelines',
      relevance: 0.95,
    }],
    agentId: 'snap-navigator-001',
    conversationId: `conv-${Date.now()}`,
    timestamp: Date.now(),
    domain: 'benefits',
    dkgValidated: true,
    dkgValidation: {
      consistent: true,
      validationConfidence: 0.92,
      supportingClaims: ['precedent-snap-2024'],
      contradictingClaims: [],
      recommendation: 'approve',
      reasoning: 'Consistent with federal guidelines.',
    },
    isVerificationSource: false,
  };

  console.log(`Classification: ${agentClaim.classificationCode}`);
  console.log(`Confidence: ${(agentClaim.confidence * 100).toFixed(0)}%`);

  // Justice decision creates precedent
  const justiceDecision: JusticeDecision = {
    caseId: 'case-snap-dispute-2024-001',
    decisionId: `decision-${Date.now()}`,
    type: 'final_judgment',
    summary: 'Employment does not categorically disqualify SNAP eligibility.',
    reasoning: 'Federal regulations establish income-based eligibility criteria.',
    principles: [
      'SNAP eligibility is income-based',
      'AI guidance aligned with regulations is authoritative',
    ],
    precedentsCited: [agentClaim.id],
    outcome: 'plaintiff_wins',
    legalDomain: 'administrative',
    confidence: 0.95,
    setsPrecedent: true,
    arbitrators: ['did:mycelix:citizen-001'],
    decidedAt: Date.now(),
  };

  console.log('\nPropagating justice decision...');
  const propagation = await feedbackLoop.propagateJusticeDecision(justiceDecision);

  console.log(`Propagation ID: ${propagation.propagationId}`);
  console.log(`Claims created: ${propagation.claimsCreated}`);
  console.log(`Precedents: ${propagation.precedentsEstablished}`);

  // Query derived claims
  const derivedClaims = feedbackLoop.queryDerivedClaims({
    domain: 'administrative',
    isCurrentLaw: true,
  });

  console.log(`\nDerived claims: ${derivedClaims.length}`);

  const stats = feedbackLoop.getStats();
  console.log(`\nFeedback Loop Stats:`);
  console.log(`  Decisions processed: ${stats.justiceDecisionsProcessed}`);
  console.log(`  Precedents: ${stats.precedentsEstablished}`);

  feedbackLoop.stop();

  return { agentClaim, justiceDecision, propagation, derivedClaims };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  epistemicCivicIntegrationExample().catch(console.error);
}
