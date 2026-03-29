// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Basic Usage Example
 *
 * This example demonstrates the core features of the Mycelix SDK:
 * - MATL trust scoring
 * - Epistemic claims
 * - Bridge messaging
 * - Federated Learning
 */

import {
  // MATL
  createPoGQ,
  compositeScore,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  isTrustworthy,
  isByzantine,

  // Epistemic
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  meetsStandard,
  Standards,

  // Bridge
  LocalBridge,
  createReputationQuery,
  createCrossHappReputation,

  // FL
  FLCoordinator,
  AggregationMethod,

  // Config
  initConfig,
  DEV_CONFIG,

  // Utils
  sdkEvents,
  MatlEventType,
} from '@mycelix/sdk';

async function main() {
  console.log('=== @mycelix/sdk Basic Usage Example ===\n');

  // Initialize with development config
  initConfig(DEV_CONFIG);

  // -------------------------------------------------------------------------
  // 1. MATL: Mycelix Adaptive Trust Layer
  // -------------------------------------------------------------------------
  console.log('--- MATL Trust Scoring ---');

  // Create a Proof of Gradient Quality
  const pogq = createPoGQ(
    0.95, // quality: gradient quality score
    0.88, // consistency: consistency across rounds
    0.12  // entropy: information entropy
  );
  console.log('PoGQ:', pogq);

  // Calculate composite trust score
  const score = compositeScore(pogq, 0.75); // 0.75 = local reputation
  console.log('Composite Score:', score.score.toFixed(4));
  console.log('Confidence:', score.confidence.toFixed(4));
  console.log('Is Byzantine?', isByzantine(score.score));

  // Build reputation over time
  let reputation = createReputation();
  console.log('\nBuilding reputation...');

  for (let i = 0; i < 10; i++) {
    // Simulate mostly positive with occasional negative
    reputation = i % 4 === 0
      ? recordNegative(reputation)
      : recordPositive(reputation);
  }

  console.log('Reputation value:', reputationValue(reputation).toFixed(4));
  console.log('Is trustworthy?', isTrustworthy(reputation));

  // -------------------------------------------------------------------------
  // 2. Epistemic: Truth Claims with 3D Classification
  // -------------------------------------------------------------------------
  console.log('\n--- Epistemic Claims ---');

  // Create a high-trust claim
  const identityClaim = claim('User verified via cryptographic proof')
    .withEmpirical(EmpiricalLevel.E3_Cryptographic)
    .withNormative(NormativeLevel.N2_Network)
    .withMateriality(MaterialityLevel.M2_Significant)
    .addEvidence({
      type: 'signature',
      data: { verified: true, algorithm: 'ed25519' },
    })
    .withTtl(3600000) // 1 hour
    .build();

  console.log('Claim ID:', identityClaim.id);
  console.log('Classification:', `E${identityClaim.classification.empirical}-N${identityClaim.classification.normative}-M${identityClaim.classification.materiality}`);
  console.log('Meets financial standard?', meetsStandard(identityClaim, Standards.Financial));
  console.log('Meets basic standard?', meetsStandard(identityClaim, Standards.Basic));

  // Create a lower-trust claim
  const testimonialClaim = claim('User reported positive experience')
    .withEmpirical(EmpiricalLevel.E1_Testimonial)
    .withNormative(NormativeLevel.N1_Local)
    .build();

  console.log('\nTestimonial claim meets basic?', meetsStandard(testimonialClaim, Standards.Basic));

  // -------------------------------------------------------------------------
  // 3. Bridge: Inter-hApp Communication
  // -------------------------------------------------------------------------
  console.log('\n--- Bridge Messaging ---');

  const bridge = new LocalBridge();

  // Register hApps
  bridge.registerHapp('identity-happ');
  bridge.registerHapp('marketplace-happ');
  bridge.registerHapp('reputation-happ');

  // Subscribe to reputation queries
  bridge.subscribe('reputation-happ', (message) => {
    console.log('Reputation hApp received:', message.type);
    if (message.type === 'reputation_query') {
      // Respond with cross-hApp reputation
      const response = createCrossHappReputation(
        message.payload.agentId,
        'marketplace-happ',
        [
          { happId: 'identity-happ', score: 0.92, weight: 0.5 },
          { happId: 'marketplace-happ', score: 0.85, weight: 0.3 },
        ]
      );
      console.log('Aggregate reputation:', response.payload.aggregateScore.toFixed(4));
    }
  });

  // Send a reputation query
  const query = createReputationQuery('agent-abc123');
  bridge.send('reputation-happ', query);

  // -------------------------------------------------------------------------
  // 4. Federated Learning
  // -------------------------------------------------------------------------
  console.log('\n--- Federated Learning ---');

  const flCoordinator = new FLCoordinator({
    roundTimeoutMs: 30000,
    minParticipants: 3,
    aggregationMethod: AggregationMethod.TrustWeighted,
    byzantineTolerance: 0.2,
  });

  // Register participants
  const participants = ['alice', 'bob', 'charlie', 'diana', 'eve'];
  participants.forEach((p) => flCoordinator.registerParticipant(p));
  console.log('Registered participants:', participants.length);

  // Start a round
  flCoordinator.startRound();
  console.log('Round started');

  // Submit gradient updates
  for (const participant of participants) {
    flCoordinator.submitUpdate({
      participantId: participant,
      gradients: new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5].map(
        (g) => g + (Math.random() - 0.5) * 0.1 // Add some noise
      )),
      timestamp: Date.now(),
    });
  }
  console.log('All updates submitted');

  // Aggregate
  const result = flCoordinator.aggregateRound();
  if (result) {
    console.log('Aggregation complete');
    console.log('  Participants:', result.participantCount);
    console.log('  Excluded:', result.excludedCount);
    console.log('  First gradient:', result.gradients[0].toFixed(4));
  }

  // -------------------------------------------------------------------------
  // 5. Event System
  // -------------------------------------------------------------------------
  console.log('\n--- Event System ---');

  // Subscribe to SDK events
  const subscription = sdkEvents.subscribe((event) => {
    if (event.type === MatlEventType.ReputationUpdated) {
      console.log('Reputation updated for:', event.payload.agentId);
    }
  });

  // Emit some events (normally done internally by SDK operations)
  sdkEvents.emit({
    type: MatlEventType.ReputationUpdated,
    timestamp: Date.now(),
    payload: {
      agentId: 'agent-demo',
      previousValue: 0.5,
      newValue: 0.75,
      reason: 'positive interaction',
    },
  });

  subscription.unsubscribe();

  console.log('\n=== Example Complete ===');
}

main().catch(console.error);
