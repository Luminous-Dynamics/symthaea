/**
 * Basic Federated Learning Example
 *
 * Demonstrates how to use the Mycelix SDK for Byzantine-resistant
 * federated learning with MATL trust integration.
 */

import {
  // MATL
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,

  // Federated Learning
  DEFAULT_CONFIG,
  fedAvg,
  trimmedMean,
  coordinateMedian,
  krum,
  trustWeightedAggregation,

  // Types
  type Reputation,
  type Participant,
} from '../src/index.js';

// =============================================================================
// Example 1: Basic Reputation Management
// =============================================================================

function demonstrateReputation() {
  console.log('\n=== Reputation Management ===\n');

  // Create reputation for a new participant
  let reputation = createReputation('participant-001');
  console.log(`Initial reputation: ${reputationValue(reputation).toFixed(4)}`);

  // Simulate successful rounds
  for (let i = 0; i < 10; i++) {
    reputation = recordPositive(reputation);
  }
  console.log(`After 10 successes: ${reputationValue(reputation).toFixed(4)}`);

  // Simulate some failures
  for (let i = 0; i < 3; i++) {
    reputation = recordNegative(reputation);
  }
  console.log(`After 3 failures: ${reputationValue(reputation).toFixed(4)}`);

  return reputation;
}

// =============================================================================
// Example 2: Gradient Aggregation
// =============================================================================

function demonstrateAggregation() {
  console.log('\n=== Gradient Aggregation ===\n');

  // Simulate gradients from 10 participants
  const gradientSize = 100;
  const participantCount = 10;
  const byzantineCount = 2; // 20% Byzantine

  const gradients: Float64Array[] = [];

  // Honest participants submit gradients close to true value (0.1)
  for (let i = 0; i < participantCount - byzantineCount; i++) {
    const gradient = new Float64Array(gradientSize);
    for (let j = 0; j < gradientSize; j++) {
      gradient[j] = 0.1 + (Math.random() - 0.5) * 0.02; // Small noise
    }
    gradients.push(gradient);
  }

  // Byzantine participants submit malicious gradients
  for (let i = 0; i < byzantineCount; i++) {
    const gradient = new Float64Array(gradientSize);
    for (let j = 0; j < gradientSize; j++) {
      gradient[j] = 100.0; // Large malicious value
    }
    gradients.push(gradient);
  }

  // Compare aggregation methods
  const avgResult = fedAvg(gradients);
  const trimmedResult = trimmedMean(gradients, 0.2);
  const medianResult = coordinateMedian(gradients);
  const krumResult = krum(gradients, byzantineCount);

  console.log(`True gradient value: 0.1000`);
  console.log(`FedAvg (vulnerable):     ${avgResult[0].toFixed(4)}`);
  console.log(`Trimmed Mean (robust):   ${trimmedResult[0].toFixed(4)}`);
  console.log(`Coordinate Median:       ${medianResult[0].toFixed(4)}`);
  console.log(`Krum (Byzantine-safe):   ${krumResult[0].toFixed(4)}`);
}

// =============================================================================
// Example 3: Trust-Weighted Aggregation
// =============================================================================

function demonstrateTrustWeighted() {
  console.log('\n=== Trust-Weighted Aggregation ===\n');

  const gradientSize = 50;

  // Create participants with varying reputation
  const participants = new Map<string, Participant>();

  // High-reputation honest participant
  let rep1 = createReputation('trusted-node-1');
  for (let i = 0; i < 50; i++) rep1 = recordPositive(rep1);
  participants.set('trusted-node-1', {
    id: 'trusted-node-1',
    reputation: rep1,
    roundsParticipated: 50,
  });

  // Medium-reputation honest participant
  let rep2 = createReputation('medium-node-1');
  for (let i = 0; i < 10; i++) rep2 = recordPositive(rep2);
  for (let i = 0; i < 2; i++) rep2 = recordNegative(rep2);
  participants.set('medium-node-1', {
    id: 'medium-node-1',
    reputation: rep2,
    roundsParticipated: 12,
  });

  // Low-reputation Byzantine participant
  let rep3 = createReputation('byzantine-node');
  for (let i = 0; i < 2; i++) rep3 = recordPositive(rep3);
  for (let i = 0; i < 8; i++) rep3 = recordNegative(rep3);
  participants.set('byzantine-node', {
    id: 'byzantine-node',
    reputation: rep3,
    roundsParticipated: 10,
  });

  // Create gradients
  const gradients: Float64Array[] = [];

  // Trusted node: accurate gradient
  const g1 = new Float64Array(gradientSize).fill(0.1);
  gradients.push(g1);

  // Medium node: slightly noisy gradient
  const g2 = new Float64Array(gradientSize);
  for (let i = 0; i < gradientSize; i++) g2[i] = 0.1 + (Math.random() - 0.5) * 0.05;
  gradients.push(g2);

  // Byzantine node: malicious gradient
  const g3 = new Float64Array(gradientSize).fill(10.0);
  gradients.push(g3);

  console.log('Participant reputations:');
  participants.forEach((p, id) => {
    console.log(`  ${id}: ${reputationValue(p.reputation).toFixed(4)}`);
  });

  // Compare unweighted vs trust-weighted
  const unweighted = fedAvg(gradients);
  const weighted = trustWeightedAggregation(gradients, participants);

  console.log(`\nTrue gradient: 0.1000`);
  console.log(`Unweighted avg: ${unweighted[0].toFixed(4)}`);
  console.log(`Trust-weighted: ${weighted[0].toFixed(4)}`);
  console.log(`\nTrust weighting reduces Byzantine influence by ${((unweighted[0] - weighted[0]) / unweighted[0] * 100).toFixed(1)}%`);
}

// =============================================================================
// Example 4: Full FL Round Simulation
// =============================================================================

function simulateFLRound() {
  console.log('\n=== Full FL Round Simulation ===\n');

  const config = {
    ...DEFAULT_CONFIG,
    minParticipants: 5,
    byzantineTolerance: 0.34,
  };

  console.log('Configuration:');
  console.log(`  Min participants: ${config.minParticipants}`);
  console.log(`  Byzantine tolerance: ${(config.byzantineTolerance * 100).toFixed(0)}%`);
  console.log(`  Aggregation: ${config.aggregationMethod}`);

  // Simulate 5 rounds
  const participants = new Map<string, Participant>();
  const numNodes = 10;
  const numByzantine = 3; // 30% - within tolerance

  // Initialize participants
  for (let i = 0; i < numNodes; i++) {
    const id = `node-${i}`;
    participants.set(id, {
      id,
      reputation: createReputation(id),
      roundsParticipated: 0,
    });
  }

  console.log('\nSimulating 5 training rounds...\n');

  for (let round = 1; round <= 5; round++) {
    const gradients: Float64Array[] = [];
    const gradientSize = 100;

    // Generate gradients
    let nodeIdx = 0;
    participants.forEach((p) => {
      const gradient = new Float64Array(gradientSize);
      const isByzantine = nodeIdx >= numNodes - numByzantine;

      for (let j = 0; j < gradientSize; j++) {
        if (isByzantine) {
          gradient[j] = Math.random() * 100; // Random malicious
        } else {
          gradient[j] = 0.1 + (Math.random() - 0.5) * 0.02; // Honest
        }
      }
      gradients.push(gradient);
      nodeIdx++;
    });

    // Aggregate using Krum
    const aggregated = krum(gradients, numByzantine);

    // Update reputations based on contribution quality
    nodeIdx = 0;
    participants.forEach((p, id) => {
      const isByzantine = nodeIdx >= numNodes - numByzantine;
      const updated = isByzantine
        ? recordNegative(p.reputation)
        : recordPositive(p.reputation);

      participants.set(id, {
        ...p,
        reputation: updated,
        roundsParticipated: p.roundsParticipated + 1,
      });
      nodeIdx++;
    });

    const avgScore = aggregated.reduce((a, b) => a + b, 0) / aggregated.length;
    console.log(`Round ${round}: Aggregated avg = ${avgScore.toFixed(4)} (target: 0.1)`);
  }

  console.log('\nFinal reputations:');
  participants.forEach((p, id) => {
    console.log(`  ${id}: ${reputationValue(p.reputation).toFixed(4)}`);
  });
}

// =============================================================================
// Run Examples
// =============================================================================

console.log('========================================');
console.log('Mycelix SDK - Federated Learning Example');
console.log('========================================');

demonstrateReputation();
demonstrateAggregation();
demonstrateTrustWeighted();
simulateFLRound();

console.log('\n========================================');
console.log('Examples completed successfully!');
console.log('========================================\n');
