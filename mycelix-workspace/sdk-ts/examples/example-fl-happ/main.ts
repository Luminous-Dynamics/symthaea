/**
 * Example Federated Learning hApp - Main Entry Point
 *
 * Demonstrates a complete FL training loop with:
 * - Byzantine-resistant aggregation (Krum)
 * - MATL reputation integration
 * - Hierarchical Byzantine detection
 */

import {
  // MATL
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  HierarchicalDetector,

  // FL
  DEFAULT_CONFIG,
  krum,
  trustWeightedAggregation,

  // Types
  type Reputation,
  type Participant,
} from '../../src/index.js';

// ============================================================================
// Configuration
// ============================================================================

const CONFIG = {
  // Training settings
  numRounds: 10,
  gradientSize: 100,
  trueGradientValue: 0.1,

  // Participants
  totalParticipants: 10,
  byzantineCount: 2, // 20% - well within tolerance

  // MATL settings
  reputationThreshold: 0.3,
  detectionLevels: 3,
  minClusterSize: 2,
};

// ============================================================================
// Participant Simulation
// ============================================================================

interface SimulatedParticipant {
  id: string;
  reputation: Reputation;
  isByzantine: boolean;
  roundsParticipated: number;
}

function createParticipants(): SimulatedParticipant[] {
  const participants: SimulatedParticipant[] = [];

  for (let i = 0; i < CONFIG.totalParticipants; i++) {
    const isByzantine = i >= CONFIG.totalParticipants - CONFIG.byzantineCount;
    participants.push({
      id: `node-${i.toString().padStart(2, '0')}`,
      reputation: createReputation(`node-${i}`),
      isByzantine,
      roundsParticipated: 0,
    });
  }

  return participants;
}

function computeGradient(participant: SimulatedParticipant): Float64Array {
  const gradient = new Float64Array(CONFIG.gradientSize);

  if (participant.isByzantine) {
    // Byzantine: Submit malicious gradient
    const attackType = Math.floor(Math.random() * 3);
    switch (attackType) {
      case 0: // Large values
        gradient.fill(100.0);
        break;
      case 1: // Sign flip
        gradient.fill(-CONFIG.trueGradientValue);
        break;
      case 2: // Random noise
        for (let i = 0; i < CONFIG.gradientSize; i++) {
          gradient[i] = (Math.random() - 0.5) * 200;
        }
        break;
    }
  } else {
    // Honest: Submit accurate gradient with small noise
    for (let i = 0; i < CONFIG.gradientSize; i++) {
      gradient[i] = CONFIG.trueGradientValue + (Math.random() - 0.5) * 0.02;
    }
  }

  return gradient;
}

// ============================================================================
// Training Loop
// ============================================================================

function runTrainingRound(
  roundNum: number,
  participants: SimulatedParticipant[]
): { error: number; detectedByzantine: string[] } {
  console.log(`\n--- Round ${roundNum} ---`);

  // 1. Collect gradients
  const gradients: Float64Array[] = [];
  const participantMap = new Map<string, Participant>();

  participants.forEach((p) => {
    const gradient = computeGradient(p);
    gradients.push(gradient);

    participantMap.set(p.id, {
      id: p.id,
      reputation: p.reputation,
      roundsParticipated: p.roundsParticipated,
    });
  });

  // 2. Aggregate using Krum (Byzantine-resistant)
  const aggregated = krum(gradients, CONFIG.byzantineCount);

  // 3. Calculate error
  const avgValue = aggregated.reduce((a, b) => a + b, 0) / aggregated.length;
  const error = Math.abs(avgValue - CONFIG.trueGradientValue);

  console.log(`  Aggregated gradient avg: ${avgValue.toFixed(4)}`);
  console.log(`  Error from true value:   ${error.toFixed(4)}`);

  // 4. Byzantine detection using hierarchical clustering
  const detector = new HierarchicalDetector(
    CONFIG.detectionLevels,
    CONFIG.minClusterSize
  );

  participants.forEach((p) => {
    const score = reputationValue(p.reputation);
    detector.assign(p.id, score);
  });

  const detectedByzantine = detector.getSuspectedByzantine();

  // 5. Update reputations
  participants.forEach((p) => {
    // Compute contribution quality (distance from aggregated)
    const pGradient = computeGradient(p);
    let distance = 0;
    for (let i = 0; i < CONFIG.gradientSize; i++) {
      distance += Math.pow(pGradient[i] - aggregated[i], 2);
    }
    distance = Math.sqrt(distance / CONFIG.gradientSize);

    // Good contributions get positive reputation
    if (distance < 1.0) {
      p.reputation = recordPositive(p.reputation);
    } else {
      p.reputation = recordNegative(p.reputation);
    }

    p.roundsParticipated++;
  });

  // 6. Log detected Byzantine
  if (detectedByzantine.length > 0) {
    console.log(`  Suspected Byzantine: ${detectedByzantine.join(', ')}`);
  }

  return { error, detectedByzantine };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log('========================================');
  console.log('   Mycelix Federated Learning Example');
  console.log('========================================');
  console.log('');
  console.log('Configuration:');
  console.log(`  Participants: ${CONFIG.totalParticipants}`);
  console.log(`  Byzantine:    ${CONFIG.byzantineCount} (${(CONFIG.byzantineCount / CONFIG.totalParticipants * 100).toFixed(0)}%)`);
  console.log(`  Rounds:       ${CONFIG.numRounds}`);
  console.log(`  Aggregation:  Krum`);

  // Create participants
  const participants = createParticipants();

  console.log('\nInitial participants:');
  participants.forEach((p) => {
    const status = p.isByzantine ? '(Byzantine)' : '(Honest)';
    console.log(`  ${p.id} ${status}`);
  });

  // Run training
  const roundResults: { error: number; detectedByzantine: string[] }[] = [];

  for (let round = 1; round <= CONFIG.numRounds; round++) {
    const result = runTrainingRound(round, participants);
    roundResults.push(result);
  }

  // Final summary
  console.log('\n========================================');
  console.log('            Final Summary');
  console.log('========================================\n');

  // Error progression
  console.log('Error progression:');
  roundResults.forEach((r, i) => {
    const bar = '█'.repeat(Math.round(r.error * 100));
    console.log(`  Round ${(i + 1).toString().padStart(2)}: ${r.error.toFixed(4)} ${bar}`);
  });

  // Final reputations
  console.log('\nFinal reputations:');
  const honestReps: number[] = [];
  const byzantineReps: number[] = [];

  participants.forEach((p) => {
    const rep = reputationValue(p.reputation);
    const bar = '▓'.repeat(Math.round(rep * 20));
    const status = p.isByzantine ? 'Byz' : 'Hon';
    console.log(`  ${p.id} [${status}]: ${rep.toFixed(4)} ${bar}`);

    if (p.isByzantine) {
      byzantineReps.push(rep);
    } else {
      honestReps.push(rep);
    }
  });

  const avgHonest = honestReps.reduce((a, b) => a + b, 0) / honestReps.length;
  const avgByzantine = byzantineReps.reduce((a, b) => a + b, 0) / byzantineReps.length;

  console.log('\nReputation summary:');
  console.log(`  Avg honest reputation:    ${avgHonest.toFixed(4)}`);
  console.log(`  Avg Byzantine reputation: ${avgByzantine.toFixed(4)}`);
  console.log(`  Separation:               ${(avgHonest - avgByzantine).toFixed(4)}`);

  // Detection accuracy
  const allDetected = new Set(roundResults.flatMap((r) => r.detectedByzantine));
  const actualByzantine = new Set(
    participants.filter((p) => p.isByzantine).map((p) => p.id)
  );
  const correctDetections = [...allDetected].filter((id) =>
    actualByzantine.has(id)
  ).length;
  const falsePositives = [...allDetected].filter(
    (id) => !actualByzantine.has(id)
  ).length;

  console.log('\nDetection summary:');
  console.log(`  Actual Byzantine:   ${actualByzantine.size}`);
  console.log(`  Detected Byzantine: ${allDetected.size}`);
  console.log(`  True positives:     ${correctDetections}`);
  console.log(`  False positives:    ${falsePositives}`);

  // Final accuracy (inverse of final error)
  const finalError = roundResults[roundResults.length - 1].error;
  const accuracy = Math.max(0, 1 - finalError * 10); // Scale for readability

  console.log('\nTraining result:');
  console.log(`  Final error:    ${finalError.toFixed(4)}`);
  console.log(`  Model accuracy: ${(accuracy * 100).toFixed(1)}%`);
  console.log(`  Byzantine-resistant: ${finalError < 0.1 ? 'YES' : 'DEGRADED'}`);

  console.log('\n========================================');
  console.log('         Example Complete!');
  console.log('========================================\n');
}

// Run
main().catch(console.error);
