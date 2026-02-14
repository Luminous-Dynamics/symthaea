/**
 * Federated Learning Example
 *
 * Demonstrates Byzantine-fault-tolerant federated learning with MATL integration.
 *
 * Run with: npx ts-node examples/02-federated-learning.ts
 */

import {
  // FL utilities
  createSimpleFLCoordinator,
  createGradientUpdate,
  runFLRound,

  // FL module
  fl,

  // Trust utilities
  buildReputation,
} from '../src/index.js';

async function main() {
  console.log('🍄 Federated Learning Example\n');

  // Create FL coordinator with MATL trust integration
  console.log('=== Setting Up FL Coordinator ===');
  const coordinator = createSimpleFLCoordinator({
    minParticipants: 3,
    maxParticipants: 10,
    byzantineTolerance: 0.33, // Tolerate up to 33% Byzantine nodes
    aggregationMethod: fl.AggregationMethod.TrustWeighted,
  });

  // Register participants with different reputation levels
  const participants = [
    { id: 'hospital-A', positiveOps: 50, negativeOps: 2 },
    { id: 'hospital-B', positiveOps: 30, negativeOps: 5 },
    { id: 'hospital-C', positiveOps: 40, negativeOps: 1 },
    { id: 'clinic-D', positiveOps: 20, negativeOps: 3 },
    { id: 'research-lab-E', positiveOps: 60, negativeOps: 0 },
  ];

  console.log('Registering participants:');
  for (const p of participants) {
    coordinator.registerParticipant(p.id);
    const rep = buildReputation(p.id, p.positiveOps, p.negativeOps);
    const repValue = (rep.positiveCount / (rep.positiveCount + rep.negativeCount)).toFixed(3);
    console.log(`  ${p.id}: reputation = ${repValue}`);
  }
  console.log();

  // Simulate gradient updates from each participant
  console.log('=== Simulating Training Round ===');

  // Model: simple 4-parameter model (e.g., linear regression)
  const updates = [
    createGradientUpdate(
      'hospital-A',
      1,
      [0.12, -0.08, 0.05, 0.03],
      { batchSize: 256, loss: 0.42, accuracy: 0.89 }
    ),
    createGradientUpdate(
      'hospital-B',
      1,
      [0.10, -0.09, 0.06, 0.02],
      { batchSize: 128, loss: 0.45, accuracy: 0.87 }
    ),
    createGradientUpdate(
      'hospital-C',
      1,
      [0.11, -0.07, 0.04, 0.04],
      { batchSize: 192, loss: 0.40, accuracy: 0.90 }
    ),
    createGradientUpdate(
      'clinic-D',
      1,
      [0.09, -0.10, 0.07, 0.01],
      { batchSize: 64, loss: 0.50, accuracy: 0.85 }
    ),
    createGradientUpdate(
      'research-lab-E',
      1,
      [0.13, -0.06, 0.03, 0.05],
      { batchSize: 512, loss: 0.38, accuracy: 0.91 }
    ),
  ];

  console.log('Gradient updates submitted:');
  for (const u of updates) {
    console.log(`  ${u.participantId}: loss=${u.metadata.loss}, accuracy=${u.metadata.accuracy}`);
  }
  console.log();

  // Run the FL round
  console.log('=== Running FL Round ===');
  const summary = runFLRound(coordinator, updates);

  console.log(`Round ${summary.roundId}:`);
  console.log(`  Status: ${summary.status}`);
  console.log(`  Participants: ${summary.participantCount}`);
  console.log(`  Updates processed: ${summary.updateCount}`);
  console.log(`  Aggregation successful: ${summary.aggregated}`);
  console.log(`  Duration: ${summary.duration}ms`);
  console.log();

  // Demonstrate Byzantine detection
  console.log('=== Byzantine Attack Simulation ===');

  const byzantineCoordinator = createSimpleFLCoordinator({
    minParticipants: 3,
    byzantineTolerance: 0.33,
    aggregationMethod: fl.AggregationMethod.Krum, // Byzantine-resilient
  });

  byzantineCoordinator.registerParticipant('honest-1');
  byzantineCoordinator.registerParticipant('honest-2');
  byzantineCoordinator.registerParticipant('honest-3');
  byzantineCoordinator.registerParticipant('attacker');

  const mixedUpdates = [
    // Honest participants with similar gradients
    createGradientUpdate('honest-1', 1, [0.10, 0.10, 0.10, 0.10], { batchSize: 100, loss: 0.5 }),
    createGradientUpdate('honest-2', 1, [0.11, 0.09, 0.10, 0.10], { batchSize: 100, loss: 0.5 }),
    createGradientUpdate('honest-3', 1, [0.09, 0.11, 0.10, 0.10], { batchSize: 100, loss: 0.5 }),
    // Attacker with wildly different gradient (trying to poison the model)
    createGradientUpdate('attacker', 1, [100, -100, 50, -50], { batchSize: 100, loss: 0.5 }),
  ];

  console.log('Mixed updates (including attacker):');
  for (const u of mixedUpdates) {
    const gradStr = Array.from(u.gradients).map((g) => g.toFixed(2)).join(', ');
    console.log(`  ${u.participantId}: [${gradStr}]`);
  }
  console.log();

  const byzantineRound = runFLRound(byzantineCoordinator, mixedUpdates);
  console.log(`Byzantine-resilient round result:`);
  console.log(`  Status: ${byzantineRound.status}`);
  console.log(`  Aggregation method: Krum (Byzantine-resilient)`);
  console.log('  Note: Krum automatically excludes outlier gradients');
  console.log();

  // Show different aggregation methods
  console.log('=== Aggregation Methods ===');
  const methods = [
    { method: fl.AggregationMethod.FedAvg, name: 'FedAvg (Federated Averaging)' },
    { method: fl.AggregationMethod.TrimmedMean, name: 'TrimmedMean (outlier resistant)' },
    { method: fl.AggregationMethod.Krum, name: 'Krum (Byzantine resilient)' },
    { method: fl.AggregationMethod.TrustWeighted, name: 'TrustWeighted (MATL integration)' },
  ];

  for (const { method, name } of methods) {
    console.log(`  • ${name}`);
  }
}

main().catch(console.error);
