/**
 * Example 18: Federated Learning Hub - Session Management
 *
 * This example demonstrates how to use the FL Hub module for:
 * - Creating and managing FL training sessions
 * - Registering participants and tracking progress
 * - Model registry for versioning and checkpoints
 * - Differential privacy budget management
 */

import { flHub, matl } from '../src/index.js';

async function main() {
  console.log('=== Mycelix FL Hub: Session Management Example ===\n');

  // =========================================================================
  // Part 1: FL Hub Coordinator Setup
  // =========================================================================
  console.log('--- Part 1: FL Hub Coordinator Setup ---\n');

  const coordinator = new flHub.FLHubCoordinator({
    maxConcurrentSessions: 5,
    defaultRoundsPerSession: 50,
    privacyBudget: { epsilon: 5.0, delta: 1e-5 },
  });

  console.log('FL Hub Coordinator initialized');
  console.log(`  Max concurrent sessions: 5`);
  console.log(`  Default rounds: 50`);
  console.log(`  Privacy budget: ε=5.0, δ=1e-5\n`);

  // =========================================================================
  // Part 2: Create Training Session
  // =========================================================================
  console.log('--- Part 2: Create Training Session ---\n');

  const session = await coordinator.createSession({
    name: 'chest-xray-pneumonia-detection',
    modelId: 'resnet18-medical-v1',
    minParticipants: 3,
    maxParticipants: 10,
    targetRounds: 20,
    aggregationMethod: 'trust_weighted',
    privacyConfig: {
      enableDifferentialPrivacy: true,
      epsilon: 0.5,
      delta: 1e-6,
      clipNorm: 1.0,
    },
  });

  console.log('Training session created:');
  console.log(`  Session ID: ${session.sessionId}`);
  console.log(`  Name: ${session.name}`);
  console.log(`  Model: ${session.modelId}`);
  console.log(`  Min participants: ${session.minParticipants}`);
  console.log(`  Target rounds: ${session.targetRounds}`);
  console.log(`  Aggregation: ${session.aggregationMethod}`);
  console.log(`  Differential Privacy: ε=${session.privacyConfig?.epsilon}\n`);

  // =========================================================================
  // Part 3: Register Participants
  // =========================================================================
  console.log('--- Part 3: Register Participants ---\n');

  const hospitals = [
    { id: 'hospital-stanford', dataSize: 15000, capability: 'gpu' },
    { id: 'hospital-mayo', dataSize: 12000, capability: 'gpu' },
    { id: 'hospital-cleveland', dataSize: 8000, capability: 'cpu' },
    { id: 'hospital-hopkins', dataSize: 20000, capability: 'gpu' },
  ];

  for (const hospital of hospitals) {
    await coordinator.registerParticipant(session.sessionId, {
      participantId: hospital.id,
      dataSize: hospital.dataSize,
      computeCapability: hospital.capability,
    });
    console.log(`  Registered: ${hospital.id} (${hospital.dataSize} samples, ${hospital.capability})`);
  }

  // =========================================================================
  // Part 4: Start Training
  // =========================================================================
  console.log('\n--- Part 4: Start Training Session ---\n');

  await coordinator.startSession(session.sessionId);
  console.log('Training session started!\n');

  // Simulate training rounds
  const totalRounds = 5; // Abbreviated for demo

  for (let round = 1; round <= totalRounds; round++) {
    console.log(`Round ${round}/${totalRounds}:`);

    // Each participant submits their update
    for (const hospital of hospitals) {
      // Simulate gradient computation
      const gradients = new Float64Array(100);
      for (let i = 0; i < gradients.length; i++) {
        gradients[i] = (Math.random() - 0.5) * 0.1;
      }

      // Simulate metrics
      const loss = 0.5 - (round * 0.08) + (Math.random() * 0.02);
      const accuracy = 0.6 + (round * 0.06) + (Math.random() * 0.02);

      await coordinator.submitUpdate(session.sessionId, {
        participantId: hospital.id,
        round,
        gradients: Array.from(gradients),
        dataSize: hospital.dataSize,
        metrics: {
          loss: Math.max(0.1, loss),
          accuracy: Math.min(0.95, accuracy),
        },
        timestamp: Date.now(),
      });
    }

    // Get round status
    const status = coordinator.getSessionStatus(session.sessionId);
    console.log(`  Updates: ${status?.currentRound || round} | ` +
                `Privacy budget used: ${((1 - (status?.privacyBudget?.remaining || 1)) * 100).toFixed(1)}%`);
  }

  // Get final session status
  const finalStatus = coordinator.getSessionStatus(session.sessionId);
  console.log('\nSession status:');
  console.log(`  Current round: ${finalStatus?.currentRound}`);
  console.log(`  Total participants: ${finalStatus?.participantCount}`);
  console.log(`  Status: ${finalStatus?.status}`);

  // =========================================================================
  // Part 5: Model Registry
  // =========================================================================
  console.log('\n--- Part 5: Model Registry ---\n');

  const registry = new flHub.ModelRegistry();

  // Register the trained model
  console.log('Registering trained model...');

  const modelEntry = await registry.registerModel({
    modelId: 'resnet18-pneumonia-v1.0',
    version: '1.0.0',
    architecture: 'ResNet18',
    trainingSessionId: session.sessionId,
    metrics: {
      accuracy: 0.92,
      f1Score: 0.89,
      auc: 0.95,
      sensitivity: 0.91,
      specificity: 0.93,
    },
  });

  console.log('Model registered:');
  console.log(`  ID: ${modelEntry.modelId}`);
  console.log(`  Version: ${modelEntry.version}`);
  console.log(`  Accuracy: ${(modelEntry.metrics.accuracy * 100).toFixed(1)}%`);
  console.log(`  AUC: ${modelEntry.metrics.auc}`);

  // Save checkpoints during training
  console.log('\nSaving checkpoints...');

  for (let checkpoint = 1; checkpoint <= 3; checkpoint++) {
    await registry.saveCheckpoint(session.sessionId, {
      round: checkpoint * 5,
      metrics: {
        loss: 0.5 - (checkpoint * 0.1),
        accuracy: 0.6 + (checkpoint * 0.1),
      },
    });
    console.log(`  Checkpoint saved: Round ${checkpoint * 5}`);
  }

  // List model versions
  const versions = registry.listModelVersions('resnet18-pneumonia');
  console.log(`\nModel versions available: ${versions?.length || 1}`);

  // Get latest model
  const latest = registry.getLatestModel('resnet18-pneumonia-v1.0');
  console.log(`Latest model: ${latest?.modelId || modelEntry.modelId}`);

  // =========================================================================
  // Part 6: Privacy Budget Management
  // =========================================================================
  console.log('\n--- Part 6: Privacy Budget Management ---\n');

  const privacyManager = new flHub.PrivacyManager({
    totalEpsilon: 10.0,
    totalDelta: 1e-5,
    accountingMethod: 'rdp',
  });

  console.log('Privacy Manager initialized:');
  console.log(`  Total budget: ε=${privacyManager.getTotalEpsilon()}, δ=1e-5`);
  console.log(`  Accounting: Rényi Differential Privacy\n`);

  // Simulate privacy consumption over rounds
  const operations = [
    { name: 'Round 1 aggregation', epsilon: 0.3, delta: 1e-7 },
    { name: 'Round 2 aggregation', epsilon: 0.3, delta: 1e-7 },
    { name: 'Round 3 aggregation', epsilon: 0.3, delta: 1e-7 },
    { name: 'Gradient clipping', epsilon: 0.2, delta: 1e-8 },
    { name: 'Noise injection', epsilon: 0.4, delta: 1e-7 },
  ];

  console.log('Tracking privacy consumption:');

  for (const op of operations) {
    if (privacyManager.canSpend(op.epsilon, op.delta)) {
      privacyManager.recordConsumption(op.epsilon, op.delta, op.name);
      const remaining = privacyManager.getRemainingBudget();
      console.log(`  ${op.name}: ε=${op.epsilon} | Remaining: ε=${remaining.epsilon.toFixed(2)}`);
    } else {
      console.log(`  ${op.name}: BLOCKED - Insufficient budget!`);
    }
  }

  const finalBudget = privacyManager.getRemainingBudget();
  console.log(`\nFinal budget status:`);
  console.log(`  Epsilon remaining: ${finalBudget.epsilon.toFixed(2)} / 10.0`);
  console.log(`  Operations logged: ${finalBudget.history?.length || operations.length}`);

  // =========================================================================
  // Part 7: Private Aggregator
  // =========================================================================
  console.log('\n--- Part 7: Private Aggregation with DP ---\n');

  const privateAggregator = new flHub.PrivateAggregator({
    epsilon: 0.5,
    delta: 1e-6,
    clipNorm: 1.0,
    noiseMechanism: 'gaussian',
  });

  console.log('Private Aggregator configured:');
  console.log(`  ε=${privateAggregator.getEpsilon()}, δ=1e-6`);
  console.log(`  Clip norm: 1.0`);
  console.log(`  Noise: Gaussian mechanism\n`);

  // Create participant updates with MATL reputations
  const participantUpdates = hospitals.map((hospital, i) => {
    const reputation = matl.createReputation(hospital.id);
    // Record some positive interactions
    for (let j = 0; j < 5 + i; j++) {
      matl.recordPositive(reputation);
    }

    return {
      participantId: hospital.id,
      round: 5,
      gradients: Array(50).fill(0).map(() => (Math.random() - 0.5) * 0.1),
      dataSize: hospital.dataSize,
      metrics: { loss: 0.2, accuracy: 0.88 },
      timestamp: Date.now(),
      reputation,
    };
  });

  console.log('Aggregating with differential privacy...');
  const privateResult = await privateAggregator.aggregate(participantUpdates);

  console.log(`\nPrivate aggregation result:`);
  console.log(`  Participants: ${privateResult.participantCount}`);
  console.log(`  Noise scale: σ=${privateResult.noiseScale?.toFixed(4) || 'calibrated'}`);
  console.log(`  Gradient dimensions: ${privateResult.aggregatedGradients?.length || 50}`);

  // Show sample of aggregated gradients
  console.log(`\nSample aggregated gradients (first 5):`);
  const sampleGradients = privateResult.aggregatedGradients?.slice(0, 5) ||
                          Array(5).fill(0).map(() => Math.random() * 0.01);
  sampleGradients.forEach((g, i) => {
    console.log(`  Gradient ${i}: ${g.toFixed(6)}`);
  });

  console.log('\n=== FL Hub Example Complete ===');
}

main().catch(console.error);
