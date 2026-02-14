/**
 * MATL Engine Correspondence (TypeScript ↔ Rust)
 *
 * This example shows how the TypeScript MATL APIs conceptually match the
 * behaviour of the Rust `MatlEngine::evaluate_node` orchestration:
 *
 *   Rust:   PoGQ + Reputation → Composite → Adaptive Threshold → Anomaly flag
 *   TS:     createPoGQ + createReputation → calculateComposite + adaptive threshold helpers
 */

import {
  createPoGQ,
  createReputation,
  recordPositive,
  recordNegative,
  reputationValue,
  calculateComposite,
  createAdaptiveThreshold,
  observe,
  getThreshold,
  isAnomalous,
} from '../src/matl/index.js';

function demoMatlEngineCorrespondence() {
  console.log('\n=== MATL Engine Correspondence (TS) ===\n');

  // Step 1: pretend we have a PoGQ measurement and a reputation value
  const pogq = createPoGQ(0.9, 0.85, 0.1);

  let rep = createReputation('node-1');
  // Node behaves well most of the time
  for (let i = 0; i < 20; i++) rep = recordPositive(rep);
  // A few bad rounds
  for (let i = 0; i < 3; i++) rep = recordNegative(rep);

  const repValue = reputationValue(rep);
  const composite = calculateComposite(pogq, rep);

  console.log(`Reputation value: ${repValue.toFixed(3)}`);
  console.log(`Composite final score: ${composite.finalScore.toFixed(3)}`);
  console.log(`Composite confidence: ${composite.confidence.toFixed(3)}`);

  // Step 2: approximate the Rust MatlEngine per-node adaptive threshold
  let at = createAdaptiveThreshold('node-1', 20, 0.5, 2.0);

  // Feed historical composite scores (simulating prior rounds)
  const historicalScores = [0.88, 0.9, 0.86, 0.89, 0.87, 0.9, 0.91, 0.88];
  for (const score of historicalScores) {
    at = observe(at, score);
  }

  const threshold = getThreshold(at);
  const anomalous = isAnomalous(at, composite.finalScore);

  console.log(`Adaptive threshold: ${threshold.toFixed(3)}`);
  console.log(`Is latest score anomalous? ${anomalous}`);
}

demoMatlEngineCorrespondence();

