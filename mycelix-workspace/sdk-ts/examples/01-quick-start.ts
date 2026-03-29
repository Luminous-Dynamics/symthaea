// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Quick Start Example
 *
 * This example shows the basics of using the Mycelix SDK for trust assessment.
 *
 * Run with: npx ts-node examples/01-quick-start.ts
 */

import {
  // Trust utilities
  checkTrust,
  buildReputation,
  runHealthChecks,
  getSdkInfo,

  // MATL primitives
  createPoGQ,
  compositeScore,
  isByzantine,
} from '../src/index.js';

async function main() {
  console.log('🍄 Mycelix SDK Quick Start\n');

  // Check SDK health
  console.log('=== SDK Health Check ===');
  const info = getSdkInfo();
  console.log(`Version: ${info.version}`);
  console.log(`Modules: ${info.modules.join(', ')}`);

  const healthResults = runHealthChecks();
  for (const result of healthResults) {
    const status = result.healthy ? '✅' : '❌';
    console.log(`${status} ${result.component}`);
  }
  console.log();

  // Quick trust assessment
  console.log('=== Quick Trust Assessment ===');

  // Agent with high quality metrics
  const trustyAgent = checkTrust(
    'alice',
    0.92, // quality
    0.88, // consistency
    0.15 // entropy (low is good)
  );

  console.log('Alice (high quality):');
  console.log(`  Trustworthy: ${trustyAgent.trustworthy}`);
  console.log(`  Score: ${trustyAgent.score.toFixed(3)}`);
  console.log(`  Byzantine: ${trustyAgent.byzantine}`);
  console.log();

  // Agent with suspicious metrics
  const suspiciousAgent = checkTrust(
    'mallory',
    0.25, // low quality
    0.30, // low consistency
    0.85 // high entropy (suspicious)
  );

  console.log('Mallory (suspicious):');
  console.log(`  Trustworthy: ${suspiciousAgent.trustworthy}`);
  console.log(`  Score: ${suspiciousAgent.score.toFixed(3)}`);
  console.log(`  Byzantine: ${suspiciousAgent.byzantine}`);
  console.log();

  // Build reputation over time
  console.log('=== Building Reputation ===');
  const reputation = buildReputation('bob', 15, 2); // 15 positive, 2 negative
  console.log(`Bob's reputation after 17 interactions:`);
  console.log(`  Positive: ${reputation.positiveCount}`);
  console.log(`  Negative: ${reputation.negativeCount}`);

  // Use existing reputation in trust check
  const bobTrust = checkTrust('bob', 0.75, 0.80, 0.20, {
    existingReputation: reputation,
  });

  console.log(`  Combined trust score: ${bobTrust.score.toFixed(3)}`);
  console.log(`  Reputation contribution: ${bobTrust.details.reputationScore.toFixed(3)}`);
  console.log();

  // Low-level MATL operations
  console.log('=== Low-Level MATL ===');
  const pogq = createPoGQ(0.85, 0.90, 0.12);
  console.log(`PoGQ: quality=${pogq.quality}, consistency=${pogq.consistency}, entropy=${pogq.entropy}`);

  const score = compositeScore(pogq, 0.7);
  console.log(`Composite score (with 0.7 reputation): ${score.toFixed(3)}`);

  const byzantine = isByzantine(pogq, 0.5);
  console.log(`Is Byzantine (threshold 0.5): ${byzantine}`);
}

main().catch(console.error);
