// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Integration Examples for Mycelix Innovation Modules
 *
 * These examples provide conceptual demonstrations of how the 6 innovation
 * modules work together to create a comprehensive civic AI infrastructure.
 *
 * Examples:
 * 1. Trust Market + Calibration - Self-correcting belief revelation
 * 2. Epistemic Agents + Civic Feedback - Claims becoming precedents
 * 3. Workflows + Constitutional AI - Governed workflow execution
 * 4. Private Queries + Analytics - Privacy-preserving cross-hApp queries
 * 5. Comprehensive Scenario - All 6 modules in a housing voucher application
 *
 * Note: These examples demonstrate integration patterns and may need adaptation
 * for specific use cases and actual API implementations.
 *
 * @module innovations/examples
 */

// Re-export individual examples
export { trustMarketCalibrationExample } from './trust-market-calibration.js';
export { epistemicCivicIntegrationExample } from './epistemic-civic-integration.js';
export { workflowConstitutionalExample } from './workflow-constitutional-integration.js';
export { privateQueryAnalyticsExample } from './private-query-analytics.js';
export { comprehensiveCivicScenario } from './comprehensive-civic-scenario.js';

/**
 * Run all integration examples sequentially
 */
export async function runAllExamples(): Promise<void> {
  const examples = [
    { name: 'Trust Market + Calibration', fn: () => import('./trust-market-calibration.js').then(m => m.trustMarketCalibrationExample()) },
    { name: 'Epistemic + Civic Feedback', fn: () => import('./epistemic-civic-integration.js').then(m => m.epistemicCivicIntegrationExample()) },
    { name: 'Workflows + Constitutional', fn: () => import('./workflow-constitutional-integration.js').then(m => m.workflowConstitutionalExample()) },
    { name: 'Private Queries + Analytics', fn: () => import('./private-query-analytics.js').then(m => m.privateQueryAnalyticsExample()) },
    { name: 'Comprehensive Civic Scenario', fn: () => import('./comprehensive-civic-scenario.js').then(m => m.comprehensiveCivicScenario()) },
  ];

  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║     MYCELIX INNOVATION MODULES - INTEGRATION EXAMPLES        ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  for (const example of examples) {
    console.log(`\n${'='.repeat(66)}`);
    console.log(`RUNNING: ${example.name}`);
    console.log(`${'='.repeat(66)}\n`);

    try {
      await example.fn();
      console.log(`\n✓ ${example.name} completed successfully\n`);
    } catch (error) {
      console.error(`\n✗ ${example.name} failed:`, error);
    }
  }

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║               ALL EXAMPLES COMPLETED                          ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}
