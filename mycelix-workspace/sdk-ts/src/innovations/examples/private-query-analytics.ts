// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Private Queries + Cross-hApp Analytics Integration Example
 *
 * Demonstrates how private queries enable privacy-preserving analytics
 * across multiple hApps without exposing individual data points.
 *
 * @module innovations/examples/private-query-analytics
 */

import { PrivateQueryService } from '../private-queries/index.js';

/**
 * Example: Privacy-Preserving Cross-hApp Eligibility Check
 */
export async function privateQueryAnalyticsExample() {
  console.log('=== Private Queries + Cross-hApp Analytics ===\n');

  // Initialize private query service with privacy budget
  const queryService = new PrivateQueryService({
    epsilon: 1.0,
    delta: 1e-7,
    accountingMethod: 'rdp',
  });
  await queryService.initialize();

  console.log('✓ Private Query Service initialized with FHE\n');

  console.log('Step 1: Privacy-Preserving Query Concepts\n');
  console.log('The Private Query Service enables:');
  console.log('  • FHE-encrypted computations on encrypted data');
  console.log('  • Differential privacy to bound information leakage');
  console.log('  • Multi-party secure aggregation');
  console.log('  • Threshold decryption requiring consensus\n');

  console.log('Step 2: Cross-hApp Aggregate Query Example\n');
  console.log('Query: "Count eligible employees across hApps"');
  console.log('  - Employment hApp: salary >= $40k');
  console.log('  - Health hApp: has chronic condition');
  console.log('  - Benefits hApp: enrolled in plan\n');

  console.log('Privacy guarantees:');
  console.log('  • No individual data points revealed');
  console.log('  • Only aggregate counts returned');
  console.log('  • Differential privacy noise added');
  console.log('  • Privacy budget (ε=1.0) tracks exposure\n');

  console.log('Step 3: Private Set Intersection (PSI)\n');
  console.log('PSI enables finding common members across datasets');
  console.log('without revealing the full sets to any party.\n');
  console.log('Example: Find people in both employment AND health datasets');
  console.log('Result: Intersection size only (e.g., "42 common members")\n');

  console.log('Step 4: Threshold Decryption\n');
  console.log('Sensitive aggregates require multi-party consent:');
  console.log('  • 5 data custodians hold key shares');
  console.log('  • 3 of 5 must approve to decrypt');
  console.log('  • No single party can access raw data\n');

  console.log('=== Key Insight ===');
  console.log('Private queries enable cross-hApp analytics while ensuring:');
  console.log('  1. No individual data points revealed');
  console.log('  2. Differential privacy bounds leakage');
  console.log('  3. PSI finds commonalities privately');
  console.log('  4. Threshold decryption requires consensus');
  console.log('  5. Complete audit trail for compliance\n');

  return { serviceInitialized: true };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  privateQueryAnalyticsExample().catch(console.error);
}
