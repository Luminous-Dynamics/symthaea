// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic Claims Example
 *
 * Demonstrates the 3D truth classification system from the Epistemic Charter v2.0.
 * Claims are classified along three dimensions:
 * - Empirical: How verifiable is the claim?
 * - Normative: How broadly accepted?
 * - Materiality: How persistent/important?
 *
 * Run with: npx ts-node examples/04-epistemic-claims.ts
 */

import {
  // Claim builders
  claim,
  createHighTrustClaim,
  createMediumTrustClaim,
  createLowTrustClaim,

  // Levels
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,

  // Utilities
  classificationCode,
  meetsMinimum,
  meetsStandard,
  isExpired,
  Standards,
} from '../src/index.js';

async function main() {
  console.log('🍄 Epistemic Claims Example\n');

  // Explain the classification system
  console.log('=== Classification Dimensions ===');
  console.log('\nEmpirical Levels (How verifiable?):');
  console.log('  E0: Unfalsifiable (opinions, beliefs)');
  console.log('  E1: Testimonial (witness accounts)');
  console.log('  E2: Privately Verifiable (can be checked with access)');
  console.log('  E3: Cryptographic (mathematically provable)');

  console.log('\nNormative Levels (How broadly accepted?):');
  console.log('  N0: Personal (individual belief)');
  console.log('  N1: Communal (group consensus)');
  console.log('  N2: Network (ecosystem-wide)');
  console.log('  N3: Universal (cross-network)');

  console.log('\nMateriality Levels (How persistent?):');
  console.log('  M0: Ephemeral (temporary, can be withdrawn)');
  console.log('  M1: Temporal (time-limited validity)');
  console.log('  M2: Persistent (long-term record)');
  console.log('  M3: Foundational (immutable, consensus-critical)');
  console.log();

  // Create claims using utility functions
  console.log('=== Quick Claim Creation ===');

  const highTrust = createHighTrustClaim(
    'Agent passed KYC verification',
    'verification-service'
  );
  console.log('High Trust Claim (E3-N2-M2):');
  console.log(`  Content: ${highTrust.content}`);
  console.log(`  Code: ${classificationCode(highTrust.classification)}`);
  console.log(`  Empirical: Cryptographic (E3)`);
  console.log(`  Normative: Network-wide (N2)`);
  console.log(`  Materiality: Persistent (M2)`);
  console.log();

  const mediumTrust = createMediumTrustClaim(
    'Project milestone completed',
    'project-coordinator'
  );
  console.log('Medium Trust Claim (E2-N1-M1):');
  console.log(`  Content: ${mediumTrust.content}`);
  console.log(`  Code: ${classificationCode(mediumTrust.classification)}`);
  console.log();

  const lowTrust = createLowTrustClaim(
    'I believe this approach will work',
    'community-member'
  );
  console.log('Low Trust Claim (E1-N0-M0):');
  console.log(`  Content: ${lowTrust.content}`);
  console.log(`  Code: ${classificationCode(lowTrust.classification)}`);
  console.log();

  // Create claims using fluent builder
  console.log('=== Fluent Builder API ===');

  const identityClaim = claim('Alice owns public key 0x1234...')
    .withClassification(
      EmpiricalLevel.E3_Cryptographic, // Cryptographic proof
      NormativeLevel.N3_Universal, // Universally verifiable
      MaterialityLevel.M3_Foundational // Foundational identity
    )
    .withIssuer('identity-registry')
    .withExpiration(new Date(Date.now() + 365 * 24 * 60 * 60 * 1000)) // 1 year
    .build();

  console.log('Identity Claim:');
  console.log(`  ID: ${identityClaim.id}`);
  console.log(`  Code: ${classificationCode(identityClaim.classification)}`);
  console.log(`  Issuer: ${identityClaim.issuer}`);
  console.log(`  Expires: ${identityClaim.expiresAt?.toISOString()}`);
  console.log();

  // Build claim with evidence
  const reputationClaim = claim('Alice has completed 100 successful trades')
    .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
    .withNormative(NormativeLevel.N2_Network)
    .withMateriality(MaterialityLevel.M2_Persistent)
    .withIssuer('marketplace-happ')
    .withEvidence({
      type: 'statistical',
      description: 'Trade completion records from on-chain data',
    })
    .withEvidence({
      type: 'attestation',
      description: 'Verified by 3 independent validators',
    })
    .build();

  console.log('Reputation Claim with Evidence:');
  console.log(`  Content: ${reputationClaim.content}`);
  console.log(`  Evidence items: ${reputationClaim.evidence.length}`);
  for (const ev of reputationClaim.evidence) {
    console.log(`    - ${ev.type}: ${ev.description}`);
  }
  console.log();

  // Validate against standards
  console.log('=== Standard Validation ===');

  const claimsToValidate = [
    { name: 'High Trust', claim: highTrust },
    { name: 'Medium Trust', claim: mediumTrust },
    { name: 'Low Trust', claim: lowTrust },
    { name: 'Identity', claim: identityClaim },
    { name: 'Reputation', claim: reputationClaim },
  ];

  console.log('Checking claims against predefined standards:\n');

  const standards = [
    { name: 'MINIMAL', standard: Standards.MINIMAL },
    { name: 'HIGH', standard: Standards.HIGH },
    { name: 'NETWORK', standard: Standards.NETWORK },
    { name: 'FOUNDATIONAL', standard: Standards.FOUNDATIONAL },
  ];

  // Header
  console.log(
    'Claim'.padEnd(15) +
      standards.map((s) => s.name.padStart(14)).join('')
  );
  console.log('-'.repeat(15 + standards.length * 14));

  for (const { name, claim: c } of claimsToValidate) {
    let row = name.padEnd(15);
    for (const { standard } of standards) {
      const meets = meetsStandard(c, standard);
      row += (meets ? '✅' : '❌').padStart(14);
    }
    console.log(row);
  }
  console.log();

  // Check specific minimum requirements
  console.log('=== Minimum Level Checks ===');

  const minCrypto = {
    empirical: EmpiricalLevel.E3_Cryptographic,
    normative: NormativeLevel.N0_Personal,
    materiality: MaterialityLevel.M0_Ephemeral,
  };

  const minNetwork = {
    empirical: EmpiricalLevel.E1_Testimonial,
    normative: NormativeLevel.N2_Network,
    materiality: MaterialityLevel.M0_Ephemeral,
  };

  console.log('Checking if claims meet minimum requirements:');
  console.log('\nMinimum: Cryptographic proof (E3):');
  for (const { name, claim: c } of claimsToValidate) {
    const meets = meetsMinimum(c.classification, minCrypto);
    console.log(`  ${name}: ${meets ? '✅' : '❌'}`);
  }

  console.log('\nMinimum: Network acceptance (N2):');
  for (const { name, claim: c } of claimsToValidate) {
    const meets = meetsMinimum(c.classification, minNetwork);
    console.log(`  ${name}: ${meets ? '✅' : '❌'}`);
  }
  console.log();

  // Check expiration
  console.log('=== Expiration Handling ===');

  // Create an expired claim
  const expiredClaim = claim('Temporary access token')
    .withClassification(
      EmpiricalLevel.E2_PrivateVerify,
      NormativeLevel.N1_Communal,
      MaterialityLevel.M0_Ephemeral
    )
    .withIssuer('auth-service')
    .withExpiration(new Date(Date.now() - 1000)) // Already expired
    .build();

  console.log('Checking claim expiration:');
  console.log(`  Identity claim expired: ${isExpired(identityClaim)}`);
  console.log(`  Expired claim expired: ${isExpired(expiredClaim)}`);
  console.log(`  High trust (no expiry) expired: ${isExpired(highTrust)}`);
  console.log();

  // Real-world use case
  console.log('=== Use Case: Trust Decision ===');

  function shouldTrustClaim(c: typeof highTrust, purpose: string): boolean {
    // Different purposes require different standards
    switch (purpose) {
      case 'financial':
        return meetsStandard(c, Standards.FOUNDATIONAL) && !isExpired(c);
      case 'governance':
        return meetsStandard(c, Standards.NETWORK) && !isExpired(c);
      case 'social':
        return meetsStandard(c, Standards.MINIMAL) && !isExpired(c);
      default:
        return meetsStandard(c, Standards.HIGH);
    }
  }

  console.log('Trust decisions for different purposes:');
  const purposes = ['financial', 'governance', 'social'];

  for (const { name, claim: c } of claimsToValidate) {
    console.log(`\n  ${name}:`);
    for (const purpose of purposes) {
      const trusted = shouldTrustClaim(c, purpose);
      console.log(`    ${purpose}: ${trusted ? '✅' : '❌'}`);
    }
  }
}

main().catch(console.error);
