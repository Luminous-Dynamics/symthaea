// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 14: GDPR Compliance with Cryptographic Shredding
 *
 * Demonstrates GDPR-compliant data handling using cryptographic shredding.
 * When a user requests data erasure, we destroy the encryption keys rather
 * than attempting to delete distributed data - making it cryptographically
 * unrecoverable.
 *
 * Key concepts:
 * - Per-subject encryption keys
 * - Key shredding for "right to be forgotten"
 * - Audit trails for compliance proof
 * - Secure key overwriting
 */

import { storage } from '@mycelix/sdk';

const {
  createCryptoShreddingManager,
  executeGDPRErasure,
  ShredReason,
} = storage;

// =============================================================================
// Setup
// =============================================================================

async function main() {
  console.log('=== GDPR Compliance with Cryptographic Shredding ===\n');

  // Create the crypto shredding manager
  const manager = await createCryptoShreddingManager({
    algorithm: 'AES-256-GCM',
    auditRetentionDays: 365 * 7, // 7 years for GDPR
    secureOverwrite: true,
    overwritePasses: 3, // DoD 5220.22-M standard
  });

  // =============================================================================
  // Example 1: Key Generation Per Data Subject
  // =============================================================================

  console.log('1. Generating encryption keys for data subjects...');

  // Each user gets their own encryption key
  const aliceKeyId = await manager.generateKey('user:alice');
  const bobKeyId = await manager.generateKey('user:bob');
  const charlieKeyId = await manager.generateKey('user:charlie');

  console.log(`   Alice's key: ${aliceKeyId}`);
  console.log(`   Bob's key: ${bobKeyId}`);
  console.log(`   Charlie's key: ${charlieKeyId}\n`);

  // =============================================================================
  // Example 2: Encrypting User Data
  // =============================================================================

  console.log('2. Encrypting personal data...');

  // Alice's personal data
  const aliceData = new TextEncoder().encode(JSON.stringify({
    email: 'alice@example.com',
    phone: '+1-555-0100',
    address: '123 Privacy Lane',
    preferences: { marketing: false, analytics: true },
  }));

  const aliceEncrypted = await manager.encrypt(aliceKeyId, aliceData);
  manager.registerResource(aliceKeyId, 'profile:alice');
  manager.registerResource(aliceKeyId, 'preferences:alice');
  manager.registerResource(aliceKeyId, 'analytics:alice');

  console.log(`   Encrypted ${aliceData.length} bytes of Alice's data`);
  console.log(`   IV: ${Buffer.from(aliceEncrypted.iv).toString('hex').slice(0, 16)}...`);
  console.log(`   Resources registered: profile, preferences, analytics\n`);

  // Bob's personal data
  const bobData = new TextEncoder().encode(JSON.stringify({
    email: 'bob@example.com',
    ssn: '123-45-6789', // Sensitive!
    bankAccount: 'IBAN123456789',
  }));

  const bobEncrypted = await manager.encrypt(bobKeyId, bobData);
  manager.registerResource(bobKeyId, 'profile:bob');
  manager.registerResource(bobKeyId, 'financial:bob');

  console.log(`   Encrypted ${bobData.length} bytes of Bob's data`);
  console.log(`   Resources registered: profile, financial\n`);

  // =============================================================================
  // Example 3: Normal Data Access (Decryption)
  // =============================================================================

  console.log('3. Normal data access (decryption)...');

  const aliceDecrypted = await manager.decrypt(
    aliceKeyId,
    aliceEncrypted.ciphertext,
    aliceEncrypted.iv
  );
  const aliceProfile = JSON.parse(new TextDecoder().decode(aliceDecrypted));

  console.log(`   Decrypted Alice's email: ${aliceProfile.email}`);
  console.log(`   Data accessible: YES\n`);

  // =============================================================================
  // Example 4: GDPR Erasure Request (Right to be Forgotten)
  // =============================================================================

  console.log('4. Processing GDPR erasure request for Alice...');

  // Alice requests her data be erased
  const erasureResult = await executeGDPRErasure(
    manager,
    'user:alice',
    'gdpr-processor:legal-team'
  );

  console.log(`   Erasure success: ${erasureResult.success}`);
  console.log(`   Keys shredded: ${erasureResult.results.length}`);
  console.log(`   Affected resources: ${erasureResult.results[0]?.affectedResources || 0}`);
  console.log(`   Audit entry created: ${erasureResult.results[0]?.auditEntry.id}\n`);

  // =============================================================================
  // Example 5: Post-Erasure Access Attempt
  // =============================================================================

  console.log('5. Attempting to access erased data...');

  try {
    await manager.decrypt(
      aliceKeyId,
      aliceEncrypted.ciphertext,
      aliceEncrypted.iv
    );
    console.log('   ERROR: Should not reach here!');
  } catch (error) {
    console.log(`   Access denied: ${(error as Error).message}`);
    console.log(`   Data is CRYPTOGRAPHICALLY UNRECOVERABLE\n`);
  }

  // The ciphertext still exists in storage, but without the key,
  // it's computationally infeasible to decrypt (AES-256)

  // =============================================================================
  // Example 6: Compliance Report
  // =============================================================================

  console.log('6. Generating compliance report for Alice...');

  const report = manager.getComplianceReport('user:alice');

  console.log(`   Data Subject: ${report.dataSubject}`);
  console.log(`   Is Erased: ${report.isErased}`);
  console.log(`   Total Keys: ${report.totalKeys}`);
  console.log(`   Shredded Keys: ${report.shreddedKeys}`);
  console.log(`   Active Keys: ${report.activeKeys}`);
  console.log(`   Affected Resources: ${report.affectedResources.join(', ')}`);
  console.log(`   Audit Trail Entries: ${report.auditTrail.length}\n`);

  // =============================================================================
  // Example 7: Audit Trail for Regulators
  // =============================================================================

  console.log('7. Audit trail (for regulatory compliance)...');

  const auditLog = manager.getFullAuditLog();
  for (const entry of auditLog) {
    console.log(`   [${new Date(entry.shreddedAt).toISOString()}]`);
    console.log(`     Subject: ${entry.dataSubject}`);
    console.log(`     Reason: ${entry.reason}`);
    console.log(`     Performed By: ${entry.performedBy}`);
    console.log(`     Key Hash: ${entry.keyHash.slice(0, 16)}...`);
    console.log(`     Resources: ${entry.affectedResources.join(', ')}`);
  }
  console.log();

  // =============================================================================
  // Example 8: Bob's Data Remains Accessible
  // =============================================================================

  console.log('8. Verifying Bob\'s data is unaffected...');

  const bobDecrypted = await manager.decrypt(
    bobKeyId,
    bobEncrypted.ciphertext,
    bobEncrypted.iv
  );
  const bobProfile = JSON.parse(new TextDecoder().decode(bobDecrypted));

  console.log(`   Bob's email still accessible: ${bobProfile.email}`);
  console.log(`   Subject isolation: VERIFIED\n`);

  // =============================================================================
  // Example 9: Manual Key Shredding (Various Reasons)
  // =============================================================================

  console.log('9. Manual shredding scenarios...');

  // Shred for data retention expiry
  await manager.shredKey(
    bobKeyId,
    ShredReason.DATA_RETENTION_EXPIRED,
    'system:retention-policy'
  );
  console.log('   Bob\'s key shredded: DATA_RETENTION_EXPIRED');

  // Shred for security incident
  await manager.shredKey(
    charlieKeyId,
    ShredReason.SECURITY_INCIDENT,
    'security:incident-response'
  );
  console.log('   Charlie\'s key shredded: SECURITY_INCIDENT');

  console.log();

  // =============================================================================
  // Example 10: Verify All Subjects Erased
  // =============================================================================

  console.log('10. Final status check...');

  for (const subject of ['user:alice', 'user:bob', 'user:charlie']) {
    const isErased = manager.isSubjectErased(subject);
    console.log(`   ${subject}: ${isErased ? 'ERASED' : 'ACTIVE'}`);
  }

  console.log('\n=== GDPR Compliance Example Complete ===');
  console.log('\nKey Takeaways:');
  console.log('- Each data subject has unique encryption keys');
  console.log('- Shredding keys makes data cryptographically unrecoverable');
  console.log('- Distributed data (DHT/IPFS) need not be deleted');
  console.log('- Complete audit trail proves compliance');
  console.log('- Subject isolation ensures other users unaffected');
}

// Run the example
main().catch(console.error);
