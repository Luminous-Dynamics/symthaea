// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 13: Epistemic Storage (UESS)
 *
 * Demonstrates the Unified Epistemic Storage System (UESS) for
 * classification-aware data storage and retrieval.
 *
 * UESS classifies data along three axes:
 * - Empirical (E0-E4): How verifiable is the claim?
 * - Normative (N0-N3): Who should have access?
 * - Materiality (M0-M3): How long should it persist?
 *
 * Classification determines storage backend, mutability, and access control.
 */

import {
  storage,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
} from '@mycelix/sdk';

// =============================================================================
// Setup
// =============================================================================

async function main() {
  console.log('=== UESS Epistemic Storage Example ===\n');

  // Create storage instance
  const store = storage.createEpistemicStorage({
    agentId: 'agent:alice',
    // Optional: custom router configuration
    routerConfig: {
      defaultTTL: 86400000, // 24 hours
    },
  });

  // =============================================================================
  // Example 1: Ephemeral Data (M0 → Memory)
  // =============================================================================

  console.log('1. Storing ephemeral session data (M0 → Memory)...');

  const sessionData = {
    userId: 'alice',
    loginTime: Date.now(),
    preferences: { theme: 'dark', language: 'en' },
  };

  const sessionReceipt = await store.store(
    'session:alice:12345',
    sessionData,
    {
      empirical: EmpiricalLevel.E0_Unverified,  // Self-reported
      normative: NormativeLevel.N0_Personal,    // Only owner access
      materiality: MaterialityLevel.M0_Ephemeral, // In-memory only
    },
    {
      schema: { id: 'session', version: '1.0.0' },
      ttlMs: 3600000, // 1 hour TTL
    }
  );

  console.log(`   Stored in: ${sessionReceipt.backend}`);
  console.log(`   CID: ${sessionReceipt.cid}`);
  console.log(`   Mutable: ${sessionReceipt.mutableMode}\n`);

  // =============================================================================
  // Example 2: Temporal Data (M1 → Local)
  // =============================================================================

  console.log('2. Storing temporal analytics (M1 → Local)...');

  const analyticsData = {
    event: 'page_view',
    page: '/dashboard',
    timestamp: Date.now(),
    metadata: { browser: 'Firefox', os: 'Linux' },
  };

  const analyticsReceipt = await store.store(
    'analytics:alice:page_view:1',
    analyticsData,
    {
      empirical: EmpiricalLevel.E1_Attested,    // System-attested
      normative: NormativeLevel.N1_Communal,    // Shared with trusted group
      materiality: MaterialityLevel.M1_Temporal, // Local storage
    },
    {
      schema: { id: 'analytics-event', version: '2.0.0' },
      ttlMs: 7 * 24 * 3600000, // 7 days
    }
  );

  console.log(`   Stored in: ${analyticsReceipt.backend}`);
  console.log(`   Namespace: local storage\n`);

  // =============================================================================
  // Example 3: Persistent Data (M2 → DHT)
  // =============================================================================

  console.log('3. Storing persistent profile (M2 → DHT)...');

  const profileData = {
    displayName: 'Alice Wonderland',
    bio: 'Exploring the epistemic frontier',
    publicKey: 'pk_alice_12345',
    createdAt: Date.now(),
  };

  const profileReceipt = await store.store(
    'profile:alice',
    profileData,
    {
      empirical: EmpiricalLevel.E3_Cryptographic, // Cryptographically signed
      normative: NormativeLevel.N2_Network,       // Network-wide access
      materiality: MaterialityLevel.M2_Persistent, // DHT replication
    },
    {
      schema: { id: 'user-profile', version: '1.0.0' },
    }
  );

  console.log(`   Stored in: ${profileReceipt.backend}`);
  console.log(`   CID: ${profileReceipt.cid}`);
  console.log(`   Replication: automatic via DHT\n`);

  // =============================================================================
  // Example 4: Immutable Data (M3 → IPFS)
  // =============================================================================

  console.log('4. Storing immutable credential (M3 → IPFS)...');

  const credentialData = {
    type: 'VerifiableCredential',
    issuer: 'did:mycelix:university',
    subject: 'did:mycelix:alice',
    claim: {
      degree: 'PhD',
      field: 'Epistemic Systems',
      graduationDate: '2025-06-15',
    },
    proof: {
      type: 'Ed25519Signature2020',
      signature: 'base64_signature_here',
    },
  };

  const credentialReceipt = await store.store(
    'credential:alice:phd',
    credentialData,
    {
      empirical: EmpiricalLevel.E4_Reproducible,   // Independently reproducible
      normative: NormativeLevel.N2_Network,        // Public credential
      materiality: MaterialityLevel.M3_Immutable,  // IPFS content-addressed
    },
    {
      schema: { id: 'verifiable-credential', version: '1.0.0' },
    }
  );

  console.log(`   Stored in: ${credentialReceipt.backend}`);
  console.log(`   CID: ${credentialReceipt.cid}`);
  console.log(`   Immutable: true (content-addressed)\n`);

  // =============================================================================
  // Example 5: Retrieval
  // =============================================================================

  console.log('5. Retrieving stored data...');

  // Retrieve session (owner access)
  const session = await store.retrieve<typeof sessionData>('session:alice:12345');
  if (session) {
    console.log(`   Session: ${session.data.userId} logged in`);
  }

  // Retrieve profile (public access)
  const profile = await store.retrieve<typeof profileData>('profile:alice');
  if (profile) {
    console.log(`   Profile: ${profile.data.displayName}`);
    console.log(`   Classification: ${profile.metadata.classification}`);
  }

  console.log();

  // =============================================================================
  // Example 6: Update (Mutable Data Only)
  // =============================================================================

  console.log('6. Updating mutable data...');

  // Update session preferences (M0 is mutable)
  const updatedSession = await store.update(
    'session:alice:12345',
    { ...sessionData, preferences: { ...sessionData.preferences, theme: 'light' } }
  );
  console.log(`   Session updated: theme changed to light`);

  // Note: M3 data cannot be updated (immutable)
  try {
    await store.update('credential:alice:phd', { ...credentialData });
  } catch (error) {
    if (error instanceof storage.ImmutableError) {
      console.log(`   Credential update rejected: ${error.message}`);
    }
  }

  console.log();

  // =============================================================================
  // Example 7: Query by Classification
  // =============================================================================

  console.log('7. Querying by classification...');

  // Find all persistent data
  const persistentData = await store.query({
    materiality: { min: MaterialityLevel.M2_Persistent },
  });
  console.log(`   Found ${persistentData.items.length} persistent items`);

  // Find all cryptographically verified claims
  const verifiedClaims = await store.query({
    empirical: { min: EmpiricalLevel.E3_Cryptographic },
  });
  console.log(`   Found ${verifiedClaims.items.length} cryptographically verified items`);

  console.log();

  // =============================================================================
  // Example 8: Verification
  // =============================================================================

  console.log('8. Verifying data integrity...');

  const verification = await store.verify('credential:alice:phd');
  console.log(`   Credential integrity: ${verification.valid ? 'VALID' : 'INVALID'}`);
  console.log(`   CID matches: ${verification.cidMatches}`);
  console.log(`   Classification: ${verification.classification}`);

  console.log();

  // =============================================================================
  // Example 9: Deletion
  // =============================================================================

  console.log('9. Deleting data...');

  // Delete ephemeral session
  const deleted = await store.delete('session:alice:12345');
  console.log(`   Session deleted: ${deleted}`);

  // Note: M3 data uses tombstones (data remains, marked deleted)
  const tombstoned = await store.delete('credential:alice:phd');
  console.log(`   Credential tombstoned: ${tombstoned} (data preserved with tombstone)`);

  console.log();

  // =============================================================================
  // Example 10: Statistics
  // =============================================================================

  console.log('10. Storage statistics...');

  const stats = await store.stats();
  console.log(`   Total items: ${stats.totalItems}`);
  console.log(`   Total size: ${stats.totalSizeBytes} bytes`);
  console.log(`   By backend:`);
  for (const [backend, count] of Object.entries(stats.byBackend)) {
    console.log(`     - ${backend}: ${count} items`);
  }

  console.log('\n=== Example Complete ===');
}

// Run the example
main().catch(console.error);
