// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 2: Batch Processing
 *
 * This example demonstrates:
 * - High-volume event ingestion
 * - Batch creation with multiple events
 * - Best-effort vs atomic processing modes
 * - Error handling and recovery
 * - Result processing
 */

import { SupplyChainClient } from '../sdk/src';

async function batchProcessing() {
  console.log('=== Batch Processing Example ===\n');

  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Step 1: Create multiple events for a production run
  console.log('📦 Creating batch of events for production run...\n');

  const events = [
    // First: Produce dark roast coffee
    client.createProducedEvent(
      'BATCH-COFFEE-DARK-001',
      'ORG-ROASTER',
      'Dark Roast Coffee',
      500,
      'kg',
      { roast_level: 'dark', temperature: '220C', duration: '15min' }
    ),

    // Second: Produce medium roast coffee
    client.createProducedEvent(
      'BATCH-COFFEE-MEDIUM-001',
      'ORG-ROASTER',
      'Medium Roast Coffee',
      500,
      'kg',
      { roast_level: 'medium', temperature: '210C', duration: '12min' }
    ),

    // Third: Create a blend from both roasts
    client.createTransformedEvent(
      'BATCH-BLEND-HOUSE-001',
      'ORG-ROASTER',
      'House Blend Coffee',
      ['BATCH-COFFEE-DARK-001', 'BATCH-COFFEE-MEDIUM-001'],
      800,
      'kg',
      { blend_ratio: '50:50', quality_grade: 'Premium' }
    ),

    // Fourth: Package the blend
    client.createTransformedEvent(
      'BATCH-PKG-RETAIL-001',
      'ORG-PACKAGER',
      'Retail Coffee Packages (250g)',
      ['BATCH-BLEND-HOUSE-001'],
      3200,
      'units',
      { package_size: '250g', package_type: 'recyclable' }
    ),
  ];

  console.log(`✓ Created ${events.length} events\n`);

  // Step 2: Submit as batch (best-effort mode)
  console.log('🚀 Submitting batch (best-effort mode)...');
  console.log('   This mode processes all valid events, allowing partial success\n');

  try {
    const batch = client.createBatch(events, 'best-effort');
    const response = await client.ingestBatch(batch);

    // Process results
    console.log('📊 Batch Processing Results:');
    console.log(`  Total events: ${response.total}`);
    console.log(`  Succeeded: ${response.succeeded} ✅`);
    console.log(`  Failed: ${response.failed} ❌`);
    console.log(`  Duration: ${response.duration_ms}ms\n`);

    // Check individual results
    console.log('Individual Results:');
    response.results.forEach((result, index) => {
      if (result.status === 'success') {
        console.log(`  ✅ Event ${index + 1}: ${result.claim_id}`);
        console.log(`     Lineage hash: ${result.lineage_hash}`);
      } else {
        console.error(`  ❌ Event ${index + 1}: ${result.error}`);
      }
    });

    // Step 3: Retry failed events (if any)
    const failedEvents = response.results
      .filter((r) => r.status === 'error')
      .map((r) => events[r.index]);

    if (failedEvents.length > 0) {
      console.log(`\n🔄 Retrying ${failedEvents.length} failed events...\n`);

      // Use atomic mode for retry to ensure all succeed together
      const retryBatch = client.createBatch(failedEvents, 'atomic');
      const retryResponse = await client.ingestBatch(retryBatch);

      console.log('Retry Results:');
      console.log(`  Succeeded: ${retryResponse.succeeded}`);
      console.log(`  Failed: ${retryResponse.failed}`);

      if (retryResponse.failed > 0) {
        console.error('  ⚠️  Some events still failing after retry');
      } else {
        console.log('  ✅ All retry events succeeded!');
      }
    } else {
      console.log('\n✅ All events succeeded on first attempt!');
    }

    // Step 4: Demonstrate atomic mode
    console.log('\n--- Atomic Mode Example ---');
    console.log('Creating a new batch with atomic processing...\n');

    const atomicEvents = [
      client.createCertifiedEvent(
        'BATCH-BLEND-HOUSE-001',
        'ORG-CERTIFIER',
        'House Blend Coffee',
        800,
        'kg',
        {
          certification_type: 'Fair Trade',
          cert_number: 'FT-2025-12345',
          certifier: 'Fair Trade International',
          expires: '2026-11-16',
        }
      ),
      client.createShippedEvent(
        'BATCH-PKG-RETAIL-001',
        'ORG-PACKAGER',
        'Retail Coffee Packages (250g)',
        3200,
        'units',
        {
          destination: 'DIST-REGIONAL',
          carrier: 'GreenTransport LLC',
          tracking: 'GT-2025-98765',
          departure_date: new Date().toISOString(),
        }
      ),
    ];

    const atomicBatch = client.createBatch(atomicEvents, 'atomic');
    const atomicResponse = await client.ingestBatch(atomicBatch);

    console.log('Atomic Batch Results:');
    console.log(`  All events: ${atomicResponse.succeeded === atomicResponse.total ? 'SUCCEEDED ✅' : 'FAILED ❌'}`);
    console.log(`  Duration: ${atomicResponse.duration_ms}ms`);

  } catch (error) {
    console.error('\n❌ Batch processing failed:', error);
    throw error;
  }

  console.log('\n✅ Batch processing example completed!');
}

// Run the example
if (require.main === module) {
  batchProcessing().catch((error) => {
    console.error('Example failed:', error);
    process.exit(1);
  });
}

export { batchProcessing };
