// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 1: Basic Event Ingestion
 *
 * This example demonstrates:
 * - SDK initialization
 * - Creating a PRODUCED event
 * - Submitting a single event
 * - Handling the response
 */

import { SupplyChainClient } from '../sdk/src';

async function basicIngestion() {
  console.log('=== Basic Event Ingestion Example ===\n');

  // Initialize the SDK client
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  console.log('📦 Creating a PRODUCED event...');

  // Create a PRODUCED event using the helper method
  const event = client.createProducedEvent(
    'BATCH-2025-001',                    // batch_id
    'ORG-ACME',                          // facility_id
    'Organic Coffee Beans',              // product_id
    1000,                                // quantity
    'kg',                                // unit
    {                                    // metadata (optional)
      origin: 'Colombia',
      grade: 'Premium',
      harvest_date: '2025-11-01',
      certification: 'USDA Organic'
    }
  );

  console.log('✓ Event created');
  console.log(`  Batch ID: ${event.batch_id}`);
  console.log(`  Product: ${event.product_id}`);
  console.log(`  Quantity: ${event.quantity} ${event.unit}\n`);

  // Submit the event to the API
  console.log('🚀 Submitting event to API...');

  try {
    const response = await client.ingestEvent(event);

    console.log('✅ Event ingested successfully!\n');
    console.log('Response:');
    console.log(`  Claim ID: ${response.claim_id}`);
    console.log(`  VC JWT: ${response.vc_jwt.substring(0, 50)}...`);
    console.log(`  Lineage Hash: ${response.lineage_hash}`);
    console.log(`  Verified: ${response.verified}`);

  } catch (error) {
    console.error('❌ Failed to ingest event:', error);
    throw error;
  }
}

// Run the example
if (require.main === module) {
  basicIngestion().catch((error) => {
    console.error('Example failed:', error);
    process.exit(1);
  });
}

export { basicIngestion };
