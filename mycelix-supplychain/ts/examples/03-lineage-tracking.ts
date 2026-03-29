// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 3: Supply Chain Lineage Tracking
 *
 * This example demonstrates:
 * - Creating a multi-stage supply chain
 * - Tracing complete lineage from source to final product
 * - Upstream/downstream relationship tracking
 * - Lineage graph visualization
 */

import { SupplyChainClient } from '../sdk/src';

async function lineageTracking() {
  console.log('=== Supply Chain Lineage Tracking Example ===\n');

  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Create a complete supply chain: Raw Material → Processing → Packaging → Distribution

  console.log('🌱 Step 1: Raw material production...');

  // 1. Raw material production (cacao beans)
  const rawMaterial = client.createProducedEvent(
    'BATCH-RAW-CACAO-001',
    'ORG-FARM',
    'Organic Cacao Beans',
    1000,
    'kg',
    {
      farm: 'Peru Highlands Cooperative',
      certification: 'USDA Organic',
      harvest_date: '2025-10-15',
      fermentation_days: 7,
      drying_method: 'solar',
    }
  );

  console.log('  ✓ Created raw material event');

  // 2. Processing/transformation (chocolate production)
  console.log('\n⚙️  Step 2: Processing into chocolate...');

  const processed = client.createTransformedEvent(
    'BATCH-PROC-CHOC-001',
    'ORG-FACTORY',
    'Dark Chocolate Bars',
    ['BATCH-RAW-CACAO-001'],
    800,
    'kg',
    {
      cocoa_content: '85%',
      process: 'stone-ground',
      conching_hours: 72,
      tempering_temp: '31C',
    }
  );

  console.log('  ✓ Created processing event');

  // 3. Packaging (retail units)
  console.log('\n📦 Step 3: Packaging for retail...');

  const packaged = client.createTransformedEvent(
    'BATCH-PKG-RETAIL-002',
    'ORG-PACKAGER',
    'Retail Chocolate Bars (100g)',
    ['BATCH-PROC-CHOC-001'],
    8000,
    'units',
    {
      packaging: 'recyclable',
      weight_per_unit: '100g',
      boxes_packed: 200,
    }
  );

  console.log('  ✓ Created packaging event');

  // 4. Certification
  console.log('\n🏆 Step 4: Quality certification...');

  const certified = client.createCertifiedEvent(
    'BATCH-PKG-RETAIL-002',
    'ORG-CERTIFIER',
    'Retail Chocolate Bars (100g)',
    8000,
    'units',
    {
      certification_type: 'Organic + Fair Trade',
      cert_number: 'ORG-FT-2025-789',
      certifier: 'International Certification Services',
      audit_date: '2025-11-10',
      expires: '2026-11-10',
    }
  );

  console.log('  ✓ Created certification event');

  // 5. Shipment
  console.log('\n🚚 Step 5: Shipping to distributor...');

  const shipped = client.createShippedEvent(
    'BATCH-PKG-RETAIL-002',
    'ORG-PACKAGER',
    'Retail Chocolate Bars (100g)',
    8000,
    'units',
    {
      destination: 'DIST-US-WEST',
      carrier: 'ColdChain Express',
      tracking: 'CCE-2025-55443',
      departure_date: new Date().toISOString(),
      expected_arrival: '2025-11-20',
      temperature_controlled: true,
    }
  );

  console.log('  ✓ Created shipment event\n');

  // Submit all events as atomic batch
  console.log('🚀 Submitting complete supply chain as atomic batch...');

  const events = [rawMaterial, processed, packaged, certified, shipped];
  const batch = client.createBatch(events, 'atomic');

  try {
    const batchResponse = await client.ingestBatch(batch);

    console.log(`\n✅ Supply chain events ingested successfully!`);
    console.log(`   Total: ${batchResponse.total}`);
    console.log(`   Succeeded: ${batchResponse.succeeded}`);
    console.log(`   Duration: ${batchResponse.duration_ms}ms\n`);

    // Query lineage for final product
    console.log('═══════════════════════════════════════════════════════');
    console.log('🔍 Querying lineage for final product...\n');

    const lineage = await client.getLineage('BATCH-PKG-RETAIL-002');

    console.log('📊 Lineage Graph Summary:');
    console.log(`  Batch ID: ${lineage.batch_id}`);
    console.log(`  Total Claims: ${lineage.total_claims}`);
    console.log(`  Graph Depth: ${lineage.depth}`);

    // Show upstream sources
    if (lineage.upstream && lineage.upstream.length > 0) {
      console.log(`\n⬆️  Upstream Sources (${lineage.upstream.length}):`);
      lineage.upstream.forEach((batch) => {
        console.log(`  • ${batch.batch_id}`);
        console.log(`    - Claims: ${batch.claim_count}`);
        console.log(`    - Depth: ${batch.depth}`);
      });
    }

    // Show downstream derivatives (if any)
    if (lineage.downstream && lineage.downstream.length > 0) {
      console.log(`\n⬇️  Downstream Products (${lineage.downstream.length}):`);
      lineage.downstream.forEach((batch) => {
        console.log(`  • ${batch.batch_id}`);
        console.log(`    - Claims: ${batch.claim_count}`);
        console.log(`    - Depth: ${batch.depth}`);
      });
    }

    // Show complete supply chain history
    console.log('\n📋 Complete Supply Chain History (Chronological):');
    console.log('═══════════════════════════════════════════════════════\n');

    lineage.claims.forEach((claim, index) => {
      const event = claim.event;
      const timestamp = new Date(event.timestamp).toLocaleString();

      console.log(`${index + 1}. ${event.event_type} Event`);
      console.log(`   ├─ Batch: ${event.batch_id}`);
      console.log(`   ├─ Product: ${event.product_id}`);
      console.log(`   ├─ Facility: ${event.facility_id}`);
      console.log(`   ├─ Quantity: ${event.quantity} ${event.unit}`);
      console.log(`   ├─ Timestamp: ${timestamp}`);

      if (event.input_batches && event.input_batches.length > 0) {
        console.log(`   ├─ Input Batches: ${event.input_batches.join(', ')}`);
      }

      if (event.metadata) {
        const metadata = JSON.parse(event.metadata);
        const keys = Object.keys(metadata);
        if (keys.length > 0) {
          console.log(`   ├─ Metadata:`);
          keys.slice(0, 3).forEach((key) => {
            console.log(`   │  • ${key}: ${metadata[key]}`);
          });
          if (keys.length > 3) {
            console.log(`   │  • ... (${keys.length - 3} more fields)`);
          }
        }
      }

      console.log(`   └─ Claim ID: ${claim.claim_id}\n`);
    });

    // Demonstrate traceability
    console.log('═══════════════════════════════════════════════════════');
    console.log('🎯 Traceability Demonstration:\n');

    const finalProduct = lineage.claims.find((c) => c.event.batch_id === 'BATCH-PKG-RETAIL-002');
    const sourceProduct = lineage.claims.find((c) => c.event.batch_id === 'BATCH-RAW-CACAO-001');

    if (finalProduct && sourceProduct) {
      console.log('From Farm to Consumer:');
      console.log(`  🌱 Source: ${sourceProduct.event.product_id}`);
      console.log(`     • Farm: ${JSON.parse(sourceProduct.event.metadata).farm}`);
      console.log(`     • Certification: ${JSON.parse(sourceProduct.event.metadata).certification}`);
      console.log(`     • Harvest: ${JSON.parse(sourceProduct.event.metadata).harvest_date}`);
      console.log('');
      console.log(`  📦 Final Product: ${finalProduct.event.product_id}`);
      console.log(`     • Units: ${finalProduct.event.quantity}`);
      console.log(`     • Package: ${JSON.parse(finalProduct.event.metadata).packaging}`);
      console.log('');
      console.log(`  ✅ Complete provenance verified!`);
      console.log(`  ✅ ${lineage.total_claims} events in supply chain`);
      console.log(`  ✅ ${lineage.depth} levels of transformation`);
    }

  } catch (error) {
    console.error('\n❌ Lineage tracking failed:', error);
    throw error;
  }

  console.log('\n═══════════════════════════════════════════════════════');
  console.log('✅ Lineage tracking example completed!');
}

// Run the example
if (require.main === module) {
  lineageTracking().catch((error) => {
    console.error('Example failed:', error);
    process.exit(1);
  });
}

export { lineageTracking };
