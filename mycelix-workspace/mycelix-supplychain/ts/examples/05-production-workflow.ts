// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 5: Complete Production Workflow
 *
 * This example demonstrates:
 * - End-to-end factory production workflow
 * - Multi-step process orchestration
 * - Error handling and recovery
 * - Audit trail generation
 * - Lineage verification
 */

import { SupplyChainClient } from '../sdk/src';

async function productionWorkflow() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🏭 Complete Production Workflow Example');
  console.log('═══════════════════════════════════════════════════════\n');

  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  try {
    // =====================================================================
    // STEP 1: Receive Raw Materials
    // =====================================================================
    console.log('📥 STEP 1: Receiving Raw Materials\n');

    const receivedMaterials = [
      client.createReceivedEvent(
        'BATCH-RM-WHEAT-001',
        'FAC-MILL',
        'Organic Wheat',
        5000,
        'kg',
        {
          source: 'Local Farm Co-op',
          quality_grade: 'A',
          moisture_content: '12%',
          protein_content: '14%',
          received_date: new Date().toISOString(),
        }
      ),
      client.createReceivedEvent(
        'BATCH-RM-SALT-001',
        'FAC-MILL',
        'Sea Salt',
        100,
        'kg',
        {
          source: 'Atlantic Salt Co.',
          purity: '99.9%',
          grain_size: 'fine',
        }
      ),
      client.createReceivedEvent(
        'BATCH-RM-YEAST-001',
        'FAC-MILL',
        'Active Dry Yeast',
        50,
        'kg',
        {
          source: 'BioYeast Industries',
          strain: 'S. cerevisiae',
          activity_level: 'high',
        }
      ),
    ];

    console.log(`   Creating batch of ${receivedMaterials.length} received materials...`);

    const receiveBatch = client.createBatch(receivedMaterials, 'atomic');
    const receiveResponse = await client.ingestBatch(receiveBatch);

    if (receiveResponse.failed > 0) {
      throw new Error('Failed to record received materials');
    }

    console.log(`   ✅ Received ${receiveResponse.succeeded} material batches`);
    console.log(`   ⏱️  Processing time: ${receiveResponse.duration_ms}ms\n`);

    // =====================================================================
    // STEP 2: Production/Transformation
    // =====================================================================
    console.log('⚙️  STEP 2: Processing Materials into Finished Goods\n');

    const processedGoods = client.createTransformedEvent(
      'BATCH-FG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour',
      ['BATCH-RM-WHEAT-001', 'BATCH-RM-SALT-001'],
      4500,
      'kg',
      {
        process: 'stone-milled',
        additives: 'salt (1%)',
        yield_rate: '90%',
        particle_size: '60 mesh',
        ash_content: '0.5%',
      }
    );

    console.log('   Processing raw materials into finished goods...');

    const processResponse = await client.ingestEvent(processedGoods);

    console.log(`   ✅ Processed goods recorded`);
    console.log(`   📋 Claim ID: ${processResponse.claim_id}`);
    console.log(`   🔗 Lineage hash: ${processResponse.lineage_hash}\n`);

    // =====================================================================
    // STEP 3: Quality Certification
    // =====================================================================
    console.log('🔬 STEP 3: Quality Certification\n');

    const certified = client.createCertifiedEvent(
      'BATCH-FG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour',
      4500,
      'kg',
      {
        certification_type: 'USDA Organic',
        certifier: 'Organic Certifiers Inc.',
        cert_number: 'ORG-2025-12345',
        audit_date: '2025-11-15',
        expires: '2026-12-31',
        lab_test_id: 'LAB-2025-998877',
        microbiological_test: 'PASS',
        heavy_metals_test: 'PASS',
      }
    );

    console.log('   Submitting certification documentation...');

    const certResponse = await client.ingestEvent(certified);

    console.log(`   ✅ Certification recorded`);
    console.log(`   📋 Claim ID: ${certResponse.claim_id}\n`);

    // =====================================================================
    // STEP 4: Packaging
    // =====================================================================
    console.log('📦 STEP 4: Packaging into Distribution Units\n');

    const packaged = client.createTransformedEvent(
      'BATCH-PKG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour (25kg bags)',
      ['BATCH-FG-FLOUR-001'],
      180,
      'units',
      {
        package_size: '25kg',
        package_type: 'recyclable paper bags',
        bags_per_pallet: 30,
        total_pallets: 6,
        batch_code: 'AF-2025-11-16',
        best_before: '2026-05-16',
      }
    );

    console.log('   Packaging finished goods...');

    const packageResponse = await client.ingestEvent(packaged);

    console.log(`   ✅ Packaging recorded`);
    console.log(`   📋 Claim ID: ${packageResponse.claim_id}\n`);

    // =====================================================================
    // STEP 5: Shipping
    // =====================================================================
    console.log('🚚 STEP 5: Shipping to Distributor\n');

    const shipped = client.createShippedEvent(
      'BATCH-PKG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour (25kg bags)',
      180,
      'units',
      {
        destination: 'DIST-REGIONAL',
        destination_address: '123 Distribution Way, Portland, OR',
        carrier: 'GreenTransport LLC',
        tracking: 'GT-2025-98765',
        departure_date: new Date().toISOString(),
        expected_arrival: '2025-11-20',
        temperature_requirements: 'ambient',
        special_handling: 'keep dry',
      }
    );

    console.log('   Recording shipment details...');

    const shipResponse = await client.ingestEvent(shipped);

    console.log(`   ✅ Shipment recorded`);
    console.log(`   📋 Claim ID: ${shipResponse.claim_id}\n`);

    // =====================================================================
    // STEP 6: Verify Complete Lineage
    // =====================================================================
    console.log('═══════════════════════════════════════════════════════');
    console.log('🔍 STEP 6: Verifying Complete Lineage\n');

    const lineage = await client.getLineage('BATCH-FG-FLOUR-001');

    console.log('📊 Production Lineage Summary:');
    console.log(`   • Batch ID: ${lineage.batch_id}`);
    console.log(`   • Total events: ${lineage.total_claims}`);
    console.log(`   • Supply chain depth: ${lineage.depth}`);
    console.log(`   • Upstream sources: ${lineage.upstream?.length || 0}`);
    console.log(`   • Downstream products: ${lineage.downstream?.length || 0}\n`);

    // Verify all required events are present
    const eventTypes = lineage.claims.map((c) => c.event.event_type);
    const requiredEvents = ['RECEIVED', 'TRANSFORMED', 'CERTIFIED', 'SHIPPED'];
    const hasAllEvents = requiredEvents.every((type) => eventTypes.includes(type));

    if (hasAllEvents) {
      console.log('   ✅ All required events present:');
      requiredEvents.forEach((type) => {
        const count = eventTypes.filter((t) => t === type).length;
        console.log(`      • ${type}: ${count} event(s)`);
      });
    } else {
      console.warn('   ⚠️  Warning: Some required events may be missing');
    }

    // Show upstream materials
    if (lineage.upstream && lineage.upstream.length > 0) {
      console.log(`\n   📦 Raw Materials Used:`);
      lineage.upstream.forEach((batch) => {
        console.log(`      • ${batch.batch_id} (${batch.claim_count} claims)`);
      });
    }

    // =====================================================================
    // STEP 7: Generate Audit Trail
    // =====================================================================
    console.log('\n═══════════════════════════════════════════════════════');
    console.log('📋 STEP 7: Generating Audit Trail\n');

    const batchClaims = await client.getBatchClaims('BATCH-FG-FLOUR-001');

    console.log(`Audit Trail for ${batchClaims.batch_id}`);
    console.log(`Total Claims: ${batchClaims.total_claims}`);
    console.log('─────────────────────────────────────────────────────\n');

    batchClaims.claims.forEach((claim, index) => {
      const event = claim.event;
      const timestamp = new Date(event.timestamp).toLocaleString();
      const metadata = event.metadata ? JSON.parse(event.metadata) : {};

      console.log(`${index + 1}. ${event.event_type} Event`);
      console.log(`   ├─ Timestamp: ${timestamp}`);
      console.log(`   ├─ Facility: ${event.facility_id}`);
      console.log(`   ├─ Product: ${event.product_id}`);
      console.log(`   ├─ Quantity: ${event.quantity} ${event.unit}`);
      console.log(`   ├─ Claim ID: ${claim.claim_id}`);
      console.log(`   ├─ Verified: ${claim.verified ? '✅ Yes' : '❌ No'}`);

      // Show relevant metadata
      const relevantKeys = Object.keys(metadata).slice(0, 3);
      if (relevantKeys.length > 0) {
        console.log(`   ├─ Key Metadata:`);
        relevantKeys.forEach((key) => {
          console.log(`   │  • ${key}: ${metadata[key]}`);
        });
      }

      console.log(`   └─ VC JWT: ${claim.vc_jwt.substring(0, 40)}...\n`);
    });

    // =====================================================================
    // STEP 8: Production Summary
    // =====================================================================
    console.log('═══════════════════════════════════════════════════════');
    console.log('📈 Production Summary\n');

    console.log('Input Materials:');
    const inputMaterials = lineage.claims.filter((c) => c.event.event_type === 'RECEIVED');
    inputMaterials.forEach((claim) => {
      console.log(`   • ${claim.event.product_id}: ${claim.event.quantity} ${claim.event.unit}`);
    });

    console.log('\nOutput Products:');
    const outputProducts = lineage.claims.filter(
      (c) => c.event.event_type === 'TRANSFORMED' || c.event.event_type === 'PRODUCED'
    );
    outputProducts.forEach((claim) => {
      console.log(`   • ${claim.event.product_id}: ${claim.event.quantity} ${claim.event.unit}`);
    });

    console.log('\nQuality Assurance:');
    const certifications = lineage.claims.filter((c) => c.event.event_type === 'CERTIFIED');
    certifications.forEach((claim) => {
      const metadata = JSON.parse(claim.event.metadata);
      console.log(`   • ${metadata.certification_type}: ${metadata.cert_number}`);
      console.log(`     Valid until: ${metadata.expires}`);
    });

    console.log('\nDistribution:');
    const shipments = lineage.claims.filter((c) => c.event.event_type === 'SHIPPED');
    shipments.forEach((claim) => {
      const metadata = JSON.parse(claim.event.metadata);
      console.log(`   • Destination: ${metadata.destination}`);
      console.log(`     Carrier: ${metadata.carrier}`);
      console.log(`     Tracking: ${metadata.tracking}`);
    });

    console.log('\n═══════════════════════════════════════════════════════');
    console.log('✅ Production Workflow Completed Successfully!');
    console.log('═══════════════════════════════════════════════════════');
    console.log(`\n🎯 Key Achievements:`);
    console.log(`   • ${receiveResponse.succeeded} raw materials received`);
    console.log(`   • ${lineage.total_claims} supply chain events recorded`);
    console.log(`   • ${certifications.length} quality certifications obtained`);
    console.log(`   • ${shipments.length} shipments dispatched`);
    console.log(`   • Complete end-to-end traceability established`);
    console.log(`   • Full audit trail generated and verified\n`);

  } catch (error) {
    console.error('\n❌ Production workflow failed:', error);
    console.error('\nError details:', error instanceof Error ? error.message : error);
    throw error;
  }
}

// Run the example
if (require.main === module) {
  productionWorkflow().catch((error) => {
    console.error('\n💥 Fatal error in production workflow');
    process.exit(1);
  });
}

export { productionWorkflow };
