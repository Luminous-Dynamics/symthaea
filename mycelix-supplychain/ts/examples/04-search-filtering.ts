// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 4: Advanced Search & Filtering
 *
 * This example demonstrates:
 * - Searching claims with various filters
 * - Filter by event type, product, facility
 * - Date range queries
 * - Pagination patterns
 * - Combining multiple filters
 */

import { SupplyChainClient } from '../sdk/src';

async function advancedSearch() {
  console.log('=== Advanced Search & Filtering Example ===\n');

  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Example 1: Find all PRODUCED events
  console.log('1️⃣  Finding all PRODUCED events...');

  try {
    const producedEvents = await client.searchClaims({
      event_type: 'PRODUCED',
      limit: 10,
    });

    console.log(`   Found ${producedEvents.total} PRODUCED events (showing ${producedEvents.claims.length})`);
    console.log(`   Has more: ${producedEvents.has_more}\n`);

    if (producedEvents.claims.length > 0) {
      const first = producedEvents.claims[0];
      console.log('   First result:');
      console.log(`   • Product: ${first.event.product_id}`);
      console.log(`   • Batch: ${first.event.batch_id}`);
      console.log(`   • Quantity: ${first.event.quantity} ${first.event.unit}\n`);
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 2: Find events for specific product
  console.log('2️⃣  Finding events for specific product...');

  try {
    const productEvents = await client.searchClaims({
      product_id: 'Organic Coffee',
      limit: 20,
    });

    console.log(`   Found ${productEvents.total} events for "Organic Coffee"`);
    console.log(`   Showing ${productEvents.claims.length} results\n`);

    if (productEvents.claims.length > 0) {
      console.log('   Events:');
      productEvents.claims.slice(0, 5).forEach((claim) => {
        console.log(`   • ${claim.event.event_type}: ${claim.event.batch_id} at ${claim.event.facility_id}`);
      });
      if (productEvents.claims.length > 5) {
        console.log(`   ... and ${productEvents.claims.length - 5} more\n`);
      } else {
        console.log('');
      }
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 3: Find events in date range
  console.log('3️⃣  Finding events in date range (2025)...');

  try {
    const startDate = new Date('2025-01-01');
    const endDate = new Date('2025-12-31');

    const rangeEvents = await client.searchClaims({
      from: startDate.toISOString(),
      to: endDate.toISOString(),
      limit: 50,
    });

    console.log(`   Found ${rangeEvents.total} events in 2025`);
    console.log(`   Date range: ${startDate.toLocaleDateString()} to ${endDate.toLocaleDateString()}\n`);
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 4: Filter by facility
  console.log('4️⃣  Finding events at specific facility...');

  try {
    const facilityEvents = await client.searchClaims({
      facility_id: 'ORG-FACTORY',
      limit: 15,
    });

    console.log(`   Found ${facilityEvents.total} events at ORG-FACTORY\n`);

    if (facilityEvents.claims.length > 0) {
      const eventTypes = new Set(facilityEvents.claims.map((c) => c.event.event_type));
      console.log(`   Event types at this facility:`);
      eventTypes.forEach((type) => {
        const count = facilityEvents.claims.filter((c) => c.event.event_type === type).length;
        console.log(`   • ${type}: ${count} events`);
      });
      console.log('');
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 5: Combine multiple filters
  console.log('5️⃣  Complex query: TRANSFORMED events at specific facility...');

  try {
    const complexQuery = await client.searchClaims({
      event_type: 'TRANSFORMED',
      facility_id: 'ORG-FACTORY',
      limit: 10,
    });

    console.log(`   Found ${complexQuery.total} TRANSFORMED events at ORG-FACTORY`);
    console.log(`   Showing ${complexQuery.claims.length} results\n`);

    if (complexQuery.claims.length > 0) {
      console.log('   Results:');
      complexQuery.claims.forEach((claim, index) => {
        const inputs = claim.event.input_batches?.join(', ') || 'none';
        console.log(`   ${index + 1}. ${claim.event.product_id}`);
        console.log(`      Batch: ${claim.event.batch_id}`);
        console.log(`      Inputs: ${inputs}`);
      });
      console.log('');
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 6: Pagination through results
  console.log('6️⃣  Paginating through all results for a product...');

  try {
    const productId = 'Organic Coffee';
    let offset = 0;
    const limit = 20;
    let allClaims = [];
    let page = 1;

    console.log(`   Fetching all claims for "${productId}"...\n`);

    while (true) {
      const pageResult = await client.searchClaims({
        product_id: productId,
        limit,
        offset,
      });

      allClaims.push(...pageResult.claims);
      console.log(`   Page ${page}: Fetched ${pageResult.claims.length} claims (offset ${offset})`);

      if (!pageResult.has_more) {
        console.log(`   ✓ Reached end of results\n`);
        break;
      }

      offset += limit;
      page++;

      // Safety limit for example
      if (page > 10) {
        console.log(`   ⚠️  Stopping after 10 pages for demo purposes\n`);
        break;
      }
    }

    console.log(`   Total claims fetched: ${allClaims.length}`);
    console.log(`   Total available: ${allClaims.length}\n`);
  } catch (error) {
    console.error('   ❌ Pagination failed:', error);
  }

  // Example 7: Filter by batch ID
  console.log('7️⃣  Finding all events for a specific batch...');

  try {
    const batchEvents = await client.searchClaims({
      batch_id: 'BATCH-COFFEE-001',
      limit: 50,
    });

    console.log(`   Found ${batchEvents.total} events for BATCH-COFFEE-001\n`);

    if (batchEvents.claims.length > 0) {
      console.log('   Event timeline:');
      batchEvents.claims.forEach((claim, index) => {
        const timestamp = new Date(claim.event.timestamp).toLocaleString();
        console.log(`   ${index + 1}. ${claim.event.event_type} - ${timestamp}`);
      });
      console.log('');
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 8: Custom pagination limits
  console.log('8️⃣  Testing different pagination limits...\n');

  try {
    const limits = [5, 10, 50, 100];

    for (const limit of limits) {
      const result = await client.searchClaims({
        limit,
        offset: 0,
      });

      console.log(`   Limit ${limit}: Got ${result.claims.length} results (total: ${result.total})`);
    }
    console.log('');
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  // Example 9: Multi-criteria advanced search
  console.log('9️⃣  Advanced multi-criteria search...');

  try {
    const advancedQuery = await client.searchClaims({
      event_type: 'SHIPPED',
      facility_id: 'ORG-PACKAGER',
      from: '2025-11-01T00:00:00Z',
      to: '2025-11-30T23:59:59Z',
      limit: 25,
    });

    console.log('   Query criteria:');
    console.log('   • Event type: SHIPPED');
    console.log('   • Facility: ORG-PACKAGER');
    console.log('   • Date range: November 2025');
    console.log('   • Limit: 25\n');

    console.log(`   Results: ${advancedQuery.claims.length} of ${advancedQuery.total} total`);
    console.log(`   Has more: ${advancedQuery.has_more}\n`);

    if (advancedQuery.claims.length > 0) {
      console.log('   Shipment details:');
      advancedQuery.claims.slice(0, 3).forEach((claim, index) => {
        const metadata = claim.event.metadata ? JSON.parse(claim.event.metadata) : {};
        console.log(`   ${index + 1}. ${claim.event.batch_id}`);
        console.log(`      Destination: ${metadata.destination || 'N/A'}`);
        console.log(`      Carrier: ${metadata.carrier || 'N/A'}`);
        console.log(`      Tracking: ${metadata.tracking || 'N/A'}`);
      });
      if (advancedQuery.claims.length > 3) {
        console.log(`      ... and ${advancedQuery.claims.length - 3} more shipments`);
      }
    }
  } catch (error) {
    console.error('   ❌ Search failed:', error);
  }

  console.log('\n✅ Advanced search & filtering example completed!');
}

// Run the example
if (require.main === module) {
  advancedSearch().catch((error) => {
    console.error('Example failed:', error);
    process.exit(1);
  });
}

export { advancedSearch };
