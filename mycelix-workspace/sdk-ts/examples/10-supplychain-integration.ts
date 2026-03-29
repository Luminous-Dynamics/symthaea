// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk SupplyChain Integration Example
 *
 * Demonstrates product provenance tracking, chain verification, and handler trust.
 *
 * Run with: npx tsx examples/10-supplychain-integration.ts
 */

import {
  getSupplyChainService,
  SupplyChainProvenanceService,
  type Checkpoint,
  type ProvenanceChain,
  type ChainVerification,
} from '../src/integrations/supplychain/index.js';

console.log('=== SupplyChain Provenance Integration ===\n');

const supplychain = getSupplyChainService();

// === Recording a Complete Supply Chain Journey ===

console.log('--- Recording Product Journey ---');

const productId = 'organic-coffee-batch-2024-001';
const handlers = {
  farm: 'handler-farm-costa-rica',
  processor: 'handler-processor-san-jose',
  exporter: 'handler-exporter-cr',
  shipper: 'handler-global-shipping',
  importer: 'handler-us-importer',
  roaster: 'handler-local-roaster',
  retailer: 'handler-cafe-downtown',
};

// Step 1: Farm Origin (use 'received' as farm initially receives harvested beans)
console.log('\n1. Farm Origin (Costa Rica)');
const farmCheckpoint = supplychain.recordCheckpoint({
  productId,
  location: 'Finca El Paraíso, Tarrazú, Costa Rica',
  coordinates: { lat: 9.6546, lng: -83.9766 },
  handler: handlers.farm,
  action: 'received',
  evidence: [
    { type: 'gps', data: { lat: 9.6546, lng: -83.9766, accuracy: 5 }, timestamp: Date.now(), verified: true },
    { type: 'photo', data: { url: 'harvest-001.jpg', hash: 'sha256:abc...' }, timestamp: Date.now(), verified: true },
    { type: 'manual_entry', data: { certifications: ['Rainforest Alliance', 'Fair Trade'] }, timestamp: Date.now(), verified: true },
  ],
});
console.log(`  ✅ Checkpoint recorded: ${farmCheckpoint.id}`);

// Step 2: Processing
console.log('\n2. Processing Facility');
supplychain.recordCheckpoint({
  productId,
  location: 'Beneficio San José, Costa Rica',
  coordinates: { lat: 9.9281, lng: -84.0907 },
  handler: handlers.processor,
  action: 'processed',
  evidence: [
    { type: 'iot_sensor', data: { humidity: 11, temperature: 22 }, timestamp: Date.now(), verified: true },
    { type: 'blockchain', data: { txHash: '0x123...', network: 'polygon' }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Processing checkpoint recorded');

// Step 3: Export
console.log('\n3. Export (Port of Limón)');
supplychain.recordCheckpoint({
  productId,
  location: 'Puerto Limón, Costa Rica',
  coordinates: { lat: 9.9907, lng: -83.0359 },
  handler: handlers.exporter,
  action: 'shipped',
  evidence: [
    { type: 'gps', data: { lat: 9.9907, lng: -83.0359 }, timestamp: Date.now(), verified: true },
    { type: 'manual_entry', data: { containerTemp: 18, sealNumber: 'CR-2024-45678' }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Export checkpoint recorded');

// Step 4: Ocean Transit
console.log('\n4. Ocean Transit');
supplychain.recordCheckpoint({
  productId,
  location: 'At Sea - Pacific Ocean',
  coordinates: { lat: 15.0, lng: -95.0 },
  handler: handlers.shipper,
  action: 'shipped',
  evidence: [
    { type: 'iot_sensor', data: { containerTemp: 18, humidity: 60 }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Transit checkpoint recorded');

// Step 5: Import (US)
console.log('\n5. Import (Port of Oakland)');
supplychain.recordCheckpoint({
  productId,
  location: 'Port of Oakland, California, USA',
  coordinates: { lat: 37.7954, lng: -122.2792 },
  handler: handlers.importer,
  action: 'received',
  evidence: [
    { type: 'gps', data: { lat: 37.7954, lng: -122.2792 }, timestamp: Date.now(), verified: true },
    { type: 'manual_entry', data: { customsEntry: 'US-2024-12345', fdaStatus: 'approved' }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Import checkpoint recorded');

// Step 6: Roasting
console.log('\n6. Roasting Facility');
supplychain.recordCheckpoint({
  productId,
  location: 'Artisan Roasters, San Francisco, CA',
  coordinates: { lat: 37.7599, lng: -122.4148 },
  handler: handlers.roaster,
  action: 'processed',
  evidence: [
    { type: 'iot_sensor', data: { roastTemp: 220, duration: 12 }, timestamp: Date.now(), verified: true },
    { type: 'manual_entry', data: { cuppingScore: 87, roastProfile: 'medium' }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Roasting checkpoint recorded');

// Step 7: Retail
console.log('\n7. Retail Location');
supplychain.recordCheckpoint({
  productId,
  location: 'Café Luminoso, Downtown SF',
  coordinates: { lat: 37.7849, lng: -122.4094 },
  handler: handlers.retailer,
  action: 'delivered',
  evidence: [
    { type: 'gps', data: { lat: 37.7849, lng: -122.4094 }, timestamp: Date.now(), verified: true },
  ],
});
console.log('  ✅ Retail checkpoint recorded');

// === Get Full Provenance Chain ===

console.log('\n--- Full Provenance Chain ---');
const chain = supplychain.getProvenanceChain(productId);

if (chain) {
  console.log(`\nProduct: ${chain.productId}`);
  console.log(`Total Checkpoints: ${chain.checkpoints.length}`);
  console.log(`Total Distance: ${chain.totalDistance.toFixed(1)} km`);
  console.log(`Chain Integrity: ${(chain.chainIntegrity * 100).toFixed(1)}%`);

  console.log('\nJourney Timeline:');
  for (let i = 0; i < chain.checkpoints.length; i++) {
    const cp = chain.checkpoints[i];
    const time = new Date(cp.timestamp).toLocaleString();
    const evidenceTypes = cp.evidence.map((e) => e.type).join(', ');
    console.log(`  ${i + 1}. [${cp.action.toUpperCase()}] ${cp.location}`);
    console.log(`     Handler: ${cp.handler}`);
    console.log(`     Time: ${time}`);
    console.log(`     Evidence: ${evidenceTypes || 'none'}`);
  }
}

// === Chain Verification ===

console.log('\n--- Chain Verification ---');
const verification = supplychain.verifyChain(productId);

console.log(`\nVerification Result:`);
console.log(`  Verified: ${verification.verified ? '✅ Yes' : '⚠️ Partial'}`);
console.log(`  Integrity Score: ${(verification.integrityScore * 100).toFixed(1)}%`);

if (verification.weakLinks.length > 0) {
  console.log(`  Weak Links:`);
  for (const weak of verification.weakLinks) {
    console.log(`    - ${weak}`);
  }
}

if (verification.recommendations.length > 0) {
  console.log(`  Recommendations:`);
  for (const rec of verification.recommendations) {
    console.log(`    - ${rec}`);
  }
}

// === Handler Profiles ===

console.log('\n--- Handler Trust Profiles ---');

for (const [role, handlerId] of Object.entries(handlers)) {
  const profile = supplychain.getHandlerProfile(handlerId);
  console.log(`\n${role}: ${handlerId}`);
  console.log(`  Trust Score: ${profile.trustScore.toFixed(3)}`);
  console.log(`  Checkpoints Recorded: ${profile.checkpointsRecorded}`);
  console.log(`  Verification Rate: ${(profile.verificationRate * 100).toFixed(1)}%`);
}

// === Trust Checks ===

console.log('\n--- Handler Trust Checks ---');
console.log(`Farm trusted (0.5): ${supplychain.isHandlerTrusted(handlers.farm, 0.5)}`);
console.log(`Shipper trusted (0.5): ${supplychain.isHandlerTrusted(handlers.shipper, 0.5)}`);
console.log(`Unknown handler trusted: ${supplychain.isHandlerTrusted('unknown-handler', 0.5)}`);

// === Real-World Pattern: Consumer QR Code Scan ===

console.log('\n--- Consumer Product Verification ---');

function generateConsumerSummary(productId: string): string {
  const chain = supplychain.getProvenanceChain(productId);
  const verification = supplychain.verifyChain(productId);

  if (!chain) {
    return '❌ Product not found in our tracking system.';
  }

  const origin = chain.checkpoints[0];
  const destination = chain.checkpoints[chain.checkpoints.length - 1];

  let summary = '☕ PRODUCT AUTHENTICITY VERIFIED\n\n';
  summary += `📍 Origin: ${origin.location}\n`;
  summary += `🏪 Current: ${destination.location}\n`;
  summary += `📏 Distance Traveled: ${chain.totalDistance.toFixed(0)} km\n`;
  summary += `🔗 Checkpoints: ${chain.checkpoints.length}\n`;
  summary += `✅ Integrity: ${(verification.integrityScore * 100).toFixed(0)}%\n`;

  if (verification.verified) {
    summary += '\n🌟 This product has complete verified provenance.';
  } else {
    summary += `\n⚠️ ${verification.weakLinks.length} checkpoint(s) have limited verification.`;
  }

  return summary;
}

console.log(generateConsumerSummary(productId));

console.log('\n=== SupplyChain Integration Complete ===');
