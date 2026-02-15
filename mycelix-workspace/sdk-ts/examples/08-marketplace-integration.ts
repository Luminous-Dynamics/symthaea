/**
 * @mycelix/sdk Marketplace Integration Example
 *
 * Demonstrates transaction reputation, seller verification, and scam detection.
 *
 * Run with: npx tsx examples/08-marketplace-integration.ts
 */

import {
  getMarketplaceService,
  MarketplaceReputationService,
  type Transaction,
  type SellerProfile,
  type ListingVerification,
} from '../src/integrations/marketplace/index.js';

console.log('=== Marketplace Reputation Integration ===\n');

const marketplace = getMarketplaceService();

// === Building Seller Reputation ===

console.log('--- Building Seller Reputation ---');

const sellers = {
  powerSeller: 'seller-powerseller-001',
  newSeller: 'seller-new-002',
  scammer: 'seller-scam-003',
};

// Power seller: 15 successful transactions
console.log('Recording power seller transactions...');
for (let i = 0; i < 15; i++) {
  marketplace.recordTransaction({
    id: `tx-power-${i}`,
    type: 'sale',
    buyerId: `buyer-${i}`,
    sellerId: sellers.powerSeller,
    amount: 50 + Math.random() * 200,
    currency: 'USD',
    itemId: `item-${i}`,
    timestamp: Date.now() - i * 86400000, // Spread over days
    success: true,
  });
}

// New seller: 3 successful transactions
console.log('Recording new seller transactions...');
for (let i = 0; i < 3; i++) {
  marketplace.recordTransaction({
    id: `tx-new-${i}`,
    type: 'sale',
    buyerId: `buyer-new-${i}`,
    sellerId: sellers.newSeller,
    amount: 25 + Math.random() * 50,
    currency: 'USD',
    itemId: `item-new-${i}`,
    timestamp: Date.now() - i * 86400000,
    success: true,
  });
}

// Scammer: Mix of failed transactions
console.log('Recording suspicious seller transactions...');
for (let i = 0; i < 8; i++) {
  const success = i < 2; // Only first 2 succeed
  marketplace.recordTransaction({
    id: `tx-scam-${i}`,
    type: 'sale',
    buyerId: `victim-${i}`,
    sellerId: sellers.scammer,
    amount: 100 + Math.random() * 500,
    currency: 'USD',
    itemId: `item-scam-${i}`,
    timestamp: Date.now() - i * 86400000,
    success,
    disputeResolution: success ? undefined : 'buyer_favor',
  });
}

// === Seller Profiles ===

console.log('\n--- Seller Profiles ---');

for (const [label, sellerId] of Object.entries(sellers)) {
  const profile = marketplace.getSellerProfile(sellerId);
  console.log(`\n${label} (${sellerId}):`);
  console.log(`  Trust Score: ${profile.trustScore.toFixed(3)}`);
  console.log(`  Transactions: ${profile.transactionCount}`);
  console.log(`  Success Rate: ${(profile.successRate * 100).toFixed(1)}%`);
  console.log(`  Verified Badge: ${profile.verified ? '✅ Yes' : '❌ No'}`);
}

// === Listing Verification ===

console.log('\n--- Listing Verification ---');

const listings = [
  { id: 'listing-001', name: 'Vintage Camera', sellerId: sellers.powerSeller, price: 299 },
  { id: 'listing-002', name: 'Handmade Jewelry', sellerId: sellers.newSeller, price: 45 },
  { id: 'listing-003', name: 'iPhone 15 Pro - 50% OFF!!!', sellerId: sellers.scammer, price: 499 },
];

for (const listing of listings) {
  const verification = marketplace.verifyListing(listing.id, listing.sellerId);

  console.log(`\n"${listing.name}" ($${listing.price}):`);
  console.log(`  Seller Trust: ${verification.sellerTrust.toFixed(3)}`);
  console.log(`  Scam Risk: ${(verification.scamRiskScore * 100).toFixed(1)}%`);
  console.log(`  Verified: ${verification.verified ? '✅' : '⚠️'}`);
  console.log(`  Recommendations:`);
  for (const rec of verification.recommendations) {
    console.log(`    - ${rec}`);
  }
}

// === Buyer Profiles ===

console.log('\n--- Buyer Profile Example ---');

// Record some buyer activity
const buyerId = 'buyer-reliable-001';
for (let i = 0; i < 5; i++) {
  marketplace.recordTransaction({
    id: `tx-buyer-${i}`,
    type: 'purchase',
    buyerId,
    sellerId: sellers.powerSeller,
    amount: 100,
    currency: 'USD',
    itemId: `item-${i}`,
    timestamp: Date.now(),
    success: true,
  });
}

const buyerProfile = marketplace.getBuyerProfile(buyerId);
console.log(`Buyer: ${buyerId}`);
console.log(`  Trust Score: ${buyerProfile.trustScore.toFixed(3)}`);
console.log(`  Transactions: ${buyerProfile.transactionCount}`);
console.log(`  Payment Reliability: ${(buyerProfile.paymentReliability * 100).toFixed(1)}%`);
console.log(`  Dispute Rate: ${(buyerProfile.disputeRate * 100).toFixed(1)}%`);

// === Trust Checks ===

console.log('\n--- Quick Trust Checks ---');
console.log(`Power seller trustworthy (0.7): ${marketplace.isSellerTrustworthy(sellers.powerSeller, 0.7)}`);
console.log(`New seller trustworthy (0.7): ${marketplace.isSellerTrustworthy(sellers.newSeller, 0.7)}`);
console.log(`Scammer trustworthy (0.7): ${marketplace.isSellerTrustworthy(sellers.scammer, 0.7)}`);

// === Federated Learning Stats ===

console.log('\n--- Federated Learning Integration ---');
const flStats = marketplace.getFLStats();
console.log(`FL Rounds Completed: ${flStats.totalRounds}`);
console.log(`Participant Count: ${flStats.participantCount}`);
console.log(`Average Participation: ${(flStats.averageParticipation * 100).toFixed(1)}%`);

// === Real-World Pattern: Purchase Decision ===

console.log('\n--- Purchase Decision Flow ---');

function shouldProceedWithPurchase(
  listingId: string,
  sellerId: string,
  amount: number
): { proceed: boolean; reason: string } {
  const verification = marketplace.verifyListing(listingId, sellerId);

  if (verification.scamRiskScore > 0.5) {
    return { proceed: false, reason: 'High scam risk detected' };
  }

  if (!verification.verified && amount > 100) {
    return { proceed: false, reason: 'Unverified seller for high-value item' };
  }

  if (verification.sellerTrust < 0.3) {
    return { proceed: false, reason: 'Seller trust too low' };
  }

  return { proceed: true, reason: 'Transaction approved' };
}

for (const listing of listings) {
  const decision = shouldProceedWithPurchase(listing.id, listing.sellerId, listing.price);
  const icon = decision.proceed ? '✅' : '🚫';
  console.log(`${icon} ${listing.name}: ${decision.reason}`);
}

console.log('\n=== Marketplace Integration Complete ===');
