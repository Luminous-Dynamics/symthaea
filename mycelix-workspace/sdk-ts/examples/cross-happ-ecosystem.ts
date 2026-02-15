/**
 * @mycelix/sdk Cross-hApp Ecosystem Example
 *
 * This example demonstrates how the Mycelix SDK enables trust to flow
 * seamlessly across different hApps (Holochain applications) in the ecosystem.
 *
 * Scenario: A supply chain product is tracked from farm to consumer,
 * with trust scores from marketplace transactions, educational credentials,
 * and email verifications all contributing to a unified reputation.
 */

// Import core modules
import { matl, epistemic, bridge, fl } from '../src/index.js';

// Import hApp-specific services
import { getMailTrustService } from '../src/integrations/mail/index.js';
import { getMarketplaceService } from '../src/integrations/marketplace/index.js';
import { getEduNetService } from '../src/integrations/edunet/index.js';
import { getSupplyChainService } from '../src/integrations/supplychain/index.js';

// ============================================================================
// Scenario Setup: Organic Coffee Supply Chain
// ============================================================================

console.log('🌿 Mycelix Cross-hApp Ecosystem Demo');
console.log('━'.repeat(50));
console.log('Scenario: Organic Coffee from Farm to Cup\n');

// Initialize services
const mailService = getMailTrustService();
const marketplaceService = getMarketplaceService();
const eduNetService = getEduNetService();
const supplyChainService = getSupplyChainService();

// Initialize local bridge for cross-hApp communication
const localBridge = new bridge.LocalBridge();
localBridge.registerHapp('mail');
localBridge.registerHapp('marketplace');
localBridge.registerHapp('edunet');
localBridge.registerHapp('supplychain');

// ============================================================================
// Step 1: Farm Origin - Farmer Credentials
// ============================================================================

console.log('📚 Step 1: Farmer Credentials (EduNet)');
console.log('-'.repeat(40));

// Farmer has completed organic farming certification
const organicCert = eduNetService.issueCertificate({
  studentId: 'farmer-carlos',
  courseId: 'organic-certification-2024',
  courseName: 'Organic Farming Certification',
  grade: 95,
});

console.log(`✅ Farmer Carlos certified: ${organicCert.claim.content}`);
console.log(`   Classification: E${organicCert.claim.classification.empirical}-N${organicCert.claim.classification.normative}-M${organicCert.claim.classification.materiality}`);

// Farmer also has coffee processing skill
const processingSkill = eduNetService.issueSkillCertification({
  holderId: 'farmer-carlos',
  skillId: 'specialty-coffee-processing',
  skillName: 'Specialty Coffee Processing',
  level: 88,
});

console.log(`✅ Processing skill certified: ${processingSkill.claim.content}\n`);

// Get farmer's learner profile
const farmerProfile = eduNetService.getLearnerProfile('farmer-carlos');
console.log(`📊 Farmer Trust Score: ${(farmerProfile.trustScore * 100).toFixed(1)}%`);
console.log(`   Courses Completed: ${farmerProfile.coursesCompleted}`);
console.log(`   Certifications: ${farmerProfile.certificationsEarned}\n`);

// ============================================================================
// Step 2: Supply Chain Tracking
// ============================================================================

console.log('🚚 Step 2: Supply Chain Provenance');
console.log('-'.repeat(40));

const productId = 'organic-coffee-batch-2024-001';

// Checkpoint 1: Farm origin
const farmCheckpoint = supplyChainService.recordCheckpoint({
  productId,
  batchId: 'BATCH-2024-COFFEE-001',
  location: 'Carlos Farm, Antioquia, Colombia',
  coordinates: { lat: 6.2442, lng: -75.5812 },
  handler: 'farmer-carlos',
  action: 'shipped',
  evidence: [
    { type: 'photo', data: { url: 'harvest-2024.jpg' }, timestamp: Date.now(), verified: true },
    { type: 'gps', data: { lat: 6.2442, lng: -75.5812 }, timestamp: Date.now(), verified: true },
    { type: 'third_party_verification', data: { certifier: 'Rainforest Alliance' }, timestamp: Date.now(), verified: true },
  ],
});
console.log(`✅ Origin recorded: ${farmCheckpoint.location}`);

// Checkpoint 2: Processing facility
const processingCheckpoint = supplyChainService.recordCheckpoint({
  productId,
  batchId: 'BATCH-2024-COFFEE-001',
  location: 'Specialty Processing Co, Medellin',
  coordinates: { lat: 6.2518, lng: -75.5636 },
  handler: 'processor-specialty-co',
  action: 'processed',
  evidence: [
    { type: 'iot_sensor', data: { temp: 22, humidity: 65, roastLevel: 'medium' }, timestamp: Date.now(), verified: true },
    { type: 'blockchain', data: { txHash: '0xabc123' }, timestamp: Date.now(), verified: true },
  ],
});
console.log(`✅ Processing recorded: ${processingCheckpoint.location}`);

// Checkpoint 3: Distribution
const distroCheckpoint = supplyChainService.recordCheckpoint({
  productId,
  batchId: 'BATCH-2024-COFFEE-001',
  location: 'Regional Distribution Center, Miami',
  coordinates: { lat: 25.7617, lng: -80.1918 },
  handler: 'distributor-americas',
  action: 'received',
  evidence: [
    { type: 'rfid', data: { tag: 'COFFEE-001-RFID' }, timestamp: Date.now(), verified: true },
    { type: 'iot_sensor', data: { temp: 18, humidity: 50 }, timestamp: Date.now(), verified: true },
  ],
});
console.log(`✅ Distribution recorded: ${distroCheckpoint.location}`);

// Checkpoint 4: Retail delivery
const retailCheckpoint = supplyChainService.recordCheckpoint({
  productId,
  batchId: 'BATCH-2024-COFFEE-001',
  location: 'Artisan Coffee Shop, Austin, TX',
  coordinates: { lat: 30.2672, lng: -97.7431 },
  handler: 'retailer-artisan-coffee',
  action: 'delivered',
  evidence: [
    { type: 'signature', data: { signedBy: 'Shop Owner', quality: 'excellent' }, timestamp: Date.now(), verified: true },
  ],
});
console.log(`✅ Retail delivery: ${retailCheckpoint.location}\n`);

// Verify the complete chain
const chainVerification = supplyChainService.verifyChain(productId);
const provenanceChain = supplyChainService.getProvenanceChain(productId);

console.log('📊 Chain Verification:');
console.log(`   Verified: ${chainVerification.verified ? '✅ YES' : '❌ NO'}`);
console.log(`   Integrity Score: ${(chainVerification.integrityScore * 100).toFixed(1)}%`);
console.log(`   Total Distance: ${provenanceChain?.totalDistance.toFixed(0)}km`);
console.log(`   Checkpoints: ${provenanceChain?.checkpoints.length}`);
console.log(`   Weak Links: ${chainVerification.weakLinks.length}\n`);

// ============================================================================
// Step 3: Marketplace Transactions
// ============================================================================

console.log('🏪 Step 3: Marketplace Transactions');
console.log('-'.repeat(40));

// Record successful sales
for (let i = 0; i < 5; i++) {
  marketplaceService.recordTransaction({
    id: `tx-coffee-${i}`,
    type: 'purchase',
    buyerId: `customer-${i}`,
    sellerId: 'retailer-artisan-coffee',
    amount: 25.99,
    currency: 'USD',
    itemId: productId,
    timestamp: Date.now(),
    success: true,
  });
}

console.log(`✅ Recorded 5 successful coffee sales`);

// Get seller profile
const sellerProfile = marketplaceService.getSellerProfile('retailer-artisan-coffee');
console.log(`📊 Seller Trust Score: ${(sellerProfile.trustScore * 100).toFixed(1)}%`);
console.log(`   Transaction Count: ${sellerProfile.transactionCount}`);
console.log(`   Success Rate: ${(sellerProfile.successRate * 100).toFixed(1)}%`);
console.log(`   Verified: ${sellerProfile.verified ? '✅ YES' : '❌ NO'}\n`);

// Verify the listing
const listingVerification = marketplaceService.verifyListing('coffee-listing-001', 'retailer-artisan-coffee');
console.log('📊 Listing Verification:');
console.log(`   Verified: ${listingVerification.verified ? '✅ YES' : '❌ NO'}`);
console.log(`   Scam Risk: ${(listingVerification.scamRiskScore * 100).toFixed(1)}%\n`);

// ============================================================================
// Step 4: Email Communications Trust
// ============================================================================

console.log('📧 Step 4: Email Trust');
console.log('-'.repeat(40));

// Record positive email interactions from the supply chain partners
const emailSenders = [
  'farmer-carlos@organicfarm.co',
  'orders@specialty-processing.co',
  'logistics@americas-distribution.com',
  'shop@artisan-coffee.com',
];

emailSenders.forEach((sender) => {
  for (let i = 0; i < 3; i++) {
    mailService.recordInteraction(sender, true);
  }
});

console.log(`✅ Recorded positive email interactions from ${emailSenders.length} partners`);

// Check sender trust
emailSenders.forEach((sender) => {
  const trust = mailService.getSenderTrust(sender);
  console.log(`   ${sender}: ${trust.level} (${(trust.score * 100).toFixed(0)}%)`);
});
console.log();

// ============================================================================
// Step 5: Cross-hApp Reputation Aggregation
// ============================================================================

console.log('🌐 Step 5: Cross-hApp Reputation');
console.log('-'.repeat(40));

// Create reputations for the retailer across all hApps
const retailerId = 'retailer-artisan-coffee';

// Get reputation from each hApp
const supplyChainHandler = supplyChainService.getHandlerProfile(retailerId);
const marketplaceSeller = marketplaceService.getSellerProfile(retailerId);

// Store reputations in bridge
localBridge.setReputation('supplychain', retailerId, supplyChainHandler.reputation);
localBridge.setReputation('marketplace', retailerId, marketplaceSeller.reputation);

// Calculate aggregate reputation
const crossHappScores = localBridge.getCrossHappReputation(retailerId);
const aggregateRep = localBridge.getAggregateReputation(retailerId);

console.log(`📊 ${retailerId} Cross-hApp Reputation:`);
crossHappScores.forEach((score) => {
  console.log(`   ${score.happ}: ${(score.score * 100).toFixed(1)}%`);
});
console.log(`   ─────────────────`);
console.log(`   Aggregate: ${(aggregateRep * 100).toFixed(1)}%\n`);

// ============================================================================
// Step 6: Federated Learning Integration
// ============================================================================

console.log('🧠 Step 6: Federated Learning');
console.log('-'.repeat(40));

// Create FL coordinator for trust model training
const flCoordinator = new fl.FLCoordinator({
  minParticipants: 3,
  aggregationMethod: 'trust_weighted',
  trustThreshold: 0.3,
});

// Register participants from the supply chain
const participants = ['farmer-carlos', 'processor-specialty-co', 'distributor-americas', 'retailer-artisan-coffee'];
participants.forEach((p) => flCoordinator.registerParticipant(p));

console.log(`✅ Registered ${participants.length} FL participants`);

// Get FL statistics
const flStats = flCoordinator.getRoundStats();
console.log(`📊 FL Statistics:`);
console.log(`   Participants: ${flStats.participantCount}`);
console.log(`   Total Rounds: ${flStats.totalRounds}`);
console.log(`   Avg Participation: ${(flStats.averageParticipation * 100).toFixed(1)}%\n`);

// ============================================================================
// Step 7: Epistemic Claims Summary
// ============================================================================

console.log('📜 Step 7: Epistemic Claims');
console.log('-'.repeat(40));

// Create a comprehensive claim about the product
const productClaim = epistemic.claim('Organic coffee verified through complete supply chain')
  .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
  .withNormative(epistemic.NormativeLevel.N2_Network)
  .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
  .withIssuer('mycelix-supplychain')
  .build();

// Add evidence from all hApps
const withFarmerEvidence = epistemic.addEvidence(productClaim, {
  type: 'credential',
  data: organicCert.id,
  source: 'edunet',
  timestamp: Date.now(),
});

const withChainEvidence = epistemic.addEvidence(withFarmerEvidence, {
  type: 'provenance',
  data: provenanceChain?.checkpoints.length,
  source: 'supplychain',
  timestamp: Date.now(),
});

console.log(`📋 Product Claim: ${productClaim.content}`);
console.log(`   Classification: ${epistemic.classificationCode(productClaim.classification)}`);
console.log(`   Evidence Items: ${withChainEvidence.evidence?.length || 0}`);

// Check against standards
const meetsHighTrust = epistemic.meetsStandard(
  productClaim,
  epistemic.EmpiricalLevel.E2_PrivateVerify,
  epistemic.NormativeLevel.N2_Network
);
console.log(`   Meets High Trust Standard: ${meetsHighTrust ? '✅ YES' : '❌ NO'}\n`);

// ============================================================================
// Summary
// ============================================================================

console.log('═'.repeat(50));
console.log('🎯 Ecosystem Integration Summary');
console.log('═'.repeat(50));
console.log(`
The Mycelix ecosystem enables:

1. 📚 Educational Credentials (EduNet)
   - Farmer certifications verified and tracked
   - Skills contribute to overall trust

2. 🚚 Supply Chain Provenance
   - Complete product journey recorded
   - Multi-evidence verification at each checkpoint
   - Chain integrity: ${(chainVerification.integrityScore * 100).toFixed(1)}%

3. 🏪 Marketplace Reputation
   - Transaction history builds trust
   - Seller verification from trading history
   - Scam risk assessment: ${(listingVerification.scamRiskScore * 100).toFixed(1)}%

4. 📧 Communication Trust (Mail)
   - Sender reputation from interactions
   - Trust levels: unknown → low → medium → high → verified

5. 🌐 Cross-hApp Integration
   - Reputations flow between hApps via Bridge
   - Aggregate scores from multiple contexts
   - Unified trust view: ${(aggregateRep * 100).toFixed(1)}%

6. 🧠 Federated Learning
   - Collaborative model training
   - Byzantine-resistant aggregation
   - Privacy-preserving reputation updates

7. 📜 Epistemic Claims
   - 3D classification: Empirical × Normative × Materiality
   - Evidence from all hApps
   - Standards-based verification

All components work together through the MATL (Mycelix Adaptive
Trust Layer) providing 34% validated Byzantine fault tolerance.
`);

console.log('Demo complete! 🌿☕\n');
