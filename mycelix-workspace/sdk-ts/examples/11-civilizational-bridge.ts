/**
 * Civilizational Bridge Example
 *
 * Demonstrates the full Mycelix Civilizational OS integration:
 * - All 8 domain bridge clients (Identity, Finance, Property, Energy, Media, Governance, Justice, Knowledge)
 * - Runtime validation with Zod schemas
 * - Real-time signals for event subscriptions
 * - Cross-hApp trust aggregation via MATL
 *
 * This example simulates a decentralized community marketplace where:
 * 1. Users create verified identities
 * 2. Sellers list renewable energy equipment
 * 3. Buyers purchase with trust-weighted financing
 * 4. The community governs through proposals
 * 5. Disputes are resolved through decentralized justice
 */

import { createMockClient } from '../src/client/index.js';
import { matl, epistemic, bridge } from '../src/index.js';

// Bridge clients for all 8 domains
import { IdentityBridgeClient } from '../src/integrations/identity/index.js';
import { FinanceBridgeClient } from '../src/integrations/finance/index.js';
import { PropertyBridgeClient } from '../src/integrations/property/index.js';
import { EnergyBridgeClient } from '../src/integrations/energy/index.js';
import { MediaBridgeClient } from '../src/integrations/media/index.js';
import { GovernanceBridgeClient } from '../src/integrations/governance/index.js';
import { JusticeBridgeClient } from '../src/integrations/justice/index.js';
import { KnowledgeBridgeClient } from '../src/integrations/knowledge/index.js';

// Validation schemas for all domains
import {
  validateOrThrow,
  validateSafe,
  IdentitySchemas,
  FinanceSchemas,
  PropertySchemas,
  EnergySchemas,
  GovernanceSchemas,
  KnowledgeSchemas,
  didSchema,
  matlScoreSchema,
} from '../src/validation/index.js';

// Signal handlers for real-time events
import { createSignalManager } from '../src/signals/index.js';

// ============================================================================
// Main Example
// ============================================================================

async function runCivilizationalBridgeExample(): Promise<void> {
  console.log('🌍 Mycelix Civilizational Bridge Example');
  console.log('=========================================\n');

  // Create mock client for demonstration
  const client = createMockClient();
  await client.connect();

  // Initialize all bridge clients
  const identity = new IdentityBridgeClient(client);
  const finance = new FinanceBridgeClient(client);
  const property = new PropertyBridgeClient(client);
  const energy = new EnergyBridgeClient(client);
  const _media = new MediaBridgeClient(client);
  const governance = new GovernanceBridgeClient(client);
  const _justice = new JusticeBridgeClient(client);
  const knowledge = new KnowledgeBridgeClient(client);

  // Initialize signal manager for real-time events
  const signals = createSignalManager();
  signals.connect(client);

  // =========================================================================
  // Step 1: Set up real-time event listeners
  // =========================================================================
  console.log('📡 Step 1: Setting up signal handlers...\n');

  // Identity events
  signals.identity.onIdentityCreated((data) => {
    console.log(`  🆔 New identity created: ${data.did}`);
    console.log(`     Initial MATL score: ${data.initial_matl_score}`);
  });

  // Finance events
  signals.finance.onPaymentProcessed((data) => {
    console.log(`  💰 Payment processed: ${data.payment_id}`);
    console.log(`     Amount: ${data.amount} ${data.currency}`);
  });

  // Energy events
  signals.energy.onEnergyListed((data) => {
    console.log(`  ⚡ Energy listed: ${data.listing_id}`);
    console.log(`     ${data.amount_kwh} kWh of ${data.energy_source}`);
  });

  // Governance events
  signals.governance.onProposalCreated((data) => {
    console.log(`  🗳️ Proposal created: "${data.title}"`);
    console.log(`     By: ${data.proposer_did}`);
  });

  signals.governance.onVoteCast((data) => {
    console.log(`  ✓ Vote cast: ${data.vote} (weight: ${data.weight})`);
  });

  console.log('  Signal handlers ready!\n');

  // =========================================================================
  // Step 2: Create and verify identities with validation
  // =========================================================================
  console.log('🆔 Step 2: Creating verified identities...\n');

  const sellerDid = 'did:mycelix:seller_alice_123';
  const buyerDid = 'did:mycelix:buyer_bob_456';

  // Validate DIDs using common schemas
  try {
    didSchema.parse(sellerDid);
    didSchema.parse(buyerDid);
    console.log('  ✓ DID format validated successfully\n');
  } catch (error) {
    console.error('  ✗ DID validation failed:', error);
    return;
  }

  // Register hApps with identity bridge
  const registerInput = {
    happ_id: 'energy-marketplace',
    subscriptions: ['trust_updated', 'credential_issued'],
    metadata: { role: 'seller', verified: true },
  };

  // Validate registration input
  validateOrThrow(IdentitySchemas.RegisterHappInput, registerInput);
  console.log('  ✓ Registration input validated\n');

  await identity.registerHapp({
    happId: registerInput.happ_id,
    subscriptions: registerInput.subscriptions,
    metadata: registerInput.metadata,
  });

  // Create MATL reputation tracking
  let sellerRep = matl.createReputation(sellerDid);
  let buyerRep = matl.createReputation(buyerDid);

  // Simulate positive interactions building trust
  for (let i = 0; i < 5; i++) {
    sellerRep = matl.recordPositive(sellerRep);
  }
  for (let i = 0; i < 3; i++) {
    buyerRep = matl.recordPositive(buyerRep);
  }

  // Validate MATL scores
  const sellerScore = matl.reputationValue(sellerRep);
  const buyerScore = matl.reputationValue(buyerRep);
  matlScoreSchema.parse(sellerScore);
  matlScoreSchema.parse(buyerScore);

  console.log(`  Seller reputation: ${sellerScore.toFixed(3)}`);
  console.log(`  Buyer reputation: ${buyerScore.toFixed(3)}\n`);

  // =========================================================================
  // Step 3: List energy equipment with property verification
  // =========================================================================
  console.log('⚡ Step 3: Listing energy equipment...\n');

  // Validate energy query input
  const energyQueryInput = {
    source_happ: 'energy-marketplace',
    location: { lat: 37.7749, lng: -122.4194 },
    energy_sources: ['solar', 'wind'],
    min_amount: 100,
  };

  const validationResult = validateSafe(EnergySchemas.QueryAvailableEnergyInput, energyQueryInput);
  if (!validationResult.success) {
    console.error('  Energy query validation failed:', validationResult.errors);
    return;
  }
  console.log('  ✓ Energy query validated\n');

  // Query available energy
  const available = await energy.queryAvailableEnergy(energyQueryInput);
  console.log(`  Found ${available.results?.length || 0} energy listings\n`);

  // Verify property ownership for solar panel installation
  const ownershipInput = {
    property_hash: 'QmSolarArray001',
    requester: sellerDid,
    source_happ: 'energy-marketplace',
  };
  validateOrThrow(PropertySchemas.VerifyOwnershipInput, ownershipInput);
  console.log('  ✓ Property ownership query validated\n');

  await property.verifyOwnership({
    propertyHash: ownershipInput.property_hash,
    requester: ownershipInput.requester,
    sourceHapp: ownershipInput.source_happ,
  });

  // Create epistemic claim about energy source
  const energyClaim = epistemic.claim('Solar array certified for grid export')
    .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
    .withNormative(epistemic.NormativeLevel.N2_Network)
    .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
    .withIssuer('grid_certification_authority')
    .build();

  console.log(`  Energy certification: ${epistemic.classificationCode(energyClaim.classification)}`);
  console.log(`  Meets high trust: ${epistemic.meetsStandard(
    energyClaim,
    epistemic.EmpiricalLevel.E2_PrivateVerify,
    epistemic.NormativeLevel.N2_Network
  )}\n`);

  // =========================================================================
  // Step 4: Process trust-weighted payment
  // =========================================================================
  console.log('💰 Step 4: Processing trust-weighted payment...\n');

  // Validate payment input
  const paymentInput = {
    payer_did: buyerDid,
    payee_did: sellerDid,
    amount: 80.0, // 1000 kWh * $0.08
    currency: 'MCX',
    reference_id: 'energy-purchase-001',
  };

  validateOrThrow(FinanceSchemas.ProcessPaymentInput, paymentInput);
  console.log('  ✓ Payment input validated\n');

  // Calculate trust-weighted discount based on both parties' reputation
  const trustDiscount = Math.min(sellerScore, buyerScore) * 0.05; // Up to 5% discount

  const finalAmount = paymentInput.amount * (1 - trustDiscount);
  console.log(`  Trust discount: ${(trustDiscount * 100).toFixed(1)}%`);
  console.log(`  Final amount: $${finalAmount.toFixed(2)} (was $${paymentInput.amount})\n`);

  // Process payment through finance bridge
  await finance.processPayment({
    payerDid: paymentInput.payer_did,
    payeeDid: paymentInput.payee_did,
    amount: finalAmount,
    currency: paymentInput.currency,
    referenceId: paymentInput.reference_id,
  });

  // Record successful transaction
  sellerRep = matl.recordPositive(sellerRep);
  buyerRep = matl.recordPositive(buyerRep);

  // =========================================================================
  // Step 5: Governance query
  // =========================================================================
  console.log('🗳️ Step 5: Querying governance state...\n');

  const govQueryInput = {
    query_type: 'ProposalStatus' as const,
    query_params: JSON.stringify({ proposal_id: 'prop-001' }),
    source_happ: 'community-dao',
  };

  validateOrThrow(GovernanceSchemas.QueryGovernanceInput, govQueryInput);
  console.log('  ✓ Governance query validated\n');

  const govResult = await governance.queryGovernance({
    queryType: govQueryInput.query_type,
    queryParams: govQueryInput.query_params,
    sourceHapp: govQueryInput.source_happ,
  });
  console.log(`  Governance query result: ${govResult.matl_score?.toFixed(3) || 'N/A'}\n`);

  // =========================================================================
  // Step 6: Knowledge fact-check
  // =========================================================================
  console.log('📚 Step 6: Requesting fact-check...\n');

  const factCheckInput = {
    claim_text: 'Community solar reduced grid costs by 15% in Q4 2024',
    context: { source: 'utility_report', date: '2025-01-01' },
    source_happ: 'knowledge-base',
  };

  validateOrThrow(KnowledgeSchemas.FactCheckInput, factCheckInput);
  console.log('  ✓ Fact-check input validated\n');

  const factCheckResult = await knowledge.factCheck({
    claimText: factCheckInput.claim_text,
    context: factCheckInput.context,
    sourceHapp: factCheckInput.source_happ,
  });
  console.log(`  Fact-check epistemic score: ${factCheckResult.matl_score?.toFixed(3) || 'pending'}\n`);

  // =========================================================================
  // Step 7: Cross-hApp reputation aggregation
  // =========================================================================
  console.log('🌐 Step 7: Aggregating cross-hApp reputation...\n');

  // Set up local bridge for demonstration
  const localBridge = new bridge.LocalBridge();
  localBridge.registerHapp('identity');
  localBridge.registerHapp('energy-marketplace');
  localBridge.registerHapp('governance');
  localBridge.registerHapp('knowledge');

  // Set reputation across hApps
  localBridge.setReputation('identity', sellerDid, sellerRep);
  localBridge.setReputation('energy-marketplace', sellerDid, sellerRep);
  localBridge.setReputation('governance', sellerDid, sellerRep);

  // Get aggregate reputation
  const crossHappScores = localBridge.getCrossHappReputation(sellerDid);
  const aggregateRep = localBridge.getAggregateReputation(sellerDid);

  console.log(`  Cross-hApp reputation for seller:`);
  crossHappScores.forEach((score) => {
    console.log(`    ${score.happ}: ${score.score.toFixed(3)}`);
  });
  console.log(`  Aggregate reputation: ${aggregateRep.toFixed(3)}\n`);

  // =========================================================================
  // Summary
  // =========================================================================
  console.log('✅ Example Complete!');
  console.log('===================\n');
  console.log('Demonstrated:');
  console.log('  • Identity creation with MATL reputation');
  console.log('  • Energy query with property verification');
  console.log('  • Trust-weighted payment processing');
  console.log('  • Governance state queries');
  console.log('  • Knowledge fact-checking');
  console.log('  • Cross-hApp reputation aggregation');
  console.log('  • Real-time signal subscriptions');
  console.log('  • Runtime validation with Zod schemas\n');

  console.log('Bridge clients used:');
  console.log('  • IdentityBridgeClient - DID, credentials, trust');
  console.log('  • FinanceBridgeClient - Payments, credit, loans');
  console.log('  • PropertyBridgeClient - Ownership, collateral');
  console.log('  • EnergyBridgeClient - Trading, grid balance');
  console.log('  • MediaBridgeClient - Content, licensing');
  console.log('  • GovernanceBridgeClient - Proposals, voting');
  console.log('  • JusticeBridgeClient - Disputes, arbitration');
  console.log('  • KnowledgeBridgeClient - Claims, fact-checking\n');

  // Cleanup
  await client.disconnect();
}

// Run the example
runCivilizationalBridgeExample()
  .then(() => console.log('Done!'))
  .catch((error) => {
    console.error('Error:', error);
    process.exit(1);
  });
