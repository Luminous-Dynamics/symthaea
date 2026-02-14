/**
 * Holochain Wallet Integration Example
 *
 * This example demonstrates the complete "Wiring the Nervous System" integration
 * connecting your wallet to a real Holochain conductor with:
 * - ConductorManager for WebSocket communication
 * - HolochainIdentityResolver for profile lookup
 * - HolochainFinanceProvider for payments
 * - OfflineQueue for resilient transactions
 * - WalletBridge for cross-hApp reputation
 *
 * Run with: npx ts-node examples/20-holochain-wallet.ts
 *
 * @module examples/holochain-wallet
 */

import {
  // Wallet factories
  createHolochainWallet,
  createDevWallet,
  isConductorAvailable,

  // Wallet bridge for cross-hApp reputation
  createWalletBridge,
  type WalletBridgeConfig,
  type TransactionRecommendation,
} from '../src/wallet/index.js';

// =============================================================================
// Example 1: Development Wallet (Mock Mode)
// =============================================================================

async function developmentWalletExample() {
  console.log('🔧 Development Wallet Example\n');
  console.log('Creating a mock wallet for UI development...\n');

  // createDevWallet() creates a fully functional wallet with mock backends
  // Perfect for UI development when you don't have a conductor running
  const { wallet, bridge, destroy } = await createDevWallet();

  // Unlock the wallet (dev wallet uses PIN '0000' by default)
  await wallet.unlockWithPin('0000');
  console.log(`✅ Wallet unlocked: ${!wallet.locked}`);

  // Check initial balances
  const state = wallet.state;
  console.log('\n📊 Initial Balances:');
  for (const [currency, balance] of state.balances) {
    console.log(`  ${currency}: ${balance.available} (${balance.pending} pending)`);
  }

  // Send a transaction (mock mode - instant)
  console.log('\n💸 Sending 50 MYC to did:test:alice...');
  const txResult = await wallet.send('did:test:alice', 50, 'MYC', 'Coffee payment');
  console.log(`  Transaction ID: ${txResult.txId || txResult.id || 'tx-' + Date.now()}`);

  // Check reputation via bridge
  console.log('\n🌐 Checking cross-hApp reputation...');
  const reputation = await bridge.getAggregateReputation('did:test:alice');
  console.log(`  Alice's aggregate score: ${reputation.aggregatedScore.toFixed(3)}`);
  console.log(`  Confidence: ${reputation.confidence.toFixed(3)}`);

  // Get transaction recommendation
  const recommendation = await bridge.getTransactionRecommendation('did:test:alice', 100);
  console.log(`\n📋 Transaction Recommendation:`);
  console.log(`  Risk Level: ${recommendation.riskLevel}`);
  console.log(`  Requires Verification: ${recommendation.requiresVerification}`);
  console.log(`  Advice: ${recommendation.recommendation}`);

  // Clean up
  await destroy();
  console.log('\n✅ Development wallet example complete!\n');
}

// =============================================================================
// Example 2: Production Wallet (Real Conductor)
// =============================================================================

async function productionWalletExample() {
  console.log('🚀 Production Wallet Example\n');

  // First, check if conductor is available
  const conductorUrl = 'ws://localhost:8888';
  const available = await isConductorAvailable(conductorUrl);

  if (!available) {
    console.log(`⚠️  Conductor not available at ${conductorUrl}`);
    console.log('   Start your conductor with: just dev');
    console.log('   Skipping production example.\n');
    return;
  }

  console.log(`✅ Conductor available at ${conductorUrl}\n`);

  // Create a fully-wired Holochain wallet
  const {
    wallet,
    conductor,
    identityResolver,
    financeProvider,
    offlineQueue,
    bridge,
    destroy,
  } = await createHolochainWallet({
    installedAppId: 'mycelix-wallet',
    appUrl: conductorUrl,
    enableOfflineQueue: true,
    refreshIntervalMs: 30000, // Auto-refresh balances every 30s
  });

  console.log('📡 Connected to Holochain conductor');
  console.log(`   Agent Key: ${conductor.agentPubKey?.slice(0, 16)}...`);

  // Unlock wallet
  await wallet.unlockWithPin('0000');
  console.log(`\n✅ Wallet unlocked`);

  // Resolve an identity
  console.log('\n👤 Resolving identity...');
  try {
    const profile = await identityResolver.resolve('test-agent');
    console.log(`   Resolved: ${profile?.nickname || 'Unknown'}`);
  } catch (e) {
    console.log(`   (Identity lookup failed - expected in demo)`);
  }

  // Check balances via finance provider
  console.log('\n💰 Checking balances...');
  const balances = await financeProvider.getAllBalances(conductor.agentPubKey?.toString() || '');
  for (const [currency, amount] of balances) {
    console.log(`   ${currency}: ${amount}`);
  }

  // Check offline queue status
  if (offlineQueue) {
    console.log('\n📴 Offline Queue Status:');
    console.log(`   Pending operations: ${offlineQueue.pendingCount}`);
    console.log(`   Processing: ${offlineQueue.isProcessing}`);
  }

  // Cross-hApp reputation
  console.log('\n🌐 Cross-hApp Reputation:');
  const rep = await bridge.getAggregateReputation('did:mycelix:test');
  console.log(`   Aggregate Score: ${rep.aggregatedScore.toFixed(3)}`);
  console.log(`   Sources: ${Object.keys(rep.scores).join(', ') || 'None'}`);

  // Clean up
  await destroy();
  console.log('\n✅ Production wallet example complete!\n');
}

// =============================================================================
// Example 3: Wallet Bridge Integration
// =============================================================================

async function walletBridgeExample() {
  console.log('🌐 Wallet Bridge Integration Example\n');

  // Create a standalone bridge (without finance provider)
  const bridgeConfig: WalletBridgeConfig = {
    defaultContextHapps: ['finance', 'identity', 'marketplace', 'governance'],
    reputationWeights: {
      identity: 1.5, // Identity verification is most important
      finance: 1.3, // Payment history matters
      marketplace: 1.2, // Transaction history
      governance: 1.1, // Participation weight
      justice: 1.4, // Dispute resolution reputation
    },
    cacheDurationMs: 60000, // Cache for 1 minute
  };

  const bridge = createWalletBridge(undefined, bridgeConfig);

  // Query reputation for multiple users
  console.log('📊 Querying reputation for test users...\n');

  const testUsers = [
    'did:test:alice',
    'did:test:bob',
    'did:test:unknown-user',
  ];

  for (const did of testUsers) {
    const rep = await bridge.getAggregateReputation(did);
    const isTrusted = await bridge.isTrustworthy(did, 0.6);

    console.log(`${did}:`);
    console.log(`  Score: ${rep.aggregatedScore.toFixed(3)}`);
    console.log(`  Confidence: ${rep.confidence.toFixed(3)}`);
    console.log(`  Trustworthy (≥0.6): ${isTrusted ? '✅' : '❌'}`);
    console.log();
  }

  // Transaction recommendations at different amounts
  console.log('💰 Transaction Recommendations:\n');
  const amounts = [10, 100, 1000, 10000];

  for (const amount of amounts) {
    const rec = await bridge.getTransactionRecommendation('did:test:new-contact', amount);
    console.log(`  $${amount.toLocaleString()}:`);
    console.log(`    Risk: ${rec.riskLevel}`);
    console.log(`    Verify: ${rec.requiresVerification ? 'Yes' : 'No'}`);
  }

  // Subscribe to reputation changes
  console.log('\n📡 Setting up reputation subscriptions...');

  const unsubscribe = bridge.onReputationChange('did:test:alice', (newRep) => {
    console.log(`  🔔 Alice reputation changed: ${newRep.aggregatedScore.toFixed(3)}`);
  });

  // Report interactions
  console.log('\n📝 Reporting interactions...');
  bridge.reportPositiveInteraction('did:test:alice', 'tx-123');
  console.log('  ✅ Reported positive interaction with Alice');

  bridge.reportNegativeInteraction('did:test:scammer', 'payment_failed');
  console.log('  ❌ Reported payment failure with Scammer');

  // Clean up
  unsubscribe();
  bridge.destroy();

  console.log('\n✅ Wallet bridge example complete!\n');
}

// =============================================================================
// Example 4: Offline-First Transactions
// =============================================================================

async function offlineFirstExample() {
  console.log('📴 Offline-First Transactions Example\n');

  // Create dev wallet with offline queue
  const { wallet, destroy } = await createDevWallet();

  await wallet.unlockWithPin('0000');

  console.log('Simulating offline scenario...\n');

  // Queue multiple transactions (would be batched when online)
  const transactions = [
    { to: 'did:test:alice', amount: 25, memo: 'Lunch share' },
    { to: 'did:test:bob', amount: 10, memo: 'Coffee' },
    { to: 'did:test:carol', amount: 15, memo: 'Snacks' },
  ];

  console.log('📤 Queuing transactions:');
  for (const tx of transactions) {
    try {
      const result = await wallet.send(tx.to, tx.amount, 'MYC', tx.memo);
      console.log(`  ✅ ${tx.memo}: ${tx.amount} MYC → ${tx.to.slice(-10)}`);
    } catch (e) {
      console.log(`  ⏳ ${tx.memo}: Queued for later (offline)`);
    }
  }

  // Show transaction history
  console.log('\n📜 Transaction History:');
  const state = wallet.state;
  for (const tx of state.transactions.slice(-5)) {
    const direction = tx.direction === 'outgoing' ? '↗️' : '↙️';
    console.log(`  ${direction} ${tx.amount} ${tx.currency} - ${tx.status}`);
  }

  await destroy();
  console.log('\n✅ Offline-first example complete!\n');
}

// =============================================================================
// Example 5: Identity Verification Integration
// =============================================================================

async function identityVerificationExample() {
  console.log('🔐 Identity Verification Example\n');

  const bridge = createWalletBridge();

  // Request identity verification
  console.log('Requesting identity verification...\n');

  const verifyDid = 'did:test:new-contact';
  const status = await bridge.requestIdentityVerification(verifyDid);

  console.log(`Identity Verification for ${verifyDid}:`);
  console.log(`  Verified: ${status.verified ? '✅' : '❌'}`);
  console.log(`  Level: ${status.level}`);
  if (status.verifiedAt) {
    console.log(`  Verified At: ${new Date(status.verifiedAt).toISOString()}`);
  }

  // Verify specific credentials
  console.log('\n📜 Credential Verification:');
  const credentials = ['diploma', 'employment', 'address'];

  for (const cred of credentials) {
    const credStatus = await bridge.verifyCredential(verifyDid, cred);
    const icon = credStatus.verified ? '✅' : '❌';
    console.log(`  ${icon} ${cred}: Level ${credStatus.level}`);
  }

  bridge.destroy();
  console.log('\n✅ Identity verification example complete!\n');
}

// =============================================================================
// Example 6: Complete Payment Flow
// =============================================================================

async function completePaymentFlowExample() {
  console.log('💳 Complete Payment Flow Example\n');

  const { wallet, bridge, destroy } = await createDevWallet();
  await wallet.unlockWithPin('0000');

  const recipientDid = 'did:test:merchant';
  const paymentAmount = 150;

  console.log(`Processing payment of ${paymentAmount} MYC to ${recipientDid}\n`);

  // Step 1: Check recipient reputation
  console.log('Step 1: Checking recipient reputation...');
  const reputation = await bridge.getAggregateReputation(recipientDid);
  console.log(`  Score: ${reputation.aggregatedScore.toFixed(3)}`);

  // Step 2: Get transaction recommendation
  console.log('\nStep 2: Getting transaction recommendation...');
  const recommendation = await bridge.getTransactionRecommendation(recipientDid, paymentAmount);
  console.log(`  Risk: ${recommendation.riskLevel}`);
  console.log(`  Advice: ${recommendation.recommendation}`);

  // Step 3: Verify identity if required
  if (recommendation.requiresVerification) {
    console.log('\nStep 3: Verifying identity (required for this amount)...');
    const verifyStatus = await bridge.requestIdentityVerification(recipientDid);
    console.log(`  Verified: ${verifyStatus.verified ? '✅' : '❌'}`);
  } else {
    console.log('\nStep 3: Identity verification not required');
  }

  // Step 4: Execute transaction
  console.log('\nStep 4: Executing transaction...');
  if (recommendation.riskLevel === 'very_high' && !recommendation.requiresVerification) {
    console.log('  ⚠️ High risk - transaction blocked for safety');
  } else {
    const result = await wallet.send(recipientDid, paymentAmount, 'MYC', 'Purchase');
    const txId = result.txId || result.id || `tx-${Date.now()}`;
    console.log(`  ✅ Transaction complete: ${txId}`);

    // Step 5: Report successful interaction
    console.log('\nStep 5: Reporting successful interaction...');
    bridge.reportPositiveInteraction(recipientDid, txId);
    console.log('  ✅ Positive interaction recorded');
  }

  // Show final balance
  console.log('\nFinal Balance:');
  const state = wallet.state;
  for (const [currency, balance] of state.balances) {
    console.log(`  ${currency}: ${balance.available} (${balance.pending} pending)`);
  }

  await destroy();
  console.log('\n✅ Complete payment flow example complete!\n');
}

// =============================================================================
// Main
// =============================================================================

async function main() {
  console.log('🍄 Mycelix Holochain Wallet Integration Examples\n');
  console.log('='.repeat(60) + '\n');

  // Run all examples
  await developmentWalletExample();
  console.log('='.repeat(60) + '\n');

  await productionWalletExample();
  console.log('='.repeat(60) + '\n');

  await walletBridgeExample();
  console.log('='.repeat(60) + '\n');

  await offlineFirstExample();
  console.log('='.repeat(60) + '\n');

  await identityVerificationExample();
  console.log('='.repeat(60) + '\n');

  await completePaymentFlowExample();
  console.log('='.repeat(60) + '\n');

  console.log('🎉 All Holochain wallet examples completed!\n');
}

main().catch(console.error);
