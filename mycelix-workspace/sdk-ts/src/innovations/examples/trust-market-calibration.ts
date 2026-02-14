/**
 * Trust Market + Calibration Integration Example
 *
 * Demonstrates how trust markets create self-correcting belief revelation
 * by integrating with the calibration engine.
 *
 * @module innovations/examples/trust-market-calibration
 */

import { TrustMarketService, type MultiDimensionalTrustStake } from '../trust-markets/index.js';

/**
 * Helper to create a stake object
 */
function createStake(monetary: number, reputationPct: number): Partial<MultiDimensionalTrustStake> {
  return {
    monetary: { amount: monetary, currency: 'MATL' },
    reputation: {
      stakePercentage: reputationPct,
      currentReputation: 0.8,
      atRisk: 0.8 * reputationPct,
    },
  };
}

/**
 * Example: Trust Market with Calibration Feedback
 */
export async function trustMarketCalibrationExample() {
  const trustMarkets = new TrustMarketService();

  console.log('=== Trust Market + Calibration Integration ===\n');

  // Create a trust market for a reputation claim
  const market = await trustMarkets.createMarket({
    subjectDid: 'did:mycelix:alice-governance-rep',
    claim: {
      type: 'reputation_maintenance',
      threshold: 0.7,
      contextHapps: ['governance', 'finance'],
      durationMs: 30 * 24 * 60 * 60 * 1000,
      operator: 'gte',
    },
    title: "Will Alice's governance reputation stay >= 0.7?",
    creatorStake: 100,
    mechanismType: 'LMSR',
    liquidityParameter: 100,
  });

  console.log(`Created market: ${market.id}`);
  console.log(`Subject: ${market.subjectDid}`);
  console.log(`Claim: ${market.claim.type} (threshold: ${market.claim.threshold})`);

  // Traders submit positions
  await trustMarkets.submitTrade({
    marketId: market.id,
    participantId: 'did:mycelix:bob',
    outcome: 'yes',
    shares: 50,
    stake: createStake(100, 0.05),
  });
  console.log('\nBob bought 50 YES shares');

  await trustMarkets.submitTrade({
    marketId: market.id,
    participantId: 'did:mycelix:carol',
    outcome: 'no',
    shares: 30,
    stake: createStake(60, 0.03),
  });
  console.log('Carol bought 30 NO shares');

  // Check market-implied probability
  const updatedMarket = trustMarkets.getMarket(market.id);
  const impliedYes = updatedMarket?.state.impliedProbabilities.get('yes') ?? 0.5;

  console.log(`\nMarket-implied YES probability: ${(impliedYes * 100).toFixed(1)}%`);

  // Resolve market
  const resolution = await trustMarkets.resolveMarket(market.id, 'yes');

  console.log(`\n=== Market Resolved: YES ===`);
  console.log(`Calibration error: ${((1 - impliedYes) * 100).toFixed(1)}%`);

  for (const [participantId, payout] of Array.from(resolution.payouts)) {
    console.log(`${participantId}: ${payout.wasCorrect ? '✓' : '✗'} (${payout.monetaryPayout.toFixed(2)})`);
  }

  return { market, resolution, impliedYes };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  trustMarketCalibrationExample().catch(console.error);
}
