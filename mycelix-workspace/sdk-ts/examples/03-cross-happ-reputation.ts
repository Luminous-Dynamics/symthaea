/**
 * Cross-hApp Reputation Example
 *
 * Demonstrates how to aggregate reputation across multiple Holochain hApps
 * using the Bridge module and ReputationAggregator utility.
 *
 * Run with: npx ts-node examples/03-cross-happ-reputation.ts
 */

import {
  // Bridge module
  LocalBridge,
  BridgeMessageType,
  createReputationQuery,
  createCrossHappReputation,
  calculateAggregateReputation,

  // Utility
  ReputationAggregator,
  buildReputation,
} from '../src/index.js';

async function main() {
  console.log('🍄 Cross-hApp Reputation Example\n');

  // Scenario: An agent participates in multiple hApps in the Mycelix ecosystem
  // Each hApp tracks reputation independently, but we can aggregate for trust decisions

  console.log('=== Setting Up Multi-hApp Ecosystem ===');

  const aggregator = new ReputationAggregator();

  // Register various hApps
  const happs = [
    { id: 'marketplace', desc: 'Peer-to-peer marketplace for goods' },
    { id: 'task-network', desc: 'Distributed task coordination' },
    { id: 'knowledge-base', desc: 'Collaborative knowledge curation' },
    { id: 'energy-trading', desc: 'Community energy trading platform' },
  ];

  for (const happ of happs) {
    aggregator.registerHapp(happ.id);
    console.log(`  📱 Registered: ${happ.id} - ${happ.desc}`);
  }
  console.log();

  // Simulate Alice's activity across hApps
  console.log('=== Alice\'s Cross-hApp Activity ===');

  // Alice has different reputation in each hApp based on her activity
  const aliceReputation = {
    'marketplace': { positive: 45, negative: 3 },     // Great seller
    'task-network': { positive: 30, negative: 8 },    // Good but sometimes misses deadlines
    'knowledge-base': { positive: 100, negative: 2 }, // Excellent contributor
    'energy-trading': { positive: 15, negative: 1 },  // New to this hApp
  };

  console.log('Alice\'s reputation in each hApp:');
  for (const [happId, rep] of Object.entries(aliceReputation)) {
    aggregator.setReputation(happId, 'alice', rep.positive, rep.negative);
    const total = rep.positive + rep.negative + 2; // +2 for Bayesian prior
    const score = ((rep.positive + 1) / total).toFixed(3);
    console.log(`  ${happId}: ${rep.positive}+ / ${rep.negative}- (score: ${score})`);
  }
  console.log();

  // Get aggregate reputation
  console.log('=== Aggregate Reputation ===');
  const aliceAggregate = aggregator.getAggregateReputation('alice');
  console.log(`Alice's aggregate reputation: ${aliceAggregate.toFixed(3)}`);
  console.log();

  // Get detailed breakdown
  console.log('=== Detailed Cross-hApp Breakdown ===');
  const aliceDetails = aggregator.getDetailedReputation('alice');
  for (const detail of aliceDetails) {
    console.log(`  ${detail.happId}: score=${detail.reputation.toFixed(3)}, weight=${detail.weight.toFixed(2)}`);
  }
  console.log();

  // Demonstrate message protocol
  console.log('=== Bridge Message Protocol ===');

  // When a new hApp wants to query Alice's reputation
  const query = createReputationQuery('new-happ', 'alice');
  console.log('Reputation Query Message:');
  console.log(`  Type: ${query.type}`);
  console.log(`  Source: ${query.sourceHappId}`);
  console.log(`  Agent: ${query.agentId}`);
  console.log();

  // Response with cross-hApp reputation data
  const response = createCrossHappReputation('alice', aliceDetails);
  console.log('Cross-hApp Reputation Response:');
  console.log(`  Agent: ${response.agentId}`);
  console.log(`  Sources: ${response.sources.length} hApps`);
  console.log();

  // Demonstrate LocalBridge for hApp integration
  console.log('=== LocalBridge Integration ===');

  const bridge = new LocalBridge();

  // Register hApps
  bridge.registerHapp('social-network');
  bridge.registerHapp('governance');

  // Set reputation
  const bobRep = buildReputation('bob', 25, 5);
  bridge.setReputation('social-network', 'bob', bobRep);

  const bobRepGov = buildReputation('bob', 40, 2);
  bridge.setReputation('governance', 'bob', bobRepGov);

  // Query cross-hApp reputation
  const bobCrossHapp = bridge.getCrossHappReputation('bob');
  console.log('Bob\'s cross-hApp reputation via LocalBridge:');
  for (const entry of bobCrossHapp) {
    console.log(`  ${entry.happId}: ${entry.reputation.toFixed(3)}`);
  }

  const bobAggregate = bridge.getAggregateReputation('bob');
  console.log(`  Aggregate: ${bobAggregate.toFixed(3)}`);
  console.log();

  // Demonstrate trust decision based on aggregate reputation
  console.log('=== Trust Decision Example ===');

  const trustThreshold = 0.7;

  const agents = [
    { name: 'alice', aggregate: aliceAggregate },
    { name: 'bob', aggregate: bobAggregate },
  ];

  console.log(`Trust threshold: ${trustThreshold}`);
  for (const agent of agents) {
    const trusted = agent.aggregate >= trustThreshold;
    const status = trusted ? '✅ TRUSTED' : '⚠️ VERIFY';
    console.log(`  ${agent.name}: ${agent.aggregate.toFixed(3)} - ${status}`);
  }
  console.log();

  // Show calculation method
  console.log('=== Aggregate Calculation Method ===');
  console.log('Aggregation uses weighted average where weight = hApp reputation score');
  console.log('This gives more weight to reputation from hApps where the agent is more active.');

  // Demonstrate manual calculation
  const manualScores = aliceDetails.map((d) => ({
    happId: d.happId,
    score: d.reputation,
    weight: d.weight,
    contribution: d.reputation * d.weight,
  }));

  const totalWeight = manualScores.reduce((sum, s) => sum + s.weight, 0);
  const weightedSum = manualScores.reduce((sum, s) => sum + s.contribution, 0);
  const calculated = weightedSum / totalWeight;

  console.log('\nManual calculation breakdown:');
  for (const s of manualScores) {
    console.log(`  ${s.happId}: ${s.score.toFixed(3)} × ${s.weight.toFixed(2)} = ${s.contribution.toFixed(3)}`);
  }
  console.log(`  Total: ${weightedSum.toFixed(3)} / ${totalWeight.toFixed(2)} = ${calculated.toFixed(3)}`);
}

main().catch(console.error);
