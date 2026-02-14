/**
 * Example 17: Fully Homomorphic Encryption for Privacy-Preserving Voting
 *
 * This example demonstrates how to use the FHE module for:
 * - Creating encrypted votes that preserve voter privacy
 * - Aggregating votes without revealing individual choices
 * - Threshold decryption with multiple authorities
 * - Secure aggregation for federated learning
 */

import { fhe } from '../src/index.js';

async function main() {
  console.log('=== Mycelix FHE: Privacy-Preserving Voting Example ===\n');

  // =========================================================================
  // Part 1: Basic FHE Operations
  // =========================================================================
  console.log('--- Part 1: Basic FHE Operations ---\n');

  // Create an FHE client with development preset (fast, for testing)
  const client = new fhe.FHEClient({ preset: 'development' });
  await client.initialize();

  console.log(`FHE Provider: ${client.getProviderName()}`);
  console.log(`Scheme: ${client.getParams().scheme}`);

  // Encrypt some values
  const encryptedA = await client.encrypt(42);
  const encryptedB = await client.encrypt(8);

  console.log('\nEncrypted values (ciphertext size):');
  console.log(`  a = 42 -> ${encryptedA.data.length} bytes`);
  console.log(`  b = 8  -> ${encryptedB.data.length} bytes`);

  // Perform operations on encrypted data
  const encryptedSum = await client.add(encryptedA, encryptedB);
  const encryptedDiff = await client.subtract(encryptedA, encryptedB);
  const encryptedProd = await client.multiply(encryptedA, encryptedB);

  // Decrypt results
  const sum = await client.decrypt(encryptedSum);
  const diff = await client.decrypt(encryptedDiff);
  const prod = await client.decrypt(encryptedProd);

  console.log('\nHomomorphic operations:');
  console.log(`  a + b = ${sum[0]} (expected: 50)`);
  console.log(`  a - b = ${diff[0]} (expected: 34)`);
  console.log(`  a * b = ${prod[0]} (expected: 336)`);

  // =========================================================================
  // Part 2: Privacy-Preserving Voting
  // =========================================================================
  console.log('\n--- Part 2: Privacy-Preserving Voting ---\n');

  // Create a voting client optimized for ballot counting
  const votingClient = await fhe.createVotingClient();

  const proposalId = 'prop-2026-001';
  const voters = ['alice', 'bob', 'carol', 'dave', 'eve'];
  const choices = [1, 1, 0, 1, 0]; // 1 = Yes, 0 = No

  console.log('Proposal: "Should we fund the community solar project?"');
  console.log('Voters casting encrypted ballots...\n');

  // Each voter creates an encrypted vote
  const encryptedVotes = await Promise.all(
    voters.map(async (voterId, i) => {
      const vote = await votingClient.createEncryptedVote(
        proposalId,
        voterId,
        choices[i]
      );
      console.log(`  ${voterId} voted (encrypted)`);
      return vote;
    })
  );

  // Aggregate all votes homomorphically
  // The aggregator NEVER sees individual votes!
  console.log('\nAggregating encrypted votes...');
  const aggregation = await votingClient.aggregateVotes(encryptedVotes);

  console.log(`  Participants: ${aggregation.participants}`);
  console.log(`  Aggregation ID: ${aggregation.aggregationId}`);

  // Only the designated authority can decrypt the final tally
  console.log('\nDecrypting final tally (authority only)...');
  const tally = await votingClient.decrypt(aggregation.sum);

  console.log(`\nResults for ${proposalId}:`);
  console.log(`  Yes votes: ${tally[0]}`);
  console.log(`  No votes: ${aggregation.participants - tally[0]}`);
  console.log(`  Outcome: ${tally[0] > aggregation.participants / 2 ? 'PASSED' : 'FAILED'}`);

  // =========================================================================
  // Part 3: Secure Aggregation with Threshold Decryption
  // =========================================================================
  console.log('\n--- Part 3: Threshold Decryption (3-of-5) ---\n');

  // Create aggregator requiring 3 of 5 authorities to decrypt
  const aggregator = new fhe.SecureAggregator({
    threshold: 3,
    totalParties: 5,
    scheme: 'BFV',
  });

  await aggregator.initialize();
  console.log('Secure aggregator initialized');
  console.log('  Threshold: 3 of 5 parties required to decrypt\n');

  // Distribute key shares to authorities
  const shares = await aggregator.distributeShares();
  const authorities = ['Authority-A', 'Authority-B', 'Authority-C', 'Authority-D', 'Authority-E'];

  console.log('Key shares distributed to:');
  authorities.forEach((auth, i) => {
    console.log(`  ${auth}: Share ${i + 1} (${shares[i].share.length} bytes)`);
  });

  // Participants submit encrypted salary data for average calculation
  console.log('\nParticipants submitting encrypted salaries...');
  const salaries = [75000, 82000, 68000, 95000, 71000];

  for (let i = 0; i < salaries.length; i++) {
    const encrypted = await aggregator.encryptValue(salaries[i]);
    await aggregator.submitEncrypted(`employee-${i + 1}`, encrypted);
    console.log(`  Employee ${i + 1} submitted encrypted salary`);
  }

  // Aggregate (result is still encrypted)
  console.log('\nAggregating encrypted values...');
  const encryptedTotal = await aggregator.aggregate();

  // Threshold decryption - requires 3 authorities
  console.log('\nPerforming threshold decryption...');
  console.log('  Authorities A, C, and E providing partial decryptions...');

  const partialDecryptions = await Promise.all([
    aggregator.partialDecrypt(encryptedTotal, shares[0]), // Authority A
    aggregator.partialDecrypt(encryptedTotal, shares[2]), // Authority C
    aggregator.partialDecrypt(encryptedTotal, shares[4]), // Authority E
  ]);

  const totalSalary = await aggregator.combinePartialDecryptions(partialDecryptions);
  const averageSalary = totalSalary / salaries.length;

  console.log('\nDecryption complete!');
  console.log(`  Total salary sum: $${totalSalary.toLocaleString()}`);
  console.log(`  Average salary: $${averageSalary.toLocaleString()}`);
  console.log(`  Participants: ${salaries.length}`);

  // =========================================================================
  // Part 4: Secure FL Gradient Aggregation
  // =========================================================================
  console.log('\n--- Part 4: Secure FL Gradient Aggregation ---\n');

  const analyticsClient = await fhe.createAnalyticsClient();

  // Simulate 3 hospitals submitting encrypted gradients
  console.log('Hospitals submitting encrypted model gradients...\n');

  const hospitalGradients = [
    [0.10, 0.20, 0.15, -0.05, 0.12],  // Hospital A
    [0.12, 0.18, 0.16, -0.03, 0.10],  // Hospital B
    [0.09, 0.22, 0.14, -0.07, 0.11],  // Hospital C
  ];

  const encryptedGradients = await Promise.all(
    hospitalGradients.map(async (gradients, i) => {
      const encrypted = await analyticsClient.encrypt(gradients);
      console.log(`  Hospital ${String.fromCharCode(65 + i)}: Encrypted ${gradients.length} gradients`);
      return encrypted;
    })
  );

  // Server aggregates without seeing raw gradients
  console.log('\nServer aggregating gradients (still encrypted)...');
  const encryptedSum2 = await analyticsClient.sum(encryptedGradients);

  // Compute average
  const encryptedAvg = await analyticsClient.computeEncryptedAverage(
    encryptedGradients,
    hospitalGradients.length
  );

  // Check noise budget
  const noiseBudget = await analyticsClient.getNoiseBudget(encryptedAvg);
  console.log(`  Noise budget remaining: ${noiseBudget} bits`);

  // Decrypt aggregated gradients
  const aggregatedGradients = await analyticsClient.decrypt(encryptedAvg);

  console.log('\nAggregated gradients (decrypted):');
  aggregatedGradients.forEach((g, i) => {
    console.log(`  Gradient ${i}: ${g.toFixed(4)}`);
  });

  // Verify correctness
  const expectedAvg = hospitalGradients[0].map((_, i) =>
    hospitalGradients.reduce((sum, h) => sum + h[i], 0) / hospitalGradients.length
  );

  console.log('\nExpected averages (for verification):');
  expectedAvg.forEach((g, i) => {
    console.log(`  Gradient ${i}: ${g.toFixed(4)}`);
  });

  console.log('\n=== FHE Example Complete ===');
}

main().catch(console.error);
