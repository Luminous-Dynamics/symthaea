/**
 * Full Ecosystem Example
 *
 * This example demonstrates a complete Mycelix ecosystem with all modules
 * working together: MATL trust, FL coordination, epistemic claims,
 * cross-hApp reputation, and security.
 *
 * Scenario: A decentralized healthcare network where hospitals contribute
 * to federated learning while maintaining reputation across multiple hApps.
 *
 * Run with: npx ts-node examples/06-full-ecosystem.ts
 */

import {
  // Configuration
  ConfigManager,
  PROD_CONFIG,

  // Trust & Reputation
  checkTrust,
  buildReputation,
  ReputationAggregator,

  // FL
  createSimpleFLCoordinator,
  createGradientUpdate,
  runFLRound,
  fl,

  // Epistemic
  claim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  meetsStandard,
  Standards,

  // Security
  signMessage,
  verifyMessage,
  createRateLimitedOperation,
  security,

  // Health
  runHealthChecks,
  getSdkInfo,
} from '../src/index.js';

// Types for our simulation
interface Hospital {
  id: string;
  name: string;
  tier: 'academic' | 'regional' | 'community';
  dataQuality: number;
  uptime: number;
}

interface TrainingRound {
  roundId: number;
  participants: string[];
  aggregatedLoss: number;
  timestamp: Date;
}

async function main() {
  console.log('🍄 Mycelix Full Ecosystem Example');
  console.log('━'.repeat(50));
  console.log('Scenario: Decentralized Healthcare FL Network\n');

  // Step 1: Initialize Configuration
  console.log('=== Step 1: System Configuration ===');

  const config = new ConfigManager(PROD_CONFIG);
  config.set({
    matl: {
      ...PROD_CONFIG.matl,
      byzantineThreshold: 0.34, // Medical data requires high BFT
    },
    fl: {
      ...PROD_CONFIG.fl,
      minParticipants: 3,
      byzantineTolerance: 0.33,
    },
    security: {
      ...PROD_CONFIG.security,
      minKeyLength: 256,
      maxRateLimitRequests: 100,
    },
  });

  console.log('Configuration initialized with production settings');
  console.log(`  Byzantine threshold: ${config.get().matl.byzantineThreshold}`);
  console.log(`  Min FL participants: ${config.get().fl.minParticipants}`);
  console.log(`  Rate limit: ${config.get().security.maxRateLimitRequests} req/window`);
  console.log();

  // Step 2: Run Health Checks
  console.log('=== Step 2: System Health Check ===');

  const health = runHealthChecks();
  const allHealthy = health.every((h) => h.healthy);
  console.log(`System status: ${allHealthy ? '✅ All modules healthy' : '⚠️ Issues detected'}`);

  const info = getSdkInfo();
  console.log(`SDK Version: ${info.version}`);
  console.log(`Active modules: ${health.filter((h) => h.healthy).length}/${health.length}`);
  console.log();

  // Step 3: Set Up Cross-hApp Reputation
  console.log('=== Step 3: Cross-hApp Reputation Setup ===');

  const reputationAggregator = new ReputationAggregator();

  // Register healthcare ecosystem hApps
  const ecosystemHapps = [
    'patient-records',
    'diagnostic-imaging',
    'clinical-trials',
    'research-collaboration',
  ];

  ecosystemHapps.forEach((happ) => reputationAggregator.registerHapp(happ));
  console.log(`Registered ${ecosystemHapps.length} healthcare hApps`);
  console.log();

  // Step 4: Register Hospitals with Reputation
  console.log('=== Step 4: Hospital Registration ===');

  const hospitals: Hospital[] = [
    { id: 'mayo-clinic', name: 'Mayo Clinic', tier: 'academic', dataQuality: 0.95, uptime: 0.99 },
    { id: 'cleveland-clinic', name: 'Cleveland Clinic', tier: 'academic', dataQuality: 0.93, uptime: 0.98 },
    { id: 'regional-med', name: 'Regional Medical Center', tier: 'regional', dataQuality: 0.85, uptime: 0.95 },
    { id: 'community-health', name: 'Community Health Network', tier: 'community', dataQuality: 0.80, uptime: 0.92 },
    { id: 'university-hospital', name: 'University Hospital', tier: 'academic', dataQuality: 0.91, uptime: 0.97 },
  ];

  // Build reputation from historical interactions
  const hospitalReputations = hospitals.map((hospital) => {
    // Simulate historical interaction counts based on tier
    const basePositive = hospital.tier === 'academic' ? 100 : hospital.tier === 'regional' ? 50 : 25;
    const baseNegative = Math.floor(basePositive * (1 - hospital.dataQuality) * 0.5);

    // Set reputation in each hApp
    for (const happ of ecosystemHapps) {
      const modifier = Math.random() * 0.2 + 0.9; // 0.9-1.1x variation
      reputationAggregator.setReputation(
        happ,
        hospital.id,
        Math.floor(basePositive * modifier),
        Math.floor(baseNegative * modifier)
      );
    }

    // Get aggregate reputation
    const aggregate = reputationAggregator.getAggregateReputation(hospital.id);

    return {
      hospital,
      aggregate,
      reputation: buildReputation(hospital.id, basePositive, baseNegative),
    };
  });

  console.log('Hospital Reputation Summary:');
  console.log('-'.repeat(60));
  for (const { hospital, aggregate } of hospitalReputations) {
    const tierIcon = hospital.tier === 'academic' ? '🏥' : hospital.tier === 'regional' ? '🏨' : '🏠';
    console.log(`${tierIcon} ${hospital.name.padEnd(30)} Aggregate: ${aggregate.toFixed(3)}`);
  }
  console.log();

  // Step 5: Issue Epistemic Claims for Compliance
  console.log('=== Step 5: Compliance Claims ===');

  const complianceClaims = hospitals.map((hospital) => {
    // Academic hospitals get cryptographic verification, others get private verification
    const empiricalLevel =
      hospital.tier === 'academic'
        ? EmpiricalLevel.E3_Cryptographic
        : EmpiricalLevel.E2_PrivateVerify;

    return claim(`${hospital.name} complies with HIPAA and research ethics standards`)
      .withClassification(empiricalLevel, NormativeLevel.N2_Network, MaterialityLevel.M2_Persistent)
      .withIssuer('healthcare-compliance-authority')
      .withEvidence({
        type: 'audit-report',
        description: `Annual compliance audit completed ${new Date().toISOString().slice(0, 10)}`,
      })
      .withExpiration(new Date(Date.now() + 365 * 24 * 60 * 60 * 1000))
      .build();
  });

  console.log('Compliance verification:');
  for (let i = 0; i < hospitals.length; i++) {
    const meetsNetwork = meetsStandard(complianceClaims[i], Standards.NETWORK);
    const meetsFoundational = meetsStandard(complianceClaims[i], Standards.FOUNDATIONAL);
    console.log(
      `  ${hospitals[i].name.padEnd(30)} Network: ${meetsNetwork ? '✅' : '❌'}  Foundational: ${meetsFoundational ? '✅' : '❌'}`
    );
  }
  console.log();

  // Step 6: Trust Assessment for FL Participation
  console.log('=== Step 6: Trust Assessment ===');

  const trustedParticipants = hospitalReputations
    .map(({ hospital, aggregate, reputation }) => {
      // Calculate trust based on data quality, consistency, and reputation
      const trustResult = checkTrust(
        hospital.id,
        hospital.dataQuality, // quality
        hospital.uptime, // consistency
        1 - hospital.dataQuality, // entropy (inverse of quality)
        { existingReputation: reputation }
      );

      return {
        hospital,
        trust: trustResult,
        eligible: trustResult.trustworthy && trustResult.score > 0.6,
      };
    })
    .filter((p) => p.eligible);

  console.log('Trust-verified FL participants:');
  for (const { hospital, trust } of trustedParticipants) {
    console.log(
      `  ✅ ${hospital.name.padEnd(30)} Score: ${trust.score.toFixed(3)} Byzantine: ${trust.byzantine}`
    );
  }

  const excluded = hospitals.length - trustedParticipants.length;
  if (excluded > 0) {
    console.log(`  ❌ ${excluded} hospital(s) excluded due to trust requirements`);
  }
  console.log();

  // Step 7: Federated Learning Round
  console.log('=== Step 7: Federated Learning Round ===');

  const flCoordinator = createSimpleFLCoordinator({
    minParticipants: 3,
    byzantineTolerance: 0.33,
    aggregationMethod: fl.AggregationMethod.TrustWeighted,
  });

  // Register trusted participants
  for (const { hospital } of trustedParticipants) {
    flCoordinator.registerParticipant(hospital.id);
  }

  // Simulate gradient updates (e.g., for a disease prediction model)
  const gradientUpdates = trustedParticipants.map(({ hospital }) => {
    // Simulate gradients - higher quality hospitals have more consistent gradients
    const baseGradient = [0.1, -0.05, 0.08, -0.03, 0.06, -0.02, 0.04, -0.01];
    const noise = hospital.dataQuality * 0.1; // Lower noise for higher quality

    const gradients = baseGradient.map((g) => g + (Math.random() - 0.5) * noise);

    return createGradientUpdate(hospital.id, 1, gradients, {
      batchSize: hospital.tier === 'academic' ? 1000 : hospital.tier === 'regional' ? 500 : 200,
      loss: 0.3 + (1 - hospital.dataQuality) * 0.2,
      accuracy: hospital.dataQuality * 0.95,
    });
  });

  const roundSummary = runFLRound(flCoordinator, gradientUpdates);

  console.log('FL Round Results:');
  console.log(`  Round ID: ${roundSummary.roundId}`);
  console.log(`  Status: ${roundSummary.status}`);
  console.log(`  Participants: ${roundSummary.participantCount}`);
  console.log(`  Updates processed: ${roundSummary.updateCount}`);
  console.log(`  Duration: ${roundSummary.duration}ms`);
  console.log();

  // Step 8: Secure Result Signing
  console.log('=== Step 8: Secure Result Signing ===');

  const signingKey = security.secureRandomBytes(32);

  const roundResult: TrainingRound = {
    roundId: roundSummary.roundId,
    participants: trustedParticipants.map((p) => p.hospital.id),
    aggregatedLoss: 0.28, // Simulated
    timestamp: new Date(),
  };

  const signedResult = await signMessage(roundResult, signingKey);

  console.log('Signed training round result:');
  console.log(`  Round: ${signedResult.payload.roundId}`);
  console.log(`  Participants: ${signedResult.payload.participants.length}`);
  console.log(`  Signature: ${signedResult.signature.slice(0, 32)}...`);

  // Verify the signed result
  const verification = await verifyMessage(signedResult, signingKey);
  console.log(`  Verification: ${verification.valid ? '✅ Valid' : '❌ Invalid'}`);
  console.log();

  // Step 9: Rate-Limited API Access
  console.log('=== Step 9: Rate-Limited Model Access ===');

  const modelAccessApi = createRateLimitedOperation(
    () => ({
      modelVersion: '1.0.0',
      lastUpdated: new Date().toISOString(),
      participantCount: trustedParticipants.length,
    }),
    5, // Max 5 requests
    10000 // Per 10 seconds
  );

  console.log('Simulating API access (5 requests allowed per 10s):');
  for (let i = 1; i <= 7; i++) {
    const result = await modelAccessApi.execute();
    if (result) {
      console.log(`  Request ${i}: ✅ Model v${result.modelVersion}`);
    } else {
      console.log(`  Request ${i}: ❌ Rate limited`);
    }
  }
  console.log();

  // Step 10: Final Summary
  console.log('=== Ecosystem Summary ===');
  console.log('━'.repeat(50));
  console.log(`Total hospitals in network: ${hospitals.length}`);
  console.log(`Trust-verified participants: ${trustedParticipants.length}`);
  console.log(`Active hApps: ${ecosystemHapps.length}`);
  console.log(`Compliance claims issued: ${complianceClaims.length}`);
  console.log(`FL rounds completed: 1`);
  console.log('');
  console.log('Key metrics:');
  const avgReputation =
    trustedParticipants.reduce((sum, p) => sum + p.trust.score, 0) / trustedParticipants.length;
  console.log(`  Average trust score: ${avgReputation.toFixed(3)}`);
  console.log(`  Byzantine tolerance: 33%`);
  console.log(`  Data integrity: ${verification.valid ? 'Verified' : 'Compromised'}`);
  console.log('');
  console.log('🎉 Healthcare FL network operating successfully!');
}

main().catch(console.error);
