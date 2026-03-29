// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Example 12: Self-Correcting Epistemic Infrastructure (SCEI) Workflow
 *
 * This example demonstrates the complete SCEI architecture:
 * 1. Calibration Engine - Tracks prediction accuracy
 * 2. Epistemic Algebra - Combines claims mathematically
 * 3. Knowledge Metabolism - Manages claim lifecycles
 * 4. Safe Belief Propagator - Circuit breakers prevent cascade failures
 * 5. Gap Detector - Finds unknown unknowns with humility
 *
 * Key principle: The system corrects itself over time through honest
 * tracking of when it's wrong.
 */

import {
  // Calibration
  CalibrationEngine,
  type CalibrationReport,

  // Epistemic Algebra
  conjunction,
  corroborate,
  chainDegradation,
  applyELevelCeiling,
  E_LEVEL_CONFIDENCE_CEILING,

  // Epistemic Core
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  createClaim,
  type EpistemicClaim,

  // Metabolism
  MetabolismEngine,
  type MetabolizingClaim,
  type Tombstone,

  // Propagation
  SafeBeliefPropagator,
  DEFAULT_CIRCUIT_BREAKER_CONFIG,

  // Discovery
  GapDetector,
  type KnowledgeGap,
} from '../src/index.js';

// ============================================================================
// SCENARIO: Scientific Knowledge Graph
// ============================================================================

async function demonstrateSCEIWorkflow() {
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('  SELF-CORRECTING EPISTEMIC INFRASTRUCTURE (SCEI) DEMONSTRATION  ');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log();

  // Initialize all SCEI components
  const calibration = new CalibrationEngine({ minSampleSize: 5 });
  const metabolism = new MetabolismEngine({ autoTransition: true });
  const propagator = new SafeBeliefPropagator(calibration);
  const gapDetector = new GapDetector({ enableCrossDomain: true });

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 1: BIRTH OF CLAIMS
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 1: Birth of Claims                                         │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  // Birth scientific claims through metabolism engine
  const climateClaim = await metabolism.birthClaim({
    content: 'Global average temperature has risen 1.1°C since pre-industrial times',
    classification: {
      empirical: EmpiricalLevel.E4_Consensus,
      normative: NormativeLevel.N3_Axiomatic,
      materiality: MaterialityLevel.M3_Foundational,
    },
    authorId: 'ipcc-2023',
    domain: 'climate_science',
    confidence: 0.95,
    tags: ['climate', 'temperature', 'global'],
  });

  const oceanClaim = await metabolism.birthClaim({
    content: 'Ocean heat content has increased significantly',
    classification: {
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    },
    authorId: 'noaa-2023',
    domain: 'oceanography',
    confidence: 0.88,
    tags: ['ocean', 'heat', 'climate'],
  });

  const controversialClaim = await metabolism.birthClaim({
    content: 'New renewable technology X will solve the climate crisis',
    classification: {
      empirical: EmpiricalLevel.E1_Testimonial, // Only testimonial evidence
      normative: NormativeLevel.N1_Communal,
      materiality: MaterialityLevel.M1_Temporal,
    },
    authorId: 'startup-founder',
    domain: 'energy',
    confidence: 0.8, // Overconfident given low evidence
    tags: ['technology', 'solution', 'energy'],
  });

  console.log(`✓ Born claim: ${climateClaim.content.slice(0, 50)}...`);
  console.log(`  Phase: ${climateClaim.phase}, Confidence: ${climateClaim.confidence}`);
  console.log();
  console.log(`✓ Born claim: ${oceanClaim.content.slice(0, 50)}...`);
  console.log(`  Phase: ${oceanClaim.phase}, Confidence: ${oceanClaim.confidence}`);
  console.log();
  console.log(`✓ Born claim: ${controversialClaim.content.slice(0, 50)}...`);
  console.log(`  Phase: ${controversialClaim.phase}, Confidence: ${controversialClaim.confidence}`);
  console.log();

  // Register claims with propagator
  propagator.registerClaim(climateClaim.id, climateClaim.confidence, [oceanClaim.id]);
  propagator.registerClaim(oceanClaim.id, oceanClaim.confidence, []);
  propagator.registerClaim(controversialClaim.id, controversialClaim.confidence, [], {
    domain: 'energy',
  });

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 2: EPISTEMIC ALGEBRA
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 2: Epistemic Algebra                                       │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  // Demonstrate weakest-link principle (conjunction)
  console.log('Weakest-Link Principle (Conjunction):');
  const combinedStrength = conjunction([
    climateClaim.confidence,
    oceanClaim.confidence,
    0.6, // Additional supporting claim
  ]);
  console.log(`  Combining confidences: ${climateClaim.confidence}, ${oceanClaim.confidence}, 0.6`);
  console.log(`  Result: ${combinedStrength.confidence.toFixed(2)} (limited by weakest link)`);
  console.log(`  ${combinedStrength.explanation}`);
  console.log();

  // Demonstrate E-level ceiling
  console.log('E-Level Confidence Ceiling:');
  const overconfidentValue = 0.95;
  const ceiling = applyELevelCeiling(overconfidentValue, EmpiricalLevel.E1_Testimonial);
  console.log(`  Claimed confidence: ${overconfidentValue} (testimonial evidence only)`);
  console.log(`  E1 ceiling: ${E_LEVEL_CONFIDENCE_CEILING[EmpiricalLevel.E1_Testimonial]}`);
  console.log(`  Adjusted: ${ceiling.confidence} (ceiling applied: ${ceiling.ceilingApplied})`);
  console.log();

  // Demonstrate chain degradation
  console.log('Chain Degradation (Inference Chain):');
  const inferenceResult = chainDegradation(0.9, 4);
  console.log(`  Starting confidence: ${inferenceResult.startingConfidence}`);
  console.log(`  Chain length: ${inferenceResult.chainLength} inference steps`);
  console.log(`  Final confidence: ${inferenceResult.finalConfidence.toFixed(3)}`);
  if (inferenceResult.warning) {
    console.log(`  ⚠️ ${inferenceResult.warning}`);
  }
  console.log();

  // Demonstrate corroboration
  console.log('Corroboration (Independent Sources):');
  const corroborationResult = corroborate({
    independent: true,
    sourceCount: 3,
    confidences: [0.75, 0.72, 0.78],
  });
  console.log(`  3 independent sources: 0.75, 0.72, 0.78`);
  console.log(`  Boosted confidence: ${corroborationResult.confidence.toFixed(3)}`);
  console.log(`  ${corroborationResult.explanation}`);
  console.log();

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 3: CALIBRATION TRACKING
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 3: Calibration Tracking                                    │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  // Simulate historical predictions and their outcomes
  console.log('Recording historical predictions and their outcomes...');

  // Well-calibrated domain (climate science)
  for (let i = 0; i < 10; i++) {
    await calibration.recordResolution(
      { id: `climate-${i}`, confidence: 0.85, domain: 'climate_science' },
      i < 8 ? 'correct' : 'incorrect' // ~80% accuracy at 85% confidence (slightly overconfident)
    );
  }

  // Poorly calibrated domain (technology predictions)
  for (let i = 0; i < 10; i++) {
    await calibration.recordResolution(
      { id: `tech-${i}`, confidence: 0.9, domain: 'technology' },
      i < 4 ? 'correct' : 'incorrect' // Only 40% accuracy at 90% confidence (very overconfident!)
    );
  }

  // Generate calibration reports
  const climateCalibration = await calibration.generateReport({
    type: 'domain',
    domain: 'climate_science',
  });

  const techCalibration = await calibration.generateReport({
    type: 'domain',
    domain: 'technology',
  });

  console.log();
  console.log('Climate Science Calibration:');
  console.log(`  Brier Score: ${climateCalibration.brierScore.toFixed(3)} (lower is better)`);
  console.log(`  Calibration Error: ${(climateCalibration.overallCalibrationError * 100).toFixed(1)}%`);
  console.log(`  Inflation Bias: ${(climateCalibration.inflationBias * 100).toFixed(1)}% (positive = overconfident)`);
  console.log();

  console.log('Technology Predictions Calibration:');
  console.log(`  Brier Score: ${techCalibration.brierScore.toFixed(3)} (lower is better)`);
  console.log(`  Calibration Error: ${(techCalibration.overallCalibrationError * 100).toFixed(1)}%`);
  console.log(`  Inflation Bias: ${(techCalibration.inflationBias * 100).toFixed(1)}% (positive = overconfident)`);
  console.log();

  // Check if new high-confidence claims are credible
  console.log('Credibility Checks:');
  const climateCredible = calibration.isConfidenceCredible(0.9, 'climate_science');
  const techCredible = calibration.isConfidenceCredible(0.9, 'technology');

  console.log(`  90% confidence in climate_science: ${climateCredible.credible ? '✓ Credible' : '✗ Not credible'}`);
  console.log(`    ${climateCredible.reason}`);
  console.log(`  90% confidence in technology: ${techCredible.credible ? '✓ Credible' : '✗ Not credible'}`);
  console.log(`    ${techCredible.reason}`);
  console.log();

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 4: SAFE BELIEF PROPAGATION
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 4: Safe Belief Propagation                                 │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  console.log('Circuit Breaker Configuration:');
  console.log(`  Max Depth: ${DEFAULT_CIRCUIT_BREAKER_CONFIG.maxDepth}`);
  console.log(`  Max Affected Claims: ${DEFAULT_CIRCUIT_BREAKER_CONFIG.maxAffectedClaims}`);
  console.log(`  Max Confidence Jump: ${DEFAULT_CIRCUIT_BREAKER_CONFIG.maxConfidenceJump}`);
  console.log(`  Rate Limit: ${DEFAULT_CIRCUIT_BREAKER_CONFIG.perClaimRateLimit} per hour per claim`);
  console.log();

  // Attempt a safe propagation
  console.log('Attempting belief propagation:');
  const safeResult = await propagator.propagate({
    id: 'prop-1',
    sourceClaimId: climateClaim.id,
    updateType: 'confidence_increase',
    newConfidence: 0.97,
    oldConfidence: 0.95,
    initiatedBy: 'new-evidence',
    createdAt: Date.now(),
    priority: 1,
  });

  console.log(`  State: ${safeResult.state}`);
  if (safeResult.state === 'completed') {
    console.log(`  Affected claims: ${safeResult.affectedClaims.length}`);
    console.log(`  Depth reached: ${safeResult.depthReached}`);
  } else if (safeResult.state === 'blocked') {
    console.log(`  Block reason: ${safeResult.blockReason}`);
  }
  console.log();

  // Attempt a dangerous propagation (too large confidence jump)
  console.log('Attempting dangerous propagation (large confidence jump):');
  const dangerousResult = await propagator.propagate({
    id: 'prop-2',
    sourceClaimId: controversialClaim.id,
    updateType: 'confidence_increase',
    newConfidence: 0.95, // Jump from 0.8 to 0.95 = 0.15 (< 0.3 limit but...)
    oldConfidence: 0.8,
    initiatedBy: 'hype-cycle',
    createdAt: Date.now(),
    priority: 1,
  });

  console.log(`  State: ${dangerousResult.state}`);
  if (dangerousResult.blockReason) {
    console.log(`  Block reason: ${dangerousResult.blockReason}`);
  }
  if (dangerousResult.pendingHumanReview) {
    console.log(`  ⚠️ Pending human review`);
  }
  console.log();

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 5: KNOWLEDGE METABOLISM
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 5: Knowledge Metabolism                                    │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  // Add evidence to grow the claim
  console.log('Adding evidence to controversial claim:');
  await metabolism.addEvidence(controversialClaim.id, {
    type: 'document',
    data: 'Independent lab verification',
    source: 'research-lab-1',
  });

  const healthAfterEvidence = metabolism.assessHealth(controversialClaim.id);
  console.log(`  Evidence strength: ${healthAfterEvidence?.evidenceStrength.toFixed(2)}`);
  console.log();

  // Challenge the claim
  console.log('Challenging controversial claim:');
  await metabolism.challengeClaim(
    controversialClaim.id,
    'skeptical-scientist',
    'The efficiency claims have not been independently verified',
    [{ type: 'document', data: 'Analysis showing overstatement', source: 'peer-review' }]
  );

  const healthAfterChallenge = metabolism.assessHealth(controversialClaim.id);
  console.log(`  Challenges: ${healthAfterChallenge?.challengeCount}`);
  console.log(`  Contradiction score: ${healthAfterChallenge?.contradictionScore.toFixed(2)}`);
  console.log(`  Overall health: ${healthAfterChallenge?.overallHealth.toFixed(2)}`);
  console.log();

  // Kill the overhyped claim
  console.log('Killing the overhyped claim:');
  const tombstone = await metabolism.killClaim(
    controversialClaim.id,
    'refuted',
    'Technology failed to deliver on promises. Lesson: Be skeptical of extraordinary claims with only testimonial evidence.',
    [{ type: 'document', data: 'Failed demonstration', source: 'independent-test' }]
  );

  console.log(`  Death reason: ${tombstone.deathReason}`);
  console.log(`  Lesson learned: ${tombstone.lessonLearned}`);
  console.log();

  // Check new claims against tombstones
  console.log('Checking new claim against tombstones:');
  const similarClaim = 'Revolutionary technology Y will solve everything';
  const tombstoneCheck = metabolism.checkAgainstTombstones(similarClaim, 'energy');
  console.log(`  Claim: "${similarClaim}"`);
  console.log(`  Similar to dead claims: ${tombstoneCheck.matches.length > 0 ? 'Yes' : 'No'}`);
  if (tombstoneCheck.warnings.length > 0) {
    console.log(`  ⚠️ Warnings: ${tombstoneCheck.warnings.join(', ')}`);
  }
  console.log();

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // PHASE 6: GAP DETECTION WITH HUMILITY
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ PHASE 6: Gap Detection with Humility                             │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  // Register claims with gap detector
  gapDetector.registerClaim(climateClaim.id, {
    content: climateClaim.content,
    domain: 'climate_science',
    empiricalLevel: climateClaim.classification.empirical,
    confidence: climateClaim.confidence,
  });

  gapDetector.registerClaim(oceanClaim.id, {
    content: oceanClaim.content,
    domain: 'oceanography',
    empiricalLevel: oceanClaim.classification.empirical,
    confidence: oceanClaim.confidence,
  });

  gapDetector.linkClaims(climateClaim.id, oceanClaim.id);

  // Detect gaps
  console.log('Detecting knowledge gaps in climate_science:');
  const climateGaps = await gapDetector.detectGaps('climate_science');

  if (climateGaps.length === 0) {
    console.log('  No gaps detected (this is itself uncertain!)');
  } else {
    for (const gap of climateGaps.slice(0, 3)) {
      console.log();
      console.log(`  Gap: ${gap.description.slice(0, 60)}...`);
      console.log(`    Type: ${gap.type}`);
      console.log(`    Severity: ${gap.severity.toFixed(2)}`);
      console.log(`    Priority: ${gap.priority}`);
      console.log();
      console.log('    Humility Flags:');
      console.log(`      Schema-relative: ${gap.humility.schemaRelative}`);
      console.log(`      Unknown unknowns acknowledged: ${gap.humility.unknownUnknownsAcknowledged}`);
      console.log(`      Detection confidence: ${(gap.humility.detectionConfidence * 100).toFixed(0)}%`);
      console.log(`      Blind spots: ${gap.humility.detectionBlindSpots.slice(0, 2).join(', ')}...`);
    }
  }
  console.log();

  // Global gap detection
  console.log('Detecting global gaps:');
  const globalGaps = await gapDetector.detectGlobalGaps();
  console.log(`  Total gaps found: ${globalGaps.length}`);

  const stats = gapDetector.getStats();
  console.log(`  By type: structural=${stats.gapsByType.structural || 0}, epistemic=${stats.gapsByType.epistemic || 0}, cross_domain=${stats.gapsByType.cross_domain || 0}`);
  console.log();

  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // SUMMARY
  // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  console.log('┌──────────────────────────────────────────────────────────────────┐');
  console.log('│ SUMMARY: SCEI in Action                                          │');
  console.log('└──────────────────────────────────────────────────────────────────┘');

  console.log(`
  The Self-Correcting Epistemic Infrastructure demonstrated:

  1. CALIBRATION: Technology predictions were identified as poorly
     calibrated (40% accuracy at 90% confidence), while climate
     science predictions were better calibrated.

  2. EPISTEMIC ALGEBRA: Claims are combined using mathematically
     honest operations (weakest-link, E-level ceilings, chain
     degradation).

  3. METABOLISM: The overhyped claim was born, challenged, and
     died, leaving a tombstone that warns against similar claims.

  4. SAFE PROPAGATION: Circuit breakers prevented dangerous belief
     updates from cascading through the knowledge graph.

  5. GAP DETECTION: Identified potential gaps while acknowledging
     the limitations of gap detection itself (humility flags).

  Key Insight: Systems become more reliable not by claiming
  perfection, but by honestly tracking their mistakes and
  building in mechanisms for self-correction.
`);

  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
}

// Run the demonstration
demonstrateSCEIWorkflow().catch(console.error);
