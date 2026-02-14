/**
 * Epistemic Module Benchmarks
 *
 * Performance tests for the Epistemic Charter framework:
 * - Claim creation and classification
 * - Evidence handling
 * - E-N-M classification operations
 * - Batch processing
 * - Claim pools
 */

import { bench, describe } from 'vitest';
import {
  claim,
  createClaim,
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  EpistemicClaimPool,
  EpistemicBatch,
  ClaimBuilder,
  classificationCode,
  parseClassificationCode,
  meetsMinimum,
  meetsStandard,
  addEvidence,
  isExpired,
} from '../src/epistemic/index.js';

// ============================================================================
// Claim Creation Benchmarks
// ============================================================================

describe('Epistemic: Claim Creation', () => {
  bench('claim() builder - minimal', () => {
    claim('Test assertion').build();
  });

  bench('claim() builder - full classification', () => {
    claim('Test assertion')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .withNormative(NormativeLevel.N1_Local)
      .withMateriality(MaterialityLevel.M1_Temporal)
      .build();
  });

  bench('claim() builder - with confidence', () => {
    claim('Test assertion')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .withConfidence(0.85)
      .build();
  });

  bench('claim() builder - with metadata', () => {
    claim('Test assertion')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .withNormative(NormativeLevel.N1_Local)
      .withMateriality(MaterialityLevel.M1_Temporal)
      .withConfidence(0.75)
      .withMetadata({
        source: 'benchmark',
        timestamp: Date.now(),
        tags: ['test', 'benchmark'],
      })
      .build();
  });

  bench('createClaim() direct - minimal', () => {
    createClaim({
      subject: 'Test assertion',
    });
  });

  bench('createClaim() direct - full', () => {
    createClaim({
      subject: 'Test assertion',
      classification: {
        empirical: EmpiricalLevel.E3_Cryptographic,
        normative: NormativeLevel.N2_Network,
        materiality: MaterialityLevel.M2_Persistent,
      },
      confidence: 0.9,
      metadata: { source: 'benchmark' },
    });
  });

  bench('ClaimBuilder (10 claims)', () => {
    for (let i = 0; i < 10; i++) {
      new ClaimBuilder(`Claim ${i}`)
        .withEmpirical(EmpiricalLevel.E1_Testimonial)
        .withNormative(NormativeLevel.N0_Personal)
        .build();
    }
  });

  bench('ClaimBuilder (100 claims)', () => {
    for (let i = 0; i < 100; i++) {
      new ClaimBuilder(`Claim ${i}`)
        .withEmpirical(EmpiricalLevel.E1_Testimonial)
        .build();
    }
  });
});

// ============================================================================
// Classification Operations Benchmarks
// ============================================================================

describe('Epistemic: Classification Operations', () => {
  bench('classificationCode() generation', () => {
    classificationCode({
      empirical: EmpiricalLevel.E2_PrivateVerify,
      normative: NormativeLevel.N1_Local,
      materiality: MaterialityLevel.M1_Temporal,
    });
  });

  bench('parseClassificationCode()', () => {
    parseClassificationCode('E2-N1-M1');
  });

  bench('classificationCode roundtrip', () => {
    const code = classificationCode({
      empirical: EmpiricalLevel.E3_Cryptographic,
      normative: NormativeLevel.N2_Network,
      materiality: MaterialityLevel.M2_Persistent,
    });
    parseClassificationCode(code);
  });

  bench('meetsMinimum check', () => {
    const cl = claim('Test')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .withNormative(NormativeLevel.N1_Local)
      .withMateriality(MaterialityLevel.M1_Temporal)
      .build();
    meetsMinimum(cl, {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    });
  });

  bench('meetsStandard check', () => {
    const cl = claim('Test')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withConfidence(0.85)
      .build();
    meetsStandard(cl, 'scientific');
  });

  bench('meetsMinimum (100 checks)', () => {
    const cl = claim('Test')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .build();
    const min = {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    };
    for (let i = 0; i < 100; i++) {
      meetsMinimum(cl, min);
    }
  });
});

// ============================================================================
// Evidence Operations Benchmarks
// ============================================================================

describe('Epistemic: Evidence Handling', () => {
  const baseClaim = claim('Test claim')
    .withEmpirical(EmpiricalLevel.E1_Testimonial)
    .build();

  bench('addEvidence (single)', () => {
    addEvidence(baseClaim, {
      type: 'testimonial',
      source: 'observer-1',
      content: 'I witnessed this event',
      timestamp: Date.now(),
    });
  });

  bench('addEvidence (cryptographic)', () => {
    addEvidence(baseClaim, {
      type: 'cryptographic',
      source: 'blockchain',
      content: '0x' + 'a'.repeat(64),
      timestamp: Date.now(),
      signature: '0x' + 'b'.repeat(128),
    });
  });

  bench('addEvidence (5 pieces)', () => {
    let c = baseClaim;
    for (let i = 0; i < 5; i++) {
      c = addEvidence(c, {
        type: 'testimonial',
        source: `observer-${i}`,
        content: `Evidence piece ${i}`,
        timestamp: Date.now(),
      });
    }
  });

  bench('addEvidence chain (10 pieces)', () => {
    let c = baseClaim;
    for (let i = 0; i < 10; i++) {
      c = addEvidence(c, {
        type: 'testimonial',
        source: `observer-${i}`,
        content: `Evidence ${i}`,
        timestamp: Date.now(),
      });
    }
  });
});

// ============================================================================
// Claim Pool Benchmarks
// ============================================================================

describe('Epistemic: Claim Pool', () => {
  bench('EpistemicClaimPool creation', () => {
    new EpistemicClaimPool({ defaultTtlMs: 60000 });
  });

  bench('pool.add() - single claim', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    pool.add(claim('Test').build());
  });

  bench('pool.add() - 50 claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 50; i++) {
      pool.add(claim(`Claim ${i}`).build());
    }
  });

  bench('pool.add() - 100 claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 100; i++) {
      pool.add(claim(`Claim ${i}`).build());
    }
  });

  bench('pool.getActive() - 100 claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 100; i++) {
      pool.add(claim(`Claim ${i}`).build());
    }
    pool.getActive();
  });

  bench('pool.cleanup() - 100 claims', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 100; i++) {
      pool.add(claim(`Claim ${i}`).build());
    }
    pool.cleanup();
  });

  bench('pool full workflow (add + get + cleanup)', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 60000 });
    for (let i = 0; i < 50; i++) {
      pool.add(claim(`Claim ${i}`).build());
    }
    pool.getActive();
    pool.cleanup();
    pool.getStats();
  });
});

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

describe('Epistemic: Batch Processing', () => {
  bench('EpistemicBatch creation', () => {
    new EpistemicBatch();
  });

  bench('batch.add() - 10 claims', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 10; i++) {
      batch.add(claim(`Claim ${i}`).build());
    }
  });

  bench('batch.add() - 50 claims', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 50; i++) {
      batch.add(claim(`Claim ${i}`).build());
    }
  });

  bench('batch.summarize() - 10 claims', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 10; i++) {
      batch.add(
        claim(`Claim ${i}`)
          .withEmpirical(EmpiricalLevel.E1_Testimonial + (i % 4))
          .withConfidence(0.5 + (i % 5) * 0.1)
          .build()
      );
    }
    batch.summarize();
  });

  bench('batch.summarize() - 50 claims', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 50; i++) {
      batch.add(
        claim(`Claim ${i}`)
          .withEmpirical(EmpiricalLevel.E1_Testimonial + (i % 4))
          .build()
      );
    }
    batch.summarize();
  });

  bench('batch full workflow', () => {
    const batch = new EpistemicBatch();
    for (let i = 0; i < 25; i++) {
      batch.add(
        claim(`Claim ${i}`)
          .withEmpirical(EmpiricalLevel.E1_Testimonial + (i % 4))
          .withNormative(NormativeLevel.N0_Personal + (i % 4))
          .withConfidence(0.6 + (i % 4) * 0.1)
          .build()
      );
    }
    batch.summarize();
    batch.getHighConfidence(0.8);
    batch.filterByEmpirical(EmpiricalLevel.E2_PrivateVerify);
  });
});

// ============================================================================
// Expiration Benchmarks
// ============================================================================

describe('Epistemic: Expiration', () => {
  bench('isExpired check (not expired)', () => {
    const cl = claim('Test')
      .withExpiration(Date.now() + 3600000)
      .build();
    isExpired(cl);
  });

  bench('isExpired check (expired)', () => {
    const cl = claim('Test')
      .withExpiration(Date.now() - 1000)
      .build();
    isExpired(cl);
  });

  bench('isExpired (100 checks)', () => {
    const claims = [];
    for (let i = 0; i < 100; i++) {
      claims.push(
        claim(`Claim ${i}`)
          .withExpiration(Date.now() + (i % 2 === 0 ? 3600000 : -1000))
          .build()
      );
    }
    for (const c of claims) {
      isExpired(c);
    }
  });
});

// ============================================================================
// Realistic Scenarios
// ============================================================================

describe('Epistemic: Realistic Scenarios', () => {
  bench('scientific paper claim workflow', () => {
    // Create claim for a research finding
    let paperClaim = claim('Hypothesis X is supported by experiment Y')
      .withEmpirical(EmpiricalLevel.E3_Cryptographic)
      .withNormative(NormativeLevel.N2_Network)
      .withMateriality(MaterialityLevel.M2_Persistent)
      .withConfidence(0.78)
      .withMetadata({
        source: 'peer-reviewed-journal',
        doi: '10.1234/example',
        authors: ['Researcher A', 'Researcher B'],
      })
      .build();

    // Add supporting evidence
    paperClaim = addEvidence(paperClaim, {
      type: 'data',
      source: 'experiment-dataset',
      content: 'Statistical analysis results',
      timestamp: Date.now(),
    });

    // Verify it meets scientific standard
    meetsStandard(paperClaim, 'scientific');
    classificationCode(paperClaim.classification);
  });

  bench('marketplace product verification', () => {
    // Verify product authenticity claim
    const authenticityClaim = claim('Product #12345 is authentic')
      .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
      .withNormative(NormativeLevel.N1_Local)
      .withMateriality(MaterialityLevel.M1_Temporal)
      .withConfidence(0.92)
      .build();

    // Check minimum requirements
    meetsMinimum(authenticityClaim, {
      empirical: EmpiricalLevel.E1_Testimonial,
      normative: NormativeLevel.N0_Personal,
      materiality: MaterialityLevel.M0_Ephemeral,
    });
  });

  bench('news article verification batch (10 articles)', () => {
    const batch = new EpistemicBatch();

    for (let i = 0; i < 10; i++) {
      const verified = i % 3 === 0;
      batch.add(
        claim(`Article ${i} claims XYZ`)
          .withEmpirical(
            verified ? EmpiricalLevel.E2_PrivateVerify : EmpiricalLevel.E1_Testimonial
          )
          .withNormative(NormativeLevel.N1_Local)
          .withConfidence(verified ? 0.85 : 0.45)
          .build()
      );
    }

    const summary = batch.summarize();
    batch.getHighConfidence(0.7);
  });

  bench('supply chain provenance (5 steps)', () => {
    const pool = new EpistemicClaimPool({ defaultTtlMs: 86400000 });

    // Track product through supply chain
    const steps = ['manufactured', 'inspected', 'shipped', 'received', 'verified'];
    for (let i = 0; i < steps.length; i++) {
      let stepClaim = claim(`Product ${steps[i]} at step ${i + 1}`)
        .withEmpirical(EmpiricalLevel.E2_PrivateVerify)
        .withNormative(NormativeLevel.N1_Local)
        .withMateriality(MaterialityLevel.M2_Persistent)
        .withConfidence(0.95 - i * 0.02)
        .build();

      stepClaim = addEvidence(stepClaim, {
        type: 'timestamp',
        source: `facility-${i}`,
        content: new Date().toISOString(),
        timestamp: Date.now(),
      });

      pool.add(stepClaim);
    }

    pool.getActive();
    pool.getStats();
  });
});
