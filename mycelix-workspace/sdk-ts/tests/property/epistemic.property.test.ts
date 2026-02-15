/**
 * Property-Based Tests for Epistemic Module
 *
 * Uses fast-check to verify epistemic classification properties.
 */

import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import {
  EmpiricalLevel,
  NormativeLevel,
  MaterialityLevel,
  classificationCode,
  parseClassificationCode,
  meetsMinimum,
  createClaim,
  addEvidence,
  claim,
} from '../../src/epistemic/index.js';

describe('Epistemic Property-Based Tests', () => {
  // Safe string arbitrary that never generates empty or whitespace-only strings
  // Must have at least 2 characters to avoid edge cases
  const safeStringArb = fc
    .string({ minLength: 2, maxLength: 50 })
    .filter((s) => /^[a-z][a-z0-9]+$/i.test(s));

  // Arbitraries for epistemic levels (using correct enum values)
  const empiricalLevelArb = fc.constantFrom(
    EmpiricalLevel.E0_Unverified,
    EmpiricalLevel.E1_Testimonial,
    EmpiricalLevel.E2_PrivateVerify,
    EmpiricalLevel.E3_Cryptographic,
    EmpiricalLevel.E4_Consensus
  );

  const normativeLevelArb = fc.constantFrom(
    NormativeLevel.N0_Personal,
    NormativeLevel.N1_Communal,
    NormativeLevel.N2_Network,
    NormativeLevel.N3_Universal
  );

  const materialityLevelArb = fc.constantFrom(
    MaterialityLevel.M0_Ephemeral,
    MaterialityLevel.M1_Temporal,
    MaterialityLevel.M2_Persistent,
    MaterialityLevel.M3_Immutable
  );

  describe('Classification Code Properties', () => {
    it('classification code should be reversible', () => {
      fc.assert(
        fc.property(
          empiricalLevelArb,
          normativeLevelArb,
          materialityLevelArb,
          (empirical, normative, materiality) => {
            const classification = { empirical, normative, materiality };
            const code = classificationCode(classification);
            const parsed = parseClassificationCode(code);

            expect(parsed).not.toBeNull();
            expect(parsed!.empirical).toBe(empirical);
            expect(parsed!.normative).toBe(normative);
            expect(parsed!.materiality).toBe(materiality);
          }
        ),
        { numRuns: 500 }
      );
    });

    it('classification code should follow E/N/M format', () => {
      fc.assert(
        fc.property(
          empiricalLevelArb,
          normativeLevelArb,
          materialityLevelArb,
          (empirical, normative, materiality) => {
            const code = classificationCode({ empirical, normative, materiality });

            // Should match format like "E1-N2-M3" (with hyphens)
            expect(code).toMatch(/^E[0-4]-N[0-3]-M[0-3]$/);
          }
        ),
        { numRuns: 300 }
      );
    });
  });

  describe('Claim Creation Properties', () => {
    it('created claim should have required fields', () => {
      fc.assert(
        fc.property(
          safeStringArb,
          safeStringArb,
          empiricalLevelArb,
          normativeLevelArb,
          materialityLevelArb,
          (content, issuer, empirical, normative, materiality) => {
            // createClaim takes: content, empirical, normative, materiality, issuer
            const epistemicClaim = createClaim(content, empirical, normative, materiality, issuer);

            expect(epistemicClaim.content).toBe(content);
            expect(epistemicClaim.classification.empirical).toBe(empirical);
            expect(epistemicClaim.classification.normative).toBe(normative);
            expect(epistemicClaim.classification.materiality).toBe(materiality);
            expect(epistemicClaim.evidence).toEqual([]);
            expect(epistemicClaim.issuedAt).toBeDefined();
          }
        ),
        { numRuns: 300 }
      );
    });

    it('claim builder should produce valid claims', () => {
      fc.assert(
        fc.property(
          safeStringArb,
          safeStringArb,
          empiricalLevelArb,
          normativeLevelArb,
          materialityLevelArb,
          (content, issuer, empirical, normative, materiality) => {
            const builtClaim = claim(content)
              .withEmpirical(empirical)
              .withNormative(normative)
              .withMateriality(materiality)
              .withIssuer(issuer)
              .build();

            expect(builtClaim.content).toBe(content);
            expect(builtClaim.classification.empirical).toBe(empirical);
            expect(builtClaim.classification.normative).toBe(normative);
            expect(builtClaim.classification.materiality).toBe(materiality);
          }
        ),
        { numRuns: 300 }
      );
    });
  });

  describe('Evidence Properties', () => {
    it('adding evidence should not modify original claim', () => {
      fc.assert(
        fc.property(
          safeStringArb,
          safeStringArb,
          safeStringArb,
          safeStringArb,
          safeStringArb,
          (content, issuer, evidenceType, evidenceData, evidenceSource) => {
            const original = createClaim(
              content,
              EmpiricalLevel.E1_Testimonial,
              NormativeLevel.N0_Personal,
              MaterialityLevel.M1_Temporal,
              issuer
            );

            const originalEvidenceCount = original.evidence.length;

            const withEvidence = addEvidence(original, {
              type: evidenceType,
              data: evidenceData,
              source: evidenceSource,
              timestamp: Date.now(),
            });

            // Original should be unchanged
            expect(original.evidence.length).toBe(originalEvidenceCount);
            // New claim should have additional evidence
            expect(withEvidence.evidence.length).toBe(originalEvidenceCount + 1);
          }
        ),
        { numRuns: 200 }
      );
    });

    it('evidence should be appended in order', () => {
      fc.assert(
        fc.property(
          safeStringArb,
          safeStringArb,
          fc.array(safeStringArb, { minLength: 1, maxLength: 5 }),
          (content, issuer, evidenceDataItems) => {
            let current = createClaim(
              content,
              EmpiricalLevel.E1_Testimonial,
              NormativeLevel.N0_Personal,
              MaterialityLevel.M1_Temporal,
              issuer
            );

            for (const eData of evidenceDataItems) {
              current = addEvidence(current, {
                type: 'test',
                data: eData,
                source: 'test-source',
                timestamp: Date.now(),
              });
            }

            expect(current.evidence.length).toBe(evidenceDataItems.length);
            for (let i = 0; i < evidenceDataItems.length; i++) {
              expect(current.evidence[i].data).toBe(evidenceDataItems[i]);
            }
          }
        ),
        { numRuns: 100 }
      );
    });
  });

  describe('MeetsMinimum Properties', () => {
    it('higher levels should meet lower minimum requirements', () => {
      fc.assert(
        fc.property(
          empiricalLevelArb,
          normativeLevelArb,
          materialityLevelArb,
          (empirical, normative, materiality) => {
            const classification = { empirical, normative, materiality };

            // Minimum with level 0 for all should always be met
            // meetsMinimum takes (classification, minE, minN, minM)
            expect(
              meetsMinimum(
                classification,
                EmpiricalLevel.E0_Unverified,
                NormativeLevel.N0_Personal,
                MaterialityLevel.M0_Ephemeral
              )
            ).toBe(true);
          }
        ),
        { numRuns: 300 }
      );
    });

    it('lower levels should not meet higher minimum requirements', () => {
      // Low empirical
      expect(
        meetsMinimum(
          {
            empirical: EmpiricalLevel.E0_Unverified,
            normative: NormativeLevel.N3_Universal,
            materiality: MaterialityLevel.M3_Immutable,
          },
          EmpiricalLevel.E4_Consensus,
          NormativeLevel.N3_Universal,
          MaterialityLevel.M3_Immutable
        )
      ).toBe(false);

      // Low normative
      expect(
        meetsMinimum(
          {
            empirical: EmpiricalLevel.E4_Consensus,
            normative: NormativeLevel.N0_Personal,
            materiality: MaterialityLevel.M3_Immutable,
          },
          EmpiricalLevel.E4_Consensus,
          NormativeLevel.N3_Universal,
          MaterialityLevel.M3_Immutable
        )
      ).toBe(false);

      // Low materiality
      expect(
        meetsMinimum(
          {
            empirical: EmpiricalLevel.E4_Consensus,
            normative: NormativeLevel.N3_Universal,
            materiality: MaterialityLevel.M0_Ephemeral,
          },
          EmpiricalLevel.E4_Consensus,
          NormativeLevel.N3_Universal,
          MaterialityLevel.M3_Immutable
        )
      ).toBe(false);
    });
  });

  describe('Level Ordering Properties', () => {
    it('empirical levels should be totally ordered', () => {
      const levels = [
        EmpiricalLevel.E0_Unverified,
        EmpiricalLevel.E1_Testimonial,
        EmpiricalLevel.E2_PrivateVerify,
        EmpiricalLevel.E3_Cryptographic,
        EmpiricalLevel.E4_Consensus,
      ];

      for (let i = 0; i < levels.length - 1; i++) {
        expect(levels[i]).toBeLessThan(levels[i + 1]);
      }
    });

    it('normative levels should be totally ordered', () => {
      const levels = [
        NormativeLevel.N0_Personal,
        NormativeLevel.N1_Communal,
        NormativeLevel.N2_Network,
        NormativeLevel.N3_Universal,
      ];

      for (let i = 0; i < levels.length - 1; i++) {
        expect(levels[i]).toBeLessThan(levels[i + 1]);
      }
    });

    it('materiality levels should be totally ordered', () => {
      const levels = [
        MaterialityLevel.M0_Ephemeral,
        MaterialityLevel.M1_Temporal,
        MaterialityLevel.M2_Persistent,
        MaterialityLevel.M3_Immutable,
      ];

      for (let i = 0; i < levels.length - 1; i++) {
        expect(levels[i]).toBeLessThan(levels[i + 1]);
      }
    });
  });

  describe('Edge Cases', () => {
    it('minimum classification should be valid', () => {
      const minClaim = createClaim(
        'Test content',
        EmpiricalLevel.E0_Unverified,
        NormativeLevel.N0_Personal,
        MaterialityLevel.M0_Ephemeral,
        'test-issuer'
      );

      expect(minClaim.content).toBe('Test content');
      expect(minClaim.classification.empirical).toBe(EmpiricalLevel.E0_Unverified);
    });

    it('maximum classification should be valid', () => {
      const maxClaim = createClaim(
        'Test content',
        EmpiricalLevel.E4_Consensus,
        NormativeLevel.N3_Universal,
        MaterialityLevel.M3_Immutable,
        'test-issuer'
      );

      expect(maxClaim.content).toBe('Test content');
      expect(maxClaim.classification.empirical).toBe(EmpiricalLevel.E4_Consensus);
    });

    it('alphanumeric content should be preserved', () => {
      fc.assert(
        fc.property(safeStringArb, safeStringArb, (content, issuer) => {
          const testClaim = createClaim(
            content,
            EmpiricalLevel.E1_Testimonial,
            NormativeLevel.N1_Communal,
            MaterialityLevel.M1_Temporal,
            issuer
          );

          expect(testClaim.content).toBe(content);
        }),
        { numRuns: 200 }
      );
    });
  });
});
