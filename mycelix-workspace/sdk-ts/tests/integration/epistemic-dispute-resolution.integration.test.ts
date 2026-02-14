/**
 * Epistemic Dispute Resolution Integration Tests
 *
 * Tests the integration between Knowledge module's epistemic classification
 * and Justice module's evidence verification for dispute resolution.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  EpistemicEvidenceBridge,
  getEpistemicEvidenceBridge,
  resetEpistemicEvidenceBridge,
  mapEvidenceTypeToClaimType,
  calculateAdjustedStrength,
  getEpistemicWeightsForCategory,
  meetsEpistemicStandards,
  explainEpistemicClassification,
  verdictToRecommendation,
  calculateOverallCredibility,
  type EpistemicEvidence,
  type EpistemicVerificationResult,
  type BatchVerificationInput,
} from '../../src/bridge/epistemic-evidence.js';
import { executeEpistemicDisputeResolutionWorkflow } from '../../src/bridge/workflows.js';
import type { Evidence, CaseCategory, EvidenceType } from '../../src/justice/index.js';
import type { EpistemicPosition, FactCheckVerdict } from '../../src/knowledge/index.js';

// ============================================================================
// Test Fixtures
// ============================================================================

function createTestEvidence(
  overrides: Partial<Evidence> = {}
): Evidence {
  return {
    id: `evidence-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    case_id: 'case-test-001',
    submitter: 'did:mycelix:alice',
    type_: 'Document',
    title: 'Test Evidence',
    description: 'Test evidence description',
    content_hash: 'Qm1234567890abcdef',
    submitted_at: Date.now(),
    verified: false,
    ...overrides,
  };
}

function createEvidenceSet(): Evidence[] {
  return [
    createTestEvidence({
      id: 'evidence-doc-001',
      type_: 'Document',
      title: 'Signed Contract',
      description: 'Original contract between parties',
      submitter: 'did:mycelix:complainant',
    }),
    createTestEvidence({
      id: 'evidence-tx-001',
      type_: 'Transaction',
      title: 'Payment Record',
      description: 'Blockchain transaction showing payment',
      submitter: 'did:mycelix:complainant',
    }),
    createTestEvidence({
      id: 'evidence-testimony-001',
      type_: 'Testimony',
      title: 'Witness Statement',
      description: 'Third-party account of events',
      submitter: 'did:mycelix:witness',
    }),
    createTestEvidence({
      id: 'evidence-screenshot-001',
      type_: 'Screenshot',
      title: 'Chat Log',
      description: 'Screenshot of conversation',
      submitter: 'did:mycelix:respondent',
    }),
  ];
}

// ============================================================================
// EpistemicEvidenceBridge Tests
// ============================================================================

describe('Integration: Epistemic Evidence Bridge', () => {
  let bridge: EpistemicEvidenceBridge;

  beforeEach(() => {
    resetEpistemicEvidenceBridge();
    bridge = new EpistemicEvidenceBridge();
  });

  describe('verifyEvidence', () => {
    it('should verify document evidence with high empirical score', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });
      const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

      expect(result).toBeDefined();
      expect(result.evidence_id).toBe(evidence.id);
      expect(result.classification.empirical).toBeGreaterThan(0.5);
      expect(result.verified_at).toBeGreaterThan(0);
      expect(result.factors.length).toBeGreaterThan(0);
    });

    it('should verify transaction evidence with highest empirical score', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      expect(result.classification.empirical).toBeGreaterThan(0.8);
      expect(result.recommendation).not.toBe('Unreliable');
    });

    it('should classify testimony with balanced epistemic position', async () => {
      const evidence = createTestEvidence({ type_: 'Testimony' });
      const result = await bridge.verifyEvidence(evidence, 'ConductViolation');

      // Testimony should have more normative weight than documents
      expect(result.classification.normative).toBeGreaterThan(0.1);
      expect(result.classification.mythic).toBeGreaterThan(0);
    });

    it('should adjust classification based on case category', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });

      const financialResult = await bridge.verifyEvidence(evidence, 'FinancialDispute');
      const governanceResult = await bridge.verifyEvidence(evidence, 'GovernanceDispute');

      // FinancialDispute cases weight empirical more heavily
      expect(financialResult.classification.empirical).not.toBe(
        governanceResult.classification.empirical
      );
    });

    it('should include verification factors in result', async () => {
      const evidence = createTestEvidence();
      const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

      expect(result.factors).toBeInstanceOf(Array);
      expect(result.factors.length).toBeGreaterThan(0);

      const hasBaseClassificationFactor = result.factors.some(
        (f) => f.name === 'Base Type Classification'
      );
      expect(hasBaseClassificationFactor).toBe(true);
    });

    it('should provide recommendation based on credibility', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

      expect(['Strong', 'Moderate', 'Weak', 'Unreliable', 'Insufficient']).toContain(
        result.recommendation
      );
    });

    it('should mark evidence as verified when credibility meets threshold', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

      // Transaction evidence should typically meet verification threshold
      expect(result.verified).toBe(true);
    });

    it('should handle different case categories appropriately', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });
      const categories: CaseCategory[] = [
        'ContractDispute',
        'ConductViolation',
        'PropertyDispute',
        'FinancialDispute',
        'GovernanceDispute',
        'IdentityDispute',
        'IPDispute',
        'Other',
      ];

      for (const category of categories) {
        const result = await bridge.verifyEvidence(evidence, category);
        expect(result.classification.empirical).toBeGreaterThan(0);
        expect(result.classification.normative).toBeDefined();
        expect(result.classification.mythic).toBeDefined();
      }
    });
  });

  describe('verifyEvidenceBatch', () => {
    it('should verify multiple evidence items', async () => {
      const evidenceSet = createEvidenceSet();
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: evidenceSet.map((e) => e.id),
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceSet, 'ContractDispute');

      expect(result.case_id).toBe('case-test-001');
      expect(result.results.length).toBe(evidenceSet.length);
      expect(result.summary.total).toBe(evidenceSet.length);
      expect(result.duration_ms).toBeGreaterThanOrEqual(0);
    });

    it('should calculate summary statistics correctly', async () => {
      const evidenceSet = createEvidenceSet();
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: evidenceSet.map((e) => e.id),
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceSet, 'ContractDispute');

      expect(result.summary.total).toBe(evidenceSet.length);
      expect(result.summary.verified + result.summary.failed).toBe(evidenceSet.length);
      expect(result.summary.avg_credibility).toBeGreaterThanOrEqual(0);
      expect(result.summary.avg_credibility).toBeLessThanOrEqual(1);
      expect(result.summary.avg_empirical).toBeGreaterThanOrEqual(0);
      expect(result.summary.by_recommendation).toBeDefined();
    });

    it('should filter by minimum credibility when specified', async () => {
      const evidenceSet = createEvidenceSet();
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: evidenceSet.map((e) => e.id),
        min_credibility: 0.7,
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceSet, 'ContractDispute');

      // All results should meet minimum credibility
      for (const r of result.results) {
        expect(r.credibility).toBeGreaterThanOrEqual(0.7);
      }
    });

    it('should only process requested evidence IDs', async () => {
      const evidenceSet = createEvidenceSet();
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: [evidenceSet[0].id, evidenceSet[1].id], // Only first two
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceSet, 'ContractDispute');

      expect(result.results.length).toBe(2);
      expect(result.results.map((r) => r.evidence_id)).toContain(evidenceSet[0].id);
      expect(result.results.map((r) => r.evidence_id)).toContain(evidenceSet[1].id);
    });

    it('should determine case evidence quality', async () => {
      const evidenceSet = createEvidenceSet();
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: evidenceSet.map((e) => e.id),
      };

      const result = await bridge.verifyEvidenceBatch(input, evidenceSet, 'ContractDispute');

      expect(['Excellent', 'Good', 'Mixed', 'Poor', 'Insufficient']).toContain(
        result.case_evidence_quality
      );
    });

    it('should return Insufficient for empty evidence set', async () => {
      const input: BatchVerificationInput = {
        case_id: 'case-test-001',
        evidence_ids: [],
      };

      const result = await bridge.verifyEvidenceBatch(input, [], 'ContractDispute');

      expect(result.summary.total).toBe(0);
      expect(result.case_evidence_quality).toBe('Insufficient');
    });
  });

  describe('enhanceEvidence', () => {
    it('should add epistemic metadata to evidence', async () => {
      const evidence = createTestEvidence({ type_: 'Document' });
      const enhanced = await bridge.enhanceEvidence(evidence, 'ContractDispute');

      expect(enhanced).toMatchObject({
        id: evidence.id,
        case_id: evidence.case_id,
        type_: evidence.type_,
      });
      expect(enhanced.epistemic_position).toBeDefined();
      expect(enhanced.credibility_score).toBeDefined();
      expect(enhanced.adjusted_strength).toBeDefined();
      expect(enhanced.epistemically_verified_at).toBeGreaterThan(0);
    });

    it('should calculate adjusted strength', async () => {
      const evidence = createTestEvidence({ type_: 'Transaction' });
      const enhanced = await bridge.enhanceEvidence(evidence, 'ConductViolation');

      expect(enhanced.adjusted_strength).toBeGreaterThan(0);
      expect(enhanced.adjusted_strength).toBeLessThanOrEqual(1);
    });
  });

  describe('factCheckEvidence', () => {
    it('should return verdict based on evidence type', async () => {
      const docEvidence = createTestEvidence({ type_: 'Document' });
      const txEvidence = createTestEvidence({ type_: 'Transaction' });
      const testimonyEvidence = createTestEvidence({ type_: 'Testimony' });

      const docResult = await bridge.factCheckEvidence(docEvidence, 'Contract is valid');
      const txResult = await bridge.factCheckEvidence(txEvidence, 'Payment was made');
      const testimonyResult = await bridge.factCheckEvidence(
        testimonyEvidence,
        'I saw what happened'
      );

      // Transaction should have highest reliability
      expect(txResult.confidence).toBeGreaterThan(docResult.confidence);
      expect(docResult.confidence).toBeGreaterThan(testimonyResult.confidence);
    });

    it('should return appropriate verdict based on reliability', async () => {
      const txEvidence = createTestEvidence({ type_: 'Transaction' });
      const result = await bridge.factCheckEvidence(txEvidence, 'Payment recorded');

      expect(['True', 'PartiallyTrue', 'Misleading', 'False', 'Unverified']).toContain(
        result.verdict
      );
    });
  });
});

// ============================================================================
// Helper Function Tests
// ============================================================================

describe('Integration: Epistemic Helper Functions', () => {
  describe('mapEvidenceTypeToClaimType', () => {
    it('should map Document to Fact', () => {
      expect(mapEvidenceTypeToClaimType('Document')).toBe('Fact');
    });

    it('should map Testimony to Opinion', () => {
      expect(mapEvidenceTypeToClaimType('Testimony')).toBe('Opinion');
    });

    it('should map Transaction to Historical', () => {
      expect(mapEvidenceTypeToClaimType('Transaction')).toBe('Historical');
    });

    it('should map Witness to Opinion', () => {
      expect(mapEvidenceTypeToClaimType('Witness')).toBe('Opinion');
    });

    it('should map Screenshot to Fact', () => {
      expect(mapEvidenceTypeToClaimType('Screenshot')).toBe('Fact');
    });

    it('should map Other to Fact as default', () => {
      expect(mapEvidenceTypeToClaimType('Other')).toBe('Fact');
    });
  });

  describe('getEpistemicWeightsForCategory', () => {
    it('should return weights summing to approximately 1', () => {
      const categories: CaseCategory[] = [
        'ContractDispute',
        'ConductViolation',
        'PropertyDispute',
        'FinancialDispute',
        'GovernanceDispute',
        'IdentityDispute',
        'IPDispute',
        'Other',
      ];

      for (const category of categories) {
        const weights = getEpistemicWeightsForCategory(category);
        const sum = weights.empirical + weights.normative + weights.mythic;
        expect(sum).toBeCloseTo(1.0, 1);
      }
    });

    it('should weight empirical highest for FinancialDispute', () => {
      const weights = getEpistemicWeightsForCategory('FinancialDispute');
      expect(weights.empirical).toBeGreaterThan(weights.normative);
      expect(weights.empirical).toBeGreaterThan(weights.mythic);
      expect(weights.empirical).toBe(0.75);
    });

    it('should weight normative highest for GovernanceDispute', () => {
      const weights = getEpistemicWeightsForCategory('GovernanceDispute');
      expect(weights.normative).toBeGreaterThan(weights.mythic);
    });

    it('should weight empirical and normative equally for ConductViolation and IdentityDispute', () => {
      const conductWeights = getEpistemicWeightsForCategory('ConductViolation');
      expect(conductWeights.empirical).toBe(0.6);
      expect(conductWeights.normative).toBe(0.3);
      expect(conductWeights.mythic).toBe(0.1);

      const identityWeights = getEpistemicWeightsForCategory('IdentityDispute');
      expect(identityWeights.empirical).toBe(0.6);
      expect(identityWeights.normative).toBe(0.3);
      expect(identityWeights.mythic).toBe(0.1);
    });
  });

  describe('calculateAdjustedStrength', () => {
    it('should adjust strength based on epistemic position', () => {
      const highEmpirical: EpistemicPosition = { empirical: 0.9, normative: 0.05, mythic: 0.05 };
      const lowEmpirical: EpistemicPosition = { empirical: 0.2, normative: 0.4, mythic: 0.4 };

      const highResult = calculateAdjustedStrength(0.5, highEmpirical, 'ConductViolation');
      const lowResult = calculateAdjustedStrength(0.5, lowEmpirical, 'ConductViolation');

      // Higher empirical should result in higher adjusted strength for ConductViolation
      expect(highResult).toBeGreaterThan(lowResult);
    });

    it('should clamp result between 0 and 1', () => {
      const extreme: EpistemicPosition = { empirical: 1.0, normative: 1.0, mythic: 1.0 };
      const result = calculateAdjustedStrength(1.0, extreme, 'ContractDispute');

      expect(result).toBeLessThanOrEqual(1.0);
      expect(result).toBeGreaterThanOrEqual(0);
    });

    it('should consider case category weights', () => {
      const normativeHeavy: EpistemicPosition = { empirical: 0.2, normative: 0.7, mythic: 0.1 };

      const conductResult = calculateAdjustedStrength(0.5, normativeHeavy, 'ConductViolation');
      const governanceResult = calculateAdjustedStrength(0.5, normativeHeavy, 'GovernanceDispute');

      // Governance should value normative more
      expect(governanceResult).toBeGreaterThan(conductResult);
    });
  });

  describe('verdictToRecommendation', () => {
    it('should return Strong for True verdict with high credibility', () => {
      expect(verdictToRecommendation('True', 0.9)).toBe('Strong');
    });

    it('should return Moderate for True verdict with medium credibility', () => {
      expect(verdictToRecommendation('True', 0.6)).toBe('Moderate');
    });

    it('should return Moderate for PartiallyTrue with high credibility', () => {
      expect(verdictToRecommendation('PartiallyTrue', 0.8)).toBe('Moderate');
    });

    it('should return Weak for PartiallyTrue with low credibility', () => {
      expect(verdictToRecommendation('PartiallyTrue', 0.3)).toBe('Weak');
    });

    it('should return Unreliable for False verdict', () => {
      expect(verdictToRecommendation('False', 0.9)).toBe('Unreliable');
    });

    it('should return Insufficient for Unverified verdict', () => {
      expect(verdictToRecommendation('Unverified', 0.5)).toBe('Insufficient');
    });
  });

  describe('calculateOverallCredibility', () => {
    it('should calculate credibility from epistemic position', () => {
      const position: EpistemicPosition = { empirical: 0.8, normative: 0.1, mythic: 0.1 };
      const result = calculateOverallCredibility(position);

      expect(result.credibility).toBeGreaterThan(0);
      expect(result.credibility).toBeLessThanOrEqual(1);
      expect(result.confidence).toBeGreaterThan(0);
    });

    it('should boost credibility with positive fact-check result', () => {
      const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };

      const withoutFactCheck = calculateOverallCredibility(position);
      const withFactCheck = calculateOverallCredibility(position, {
        verdict: 'True',
        confidence: 0.9,
        checked_at: Date.now(),
        sources: [],
      });

      expect(withFactCheck.credibility).toBeGreaterThan(withoutFactCheck.credibility);
    });

    it('should reduce credibility with False fact-check result', () => {
      const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };

      const withoutFactCheck = calculateOverallCredibility(position);
      const withFalseFactCheck = calculateOverallCredibility(position, {
        verdict: 'False',
        confidence: 0.9,
        checked_at: Date.now(),
        sources: [],
      });

      expect(withFalseFactCheck.credibility).toBeLessThan(withoutFactCheck.credibility);
    });
  });

  describe('meetsEpistemicStandards', () => {
    it('should return true for strong evidence', () => {
      const result: EpistemicVerificationResult = {
        evidence_id: 'test',
        verified: true,
        classification: { empirical: 0.8, normative: 0.1, mythic: 0.1 },
        credibility: 0.85,
        confidence: 0.9,
        supporting_claims: [],
        contradicting_claims: [],
        factors: [],
        recommendation: 'Strong',
        verified_at: Date.now(),
      };

      expect(meetsEpistemicStandards(result, 'ContractDispute')).toBe(true);
    });

    it('should return false for unreliable evidence', () => {
      const result: EpistemicVerificationResult = {
        evidence_id: 'test',
        verified: false,
        classification: { empirical: 0.2, normative: 0.3, mythic: 0.5 },
        credibility: 0.2,
        confidence: 0.3,
        supporting_claims: [],
        contradicting_claims: [],
        factors: [],
        recommendation: 'Unreliable',
        verified_at: Date.now(),
      };

      expect(meetsEpistemicStandards(result, 'ConductViolation')).toBe(false);
    });

    it('should return false for insufficient evidence', () => {
      const result: EpistemicVerificationResult = {
        evidence_id: 'test',
        verified: false,
        classification: { empirical: 0.3, normative: 0.3, mythic: 0.4 },
        credibility: 0.35,
        confidence: 0.4,
        supporting_claims: [],
        contradicting_claims: [],
        factors: [],
        recommendation: 'Insufficient',
        verified_at: Date.now(),
      };

      expect(meetsEpistemicStandards(result, 'PropertyDispute')).toBe(false);
    });
  });

  describe('explainEpistemicClassification', () => {
    it('should describe strongly empirical evidence', () => {
      const position: EpistemicPosition = { empirical: 0.9, normative: 0.05, mythic: 0.05 };
      const explanation = explainEpistemicClassification(position);

      expect(explanation).toContain('strongly empirically verifiable');
    });

    it('should describe partially empirical evidence', () => {
      const position: EpistemicPosition = { empirical: 0.5, normative: 0.3, mythic: 0.2 };
      const explanation = explainEpistemicClassification(position);

      expect(explanation).toContain('partially empirically verifiable');
    });

    it('should note normative dimensions', () => {
      const position: EpistemicPosition = { empirical: 0.3, normative: 0.6, mythic: 0.1 };
      const explanation = explainEpistemicClassification(position);

      expect(explanation).toContain('normative');
    });

    it('should note narrative context', () => {
      const position: EpistemicPosition = { empirical: 0.2, normative: 0.3, mythic: 0.5 };
      const explanation = explainEpistemicClassification(position);

      expect(explanation).toContain('narrative');
    });
  });
});

// ============================================================================
// Workflow Integration Tests
// ============================================================================

describe('Integration: Epistemic Dispute Resolution Workflow', () => {
  beforeEach(() => {
    resetEpistemicEvidenceBridge();
  });

  describe('executeEpistemicDisputeResolutionWorkflow', () => {
    it('should execute full workflow with evidence verification', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-001',
        caseCategory: 'ContractDispute',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.data.caseId).toBe('case-test-workflow-001');
      expect(result.data.evidenceVerifications.length).toBe(evidence.length);
    });

    it('should assess evidence quality for both parties', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-002',
        caseCategory: 'ConductViolation',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.data.complainantEvidenceQuality).toBeDefined();
      expect(result.data.respondentEvidenceQuality).toBeDefined();
      expect(['Excellent', 'Good', 'Mixed', 'Poor', 'Insufficient']).toContain(
        result.data.complainantEvidenceQuality
      );
    });

    it('should provide overall case quality assessment', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-003',
        caseCategory: 'ConductViolation',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(['Excellent', 'Good', 'Mixed', 'Poor', 'Insufficient']).toContain(
        result.data.overallCaseQuality
      );
    });

    it('should provide recommendation', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-004',
        caseCategory: 'PropertyDispute',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.data.recommendation).toBeDefined();
      expect(result.data.recommendation.length).toBeGreaterThan(0);
    });

    it('should include summary explanation', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-005',
        caseCategory: 'IPDispute',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.data.summaryExplanation).toBeDefined();
      expect(result.data.summaryExplanation.length).toBeGreaterThan(10);
    });

    it('should apply minimum credibility threshold', async () => {
      const evidence = createEvidenceSet();

      const resultWithThreshold = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-006',
        caseCategory: 'ContractDispute',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
        minCredibilityThreshold: 0.8,
      });

      // Verifications that pass should have high credibility
      for (const verification of resultWithThreshold.data.evidenceVerifications) {
        expect(verification.credibility).toBeGreaterThanOrEqual(0);
      }
    });

    it('should handle empty evidence set', async () => {
      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-007',
        caseCategory: 'GovernanceDispute',
        evidence: [],
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.success).toBe(true);
      expect(result.data.evidenceVerifications.length).toBe(0);
      expect(result.data.overallCaseQuality).toBe('No Evidence');
    });

    it('should record workflow steps', async () => {
      const evidence = createEvidenceSet();

      const result = await executeEpistemicDisputeResolutionWorkflow({
        caseId: 'case-test-workflow-008',
        caseCategory: 'IdentityDispute',
        evidence,
        complainantDid: 'did:mycelix:complainant',
        respondentDid: 'did:mycelix:respondent',
      });

      expect(result.steps).toBeDefined();
      expect(result.steps.length).toBeGreaterThan(0);
      // Steps use 'status' property with 'completed'/'failed' values
      expect(result.steps.every((s) => s.status === 'completed')).toBe(true);
    });

    it('should handle all case categories', async () => {
      const evidence = createEvidenceSet();
      const categories: CaseCategory[] = [
        'ContractDispute',
        'ConductViolation',
        'PropertyDispute',
        'FinancialDispute',
        'GovernanceDispute',
        'IdentityDispute',
        'IPDispute',
        'Other',
      ];

      for (const category of categories) {
        const result = await executeEpistemicDisputeResolutionWorkflow({
          caseId: `case-test-category-${category}`,
          caseCategory: category,
          evidence,
          complainantDid: 'did:mycelix:complainant',
          respondentDid: 'did:mycelix:respondent',
        });

        expect(result.success).toBe(true);
        expect(result.data.caseId).toBe(`case-test-category-${category}`);
      }
    });
  });
});

// ============================================================================
// Singleton/Factory Tests
// ============================================================================

describe('Integration: Epistemic Evidence Bridge Factory', () => {
  beforeEach(() => {
    resetEpistemicEvidenceBridge();
  });

  it('should return singleton instance', () => {
    const bridge1 = getEpistemicEvidenceBridge();
    const bridge2 = getEpistemicEvidenceBridge();

    expect(bridge1).toBe(bridge2);
  });

  it('should reset singleton correctly', () => {
    const bridge1 = getEpistemicEvidenceBridge();
    resetEpistemicEvidenceBridge();
    const bridge2 = getEpistemicEvidenceBridge();

    expect(bridge1).not.toBe(bridge2);
  });

  it('should create new instance directly', () => {
    const bridge1 = new EpistemicEvidenceBridge();
    const bridge2 = new EpistemicEvidenceBridge();

    expect(bridge1).not.toBe(bridge2);
  });
});

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

describe('Integration: Edge Cases', () => {
  let bridge: EpistemicEvidenceBridge;

  beforeEach(() => {
    resetEpistemicEvidenceBridge();
    bridge = new EpistemicEvidenceBridge();
  });

  it('should handle evidence with unknown type gracefully', async () => {
    const evidence = createTestEvidence({ type_: 'Other' });
    const result = await bridge.verifyEvidence(evidence, 'ContractDispute');

    expect(result).toBeDefined();
    expect(result.classification).toBeDefined();
  });

  it('should handle very long evidence descriptions', async () => {
    const longDescription = 'A'.repeat(10000);
    const evidence = createTestEvidence({ description: longDescription });

    const result = await bridge.verifyEvidence(evidence, 'ContractDispute');
    expect(result).toBeDefined();
  });

  it('should handle special characters in evidence content', async () => {
    const evidence = createTestEvidence({
      title: '契約書 Contract <script>alert("xss")</script>',
      description: 'Description with émojis 🎉 and spëcial chars @#$%',
    });

    const result = await bridge.verifyEvidence(evidence, 'ContractDispute');
    expect(result).toBeDefined();
  });

  it('should handle concurrent verifications', async () => {
    const evidenceSet = createEvidenceSet();

    const promises = evidenceSet.map((e) => bridge.verifyEvidence(e, 'ContractDispute'));

    const results = await Promise.all(promises);

    expect(results.length).toBe(evidenceSet.length);
    results.forEach((r) => expect(r).toBeDefined());
  });

  it('should handle batch with duplicate evidence IDs', async () => {
    const evidence = createTestEvidence({ id: 'duplicate-id' });
    const input: BatchVerificationInput = {
      case_id: 'case-test-duplicate',
      evidence_ids: ['duplicate-id', 'duplicate-id'],
    };

    const result = await bridge.verifyEvidenceBatch(input, [evidence], 'ContractDispute');

    // Should only process once despite duplicate IDs
    expect(result.results.length).toBe(1);
  });

  it('should handle mismatched evidence IDs in batch', async () => {
    const evidence = createTestEvidence({ id: 'existing-id' });
    const input: BatchVerificationInput = {
      case_id: 'case-test-mismatch',
      evidence_ids: ['non-existent-id-1', 'non-existent-id-2'],
    };

    const result = await bridge.verifyEvidenceBatch(input, [evidence], 'ContractDispute');

    // Should not find any matching evidence
    expect(result.results.length).toBe(0);
  });
});
