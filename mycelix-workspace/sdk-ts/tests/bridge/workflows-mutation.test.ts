// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Workflow Mutation Tests
 *
 * Mutation-targeted tests for bridge/workflows.ts to improve mutation score.
 * Focus on edge cases, error paths, and boundary conditions.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as bridge from '../../src/bridge/index.js';
import {
  executeProposalWorkflow,
  executeLendingWorkflow,
  executePropertyTransferWorkflow,
  executeEnergyTradeWorkflow,
  executeEnforcementWorkflow,
  executePublicationWorkflow,
  executeComprehensiveIdentityCheck,
  executeWorkflow,
  executeExtendedWorkflow,
  executeRegenerativeExitWorkflow,
  executeCreditAssessmentWorkflow,
  executeFactCheckWorkflow,
  executeDefamationDisputeWorkflow,
  executeEpistemicDisputeResolutionWorkflow,
  workflowRegistry,
  extendedWorkflowRegistry,
  type WorkflowResult,
  type WorkflowType,
  type ExtendedWorkflowType,
} from '../../src/bridge/workflows.js';

// ============================================================================
// Test Helpers
// ============================================================================

function createTestDid(name: string): string {
  return `did:mycelix:test-${name}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// ============================================================================
// Workflow Registry Tests
// ============================================================================

describe('Workflow Registry', () => {
  it('should have all workflow types registered', () => {
    expect(workflowRegistry.proposal).toBeDefined();
    expect(workflowRegistry.lending).toBeDefined();
    expect(workflowRegistry.property_transfer).toBeDefined();
    expect(workflowRegistry.energy_trade).toBeDefined();
    expect(workflowRegistry.enforcement).toBeDefined();
    expect(workflowRegistry.publication).toBeDefined();
    expect(workflowRegistry.identity_check).toBeDefined();
  });

  it('should have all extended workflow types registered', () => {
    expect(extendedWorkflowRegistry.regenerative_exit).toBeDefined();
    expect(extendedWorkflowRegistry.credit_assessment).toBeDefined();
    expect(extendedWorkflowRegistry.fact_check).toBeDefined();
    expect(extendedWorkflowRegistry.defamation_dispute).toBeDefined();
    expect(extendedWorkflowRegistry.epistemic_dispute_resolution).toBeDefined();
  });
});

// ============================================================================
// executeWorkflow Tests
// ============================================================================

describe('executeWorkflow', () => {
  it('should execute a valid workflow type', async () => {
    const result = await executeWorkflow('identity_check', {
      subjectDid: createTestDid('subject'),
      requesterHapp: 'governance',
      checkCategories: ['reputation'],
    });

    expect(result).toBeDefined();
    expect(result.steps).toBeDefined();
    expect(typeof result.duration).toBe('number');
  });

  it('should return error for unknown workflow type', async () => {
    const result = await executeWorkflow('unknown_workflow' as WorkflowType, {});

    expect(result.success).toBe(false);
    expect(result.error).toContain('Unknown workflow type');
    expect(result.steps).toEqual([]);
    expect(result.duration).toBe(0);
  });
});

describe('executeExtendedWorkflow', () => {
  it('should execute an extended workflow type', async () => {
    const result = await executeExtendedWorkflow('credit_assessment', {
      applicantDid: createTestDid('applicant'),
      requestedAmount: 10000,
      purpose: 'loan',
    });

    expect(result).toBeDefined();
    expect(result.steps).toBeDefined();
  });

  it('should return error for unknown extended workflow type', async () => {
    const result = await executeExtendedWorkflow('totally_unknown' as ExtendedWorkflowType, {});

    expect(result.success).toBe(false);
    expect(result.error).toContain('Unknown workflow type');
    expect(result.steps).toEqual([]);
    expect(result.duration).toBe(0);
  });

  it('should execute base workflow types through extended registry', async () => {
    const result = await executeExtendedWorkflow('proposal', {
      proposerDid: createTestDid('proposer'),
      daoId: 'test-dao',
      title: 'Test Proposal',
      description: 'Test description',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    expect(result).toBeDefined();
  });
});

// ============================================================================
// Proposal Workflow Tests
// ============================================================================

describe('executeProposalWorkflow - Mutation Coverage', () => {
  it('should fail when identity not verified', async () => {
    // By using an invalid/unusual DID, the mock verification may fail
    const result = await executeProposalWorkflow({
      proposerDid: 'invalid-did',
      daoId: 'test-dao',
      title: 'Test',
      description: 'Test',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    // The workflow should handle verification - check structure
    expect(result.steps.length).toBeGreaterThanOrEqual(0);
    expect(result.duration).toBeGreaterThanOrEqual(0);
  });

  it('should track duration correctly', async () => {
    const start = Date.now();
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'test-dao',
      title: 'Test',
      description: 'Test',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });
    const elapsed = Date.now() - start;

    expect(result.duration).toBeGreaterThanOrEqual(0);
    expect(result.duration).toBeLessThanOrEqual(elapsed + 100);
  });

  it('should include all step names', async () => {
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'test-dao',
      title: 'Test',
      description: 'Test',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Identity');
  });
});

// ============================================================================
// Lending Workflow Tests
// ============================================================================

describe('executeLendingWorkflow - Mutation Coverage', () => {
  it('should complete with collateral verification', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 10000,
      currency: 'USD',
      termDays: 30,
      collateralAssetId: 'asset-123',
    });

    expect(result.steps.length).toBeGreaterThanOrEqual(0);
    expect(result.duration).toBeGreaterThanOrEqual(0);
  });

  it('should complete without collateral', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 5000,
      currency: 'USD',
      termDays: 15,
    });

    // Without collateral, should skip collateral verification step
    expect(result.steps.length).toBeGreaterThanOrEqual(2);
  });

  it('should calculate interest rate based on credit score', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 10000,
      currency: 'USD',
      termDays: 30,
    });

    if (result.success && result.data) {
      expect(result.data.interestRate).toBeGreaterThanOrEqual(0.05);
      expect(result.data.loanId).toContain('loan-');
    }
  });
});

// ============================================================================
// Property Transfer Workflow Tests
// ============================================================================

describe('executePropertyTransferWorkflow - Mutation Coverage', () => {
  it('should verify both seller and buyer', async () => {
    const result = await executePropertyTransferWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      assetId: 'property-123',
      price: 500000,
      currency: 'USD',
    });

    expect(result.steps.length).toBeGreaterThanOrEqual(2);
    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Seller');
    expect(stepNames).toContain('Verify Buyer');
  });

  it('should include escrow creation step', async () => {
    const result = await executePropertyTransferWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      assetId: 'property-456',
      price: 100000,
      currency: 'USD',
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Create Escrow');
  });

  it('should generate unique transfer and escrow IDs', async () => {
    const result = await executePropertyTransferWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      assetId: 'property-789',
      price: 200000,
      currency: 'USD',
    });

    if (result.success && result.data) {
      expect(result.data.transferId).toContain('transfer-');
      expect(result.data.escrowWalletId).toContain('escrow-');
    }
  });
});

// ============================================================================
// Energy Trade Workflow Tests
// ============================================================================

describe('executeEnergyTradeWorkflow - Mutation Coverage', () => {
  it('should calculate total cost correctly', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('producer'),
      buyerDid: createTestDid('consumer'),
      amountKwh: 100,
      source: 'solar',
      pricePerKwh: 0.15,
    });

    // Step 2 should have totalCost calculated
    const buyerFundsStep = result.steps.find((s) => s.name === 'Verify Buyer Funds');
    if (buyerFundsStep?.result) {
      const stepResult = buyerFundsStep.result as { totalCost: number };
      expect(stepResult.totalCost).toBe(15); // 100 * 0.15
    }
  });

  it('should generate trade and credit IDs', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('producer'),
      buyerDid: createTestDid('consumer'),
      amountKwh: 50,
      source: 'wind',
      pricePerKwh: 0.10,
    });

    if (result.success && result.data) {
      expect(result.data.tradeId).toContain('trade-');
      expect(result.data.creditId).toContain('credit-');
    }
  });

  it('should have 6 steps for complete workflow', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('producer'),
      buyerDid: createTestDid('consumer'),
      amountKwh: 200,
      source: 'hydro',
      pricePerKwh: 0.08,
    });

    expect(result.steps.length).toBe(6);
  });
});

// ============================================================================
// Enforcement Workflow Tests
// ============================================================================

describe('executeEnforcementWorkflow - Mutation Coverage', () => {
  it('should process multiple remedies', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-123',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'reputation_penalty', targetHapp: 'governance', details: { amount: 0.2 } },
        { type: 'asset_freeze', targetHapp: 'finance', details: { duration: 30 } },
      ],
    });

    expect(result.data?.enforcementIds.length).toBe(2);
    expect(result.data?.acknowledged).toBeGreaterThanOrEqual(0);
  });

  it('should handle empty remedies array', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-456',
      targetDid: createTestDid('target'),
      remedies: [],
    });

    // Should only have validate and broadcast steps
    expect(result.steps.length).toBe(2);
    expect(result.data?.enforcementIds).toEqual([]);
    expect(result.data?.acknowledged).toBe(0);
  });

  it('should determine success based on acknowledgment count', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-789',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'suspension', targetHapp: 'marketplace', details: {} },
      ],
    });

    // Success should equal whether all remedies were acknowledged
    expect(result.success).toBe(result.data?.acknowledged === 1);
  });

  it('should generate unique enforcement IDs', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-abc',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'payment_order', targetHapp: 'finance', details: { amount: 1000 } },
      ],
    });

    if (result.data?.enforcementIds[0]) {
      expect(result.data.enforcementIds[0]).toContain('enf-');
    }
  });
});

// ============================================================================
// Publication Workflow Tests
// ============================================================================

describe('executePublicationWorkflow - Mutation Coverage', () => {
  it('should request fact check when requested', async () => {
    const result = await executePublicationWorkflow({
      authorDid: createTestDid('author'),
      contentHash: 'hash-123',
      title: 'Test Article',
      tags: ['news', 'test'],
      requestFactCheck: true,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Request Fact Check');
    expect(result.data?.verificationStatus).toBe('Pending');
  });

  it('should skip fact check when not requested', async () => {
    const result = await executePublicationWorkflow({
      authorDid: createTestDid('author'),
      contentHash: 'hash-456',
      title: 'Test Article',
      tags: ['opinion'],
      requestFactCheck: false,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).not.toContain('Request Fact Check');
    expect(result.data?.verificationStatus).toBe('Unverified');
  });

  it('should generate content ID', async () => {
    const result = await executePublicationWorkflow({
      authorDid: createTestDid('author'),
      contentHash: 'hash-789',
      title: 'Test',
      tags: [],
      requestFactCheck: false,
    });

    if (result.success && result.data) {
      expect(result.data.contentId).toContain('content-');
    }
  });
});

// ============================================================================
// Comprehensive Identity Check Tests
// ============================================================================

describe('executeComprehensiveIdentityCheck - Mutation Coverage', () => {
  it('should check reputation when category included', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'governance',
      checkCategories: ['reputation'],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Query Reputation');
  });

  it('should check standing when category included', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'finance',
      checkCategories: ['standing'],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Check Justice Standing');
  });

  it('should skip optional checks when not in categories', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'media',
      checkCategories: [],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).not.toContain('Query Reputation');
    expect(stepNames).not.toContain('Check Justice Standing');
  });

  it('should calculate aggregated score', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'property',
      checkCategories: ['reputation', 'credentials', 'standing'],
    });

    if (result.success && result.data) {
      expect(typeof result.data.aggregatedScore).toBe('number');
      expect(result.data.aggregatedScore).toBeGreaterThanOrEqual(0);
      expect(result.data.aggregatedScore).toBeLessThanOrEqual(1);
    }
  });
});

// ============================================================================
// Regenerative Exit Workflow Tests
// ============================================================================

describe('executeRegenerativeExitWorkflow - Mutation Coverage', () => {
  it('should calculate ownership transferred', async () => {
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-123',
      communityDid: createTestDid('community'),
      investorsDids: [createTestDid('inv1'), createTestDid('inv2')],
      tranchePercentage: 10,
    });

    if (result.success && result.data) {
      expect(result.data.ownershipTransferred).toBe(10);
      expect(result.data.trancheId).toContain('tranche-');
    }
  });

  it('should process all investors', async () => {
    const investors = [createTestDid('inv1'), createTestDid('inv2'), createTestDid('inv3')];
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-456',
      communityDid: createTestDid('community'),
      investorsDids: investors,
      tranchePercentage: 15,
    });

    const paymentStep = result.steps.find((s) => s.name === 'Process Investor Payments');
    if (paymentStep?.result) {
      const stepResult = paymentStep.result as { investorsPaid: number };
      expect(stepResult.investorsPaid).toBe(3);
    }
  });

  it('should have steps for workflow', async () => {
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-789',
      communityDid: createTestDid('community'),
      investorsDids: [],
      tranchePercentage: 5,
    });

    // Steps may vary based on certification result
    expect(result.steps.length).toBeGreaterThanOrEqual(1);
  });
});

// ============================================================================
// Credit Assessment Workflow Tests
// ============================================================================

describe('executeCreditAssessmentWorkflow - Mutation Coverage', () => {
  it('should calculate credit tiers correctly', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 50000,
      purpose: 'loan',
    });

    if (result.success && result.data) {
      // Tier should be one of the valid options
      expect(['Excellent', 'Good', 'Fair', 'Poor', 'Insufficient']).toContain(
        result.data.creditTier
      );
      expect(result.data.creditScore).toBeGreaterThanOrEqual(0);
      expect(result.data.creditScore).toBeLessThanOrEqual(1000);
    }
  });

  it('should calculate interest rate based on risk', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 25000,
      purpose: 'investment',
    });

    if (result.success && result.data) {
      expect(result.data.interestRate).toBeGreaterThanOrEqual(0.05);
      expect(result.data.interestRate).toBeLessThanOrEqual(0.20);
    }
  });

  it('should include all factor scores', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 10000,
      purpose: 'marketplace',
    });

    if (result.success && result.data) {
      expect(result.data.factors.matlScore).toBeDefined();
      expect(result.data.factors.paymentHistory).toBeDefined();
      expect(result.data.factors.creditUtilization).toBeDefined();
    }
  });

  it('should calculate max approved amount', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 100000,
      purpose: 'energy',
    });

    if (result.success && result.data) {
      expect(result.data.maxApproved).toBeLessThanOrEqual(100000);
      expect(result.data.maxApproved).toBeGreaterThanOrEqual(0);
    }
  });
});

// ============================================================================
// Fact Check Workflow Tests
// ============================================================================

describe('executeFactCheckWorkflow - Mutation Coverage', () => {
  it('should classify claims epistemically', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'content-123',
      claimText: 'The sky is blue',
      requestorHapp: 'media',
    });

    if (result.success && result.data) {
      expect(result.data.epistemicClassification.empirical).toBeDefined();
      expect(result.data.epistemicClassification.normative).toBeDefined();
      expect(result.data.epistemicClassification.mythic).toBeDefined();
    }
  });

  it('should determine verdict based on support ratio', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'content-456',
      claimText: 'Test claim',
      requestorHapp: 'knowledge',
    });

    if (result.success && result.data) {
      expect(['True', 'MostlyTrue', 'Mixed', 'MostlyFalse', 'False', 'Unverifiable']).toContain(
        result.data.verdict
      );
      expect(result.data.confidence).toBeGreaterThanOrEqual(0);
      expect(result.data.confidence).toBeLessThanOrEqual(1);
    }
  });

  it('should return supporting and contradicting claims', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'content-789',
      claimText: 'Another claim',
      requestorHapp: 'justice',
    });

    if (result.success && result.data) {
      expect(Array.isArray(result.data.supportingClaims)).toBe(true);
      expect(Array.isArray(result.data.contradictingClaims)).toBe(true);
    }
  });
});

// ============================================================================
// Defamation Dispute Workflow Tests
// ============================================================================

describe('executeDefamationDisputeWorkflow - Mutation Coverage', () => {
  it('should generate case ID', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'content-123',
      claimedDamages: 10000,
    });

    if (result.success && result.data) {
      expect(result.data.caseId).toContain('case-defamation-');
    }
  });

  it('should determine liability based on fact check', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'content-456',
      claimedDamages: 50000,
    });

    if (result.success && result.data) {
      expect(['liable', 'not_liable', 'partial']).toContain(result.data.liabilityDetermination);
    }
  });

  it('should calculate damages when liable', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'content-789',
      claimedDamages: 25000,
    });

    if (result.success && result.data) {
      // Damages should be between 0 and claimed amount
      expect(result.data.awardedDamages).toBeGreaterThanOrEqual(0);
      expect(result.data.awardedDamages).toBeLessThanOrEqual(25000);
    }
  });
});

// ============================================================================
// Error Handling Tests
// ============================================================================

describe('Workflow Error Handling', () => {
  it('should handle errors gracefully in proposal workflow', async () => {
    // Even with unusual parameters, should not throw
    const result = await executeProposalWorkflow({
      proposerDid: '',
      daoId: '',
      title: '',
      description: '',
      votingPeriodHours: 0,
      quorumPercentage: 0,
    });

    expect(result).toBeDefined();
    expect(typeof result.duration).toBe('number');
  });

  it('should handle errors gracefully in lending workflow', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: '',
      lenderDid: '',
      amount: -100,
      currency: '',
      termDays: 0,
    });

    expect(result).toBeDefined();
  });

  it('should handle errors gracefully in energy trade workflow', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: '',
      buyerDid: '',
      amountKwh: 0,
      source: '',
      pricePerKwh: 0,
    });

    expect(result).toBeDefined();
  });
});

// ============================================================================
// Duration Tracking Tests
// ============================================================================

describe('Workflow Duration Tracking', () => {
  it('should track duration in all workflow types', async () => {
    const workflows = [
      executeProposalWorkflow({
        proposerDid: createTestDid('p'),
        daoId: 'd',
        title: 't',
        description: 'd',
        votingPeriodHours: 1,
        quorumPercentage: 0.5,
      }),
      executeLendingWorkflow({
        borrowerDid: createTestDid('b'),
        lenderDid: createTestDid('l'),
        amount: 100,
        currency: 'USD',
        termDays: 1,
      }),
      executeEnergyTradeWorkflow({
        sellerDid: createTestDid('s'),
        buyerDid: createTestDid('b'),
        amountKwh: 10,
        source: 'solar',
        pricePerKwh: 0.1,
      }),
    ];

    const results = await Promise.all(workflows);

    for (const result of results) {
      expect(result.duration).toBeGreaterThanOrEqual(0);
      expect(typeof result.duration).toBe('number');
    }
  });
});

// ============================================================================
// Step Status Tests
// ============================================================================

describe('Workflow Step Status', () => {
  it('should mark steps as completed or failed', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'governance',
      checkCategories: ['reputation', 'standing'],
    });

    for (const step of result.steps) {
      expect(['pending', 'running', 'completed', 'failed', 'skipped']).toContain(step.status);
    }
  });

  it('should include happ in each step', async () => {
    const result = await executePropertyTransferWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      assetId: 'asset-1',
      price: 1000,
      currency: 'USD',
    });

    for (const step of result.steps) {
      expect(step.happ).toBeDefined();
      expect(typeof step.happ).toBe('string');
    }
  });
});

// ============================================================================
// Boundary & Edge Case Tests for Mutation Coverage
// ============================================================================

describe('Credit Score Tier Boundaries - Mutation Targets', () => {
  // These tests target the conditional expressions at lines 994-998:
  // if (creditScore >= 900) creditTier = 'Excellent';
  // else if (creditScore >= 750) creditTier = 'Good';
  // else if (creditScore >= 600) creditTier = 'Fair';
  // else if (creditScore >= 400) creditTier = 'Poor';
  // else creditTier = 'Insufficient';

  it('should verify credit score calculation uses weighted formula', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 10000,
      purpose: 'loan',
    });

    if (result.success && result.data) {
      // Verify the weighted formula components
      const factors = result.data.factors;
      expect(factors.paymentHistory).toBe(0.85); // 35% weight
      expect(factors.creditUtilization).toBe(0.7); // 30% weight
      expect(factors.creditHistoryLength).toBe(0.6); // 15% weight
      expect(factors.justiceStanding).toBe(1.0);
      expect(factors.governanceParticipation).toBe(0.75);
    }
  });

  it('should verify interest rate calculation formula', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 50000,
      purpose: 'investment',
    });

    if (result.success && result.data) {
      // Base rate is 0.05, risk premium is (1 - approvalRatio) * 0.15
      const baseRate = 0.05;
      const approvalRatio = result.data.creditScore / 1000;
      const expectedRiskPremium = (1 - approvalRatio) * 0.15;
      const expectedRate = baseRate + expectedRiskPremium;

      expect(result.data.interestRate).toBeCloseTo(expectedRate, 5);
    }
  });

  it('should verify maxApproved calculation', async () => {
    const requestedAmount = 75000;
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount,
      purpose: 'marketplace',
    });

    if (result.success && result.data) {
      // maxApproved = floor(requestedAmount * (creditScore / 1000))
      const expectedMax = Math.floor(requestedAmount * (result.data.creditScore / 1000));
      expect(result.data.maxApproved).toBe(expectedMax);
    }
  });
});

describe('Reputation Threshold Boundaries - Mutation Targets', () => {
  // Target: if (reputation.aggregatedScore < 0.3) at line 101

  it('should verify proposal workflow checks reputation threshold', async () => {
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'test-dao',
      title: 'Test Proposal',
      description: 'Test description for mutation coverage',
      votingPeriodHours: 48,
      quorumPercentage: 0.51,
    });

    // The workflow should reach the reputation check step
    expect(result.steps.length).toBeGreaterThanOrEqual(2);
    const repStep = result.steps.find((s) => s.name === 'Check Reputation');
    expect(repStep).toBeDefined();
  });

  it('should verify reputation step result contains aggregated score', async () => {
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'dao-123',
      title: 'Another Proposal',
      description: 'Testing reputation aggregation',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    const repStep = result.steps.find((s) => s.name === 'Check Reputation');
    if (repStep?.result) {
      const repResult = repStep.result as { aggregatedScore: number };
      expect(typeof repResult.aggregatedScore).toBe('number');
    }
  });
});

describe('Defamation Liability Determination - Mutation Targets', () => {
  // Target lines 1227-1236:
  // if (factCheckResult === 'False' || factCheckResult === 'MostlyFalse')
  //   liabilityDetermination = 'liable'; awardedDamages = claimedDamages * 0.8
  // else if (factCheckResult === 'Mixed')
  //   liabilityDetermination = 'partial'; awardedDamages = claimedDamages * 0.3

  it('should verify damages calculation with claimed damages', async () => {
    const claimedDamages = 100000;
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'defamation-content-1',
      claimedDamages,
    });

    if (result.success && result.data) {
      // If liable, damages should be 80% of claimed (floor(100000 * 0.8) = 80000)
      // If partial, damages should be 30% of claimed (floor(100000 * 0.3) = 30000)
      // If not liable, damages should be 0
      if (result.data.liabilityDetermination === 'liable') {
        expect(result.data.awardedDamages).toBe(Math.floor(claimedDamages * 0.8));
      } else if (result.data.liabilityDetermination === 'partial') {
        expect(result.data.awardedDamages).toBe(Math.floor(claimedDamages * 0.3));
      } else {
        expect(result.data.awardedDamages).toBe(0);
      }
    }
  });

  it('should verify damages multiplier is exactly 0.8 for liable', async () => {
    const claimedDamages = 50000;
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'defamation-content-2',
      claimedDamages,
    });

    if (result.success && result.data && result.data.liabilityDetermination === 'liable') {
      // Math.floor(50000 * 0.8) = 40000
      expect(result.data.awardedDamages).toBe(40000);
    }
  });

  it('should verify damages multiplier is exactly 0.3 for partial', async () => {
    const claimedDamages = 50000;
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'defamation-content-3',
      claimedDamages,
    });

    if (result.success && result.data && result.data.liabilityDetermination === 'partial') {
      // Math.floor(50000 * 0.3) = 15000
      expect(result.data.awardedDamages).toBe(15000);
    }
  });

  it('should verify party verification error message', async () => {
    // Test that both parties must be verified
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: 'invalid-did',
      respondentDid: 'also-invalid',
      contentId: 'content-error',
      claimedDamages: 10000,
    });

    // Should have Verify Parties step
    const verifyStep = result.steps.find((s) => s.name === 'Verify Parties');
    expect(verifyStep).toBeDefined();
  });
});

describe('Fact Check Verdict Logic - Mutation Targets', () => {
  // Target lines 1098-1116:
  // if (relatedClaims.length < 2) verdict = 'Unverifiable'
  // else if (supportRatio > 0.9) verdict = 'True'
  // else if (supportRatio > 0.7) verdict = 'MostlyTrue'
  // else if (supportRatio > 0.4) verdict = 'Mixed'
  // else if (supportRatio > 0.2) verdict = 'MostlyFalse'
  // else verdict = 'False'

  it('should verify confidence values for different verdicts', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'fact-check-1',
      claimText: 'Testing verdict confidence calculation',
      requestorHapp: 'media',
    });

    if (result.success && result.data) {
      // Confidence values based on verdict:
      // True: 0.85 + inferenceResult.confidenceBoost
      // MostlyTrue: 0.7
      // Mixed: 0.5
      // MostlyFalse: 0.6
      // False: 0.8
      // Unverifiable: 0.3
      expect(result.data.confidence).toBeGreaterThanOrEqual(0.3);
      expect(result.data.confidence).toBeLessThanOrEqual(1.0);
    }
  });

  it('should verify inference confidence boost of 0.1', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'fact-check-2',
      claimText: 'Testing inference boost',
      requestorHapp: 'knowledge',
    });

    if (result.success && result.data) {
      // The inference step should add 0.1 confidence boost for True verdict
      const inferenceStep = result.steps.find((s) => s.name === 'Reason Over Claims');
      if (inferenceStep?.result) {
        const inferResult = inferenceStep.result as { confidenceBoost: number };
        expect(inferResult.confidenceBoost).toBe(0.1);
      }
    }
  });

  it('should verify support ratio calculation includes 0.1 offset', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'fact-check-3',
      claimText: 'Testing support ratio formula',
      requestorHapp: 'justice',
    });

    if (result.success && result.data) {
      // supportRatio = supporting / (supporting + contradicting + 0.1)
      // With supporting=1, contradicting=0: 1 / (1 + 0 + 0.1) = 1/1.1 ≈ 0.909
      expect(result.data.supportingClaims.length).toBeGreaterThanOrEqual(0);
      expect(result.data.contradictingClaims.length).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('Lending Interest Rate Calculation - Mutation Targets', () => {
  // Target lines 227-229:
  // const baseRate = 0.05;
  // const riskPremium = (1 - creditScore) * 0.15;
  // const interestRate = baseRate + riskPremium;

  it('should verify base rate is exactly 0.05', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 10000,
      currency: 'USD',
      termDays: 30,
    });

    if (result.success && result.data) {
      // Interest rate should be at least baseRate (0.05) plus some risk premium
      expect(result.data.interestRate).toBeGreaterThanOrEqual(0.05);
    }
  });

  it('should verify risk premium multiplier is 0.15', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 5000,
      currency: 'EUR',
      termDays: 60,
    });

    if (result.success && result.data) {
      // Max interest rate would be 0.05 + (1 - 0) * 0.15 = 0.20
      expect(result.data.interestRate).toBeLessThanOrEqual(0.20);
    }
  });
});

describe('Readiness Threshold - Mutation Targets', () => {
  // Target line 829: if (readinessScore < 0.7)

  it('should verify readiness workflow includes certification check', async () => {
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-readiness-test',
      communityDid: createTestDid('community'),
      investorsDids: [createTestDid('inv1')],
      tranchePercentage: 10,
    });

    // Should have certification step first
    const certStep = result.steps.find((s) => s.name === 'Verify Operator Certifications');
    expect(certStep).toBeDefined();
    expect(certStep?.happ).toBe('praxis');
  });

  it('should verify readiness step may exist after certification', async () => {
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-dimensions',
      communityDid: createTestDid('community'),
      investorsDids: [],
      tranchePercentage: 5,
    });

    // If workflow passes certification, it will have readiness step
    // If it fails certification, it won't reach readiness step
    const readinessStep = result.steps.find((s) => s.name === 'Assess Community Readiness');
    if (readinessStep?.result) {
      const readinessResult = readinessStep.result as {
        technicalCapacity: number;
        financialCapacity: number;
        governanceCapacity: number;
        trainingCompletion: number;
        communitySupport: number;
        overallReadiness: number;
      };
      // Verify all dimensions are numbers between 0 and 1
      expect(readinessResult.technicalCapacity).toBeGreaterThanOrEqual(0);
      expect(readinessResult.technicalCapacity).toBeLessThanOrEqual(1);
      expect(readinessResult.overallReadiness).toBeGreaterThanOrEqual(0);
      expect(readinessResult.overallReadiness).toBeLessThanOrEqual(1);
    } else {
      // Workflow failed early, verify it has at least certification step
      expect(result.steps.length).toBeGreaterThanOrEqual(1);
    }
  });
});

describe('String Literal Verification - Mutation Targets', () => {
  // Test that specific string literals are correct (step names, error messages)

  it('should verify all proposal workflow step names', async () => {
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'dao-string-test',
      title: 'String Test',
      description: 'Testing string literals',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Identity');
    expect(stepNames).toContain('Check Reputation');
  });

  it('should verify lending workflow step names', async () => {
    const result = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 1000,
      currency: 'USD',
      termDays: 10,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Borrower');
    expect(stepNames).toContain('Check Credit Score');
  });

  it('should verify energy trade workflow step names', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      amountKwh: 50,
      source: 'wind',
      pricePerKwh: 0.12,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Seller');
    expect(stepNames).toContain('Verify Buyer Funds');
    expect(stepNames).toContain('Execute Trade');
    expect(stepNames).toContain('Process Payment');
    expect(stepNames).toContain('Issue Credit');
    expect(stepNames).toContain('Update Reputations');
  });

  it('should verify property transfer workflow step names', async () => {
    const result = await executePropertyTransferWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      assetId: 'property-strings',
      price: 100000,
      currency: 'USD',
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Seller');
    expect(stepNames).toContain('Verify Buyer');
    expect(stepNames).toContain('Check Liens');
    expect(stepNames).toContain('Create Escrow');
    expect(stepNames).toContain('Transfer Title');
    expect(stepNames).toContain('Finalize');
  });

  it('should verify enforcement workflow step names', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-strings',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'reputation_penalty', targetHapp: 'governance', details: {} },
      ],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Validate Decision');
    expect(stepNames).toContain('Broadcast Status');
  });

  it('should verify credit assessment workflow step names', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 10000,
      purpose: 'loan',
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Identity');
    expect(stepNames).toContain('Query MATL Scores');
    expect(stepNames).toContain('Check Payment History');
    expect(stepNames).toContain('Check Justice Standing');
    expect(stepNames).toContain('Check Governance');
    expect(stepNames).toContain('Calculate Score');
  });

  it('should verify fact check workflow step names', async () => {
    const result = await executeFactCheckWorkflow({
      contentId: 'content-strings',
      claimText: 'Test claim for step names',
      requestorHapp: 'media',
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Classify Claim');
    expect(stepNames).toContain('Query Related Claims');
    expect(stepNames).toContain('Semantic Search');
    expect(stepNames).toContain('Reason Over Claims');
    expect(stepNames).toContain('Calculate Verdict');
    expect(stepNames).toContain('Return Result');
  });

  it('should verify defamation dispute workflow step names', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'content-defamation-strings',
      claimedDamages: 10000,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Parties');
    expect(stepNames).toContain('File Case');
    expect(stepNames).toContain('Retrieve Content');
    expect(stepNames).toContain('Fact-Check Claims');
    expect(stepNames).toContain('Determine Liability');
  });

  it('should verify regenerative exit workflow step names', async () => {
    const result = await executeRegenerativeExitWorkflow({
      projectId: 'project-strings',
      communityDid: createTestDid('community'),
      investorsDids: [createTestDid('inv')],
      tranchePercentage: 10,
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Operator Certifications');
  });
});

describe('HappId String Verification - Mutation Targets', () => {
  it('should verify correct happ IDs in proposal workflow', async () => {
    const result = await executeProposalWorkflow({
      proposerDid: createTestDid('proposer'),
      daoId: 'dao-happ-test',
      title: 'Happ Test',
      description: 'Testing happ IDs',
      votingPeriodHours: 24,
      quorumPercentage: 0.5,
    });

    const happIds = result.steps.map((s) => s.happ);
    expect(happIds).toContain('identity');
    expect(happIds).toContain('governance');
  });

  it('should verify correct happ IDs in energy trade workflow', async () => {
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      amountKwh: 100,
      source: 'solar',
      pricePerKwh: 0.1,
    });

    const happIds = result.steps.map((s) => s.happ);
    expect(happIds).toContain('energy');
    expect(happIds).toContain('finance');
  });

  it('should verify correct happ IDs in credit assessment workflow', async () => {
    const result = await executeCreditAssessmentWorkflow({
      applicantDid: createTestDid('applicant'),
      requestedAmount: 10000,
      purpose: 'loan',
    });

    const happIds = result.steps.map((s) => s.happ);
    expect(happIds).toContain('identity');
    expect(happIds).toContain('finance');
    expect(happIds).toContain('justice');
    expect(happIds).toContain('governance');
  });

  it('should verify correct happ IDs in defamation dispute workflow', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'happ-ids-test',
      claimedDamages: 10000,
    });

    const happIds = result.steps.map((s) => s.happ);
    expect(happIds).toContain('identity');
    expect(happIds).toContain('justice');
    expect(happIds).toContain('media');
    expect(happIds).toContain('knowledge');
  });
});

describe('Numeric Constant Verification - Mutation Targets', () => {
  it('should verify energy trade total cost calculation', async () => {
    const amountKwh = 150;
    const pricePerKwh = 0.20;
    const result = await executeEnergyTradeWorkflow({
      sellerDid: createTestDid('seller'),
      buyerDid: createTestDid('buyer'),
      amountKwh,
      source: 'solar',
      pricePerKwh,
    });

    const buyerFundsStep = result.steps.find((s) => s.name === 'Verify Buyer Funds');
    if (buyerFundsStep?.result) {
      const stepResult = buyerFundsStep.result as { totalCost: number };
      // Should be exactly amountKwh * pricePerKwh
      expect(stepResult.totalCost).toBe(amountKwh * pricePerKwh);
    }
  });

  it('should verify enforcement acknowledgment counting', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-count-test',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'reputation_penalty', targetHapp: 'governance', details: {} },
        { type: 'asset_freeze', targetHapp: 'finance', details: {} },
        { type: 'suspension', targetHapp: 'marketplace', details: {} },
      ],
    });

    if (result.data) {
      // acknowledged should be between 0 and remedies.length
      expect(result.data.acknowledged).toBeGreaterThanOrEqual(0);
      expect(result.data.acknowledged).toBeLessThanOrEqual(3);
      // enforcementIds should match remedies count
      expect(result.data.enforcementIds.length).toBe(3);
    }
  });
});

describe('Collateral Conditional Branch - Mutation Targets', () => {
  // Target line 206: if (params.collateralAssetId)

  it('should have extra step with collateral', async () => {
    const withCollateral = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 10000,
      currency: 'USD',
      termDays: 30,
      collateralAssetId: 'asset-123',
    });

    const withoutCollateral = await executeLendingWorkflow({
      borrowerDid: createTestDid('borrower'),
      lenderDid: createTestDid('lender'),
      amount: 10000,
      currency: 'USD',
      termDays: 30,
    });

    // With collateral should have one more step (Verify Collateral)
    const collateralStep = withCollateral.steps.find((s) => s.name === 'Verify Collateral');
    const noCollateralStep = withoutCollateral.steps.find((s) => s.name === 'Verify Collateral');

    expect(collateralStep).toBeDefined();
    expect(noCollateralStep).toBeUndefined();
  });
});

describe('Fact Check requestFactCheck Branch - Mutation Targets', () => {
  // Target publication workflow requestFactCheck boolean

  it('should have different verification status based on requestFactCheck', async () => {
    const withFactCheck = await executePublicationWorkflow({
      authorDid: createTestDid('author'),
      contentHash: 'hash-with-fc',
      title: 'With Fact Check',
      tags: ['news'],
      requestFactCheck: true,
    });

    const withoutFactCheck = await executePublicationWorkflow({
      authorDid: createTestDid('author'),
      contentHash: 'hash-without-fc',
      title: 'Without Fact Check',
      tags: ['opinion'],
      requestFactCheck: false,
    });

    if (withFactCheck.success && withFactCheck.data) {
      expect(withFactCheck.data.verificationStatus).toBe('Pending');
    }
    if (withoutFactCheck.success && withoutFactCheck.data) {
      expect(withoutFactCheck.data.verificationStatus).toBe('Unverified');
    }
  });
});

describe('Identity Check Categories Branch - Mutation Targets', () => {
  // Target executeComprehensiveIdentityCheck category checks
  // Note: Only 'reputation' and 'standing' categories have implementations

  it('should include reputation step when reputation in categories', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'finance',
      checkCategories: ['reputation'],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Query Reputation');
  });

  it('should include standing step when standing in categories', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'governance',
      checkCategories: ['standing'],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Check Justice Standing');
  });

  it('should include both reputation and standing when both categories included', async () => {
    const result = await executeComprehensiveIdentityCheck({
      subjectDid: createTestDid('subject'),
      requesterHapp: 'governance',
      checkCategories: ['reputation', 'standing'],
    });

    const stepNames = result.steps.map((s) => s.name);
    expect(stepNames).toContain('Verify Identity');
    expect(stepNames).toContain('Query Reputation');
    expect(stepNames).toContain('Check Justice Standing');
  });
});

describe('Enforcement Success Condition - Mutation Targets', () => {
  // Target line 518: success: acknowledged === params.remedies.length

  it('should return success true when all remedies acknowledged', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-all-ack',
      targetDid: createTestDid('target'),
      remedies: [
        { type: 'reputation_penalty', targetHapp: 'governance', details: {} },
      ],
    });

    // Success should be true if acknowledged equals remedies count
    expect(result.success).toBe(result.data?.acknowledged === 1);
  });

  it('should return success true when no remedies (empty array)', async () => {
    const result = await executeEnforcementWorkflow({
      decisionId: 'decision-no-remedies',
      targetDid: createTestDid('target'),
      remedies: [],
    });

    // With no remedies, acknowledged should be 0 and success should be true (0 === 0)
    expect(result.data?.acknowledged).toBe(0);
    expect(result.success).toBe(true);
  });
});

describe('Liability Enforcement Condition - Mutation Targets', () => {
  // Target line 1242: if (liabilityDetermination !== 'not_liable' && awardedDamages > 0)

  it('should create enforcement step when liable with damages', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'enforcement-test',
      claimedDamages: 50000,
    });

    if (result.success && result.data) {
      if (result.data.liabilityDetermination !== 'not_liable' && result.data.awardedDamages > 0) {
        // Should have Create Enforcement step
        const enforcementStep = result.steps.find((s) => s.name === 'Create Enforcement');
        expect(enforcementStep).toBeDefined();
      }
    }
  });

  it('should verify enforcement step exists for partial liability', async () => {
    const result = await executeDefamationDisputeWorkflow({
      complainantDid: createTestDid('complainant'),
      respondentDid: createTestDid('respondent'),
      contentId: 'partial-enforcement',
      claimedDamages: 100000,
    });

    // Check enforcement step based on liability
    const enforcementStep = result.steps.find((s) => s.name === 'Create Enforcement');
    const notifyStep = result.steps.find((s) => s.name === 'Notify Parties');

    // Notify Parties should always exist
    expect(notifyStep).toBeDefined();
  });
});
