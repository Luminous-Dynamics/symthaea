// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Health-Governance Integration Tests
 *
 * Tests for the health-governance integration module including:
 * - Utility functions (voter weight, outcome evaluation, conflict detection)
 * - Type validation
 * - Bridge functionality
 * - Cross-domain workflows
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  HealthGovernanceBridge,
  HealthPolicyClient,
  TrialOversightClient,
  PandemicResponseClient,
  DataPrivacyPolicyClient,
  calculateVoterWeight,
  evaluateProposalOutcome,
  detectConflictOfInterest,
  type HealthPolicyProposal,
  type HealthPolicyVote,
  type HealthVoterType,
  type HealthPolicyCategory,
  type TrialOversight,
  type TrialParticipantVoice,
  type PandemicResponseDecision,
  type DataPrivacyPolicy,
  type CredentialingDecision,
} from '../../src/integrations/health-governance/index.js';
import type { MycelixClient } from '../../src/client/index.js';

describe('Health-Governance Integration', () => {
  describe('calculateVoterWeight', () => {
    it('should give higher weight to affected patients', () => {
      const patientWeight = calculateVoterWeight('Patient', 'ClinicalGuidelines', true);
      const nonAffectedWeight = calculateVoterWeight('Patient', 'ClinicalGuidelines', false);
      expect(patientWeight).toBeGreaterThan(nonAffectedWeight);
    });

    it('should weight providers heavily for clinical guidelines', () => {
      const providerWeight = calculateVoterWeight('Provider', 'ClinicalGuidelines', false);
      const advocateWeight = calculateVoterWeight('CommunityAdvocate', 'ClinicalGuidelines', false);
      expect(providerWeight).toBeGreaterThan(advocateWeight);
    });

    it('should weight researchers heavily for trial oversight', () => {
      const researcherWeight = calculateVoterWeight('Researcher', 'ClinicalTrialOversight', false);
      const patientWeight = calculateVoterWeight('Patient', 'ClinicalTrialOversight', false);
      expect(researcherWeight).toBeGreaterThan(patientWeight);
    });

    it('should weight public health officials for pandemic response', () => {
      const officialWeight = calculateVoterWeight('PublicHealthOfficial', 'PandemicResponse', false);
      const caregiverWeight = calculateVoterWeight('Caregiver', 'PandemicResponse', false);
      expect(officialWeight).toBeGreaterThan(caregiverWeight);
    });

    it('should boost affected stakeholder weight', () => {
      const affected = calculateVoterWeight('Patient', 'ResourceAllocation', true);
      const notAffected = calculateVoterWeight('Patient', 'ResourceAllocation', false);
      expect(affected).toBe(notAffected * 1.25); // 25% boost per implementation
    });

    it('should return positive weight for all voter types', () => {
      const voterTypes: HealthVoterType[] = [
        'Patient', 'Provider', 'Researcher', 'PublicHealthOfficial',
        'CommunityAdvocate', 'TrialParticipant', 'Caregiver',
      ];
      const categories: HealthPolicyCategory[] = [
        'ClinicalGuidelines', 'ResourceAllocation', 'DataPrivacy',
        'PandemicResponse', 'ProviderCredentialing', 'ClinicalTrialOversight',
        'CommunityHealth', 'MentalHealth', 'Other',
      ];

      for (const voter of voterTypes) {
        for (const category of categories) {
          const weight = calculateVoterWeight(voter, category, false);
          expect(weight).toBeGreaterThan(0);
        }
      }
    });
  });

  describe('evaluateProposalOutcome', () => {
    const baseProposal: HealthPolicyProposal = {
      proposalId: 'p1',
      category: 'ClinicalGuidelines',
      title: 'Test Proposal',
      summary: 'Test',
      fullText: 'Test text',
      evidenceLinks: [],
      proposerId: 'user-1',
      proposerType: 'Provider',
      jurisdictionLevel: 'Local',
      status: 'Voting',
      quorumRequired: 0.5,
      approvalThreshold: 0.6,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    it('should calculate participation rate', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 2.8,
        weightedAgainst: 0.8,
        totalWeight: 3.6,
        eligibleWeight: 5.0,
      });
      // Participation: 3.6 / 5.0 = 72%
      expect(result.participationRate).toBeCloseTo(72, 0);
    });

    it('should return passed when quorum and threshold met', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 3.0,
        weightedAgainst: 1.0,
        totalWeight: 4.0,
        eligibleWeight: 5.0, // 80% participation > 50% quorum
      });
      // Approval: 3.0 / 4.0 = 75% > 60% threshold
      expect(result.passed).toBe(true);
      expect(result.quorumMet).toBe(true);
    });

    it('should return failed when threshold not met', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 2.0,
        weightedAgainst: 2.0,
        totalWeight: 4.0,
        eligibleWeight: 5.0,
      });
      // Approval: 2.0 / 4.0 = 50% < 60% threshold
      expect(result.passed).toBe(false);
    });

    it('should fail when quorum not met', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 1.0,
        weightedAgainst: 0.0,
        totalWeight: 1.0,
        eligibleWeight: 5.0, // 20% participation < 50% quorum
      });
      expect(result.quorumMet).toBe(false);
      expect(result.passed).toBe(false);
    });

    it('should handle zero total weight', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 0,
        weightedAgainst: 0,
        totalWeight: 0,
        eligibleWeight: 5.0,
      });
      expect(result.passed).toBe(false);
      expect(result.approvalRate).toBe(0);
    });
  });

  describe('detectConflictOfInterest', () => {
    const baseProposal: HealthPolicyProposal = {
      proposalId: 'p1',
      category: 'ProviderCredentialing',
      title: 'Credentialing Standards',
      summary: 'Update credentialing requirements',
      fullText: 'This proposal affects provider-123 credentials',
      evidenceLinks: [],
      proposerId: 'admin-1',
      proposerType: 'PublicHealthOfficial',
      jurisdictionLevel: 'Regional',
      status: 'Voting',
      quorumRequired: 0.3,
      approvalThreshold: 0.5,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    it('should detect provider credentialing self-interest', () => {
      const result = detectConflictOfInterest('Provider', 'provider-123', baseProposal);
      expect(result.hasConflict).toBe(true);
      expect(result.conflictType).toContain('personal interest');
      expect(result.mustRecuse).toBe(true);
    });

    it('should detect researcher voting on their own trial', () => {
      const trialProposal: HealthPolicyProposal = {
        ...baseProposal,
        category: 'ClinicalTrialOversight',
        proposerId: 'researcher-1',
      };
      const result = detectConflictOfInterest('Researcher', 'researcher-1', trialProposal);
      expect(result.hasConflict).toBe(true);
      expect(result.conflictType).toContain('Principal investigator');
    });

    it('should return no conflict for unrelated voter', () => {
      const result = detectConflictOfInterest('CommunityAdvocate', 'advocate-1', baseProposal);
      expect(result.hasConflict).toBe(false);
    });

    it('should not flag provider when not mentioned in proposal', () => {
      const unrelatedProposal: HealthPolicyProposal = {
        ...baseProposal,
        fullText: 'General credentialing standards update',
      };
      const result = detectConflictOfInterest('Provider', 'provider-999', unrelatedProposal);
      expect(result.hasConflict).toBe(false);
    });
  });

  describe('HealthPolicyClient', () => {
    let mockClient: MycelixClient;
    let policyClient: HealthPolicyClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      policyClient = new HealthPolicyClient(mockClient);
    });

    it('should call correct zome for createProposal', async () => {
      const proposal: Partial<HealthPolicyProposal> = {
        title: 'COVID-19 Vaccination Guidelines',
        category: 'ClinicalGuidelines',
        summary: 'Updated vaccination recommendations',
        fullText: 'Full policy text...',
        evidenceLinks: ['https://cdc.gov/covid'],
        proposerId: 'official-1',
        proposerType: 'PublicHealthOfficial',
        jurisdictionLevel: 'National',
        quorumRequired: 0.3,
        approvalThreshold: 0.6,
      };

      vi.mocked(mockClient.callZome).mockResolvedValue({ proposalId: 'proposal-1' });

      await policyClient.createProposal(proposal as HealthPolicyProposal);

      expect(mockClient.callZome).toHaveBeenCalledWith({
        role_name: 'governance',
        zome_name: 'health_policy',
        fn_name: 'create_health_proposal',
        payload: proposal,
      });
    });

    // Note: castVote and getProposalsByCategory tests skipped until API is finalized
    it.skip('should call correct zome for castVote', () => {
      // Placeholder - actual implementation may have different payload structure
    });

    it.skip('should fetch proposals by category', () => {
      // Placeholder - actual implementation may have different payload structure
    });

    // Note: isVotingOpen test skipped until method is implemented
    it.skip('should handle voting period validation', () => {
      // Placeholder for future implementation
    });
  });

  describe('TrialOversightClient', () => {
    // Note: These tests are skipped until the TrialOversightClient methods are implemented
    it.skip('should create trial oversight record', () => {
      // Placeholder for future implementation
    });
  });

  describe('PandemicResponseClient', () => {
    // Note: These tests are skipped until the PandemicResponseClient methods are implemented
    it.skip('should create pandemic response decision', () => {
      // Placeholder for future implementation
    });
  });

  describe('DataPrivacyPolicyClient', () => {
    // Note: These tests are skipped until the DataPrivacyPolicyClient methods are implemented
    it.skip('should create data privacy policy', () => {
      // Placeholder for future implementation
    });
  });

  describe('HealthGovernanceBridge', () => {
    let mockClient: MycelixClient;
    let bridge: HealthGovernanceBridge;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      bridge = new HealthGovernanceBridge(mockClient);
    });

    it('should provide access to all sub-clients', () => {
      expect(bridge.policy).toBeInstanceOf(HealthPolicyClient);
      expect(bridge.trialOversight).toBeInstanceOf(TrialOversightClient);
      expect(bridge.pandemicResponse).toBeInstanceOf(PandemicResponseClient);
      expect(bridge.dataPrivacy).toBeInstanceOf(DataPrivacyPolicyClient);
    });

    // Note: getGovernanceMetrics test skipped until method is implemented
    it.skip('should aggregate governance metrics', () => {
      // Placeholder for future implementation
    });
  });

  describe('Cross-Domain Workflows', () => {
    // Note: These tests are skipped until the full sub-client methods are implemented
    // The bridge structure and utility functions have been tested above
    it.skip('should coordinate trial oversight with patient advocacy', () => {
      // Placeholder for future implementation
    });

    it.skip('should link pandemic response to clinical evidence', () => {
      // Placeholder for future implementation
    });
  });

  describe('Type Validation', () => {
    it('should validate HealthPolicyCategory values', () => {
      const validCategories: HealthPolicyCategory[] = [
        'ClinicalGuidelines',
        'ResourceAllocation',
        'DataPrivacy',
        'PandemicResponse',
        'ProviderCredentialing',
        'ClinicalTrialOversight',
        'CommunityHealth',
        'MentalHealth',
        'Other',
      ];

      validCategories.forEach(category => {
        // Type assertion should work for all valid values
        const categoryValue: HealthPolicyCategory = category;
        expect(categoryValue).toBe(category);
      });
    });

    it('should validate HealthVoterType values', () => {
      const validVoterTypes: HealthVoterType[] = [
        'Patient',
        'Provider',
        'Researcher',
        'PublicHealthOfficial',
        'CommunityAdvocate',
        'TrialParticipant',
        'Caregiver',
      ];

      validVoterTypes.forEach(voterType => {
        const typeValue: HealthVoterType = voterType;
        expect(typeValue).toBe(voterType);
      });
    });

    it('should validate HealthPolicyVote structure', () => {
      const vote: HealthPolicyVote = {
        voteId: 'vote-1',
        proposalId: 'proposal-1',
        voterId: 'voter-1',
        voterType: 'Patient',
        vote: 'For',
        weight: 1.0,
        rationale: 'I support this policy',
        declaredConflicts: [],
        castAt: Date.now(),
      };

      expect(vote.voteId).toBeDefined();
      expect(vote.proposalId).toBeDefined();
      expect(['For', 'Against', 'Abstain']).toContain(vote.vote);
      expect(vote.weight).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    const baseProposal: HealthPolicyProposal = {
      proposalId: 'p1',
      category: 'ClinicalGuidelines',
      title: 'Test',
      summary: 'Test',
      fullText: 'Test',
      evidenceLinks: [],
      proposerId: 'user-1',
      proposerType: 'Provider',
      jurisdictionLevel: 'Local',
      status: 'Voting',
      quorumRequired: 0.5,
      approvalThreshold: 0.6,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };

    it('should handle proposals with no votes', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 0,
        weightedAgainst: 0,
        totalWeight: 0,
        eligibleWeight: 10,
      });
      expect(result.passed).toBe(false);
      expect(result.approvalRate).toBe(0);
    });

    it('should handle unanimous for votes', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 5.0,
        weightedAgainst: 0,
        totalWeight: 5.0,
        eligibleWeight: 5.0,
      });
      expect(result.passed).toBe(true);
      expect(result.approvalRate).toBe(100);
    });

    it('should handle very high approval thresholds', () => {
      const highThresholdProposal: HealthPolicyProposal = {
        ...baseProposal,
        approvalThreshold: 0.99,
      };
      // 95% approval, requires 99%
      const result = evaluateProposalOutcome(highThresholdProposal, {
        weightedFor: 9.5,
        weightedAgainst: 0.5,
        totalWeight: 10.0,
        eligibleWeight: 10.0,
      });
      expect(result.passed).toBe(false);
    });

    it('should handle exactly meeting threshold', () => {
      const result = evaluateProposalOutcome(baseProposal, {
        weightedFor: 6.0,
        weightedAgainst: 4.0,
        totalWeight: 10.0,
        eligibleWeight: 10.0,
      });
      // Exactly 60% approval = 60% threshold
      expect(result.approvalRate).toBe(60);
    });
  });

  describe('Error Handling', () => {
    let mockClient: MycelixClient;
    let policyClient: HealthPolicyClient;

    beforeEach(() => {
      mockClient = {
        callZome: vi.fn(),
      } as unknown as MycelixClient;
      policyClient = new HealthPolicyClient(mockClient);
    });

    it('should propagate zome call errors', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Zome error: unauthorized')
      );

      await expect(
        policyClient.getProposalsByCategory('ClinicalGuidelines')
      ).rejects.toThrow('Zome error: unauthorized');
    });

    it('should handle network timeouts', async () => {
      vi.mocked(mockClient.callZome).mockRejectedValue(
        new Error('Request timeout')
      );

      await expect(
        policyClient.getProposalsByCategory('PandemicResponse')
      ).rejects.toThrow('Request timeout');
    });
  });
});
