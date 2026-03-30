// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Validation Module Tests
 *
 * Comprehensive tests for Zod validation schemas.
 */

import { describe, it, expect } from 'vitest';
import {
  // Common schemas
  didSchema,
  happIdSchema,
  matlScoreSchema,
  amountSchema,
  timestampSchema,
  locationSchema,
  // Identity schemas
  registerHappInputSchema,
  queryIdentityInputSchema,
  reportReputationInputSchema,
  bridgeEventTypeSchema,
  broadcastIdentityEventInputSchema,
  // Finance schemas
  creditPurposeSchema,
  queryCreditInputSchema,
  processPaymentInputSchema,
  registerCollateralInputSchema,
  // Property schemas
  verifyOwnershipInputSchema,
  pledgeCollateralInputSchema,
  // Energy schemas
  energySourceSchema,
  queryAvailableEnergyInputSchema,
  requestEnergyPurchaseInputSchema,
  reportProductionInputSchema,
  // Media schemas
  contentTypeSchema,
  licenseTypeSchema,
  queryContentInputSchema,
  requestLicenseInputSchema,
  distributeRoyaltiesInputSchema,
  // Governance schemas
  queryGovernanceInputSchema,
  requestExecutionInputSchema,
  // Justice schemas
  fileCrossHappDisputeInputSchema,
  requestEnforcementInputSchema,
  disputeHistoryQuerySchema,
  // Knowledge schemas
  queryKnowledgeInputSchema,
  factCheckInputSchema,
  registerExternalClaimInputSchema,
  // Utilities
  validateOrThrow,
  validateSafe,
  withValidation,
  // Collections
  IdentitySchemas,
  FinanceSchemas,
  PropertySchemas,
  EnergySchemas,
  MediaSchemas,
  GovernanceSchemas,
  JusticeSchemas,
  KnowledgeSchemas,
} from '../src/validation/index.js';

// ============================================================================
// Common Schema Tests
// ============================================================================

describe('Common Schemas', () => {
  describe('didSchema', () => {
    it('accepts valid Mycelix DIDs', () => {
      const validDid = 'did:mycelix:abc123def456xyz789';
      expect(didSchema.parse(validDid)).toBe(validDid);
    });

    it('rejects DIDs without mycelix prefix', () => {
      expect(() => didSchema.parse('did:key:abc123')).toThrow();
    });

    it('rejects DIDs that are too short', () => {
      expect(() => didSchema.parse('did:mycelix:abc')).toThrow();
    });
  });

  describe('happIdSchema', () => {
    it('accepts valid hApp IDs', () => {
      expect(happIdSchema.parse('my-marketplace')).toBe('my-marketplace');
      expect(happIdSchema.parse('mycelix-identity')).toBe('mycelix-identity');
    });

    it('rejects IDs with invalid characters', () => {
      expect(() => happIdSchema.parse('My_App')).toThrow();
      expect(() => happIdSchema.parse('my app')).toThrow();
    });

    it('rejects empty IDs', () => {
      expect(() => happIdSchema.parse('')).toThrow();
    });
  });

  describe('matlScoreSchema', () => {
    it('accepts scores between 0 and 1', () => {
      expect(matlScoreSchema.parse(0)).toBe(0);
      expect(matlScoreSchema.parse(0.5)).toBe(0.5);
      expect(matlScoreSchema.parse(1)).toBe(1);
    });

    it('rejects scores outside valid range', () => {
      expect(() => matlScoreSchema.parse(-0.1)).toThrow();
      expect(() => matlScoreSchema.parse(1.1)).toThrow();
    });
  });

  describe('amountSchema', () => {
    it('accepts positive numbers', () => {
      expect(amountSchema.parse(100)).toBe(100);
      expect(amountSchema.parse(0.01)).toBe(0.01);
    });

    it('rejects zero and negative numbers', () => {
      expect(() => amountSchema.parse(0)).toThrow();
      expect(() => amountSchema.parse(-10)).toThrow();
    });
  });

  describe('timestampSchema', () => {
    it('accepts valid timestamps', () => {
      const ts = Date.now() * 1000; // microseconds
      expect(timestampSchema.parse(ts)).toBe(ts);
    });

    it('rejects negative timestamps', () => {
      expect(() => timestampSchema.parse(-1)).toThrow();
    });

    it('rejects non-integer timestamps', () => {
      expect(() => timestampSchema.parse(123.456)).toThrow();
    });
  });

  describe('locationSchema', () => {
    it('accepts valid coordinates', () => {
      const loc = { lat: 37.7749, lng: -122.4194 };
      expect(locationSchema.parse(loc)).toEqual(loc);
    });

    it('accepts optional radius', () => {
      const loc = { lat: 37.7749, lng: -122.4194, radius_km: 10 };
      expect(locationSchema.parse(loc)).toEqual(loc);
    });

    it('rejects invalid latitudes', () => {
      expect(() => locationSchema.parse({ lat: 91, lng: 0 })).toThrow();
      expect(() => locationSchema.parse({ lat: -91, lng: 0 })).toThrow();
    });

    it('rejects invalid longitudes', () => {
      expect(() => locationSchema.parse({ lat: 0, lng: 181 })).toThrow();
      expect(() => locationSchema.parse({ lat: 0, lng: -181 })).toThrow();
    });
  });
});

// ============================================================================
// Identity Bridge Schema Tests
// ============================================================================

describe('Identity Bridge Schemas', () => {
  describe('registerHappInputSchema', () => {
    it('accepts valid registration', () => {
      const input = {
        happ_name: 'my-marketplace',
        capabilities: ['identity_query', 'reputation_report'],
      };
      expect(registerHappInputSchema.parse(input)).toEqual(input);
    });

    it('rejects empty capabilities', () => {
      expect(() =>
        registerHappInputSchema.parse({
          happ_name: 'my-app',
          capabilities: [],
        })
      ).toThrow();
    });
  });

  describe('queryIdentityInputSchema', () => {
    it('accepts valid query with default fields', () => {
      const input = {
        did: 'did:mycelix:abc123def456xyz789',
        source_happ: 'my-marketplace',
      };
      const result = queryIdentityInputSchema.parse(input);
      expect(result.requested_fields).toEqual([]);
    });

    it('accepts query with specific fields', () => {
      const input = {
        did: 'did:mycelix:abc123def456xyz789',
        source_happ: 'my-marketplace',
        requested_fields: ['matl_score', 'credential_count'],
      };
      expect(queryIdentityInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('bridgeEventTypeSchema', () => {
    it('accepts all valid event types', () => {
      const validTypes = [
        'IdentityCreated',
        'IdentityUpdated',
        'CredentialIssued',
        'CredentialRevoked',
        'TrustAttested',
        'RecoveryInitiated',
      ];
      validTypes.forEach((type) => {
        expect(bridgeEventTypeSchema.parse(type)).toBe(type);
      });
    });

    it('rejects invalid event types', () => {
      expect(() => bridgeEventTypeSchema.parse('InvalidEvent')).toThrow();
    });
  });
});

// ============================================================================
// Finance Bridge Schema Tests
// ============================================================================

describe('Finance Bridge Schemas', () => {
  describe('creditPurposeSchema', () => {
    it('accepts all valid purposes', () => {
      const validPurposes = [
        'LoanApplication',
        'TrustVerification',
        'MarketplaceTransaction',
        'PropertyPurchase',
        'EnergyInvestment',
      ];
      validPurposes.forEach((purpose) => {
        expect(creditPurposeSchema.parse(purpose)).toBe(purpose);
      });
    });
  });

  describe('queryCreditInputSchema', () => {
    it('accepts valid credit query', () => {
      const input = {
        did: 'did:mycelix:abc123def456xyz789',
        purpose: 'LoanApplication',
        amount_requested: 10000,
      };
      expect(queryCreditInputSchema.parse(input)).toEqual(input);
    });

    it('accepts query without amount', () => {
      const input = {
        did: 'did:mycelix:abc123def456xyz789',
        purpose: 'TrustVerification',
      };
      expect(queryCreditInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('processPaymentInputSchema', () => {
    it('accepts valid payment with default currency', () => {
      const input = {
        payee_did: 'did:mycelix:abc123def456xyz789',
        amount: 100,
        target_happ: 'my-marketplace',
        reference_id: 'order-123',
      };
      const result = processPaymentInputSchema.parse(input);
      expect(result.currency).toBe('MCX');
    });

    it('accepts payment with custom currency', () => {
      const input = {
        payee_did: 'did:mycelix:abc123def456xyz789',
        amount: 100,
        currency: 'USD',
        target_happ: 'my-marketplace',
        reference_id: 'order-123',
      };
      expect(processPaymentInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('registerCollateralInputSchema', () => {
    it('accepts valid collateral registration', () => {
      const input = {
        asset_type: 'RealEstate',
        asset_id: 'property-123',
        valuation: 500000,
      };
      expect(registerCollateralInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Property Bridge Schema Tests
// ============================================================================

describe('Property Bridge Schemas', () => {
  describe('verifyOwnershipInputSchema', () => {
    it('accepts valid ownership verification', () => {
      const input = {
        asset_id: 'property-123',
        verification_type: 'Current',
      };
      expect(verifyOwnershipInputSchema.parse(input)).toEqual(input);
    });

    it('accepts all verification types', () => {
      ['Current', 'Historical', 'Encumbrances'].forEach((type) => {
        const input = { asset_id: 'prop-1', verification_type: type };
        expect(verifyOwnershipInputSchema.parse(input).verification_type).toBe(type);
      });
    });
  });

  describe('pledgeCollateralInputSchema', () => {
    it('accepts valid collateral pledge', () => {
      const input = {
        asset_id: 'property-123',
        pledge_to_happ: 'mycelix-finance',
        pledge_amount: 100000,
        loan_reference: 'loan-456',
      };
      expect(pledgeCollateralInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Energy Bridge Schema Tests
// ============================================================================

describe('Energy Bridge Schemas', () => {
  describe('energySourceSchema', () => {
    it('accepts all valid energy sources', () => {
      const sources = ['solar', 'wind', 'hydro', 'battery', 'grid', 'other_renewable'];
      sources.forEach((source) => {
        expect(energySourceSchema.parse(source)).toBe(source);
      });
    });
  });

  describe('queryAvailableEnergyInputSchema', () => {
    it('accepts query with all options', () => {
      const input = {
        source_happ: 'energy-grid',
        location: { lat: 37.7749, lng: -122.4194, radius_km: 50 },
        energy_sources: ['solar', 'wind'],
        min_amount: 100,
      };
      expect(queryAvailableEnergyInputSchema.parse(input)).toEqual(input);
    });

    it('accepts minimal query', () => {
      const input = { source_happ: 'energy-grid' };
      expect(queryAvailableEnergyInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('reportProductionInputSchema', () => {
    it('accepts valid production report', () => {
      const input = {
        participant_id: 'solar-farm-1',
        production_kwh: 5000,
        consumption_kwh: 1000,
        energy_source: 'solar',
        timestamp: Date.now() * 1000,
      };
      expect(reportProductionInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Media Bridge Schema Tests
// ============================================================================

describe('Media Bridge Schemas', () => {
  describe('contentTypeSchema', () => {
    it('accepts all valid content types', () => {
      const types = ['Article', 'Opinion', 'Investigation', 'Review', 'Analysis', 'Interview', 'Report', 'Editorial', 'Other'];
      types.forEach((type) => {
        expect(contentTypeSchema.parse(type)).toBe(type);
      });
    });
  });

  describe('licenseTypeSchema', () => {
    it('accepts all valid license types', () => {
      const types = ['CC0', 'CCBY', 'CCBYSA', 'CCBYNC', 'CCBYNCSA', 'AllRightsReserved', 'Custom'];
      types.forEach((type) => {
        expect(licenseTypeSchema.parse(type)).toBe(type);
      });
    });
  });

  describe('requestLicenseInputSchema', () => {
    it('accepts valid license request', () => {
      const input = {
        content_id: 'content-123',
        licensee_did: 'did:mycelix:abc123def456xyz789',
        license_type: 'CCBY',
        purpose: 'Educational use in online course',
        duration_days: 365,
      };
      expect(requestLicenseInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('distributeRoyaltiesInputSchema', () => {
    it('accepts valid royalty distribution', () => {
      const input = {
        content_id: 'content-123',
        total_amount: 1000,
        currency: 'MCX',
        reason: 'view',
      };
      expect(distributeRoyaltiesInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Governance Bridge Schema Tests
// ============================================================================

describe('Governance Bridge Schemas', () => {
  describe('queryGovernanceInputSchema', () => {
    it('accepts valid governance query', () => {
      const input = {
        query_type: 'ProposalStatus',
        query_params: JSON.stringify({ proposal_id: 'prop-123' }),
        source_happ: 'dao-app',
      };
      expect(queryGovernanceInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('requestExecutionInputSchema', () => {
    it('accepts valid execution request', () => {
      const input = {
        proposal_hash: 'QmXyz123',
        target_happ: 'mycelix-finance',
        action: 'TransferFunds',
        payload: JSON.stringify({ amount: 1000, recipient: 'did:mycelix:abc' }),
      };
      expect(requestExecutionInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Justice Bridge Schema Tests
// ============================================================================

describe('Justice Bridge Schemas', () => {
  describe('fileCrossHappDisputeInputSchema', () => {
    it('accepts valid dispute filing', () => {
      const input = {
        dispute_type: 'ContractBreach',
        respondent_did: 'did:mycelix:abc123def456xyz789',
        related_happs: ['my-marketplace', 'mycelix-finance'],
        title: 'Failure to deliver goods',
        description: 'Seller failed to deliver goods after payment was made',
        evidence_hashes: ['QmHash1', 'QmHash2'],
      };
      expect(fileCrossHappDisputeInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('requestEnforcementInputSchema', () => {
    it('accepts valid enforcement request', () => {
      const input = {
        dispute_id: 'dispute-123',
        target_happ: 'my-marketplace',
        target_did: 'did:mycelix:abc123def456xyz789',
        action_type: 'ReputationPenalty',
        details: 'Reduce reputation by 20% for breach',
      };
      expect(requestEnforcementInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('disputeHistoryQuerySchema', () => {
    it('accepts valid history query', () => {
      const input = {
        did: 'did:mycelix:abc123def456xyz789',
        role: 'both',
        limit: 50,
      };
      expect(disputeHistoryQuerySchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Knowledge Bridge Schema Tests
// ============================================================================

describe('Knowledge Bridge Schemas', () => {
  describe('queryKnowledgeInputSchema', () => {
    it('accepts valid knowledge query', () => {
      const input = {
        query_type: 'VerifyClaim',
        query_params: JSON.stringify({ claim_hash: 'QmClaim123' }),
        source_happ: 'fact-checker',
      };
      expect(queryKnowledgeInputSchema.parse(input)).toEqual(input);
    });
  });

  describe('factCheckInputSchema', () => {
    it('accepts valid fact check request', () => {
      const input = {
        claim_text: 'The Earth orbits the Sun.',
        source_happ: 'fact-checker',
      };
      expect(factCheckInputSchema.parse(input)).toEqual(input);
    });

    it('rejects overly long claims', () => {
      const longClaim = 'x'.repeat(5000);
      expect(() =>
        factCheckInputSchema.parse({
          claim_text: longClaim,
          source_happ: 'app',
        })
      ).toThrow();
    });
  });

  describe('registerExternalClaimInputSchema', () => {
    it('accepts valid external claim registration', () => {
      const input = {
        claim_hash: 'QmClaim123',
        source_happ: 'research-hub',
        title: 'Climate Impact Study 2024',
        empirical: 0.9,
        normative: 0.7,
        mythic: 0.3,
        author_did: 'did:mycelix:researcher123abc',
      };
      expect(registerExternalClaimInputSchema.parse(input)).toEqual(input);
    });
  });
});

// ============================================================================
// Validation Utility Tests
// ============================================================================

describe('Validation Utilities', () => {
  describe('validateOrThrow', () => {
    it('returns data when valid', () => {
      const data = { lat: 37.7749, lng: -122.4194 };
      expect(validateOrThrow(locationSchema, data)).toEqual(data);
    });

    it('throws with detailed message when invalid', () => {
      expect(() => validateOrThrow(locationSchema, { lat: 100, lng: 0 })).toThrow(
        /Validation failed/
      );
    });
  });

  describe('validateSafe', () => {
    it('returns success result when valid', () => {
      const result = validateSafe(matlScoreSchema, 0.5);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toBe(0.5);
      }
    });

    it('returns failure result with errors when invalid', () => {
      const result = validateSafe(matlScoreSchema, 2);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors.length).toBeGreaterThan(0);
      }
    });
  });

  describe('withValidation', () => {
    it('validates input before calling function', async () => {
      const mockFn = async (input: { lat: number; lng: number }) => ({
        ...input,
        processed: true,
      });

      const wrapped = withValidation(locationSchema, mockFn);

      const result = await wrapped({ lat: 37.7749, lng: -122.4194 });
      expect(result).toEqual({ lat: 37.7749, lng: -122.4194, processed: true });
    });

    it('throws on invalid input before calling function', async () => {
      const mockFn = async (input: { lat: number; lng: number }) => input;
      const wrapped = withValidation(locationSchema, mockFn);

      await expect(wrapped({ lat: 200, lng: 0 })).rejects.toThrow(/Validation failed/);
    });
  });
});

// ============================================================================
// Schema Collection Tests
// ============================================================================

describe('Schema Collections', () => {
  it('IdentitySchemas contains all identity schemas', () => {
    expect(IdentitySchemas.RegisterHappInput).toBeDefined();
    expect(IdentitySchemas.QueryIdentityInput).toBeDefined();
    expect(IdentitySchemas.ReportReputationInput).toBeDefined();
    expect(IdentitySchemas.BroadcastEventInput).toBeDefined();
  });

  it('FinanceSchemas contains all finance schemas', () => {
    expect(FinanceSchemas.QueryCreditInput).toBeDefined();
    expect(FinanceSchemas.ProcessPaymentInput).toBeDefined();
    expect(FinanceSchemas.RegisterCollateralInput).toBeDefined();
  });

  it('PropertySchemas contains all property schemas', () => {
    expect(PropertySchemas.VerifyOwnershipInput).toBeDefined();
    expect(PropertySchemas.PledgeCollateralInput).toBeDefined();
  });

  it('EnergySchemas contains all energy schemas', () => {
    expect(EnergySchemas.QueryAvailableEnergyInput).toBeDefined();
    expect(EnergySchemas.RequestEnergyPurchaseInput).toBeDefined();
    expect(EnergySchemas.ReportProductionInput).toBeDefined();
  });

  it('MediaSchemas contains all media schemas', () => {
    expect(MediaSchemas.QueryContentInput).toBeDefined();
    expect(MediaSchemas.RequestLicenseInput).toBeDefined();
    expect(MediaSchemas.DistributeRoyaltiesInput).toBeDefined();
    expect(MediaSchemas.VerifyLicenseInput).toBeDefined();
  });

  it('GovernanceSchemas contains all governance schemas', () => {
    expect(GovernanceSchemas.QueryGovernanceInput).toBeDefined();
    expect(GovernanceSchemas.RequestExecutionInput).toBeDefined();
    expect(GovernanceSchemas.BroadcastEventInput).toBeDefined();
  });

  it('JusticeSchemas contains all justice schemas', () => {
    expect(JusticeSchemas.FileCrossHappDisputeInput).toBeDefined();
    expect(JusticeSchemas.RequestEnforcementInput).toBeDefined();
    expect(JusticeSchemas.DisputeHistoryQuery).toBeDefined();
  });

  it('KnowledgeSchemas contains all knowledge schemas', () => {
    expect(KnowledgeSchemas.QueryKnowledgeInput).toBeDefined();
    expect(KnowledgeSchemas.FactCheckInput).toBeDefined();
    expect(KnowledgeSchemas.RegisterExternalClaimInput).toBeDefined();
    expect(KnowledgeSchemas.BroadcastEventInput).toBeDefined();
  });
});
