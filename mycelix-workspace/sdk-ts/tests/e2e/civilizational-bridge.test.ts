// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Civilizational Bridge E2E Tests
 *
 * End-to-end tests for the complete Mycelix Civilizational OS integration:
 * - Validation module with Zod schemas for all 8 domains
 * - Signals module with event handlers
 * - Cross-domain workflows with MATL and epistemic claims
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import * as matl from '../../src/matl/index.js';
import * as epistemic from '../../src/epistemic/index.js';
import * as bridge from '../../src/bridge/index.js';

// Validation
import {
  validateOrThrow,
  validateSafe,
  IdentitySchemas,
  FinanceSchemas,
  PropertySchemas,
  EnergySchemas,
  GovernanceSchemas,
  KnowledgeSchemas,
  JusticeSchemas,
  MediaSchemas,
  didSchema,
  matlScoreSchema,
  happIdSchema,
} from '../../src/validation/index.js';

// Signals
import {
  createSignalManager,
  createIdentitySignals,
  createFinanceSignals,
  createEnergySignals,
  createGovernanceSignals,
  createKnowledgeSignals,
  IdentitySignalHandler,
  FinanceSignalHandler,
  type IdentitySignalPayload,
  type FinanceSignalPayload,
} from '../../src/signals/index.js';

describe('E2E: Validation Module', () => {
  describe('Common Schemas', () => {
    it('should validate DIDs correctly', () => {
      // Valid DIDs
      expect(() => didSchema.parse('did:mycelix:abc123def456')).not.toThrow();
      expect(() => didSchema.parse('did:mycelix:user_001_verified')).not.toThrow();

      // Invalid DIDs
      expect(() => didSchema.parse('abc123')).toThrow();
      expect(() => didSchema.parse('did:other:abc')).toThrow();
      expect(() => didSchema.parse('')).toThrow();
    });

    it('should validate MATL scores correctly', () => {
      // Valid scores
      expect(() => matlScoreSchema.parse(0)).not.toThrow();
      expect(() => matlScoreSchema.parse(0.5)).not.toThrow();
      expect(() => matlScoreSchema.parse(1)).not.toThrow();

      // Invalid scores
      expect(() => matlScoreSchema.parse(-0.1)).toThrow();
      expect(() => matlScoreSchema.parse(1.1)).toThrow();
      expect(() => matlScoreSchema.parse(NaN)).toThrow();
    });

    it('should validate hApp IDs correctly', () => {
      expect(() => happIdSchema.parse('my-happ')).not.toThrow();
      expect(() => happIdSchema.parse('marketplace-v2')).not.toThrow();

      expect(() => happIdSchema.parse('My-Happ')).toThrow(); // uppercase
      expect(() => happIdSchema.parse('my happ')).toThrow(); // spaces
    });
  });

  describe('Identity Schemas', () => {
    it('should validate RegisterHappInput', () => {
      const valid = {
        happ_name: 'my-happ',
        capabilities: ['trust_updated', 'credential_issued'],
      };
      expect(() => validateOrThrow(IdentitySchemas.RegisterHappInput, valid)).not.toThrow();
    });

    it('should validate QueryIdentityInput', () => {
      const valid = {
        did: 'did:mycelix:user12345678',  // min 20 chars
        source_happ: 'marketplace',
      };
      expect(() => validateOrThrow(IdentitySchemas.QueryIdentityInput, valid)).not.toThrow();
    });

    it('should reject invalid identity inputs', () => {
      const invalid = {
        happ_name: '', // empty string should fail min(1)
        capabilities: 'not-an-array',
      };
      expect(() => validateOrThrow(IdentitySchemas.RegisterHappInput, invalid)).toThrow();
    });
  });

  describe('Finance Schemas', () => {
    it('should validate ProcessPaymentInput', () => {
      const valid = {
        payee_did: 'did:mycelix:payee456',
        amount: 100.50,
        currency: 'MCX',
        target_happ: 'energy-marketplace',
        reference_id: 'order-001',
      };
      expect(() => validateOrThrow(FinanceSchemas.ProcessPaymentInput, valid)).not.toThrow();
    });

    it('should reject negative payment amounts', () => {
      const invalid = {
        payee_did: 'did:mycelix:payee456',
        amount: -50,
        currency: 'MCX',
        target_happ: 'marketplace',
        reference_id: 'order-001',
      };
      expect(() => validateOrThrow(FinanceSchemas.ProcessPaymentInput, invalid)).toThrow();
    });
  });

  describe('Energy Schemas', () => {
    it('should validate QueryAvailableEnergyInput', () => {
      const valid = {
        source_happ: 'energy-marketplace',
        location: { lat: 37.7749, lng: -122.4194 },
        energy_sources: ['solar', 'wind'],
        min_amount: 100,
      };
      expect(() => validateOrThrow(EnergySchemas.QueryAvailableEnergyInput, valid)).not.toThrow();
    });

    it('should validate location coordinates', () => {
      const invalidLat = {
        source_happ: 'energy',
        location: { lat: 200, lng: 0 }, // Invalid latitude
      };
      expect(() => validateOrThrow(EnergySchemas.QueryAvailableEnergyInput, invalidLat)).toThrow();
    });
  });

  describe('All Domain Schemas', () => {
    it('should have Property schemas', () => {
      expect(PropertySchemas.VerifyOwnershipInput).toBeDefined();
      expect(PropertySchemas.PledgeCollateralInput).toBeDefined();
    });

    it('should have Governance schemas', () => {
      expect(GovernanceSchemas.QueryGovernanceInput).toBeDefined();
      expect(GovernanceSchemas.RequestExecutionInput).toBeDefined();
    });

    it('should have Justice schemas', () => {
      expect(JusticeSchemas.FileCrossHappDisputeInput).toBeDefined();
      expect(JusticeSchemas.RequestEnforcementInput).toBeDefined();
    });

    it('should have Knowledge schemas', () => {
      expect(KnowledgeSchemas.QueryKnowledgeInput).toBeDefined();
      expect(KnowledgeSchemas.FactCheckInput).toBeDefined();
    });

    it('should have Media schemas', () => {
      expect(MediaSchemas.QueryContentInput).toBeDefined();
      expect(MediaSchemas.RequestLicenseInput).toBeDefined();
    });
  });

  describe('validateSafe helper', () => {
    it('should return success with valid data', () => {
      const result = validateSafe(IdentitySchemas.RegisterHappInput, {
        happ_name: 'my-happ',
        capabilities: ['trust', 'credentials'],
      });
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.happ_name).toBe('my-happ');
      }
    });

    it('should return errors with invalid data', () => {
      const result = validateSafe(IdentitySchemas.RegisterHappInput, {
        happ_name: 123, // should be string
      });
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors.length).toBeGreaterThan(0);
      }
    });
  });
});

describe('E2E: Signals Module', () => {
  describe('IdentitySignalHandler', () => {
    it('should handle IdentityCreated events', () => {
      const handler = createIdentitySignals();
      const callback = vi.fn();
      handler.onIdentityCreated(callback);

      const payload: IdentitySignalPayload = {
        type: 'IdentityCreated',
        data: {
          did: 'did:mycelix:new_user',
          initial_matl_score: 0.5,
          timestamp: Date.now(),
        },
      };

      // Simulate signal from conductor
      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(32), new Uint8Array(32)],
        zomeName: 'identity_bridge',
        signalName: 'IdentityCreated',
        payload,
      });

      expect(callback).toHaveBeenCalledWith(payload.data);
    });

    it('should support unsubscribe', () => {
      const handler = createIdentitySignals();
      const callback = vi.fn();
      const unsubscribe = handler.onIdentityCreated(callback);

      unsubscribe();

      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(32), new Uint8Array(32)],
        zomeName: 'identity_bridge',
        signalName: 'IdentityCreated',
        payload: { type: 'IdentityCreated', data: {} },
      });

      expect(callback).not.toHaveBeenCalled();
    });

    it('should buffer recent signals', () => {
      const handler = createIdentitySignals({ bufferSize: 10 });

      // Add some signals
      for (let i = 0; i < 5; i++) {
        handler.handle({
          type: 'app',
          cellId: [new Uint8Array(32), new Uint8Array(32)],
          zomeName: 'identity_bridge',
          signalName: 'IdentityCreated',
          payload: { type: 'IdentityCreated', data: { index: i } },
        });
      }

      const buffer = handler.getBuffer();
      expect(buffer.length).toBe(5);
    });
  });

  describe('FinanceSignalHandler', () => {
    it('should handle PaymentProcessed events', () => {
      const handler = createFinanceSignals();
      const callback = vi.fn();
      handler.onPaymentProcessed(callback);

      const payload: FinanceSignalPayload = {
        type: 'PaymentProcessed',
        data: {
          payment_id: 'pay-001',
          payer_did: 'did:mycelix:payer',
          payee_did: 'did:mycelix:payee',
          amount: 100,
          currency: 'MCX',
          reference_id: 'order-001',
          timestamp: Date.now(),
        },
      };

      handler.handle({
        type: 'app',
        cellId: [new Uint8Array(32), new Uint8Array(32)],
        zomeName: 'finance_bridge',
        signalName: 'PaymentProcessed',
        payload,
      });

      expect(callback).toHaveBeenCalledWith(payload.data);
    });
  });

  describe('All Domain Signal Handlers', () => {
    it('should create Energy signal handler', () => {
      const handler = createEnergySignals();
      expect(handler).toBeDefined();
      expect(typeof handler.onEnergyListed).toBe('function');
    });

    it('should create Governance signal handler', () => {
      const handler = createGovernanceSignals();
      expect(handler).toBeDefined();
      expect(typeof handler.onProposalCreated).toBe('function');
      expect(typeof handler.onVoteCast).toBe('function');
    });

    it('should create Knowledge signal handler', () => {
      const handler = createKnowledgeSignals();
      expect(handler).toBeDefined();
      expect(typeof handler.onClaimCreated).toBe('function');
      expect(typeof handler.onFactCheckResult).toBe('function');
    });
  });

  describe('BridgeSignalManager', () => {
    it('should create all domain handlers', () => {
      const manager = createSignalManager();

      expect(manager.identity).toBeInstanceOf(IdentitySignalHandler);
      expect(manager.finance).toBeInstanceOf(FinanceSignalHandler);
      expect(manager.energy).toBeDefined();
      expect(manager.governance).toBeDefined();
      expect(manager.knowledge).toBeDefined();
    });

    it('should not be connected without client', () => {
      const manager = createSignalManager();
      expect(manager.isConnected()).toBe(false);
    });
  });
});

describe('E2E: Cross-Domain Workflows', () => {
  let localBridge: bridge.LocalBridge;

  beforeEach(() => {
    localBridge = new bridge.LocalBridge();
    localBridge.registerHapp('identity');
    localBridge.registerHapp('finance');
    localBridge.registerHapp('property');
    localBridge.registerHapp('energy');
    localBridge.registerHapp('governance');
    localBridge.registerHapp('knowledge');
  });

  it('should complete energy marketplace transaction with validation', () => {
    const sellerDid = 'did:mycelix:seller_001';
    const buyerDid = 'did:mycelix:buyer_001';

    // Step 1: Validate identities
    didSchema.parse(sellerDid);
    didSchema.parse(buyerDid);

    // Step 2: Validate registration input
    const registerInput = {
      happ_name: 'energy-marketplace',
      capabilities: ['trade_completed', 'listing_created'],
    };
    validateOrThrow(IdentitySchemas.RegisterHappInput, registerInput);

    // Step 3: Build reputation
    let sellerRep = matl.createReputation(sellerDid);
    for (let i = 0; i < 5; i++) {
      sellerRep = matl.recordPositive(sellerRep);
    }
    matlScoreSchema.parse(matl.reputationValue(sellerRep));

    // Step 4: Validate energy query
    const energyQuery = {
      source_happ: 'energy-marketplace',
      energy_sources: ['solar'],
      min_amount: 100,
    };
    validateOrThrow(EnergySchemas.QueryAvailableEnergyInput, energyQuery);

    // Step 5: Validate and process payment
    const paymentInput = {
      payee_did: sellerDid,
      amount: 50.0,
      currency: 'MCX',
      target_happ: 'energy-marketplace',
      reference_id: 'energy-purchase-001',
    };
    validateOrThrow(FinanceSchemas.ProcessPaymentInput, paymentInput);

    // Step 6: Store reputation in bridge
    localBridge.setReputation('energy', sellerDid, sellerRep);
    const aggregate = localBridge.getAggregateReputation(sellerDid);
    expect(aggregate).toBeGreaterThan(0);
  });

  it('should handle governance workflow with fact-checking', () => {
    const proposerDid = 'did:mycelix:proposer_001';
    didSchema.parse(proposerDid);

    // Step 1: Validate governance query
    const govQuery = {
      query_type: 'ProposalStatus' as const,
      query_params: JSON.stringify({ proposal_id: 'prop-001' }),
      source_happ: 'community-dao',
    };
    validateOrThrow(GovernanceSchemas.QueryGovernanceInput, govQuery);

    // Step 2: Validate fact-check input
    const factCheckInput = {
      claim_text: 'Proposal benefits 90% of community members',
      context: { proposal_id: 'prop-001' },
      source_happ: 'knowledge-base',
    };
    validateOrThrow(KnowledgeSchemas.FactCheckInput, factCheckInput);

    // Step 3: Create epistemic claim for verification
    const claim = epistemic.claim('Proposal verified by community analysis')
      .withEmpirical(epistemic.EmpiricalLevel.E2_PrivateVerify)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M2_Persistent)
      .withIssuer(proposerDid)
      .build();

    expect(epistemic.classificationCode(claim.classification)).toBe('E2-N2-M2');
  });

  it('should aggregate reputation across all domains', () => {
    const agentDid = 'did:mycelix:multi_domain_agent';
    didSchema.parse(agentDid);

    // Create reputation in multiple domains
    let identityRep = matl.createReputation(agentDid);
    let financeRep = matl.createReputation(agentDid);
    let energyRep = matl.createReputation(agentDid);
    let governanceRep = matl.createReputation(agentDid);

    // Simulate different activity levels
    for (let i = 0; i < 10; i++) identityRep = matl.recordPositive(identityRep);
    for (let i = 0; i < 5; i++) financeRep = matl.recordPositive(financeRep);
    for (let i = 0; i < 8; i++) energyRep = matl.recordPositive(energyRep);
    for (let i = 0; i < 3; i++) governanceRep = matl.recordPositive(governanceRep);

    // Store in bridge
    localBridge.setReputation('identity', agentDid, identityRep);
    localBridge.setReputation('finance', agentDid, financeRep);
    localBridge.setReputation('energy', agentDid, energyRep);
    localBridge.setReputation('governance', agentDid, governanceRep);

    // Get cross-domain reputation
    const scores = localBridge.getCrossHappReputation(agentDid);
    expect(scores.length).toBe(4);

    const aggregate = localBridge.getAggregateReputation(agentDid);
    expect(aggregate).toBeGreaterThan(0.5);
    expect(aggregate).toBeLessThanOrEqual(1);
  });

  it('should handle property collateral with finance integration', () => {
    const ownerDid = 'did:mycelix:property_owner';
    didSchema.parse(ownerDid);

    // Step 1: Validate property ownership query
    const ownershipInput = {
      asset_id: 'QmProperty123',
      verification_type: 'Current' as const,
    };
    validateOrThrow(PropertySchemas.VerifyOwnershipInput, ownershipInput);

    // Step 2: Validate collateral pledge
    const pledgeInput = {
      asset_id: 'QmProperty123',
      pledge_to_happ: 'finance-loans',
      pledge_amount: 10000,
      loan_reference: 'loan-001',
    };
    validateOrThrow(PropertySchemas.PledgeCollateralInput, pledgeInput);

    // Step 3: Create epistemic claim about collateral
    const claim = epistemic.claim('Property QmProperty123 pledged as collateral')
      .withEmpirical(epistemic.EmpiricalLevel.E3_Cryptographic)
      .withNormative(epistemic.NormativeLevel.N2_Network)
      .withMateriality(epistemic.MaterialityLevel.M3_Immutable)
      .withIssuer('property-registry')
      .build();

    expect(epistemic.meetsStandard(
      claim,
      epistemic.EmpiricalLevel.E2_PrivateVerify,
      epistemic.NormativeLevel.N2_Network
    )).toBe(true);
  });
});

describe('E2E: Validation + Signals Integration', () => {
  it('should validate and handle identity creation flow', () => {
    const handler = createIdentitySignals();
    const createdIdentities: unknown[] = [];

    handler.onIdentityCreated((data) => {
      // Validate incoming signal data
      didSchema.parse(data.did);
      matlScoreSchema.parse(data.initial_matl_score);
      createdIdentities.push(data);
    });

    // Simulate valid signal
    const validPayload: IdentitySignalPayload = {
      type: 'IdentityCreated',
      data: {
        did: 'did:mycelix:new_user_001',
        initial_matl_score: 0.5,
        timestamp: Date.now(),
      },
    };

    handler.handle({
      type: 'app',
      cellId: [new Uint8Array(32), new Uint8Array(32)],
      zomeName: 'identity_bridge',
      signalName: 'IdentityCreated',
      payload: validPayload,
    });

    expect(createdIdentities.length).toBe(1);
  });

  it('should handle payment signals with amount validation', () => {
    const handler = createFinanceSignals();
    const processedPayments: unknown[] = [];

    handler.onPaymentProcessed((data) => {
      // Validate payment amount is positive
      if (data.amount <= 0) {
        throw new Error('Invalid payment amount');
      }
      processedPayments.push(data);
    });

    // Valid payment
    handler.handle({
      type: 'app',
      cellId: [new Uint8Array(32), new Uint8Array(32)],
      zomeName: 'finance_bridge',
      signalName: 'PaymentProcessed',
      payload: {
        type: 'PaymentProcessed',
        data: {
          payment_id: 'pay-001',
          payer_did: 'did:mycelix:payer',
          payee_did: 'did:mycelix:payee',
          amount: 100,
          currency: 'MCX',
          reference_id: 'order-001',
          timestamp: Date.now(),
        },
      },
    });

    expect(processedPayments.length).toBe(1);
  });
});
