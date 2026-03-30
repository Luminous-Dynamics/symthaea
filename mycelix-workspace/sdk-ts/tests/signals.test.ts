// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Signals Module Tests
 *
 * Tests for typed signal handlers for bridge events.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  IdentitySignalHandler,
  FinanceSignalHandler,
  EnergySignalHandler,
  GovernanceSignalHandler,
  KnowledgeSignalHandler,
  BridgeSignalManager,
  createSignalManager,
  createIdentitySignals,
  createFinanceSignals,
  createEnergySignals,
  createGovernanceSignals,
  createKnowledgeSignals,
  type IdentitySignalPayload,
  type FinanceSignalPayload,
  type EnergySignalPayload,
  type GovernanceSignalPayload,
  type KnowledgeSignalPayload,
} from '../src/signals/index.js';
import type { HolochainSignal } from '../src/utils/index.js';

// Helper to create a mock Holochain signal
function createMockSignal<T>(
  zomeName: string,
  signalName: string,
  payload: T
): HolochainSignal<T> {
  return {
    type: 'app',
    cellId: [new Uint8Array(32), new Uint8Array(32)],
    zomeName,
    signalName,
    payload,
  };
}

describe('IdentitySignalHandler', () => {
  let handler: IdentitySignalHandler;

  beforeEach(() => {
    handler = new IdentitySignalHandler();
  });

  it('should handle IdentityCreated events', () => {
    const callback = vi.fn();
    handler.onIdentityCreated(callback);

    const payload: IdentitySignalPayload = {
      type: 'IdentityCreated',
      data: {
        did: 'did:mycelix:abc123',
        initial_matl_score: 0.5,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('identity_bridge', 'IdentityCreated', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle IdentityUpdated events', () => {
    const callback = vi.fn();
    handler.onIdentityUpdated(callback);

    const payload: IdentitySignalPayload = {
      type: 'IdentityUpdated',
      data: {
        did: 'did:mycelix:abc123',
        updated_fields: ['profile', 'credentials'],
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('identity_bridge', 'IdentityUpdated', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle CredentialIssued events', () => {
    const callback = vi.fn();
    handler.onCredentialIssued(callback);

    const payload: IdentitySignalPayload = {
      type: 'CredentialIssued',
      data: {
        credential_id: 'cred-123',
        issuer_did: 'did:mycelix:issuer',
        subject_did: 'did:mycelix:subject',
        credential_type: 'VerifiedEmail',
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('identity_bridge', 'CredentialIssued', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should unsubscribe when unsubscribe function is called', () => {
    const callback = vi.fn();
    const unsubscribe = handler.onIdentityCreated(callback);

    unsubscribe();

    const payload: IdentitySignalPayload = {
      type: 'IdentityCreated',
      data: {
        did: 'did:mycelix:abc123',
        initial_matl_score: 0.5,
        timestamp: Date.now(),
      },
    };

    handler.handle(createMockSignal('identity_bridge', 'IdentityCreated', payload));

    expect(callback).not.toHaveBeenCalled();
  });

  it('should support onAny for all events', () => {
    const callback = vi.fn();
    handler.onAny(callback);

    const payload: IdentitySignalPayload = {
      type: 'TrustAttested',
      data: {
        attester_did: 'did:mycelix:attester',
        subject_did: 'did:mycelix:subject',
        trust_level: 0.8,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('identity_bridge', 'TrustAttested', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(signal);
  });

  it('should buffer recent signals', () => {
    const payload: IdentitySignalPayload = {
      type: 'IdentityCreated',
      data: {
        did: 'did:mycelix:abc123',
        initial_matl_score: 0.5,
        timestamp: Date.now(),
      },
    };

    handler.handle(createMockSignal('identity_bridge', 'IdentityCreated', payload));

    const buffer = handler.getBuffer();
    expect(buffer).toHaveLength(1);
    expect(buffer[0].payload).toEqual(payload);
  });
});

describe('FinanceSignalHandler', () => {
  let handler: FinanceSignalHandler;

  beforeEach(() => {
    handler = new FinanceSignalHandler();
  });

  it('should handle PaymentProcessed events', () => {
    const callback = vi.fn();
    handler.onPaymentProcessed(callback);

    const payload: FinanceSignalPayload = {
      type: 'PaymentProcessed',
      data: {
        payment_id: 'pay-123',
        payer_did: 'did:mycelix:payer',
        payee_did: 'did:mycelix:payee',
        amount: 100,
        currency: 'MCX',
        reference_id: 'order-456',
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('finance_bridge', 'PaymentProcessed', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle CreditScoreUpdated events', () => {
    const callback = vi.fn();
    handler.onCreditScoreUpdated(callback);

    const payload: FinanceSignalPayload = {
      type: 'CreditScoreUpdated',
      data: {
        did: 'did:mycelix:user',
        old_score: 0.6,
        new_score: 0.75,
        reason: 'Successful loan repayment',
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('finance_bridge', 'CreditScoreUpdated', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });
});

describe('EnergySignalHandler', () => {
  let handler: EnergySignalHandler;

  beforeEach(() => {
    handler = new EnergySignalHandler();
  });

  it('should handle EnergyListed events', () => {
    const callback = vi.fn();
    handler.onEnergyListed(callback);

    const payload: EnergySignalPayload = {
      type: 'EnergyListed',
      data: {
        listing_id: 'listing-123',
        seller_did: 'did:mycelix:seller',
        amount_kwh: 500,
        price_per_kwh: 0.12,
        energy_source: 'solar',
        location: { lat: 37.7749, lng: -122.4194 },
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('energy_bridge', 'EnergyListed', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle GridBalanceRequest events', () => {
    const callback = vi.fn();
    handler.onGridBalanceRequest(callback);

    const payload: EnergySignalPayload = {
      type: 'GridBalanceRequest',
      data: {
        request_id: 'req-123',
        requester_did: 'did:mycelix:grid',
        needed_kwh: 1000,
        max_price: 0.15,
        deadline: Date.now() + 3600000,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('energy_bridge', 'GridBalanceRequest', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });
});

describe('GovernanceSignalHandler', () => {
  let handler: GovernanceSignalHandler;

  beforeEach(() => {
    handler = new GovernanceSignalHandler();
  });

  it('should handle ProposalCreated events', () => {
    const callback = vi.fn();
    handler.onProposalCreated(callback);

    const payload: GovernanceSignalPayload = {
      type: 'ProposalCreated',
      data: {
        proposal_hash: 'QmProposal123',
        proposer_did: 'did:mycelix:proposer',
        title: 'Fund Community Garden',
        voting_ends_at: Date.now() + 7 * 24 * 60 * 60 * 1000,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('governance_bridge', 'ProposalCreated', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle VoteCast events', () => {
    const callback = vi.fn();
    handler.onVoteCast(callback);

    const payload: GovernanceSignalPayload = {
      type: 'VoteCast',
      data: {
        proposal_hash: 'QmProposal123',
        voter_did: 'did:mycelix:voter',
        vote: 'for',
        weight: 1.5,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('governance_bridge', 'VoteCast', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });
});

describe('KnowledgeSignalHandler', () => {
  let handler: KnowledgeSignalHandler;

  beforeEach(() => {
    handler = new KnowledgeSignalHandler();
  });

  it('should handle ClaimCreated events', () => {
    const callback = vi.fn();
    handler.onClaimCreated(callback);

    const payload: KnowledgeSignalPayload = {
      type: 'ClaimCreated',
      data: {
        claim_hash: 'QmClaim123',
        author_did: 'did:mycelix:author',
        title: 'Climate Change Impact Study',
        empirical: 0.9,
        normative: 0.7,
        mythic: 0.3,
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('knowledge_bridge', 'ClaimCreated', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });

  it('should handle FactCheckResult events', () => {
    const callback = vi.fn();
    handler.onFactCheckResult(callback);

    const payload: KnowledgeSignalPayload = {
      type: 'FactCheckResult',
      data: {
        request_id: 'req-123',
        claim_text: 'Water boils at 100C at sea level',
        verdict: 'true',
        confidence: 0.99,
        supporting_claims: ['QmClaim456', 'QmClaim789'],
        timestamp: Date.now(),
      },
    };

    const signal = createMockSignal('knowledge_bridge', 'FactCheckResult', payload);
    handler.handle(signal);

    expect(callback).toHaveBeenCalledWith(payload.data);
  });
});

describe('BridgeSignalManager', () => {
  it('should create all domain handlers', () => {
    const manager = new BridgeSignalManager();

    expect(manager.identity).toBeInstanceOf(IdentitySignalHandler);
    expect(manager.finance).toBeInstanceOf(FinanceSignalHandler);
    expect(manager.energy).toBeInstanceOf(EnergySignalHandler);
    expect(manager.governance).toBeInstanceOf(GovernanceSignalHandler);
    expect(manager.knowledge).toBeInstanceOf(KnowledgeSignalHandler);
  });

  it('should not be connected without a client', () => {
    const manager = new BridgeSignalManager();
    expect(manager.isConnected()).toBe(false);
  });
});

describe('Factory Functions', () => {
  it('createSignalManager should create a BridgeSignalManager', () => {
    const manager = createSignalManager();
    expect(manager).toBeInstanceOf(BridgeSignalManager);
  });

  it('createIdentitySignals should create an IdentitySignalHandler', () => {
    const handler = createIdentitySignals();
    expect(handler).toBeInstanceOf(IdentitySignalHandler);
  });

  it('createFinanceSignals should create a FinanceSignalHandler', () => {
    const handler = createFinanceSignals();
    expect(handler).toBeInstanceOf(FinanceSignalHandler);
  });

  it('createEnergySignals should create an EnergySignalHandler', () => {
    const handler = createEnergySignals();
    expect(handler).toBeInstanceOf(EnergySignalHandler);
  });

  it('createGovernanceSignals should create a GovernanceSignalHandler', () => {
    const handler = createGovernanceSignals();
    expect(handler).toBeInstanceOf(GovernanceSignalHandler);
  });

  it('createKnowledgeSignals should create a KnowledgeSignalHandler', () => {
    const handler = createKnowledgeSignals();
    expect(handler).toBeInstanceOf(KnowledgeSignalHandler);
  });

  it('factory functions should accept options', () => {
    const handler = createIdentitySignals({ bufferSize: 50 });
    expect(handler).toBeInstanceOf(IdentitySignalHandler);
  });
});
