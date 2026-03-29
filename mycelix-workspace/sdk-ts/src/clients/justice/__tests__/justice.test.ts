// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Justice Client Tests
 *
 * Verifies zome call arguments and response pass-through for
 * CasesClient, EvidenceClient, ArbitrationClient, and EnforcementClient.
 *
 * Justice clients use ZomeCallable interface and return HolochainRecord<T> directly.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  type ZomeCallable,
} from '../index';

// ============================================================================
// MOCK HELPERS
// ============================================================================

function createMockCallable(): ZomeCallable {
  return {
    callZome: vi.fn(),
  } as unknown as ZomeCallable;
}

function mockRecord(entry: Record<string, unknown>) {
  return {
    entry: { Present: entry },
    signed_action: { hashed: { hash: 'mock-hash', content: {} }, signature: 'mock-sig' },
  };
}

// ============================================================================
// MOCK ENTRIES
// ============================================================================

const CASE_ENTRY = {
  id: 'case-001',
  title: 'Property boundary dispute',
  description: 'Fence encroaches on neighbor lot by 2 feet',
  category: 'PropertyDispute',
  complainant: 'did:mycelix:alice',
  respondent: 'did:mycelix:bob',
  phase: 'Filed',
  status: 'Active',
  arbitrators: [],
  evidence_count: 0,
  filed_at: 1708200000,
  updated_at: 1708200000,
};

const EVIDENCE_ENTRY = {
  id: 'ev-001',
  case_id: 'case-001',
  submitter: 'did:mycelix:alice',
  type_: 'Document',
  title: 'Property Survey',
  description: 'Official survey showing boundary line',
  content_hash: 'sha256:survey123',
  storage_ref: 'ipfs://QmSurvey',
  submitted_at: 1708200000,
  verified: false,
};

const MEDIATOR_ENTRY = {
  did: 'did:mycelix:mediator',
  reputation_score: 0.92,
  cases_mediated: 15,
  success_rate: 0.87,
  specializations: ['PropertyDispute', 'ContractDispute'],
  available: true,
  certified_at: 1700000000,
};

const DECISION_ENTRY = {
  id: 'dec-001',
  case_id: 'case-001',
  outcome: 'ForComplainant',
  reasoning: 'Survey evidence conclusively shows boundary encroachment',
  remedies: [
    {
      type_: 'Action',
      target: 'did:mycelix:bob',
      description: 'Remove fence within 30 days',
      deadline: 1710800000,
      completed: false,
    },
  ],
  arbitrators: ['did:mycelix:arb1'],
  appealable: true,
  appeal_deadline: 1709500000,
  issued_at: 1708300000,
  finalized: false,
};

const ENFORCEMENT_ENTRY = {
  id: 'enf-001',
  decision_id: 'dec-001',
  target_happ: 'mycelix-property',
  target_did: 'did:mycelix:bob',
  action_type: 'RestrictAction',
  details: 'Restrict property transfer until fence removed',
  deadline: 1710800000,
  status: 'Pending',
  requested_at: 1708300000,
};

// ============================================================================
// CASES CLIENT TESTS
// ============================================================================

describe('CasesClient', () => {
  let mockCallable: ZomeCallable;
  let cases: CasesClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    cases = new CasesClient(mockCallable);
  });

  it('fileCase calls correct zome', async () => {
    const record = mockRecord(CASE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      title: 'Property boundary dispute',
      description: 'Fence encroaches on neighbor lot by 2 feet',
      category: 'PropertyDispute' as const,
      respondent: 'did:mycelix:bob',
    };
    const result = await cases.fileCase(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_cases',
      fn_name: 'file_case',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Active');
    expect(result.entry.Present.phase).toBe('Filed');
  });

  it('getCasesByComplainant returns case list', async () => {
    const record = mockRecord(CASE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await cases.getCasesByComplainant('did:mycelix:alice');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_cases',
      fn_name: 'get_cases_by_complainant',
      payload: 'did:mycelix:alice',
    });
    expect(result).toHaveLength(1);
    expect(result[0].entry.Present.complainant).toBe('did:mycelix:alice');
  });
});

// ============================================================================
// EVIDENCE CLIENT TESTS
// ============================================================================

describe('EvidenceClient', () => {
  let mockCallable: ZomeCallable;
  let evidence: EvidenceClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    evidence = new EvidenceClient(mockCallable);
  });

  it('submitEvidence calls correct zome', async () => {
    const record = mockRecord(EVIDENCE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      case_id: 'case-001',
      type_: 'Document' as const,
      title: 'Property Survey',
      description: 'Official survey showing boundary line',
      content_hash: 'sha256:survey123',
      storage_ref: 'ipfs://QmSurvey',
    };
    const result = await evidence.submitEvidence(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_evidence',
      fn_name: 'submit_evidence',
      payload: input,
    });
    expect(result.entry.Present.type_).toBe('Document');
  });

  it('getEvidenceForCase returns evidence list', async () => {
    const record = mockRecord(EVIDENCE_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await evidence.getEvidenceForCase('case-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_evidence',
      fn_name: 'get_evidence_for_case',
      payload: 'case-001',
    });
    expect(result).toHaveLength(1);
  });
});

// ============================================================================
// ARBITRATION CLIENT TESTS
// ============================================================================

describe('ArbitrationClient', () => {
  let mockCallable: ZomeCallable;
  let arbitration: ArbitrationClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    arbitration = new ArbitrationClient(mockCallable);
  });

  it('registerAsMediator calls correct zome', async () => {
    const record = mockRecord(MEDIATOR_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      specializations: ['PropertyDispute', 'ContractDispute'],
    };
    const result = await arbitration.registerAsMediator(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_arbitration',
      fn_name: 'register_as_mediator',
      payload: input,
    });
    expect(result.entry.Present.available).toBe(true);
    expect(result.entry.Present.specializations).toHaveLength(2);
  });

  it('getDecision returns decision record', async () => {
    const record = mockRecord(DECISION_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const result = await arbitration.getDecision('case-001');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_arbitration',
      fn_name: 'get_decision',
      payload: 'case-001',
    });
    expect(result!.entry.Present.outcome).toBe('ForComplainant');
    expect(result!.entry.Present.remedies).toHaveLength(1);
  });
});

// ============================================================================
// ENFORCEMENT CLIENT TESTS
// ============================================================================

describe('EnforcementClient', () => {
  let mockCallable: ZomeCallable;
  let enforcement: EnforcementClient;

  beforeEach(() => {
    mockCallable = createMockCallable();
    enforcement = new EnforcementClient(mockCallable);
  });

  it('requestEnforcement calls correct zome', async () => {
    const record = mockRecord(ENFORCEMENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce(record);

    const input = {
      decision_id: 'dec-001',
      target_happ: 'mycelix-property',
      target_did: 'did:mycelix:bob',
      action_type: 'RestrictAction',
      details: 'Restrict property transfer until fence removed',
      deadline: 1710800000,
    };
    const result = await enforcement.requestEnforcement(input);

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_enforcement',
      fn_name: 'request_enforcement',
      payload: input,
    });
    expect(result.entry.Present.status).toBe('Pending');
  });

  it('getPendingEnforcements returns enforcement list', async () => {
    const record = mockRecord(ENFORCEMENT_ENTRY);
    (mockCallable.callZome as ReturnType<typeof vi.fn>).mockResolvedValueOnce([record]);

    const result = await enforcement.getPendingEnforcements('mycelix-property');

    expect(mockCallable.callZome).toHaveBeenCalledWith({
      role_name: 'civic',
      zome_name: 'justice_enforcement',
      fn_name: 'get_pending_enforcements',
      payload: 'mycelix-property',
    });
    expect(result).toHaveLength(1);
  });
});
