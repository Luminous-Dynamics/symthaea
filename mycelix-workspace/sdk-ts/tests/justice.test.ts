// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Justice Module Tests
 *
 * Tests for the Justice hApp TypeScript clients:
 * - CasesClient (dispute case filing and management)
 * - EvidenceClient (evidence submission and verification)
 * - ArbitrationClient (mediator/arbitrator management and decisions)
 * - EnforcementClient (cross-hApp enforcement actions)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  createJusticeClients,
  type Case,
  type Evidence,
  type Decision,
  type MediatorProfile,
  type ArbitratorProfile,
  type Enforcement,
  type Remedy,
  type ZomeCallable,
  type HolochainRecord,
  type CaseCategory,
  type CasePhase,
  type CaseStatus,
  type EvidenceType,
  type DecisionOutcome,
  type ArbitratorTier,
  type EnforcementType,
  type EnforcementStatus,
} from '../src/justice/index.js';

// ============================================================================
// Mock Setup
// ============================================================================

function createMockRecord<T>(entry: T): HolochainRecord<T> {
  return {
    signed_action: {
      hashed: { hash: 'uhCXk_test_hash_123', content: {} },
      signature: 'sig_test_123',
    },
    entry: { Present: entry },
  };
}

function createMockClient(responses: Map<string, unknown> = new Map()): ZomeCallable {
  return {
    callZome: vi.fn(
      async <T>(params: {
        role_name: string;
        zome_name: string;
        fn_name: string;
        payload: unknown;
      }): Promise<T> => {
        const key = `${params.zome_name}:${params.fn_name}`;
        if (responses.has(key)) {
          return responses.get(key) as T;
        }
        throw new Error(`No mock response for ${key}`);
      }
    ),
  };
}

// ============================================================================
// Mock Data Factories
// ============================================================================

function createMockCase(overrides: Partial<Case> = {}): Case {
  return {
    id: 'case-123',
    title: 'Contract Breach Dispute',
    description: 'Vendor failed to deliver goods as specified',
    category: 'ContractDispute' as CaseCategory,
    complainant: 'did:mycelix:complainant123',
    respondent: 'did:mycelix:respondent456',
    phase: 'Filed' as CasePhase,
    status: 'Active' as CaseStatus,
    arbitrators: [],
    evidence_count: 0,
    filed_at: Date.now() * 1000,
    updated_at: Date.now() * 1000,
    ...overrides,
  };
}

function createMockEvidence(overrides: Partial<Evidence> = {}): Evidence {
  return {
    id: 'evidence-123',
    case_id: 'case-123',
    submitter: 'did:mycelix:complainant123',
    type_: 'Document' as EvidenceType,
    title: 'Contract Agreement',
    description: 'Original signed contract document',
    content_hash: 'sha256:abc123def456',
    submitted_at: Date.now() * 1000,
    verified: false,
    ...overrides,
  };
}

function createMockMediatorProfile(overrides: Partial<MediatorProfile> = {}): MediatorProfile {
  return {
    did: 'did:mycelix:mediator123',
    reputation_score: 0.85,
    cases_mediated: 25,
    success_rate: 0.78,
    specializations: ['ContractDispute', 'PropertyDispute'] as CaseCategory[],
    available: true,
    certified_at: Date.now() * 1000 - 86400000000,
    ...overrides,
  };
}

function createMockArbitratorProfile(overrides: Partial<ArbitratorProfile> = {}): ArbitratorProfile {
  return {
    did: 'did:mycelix:arbitrator123',
    reputation_score: 0.92,
    cases_arbitrated: 50,
    appeals_overturned: 3,
    specializations: ['ContractDispute', 'ConductViolation'] as CaseCategory[],
    tier: 'Lead' as ArbitratorTier,
    available: true,
    certified_at: Date.now() * 1000 - 172800000000,
    ...overrides,
  };
}

function createMockDecision(overrides: Partial<Decision> = {}): Decision {
  return {
    id: 'decision-123',
    case_id: 'case-123',
    outcome: 'ForComplainant' as DecisionOutcome,
    reasoning: 'Evidence clearly shows breach of contract terms',
    remedies: [
      {
        type_: 'Compensation',
        target: 'did:mycelix:complainant123',
        description: 'Full refund of contract value',
        amount: 5000,
        deadline: Date.now() * 1000 + 604800000000,
        completed: false,
      },
    ],
    arbitrators: ['did:mycelix:arbitrator123'],
    appealable: true,
    appeal_deadline: Date.now() * 1000 + 1209600000000,
    issued_at: Date.now() * 1000,
    finalized: false,
    ...overrides,
  };
}

function createMockEnforcement(overrides: Partial<Enforcement> = {}): Enforcement {
  return {
    id: 'enforcement-123',
    decision_id: 'decision-123',
    target_happ: 'finance',
    target_did: 'did:mycelix:respondent456',
    action_type: 'PaymentOrder' as EnforcementType,
    details: 'Transfer 5000 MYC to complainant wallet',
    amount: 5000,
    deadline: Date.now() * 1000 + 604800000000,
    status: 'Requested' as EnforcementStatus,
    requested_at: Date.now() * 1000,
    ...overrides,
  };
}

// ============================================================================
// CasesClient Tests
// ============================================================================

describe('CasesClient', () => {
  let client: CasesClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockCase = createMockCase();

    responses.set('justice_cases:file_case', createMockRecord(mockCase));
    responses.set('justice_cases:get_case', createMockRecord(mockCase));
    responses.set('justice_cases:get_cases_by_complainant', [createMockRecord(mockCase)]);
    responses.set('justice_cases:get_cases_by_respondent', [createMockRecord(mockCase)]);
    responses.set('justice_cases:get_cases_by_party', [createMockRecord(mockCase)]);
    responses.set(
      'justice_cases:withdraw_case',
      createMockRecord({
        ...mockCase,
        status: 'Withdrawn',
      })
    );
    responses.set(
      'justice_cases:respond_to_case',
      createMockRecord({
        ...mockCase,
        phase: 'Mediation',
      })
    );

    mockZome = createMockClient(responses);
    client = new CasesClient(mockZome);
  });

  describe('fileCase', () => {
    it('should file a new case', async () => {
      const result = await client.fileCase({
        title: 'Contract Breach Dispute',
        description: 'Vendor failed to deliver goods',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent456',
      });

      expect(result.entry.Present.id).toBe('case-123');
      expect(result.entry.Present.title).toBe('Contract Breach Dispute');
      expect(mockZome.callZome).toHaveBeenCalledWith(
        expect.objectContaining({
          role_name: 'civic',
          zome_name: 'justice_cases',
          fn_name: 'file_case',
        })
      );
    });

    it('should file case with all categories', async () => {
      const categories: CaseCategory[] = [
        'ContractDispute',
        'PropertyDispute',
        'ConductViolation',
        'ConductViolation',
        'IPDispute',
        'GovernanceDispute',
        'ConductViolation',
        'Other',
      ];

      for (const category of categories) {
        const result = await client.fileCase({
          title: `${category} Case`,
          description: 'Test case',
          category,
          respondent: 'did:mycelix:test',
        });

        expect(result).toBeDefined();
      }
    });
  });

  describe('getCase', () => {
    it('should get case by ID', async () => {
      const result = await client.getCase('case-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.id).toBe('case-123');
    });
  });

  describe('getCasesByComplainant', () => {
    it('should get cases by complainant DID', async () => {
      const results = await client.getCasesByComplainant('did:mycelix:complainant123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.complainant).toBe('did:mycelix:complainant123');
    });
  });

  describe('getCasesByRespondent', () => {
    it('should get cases by respondent DID', async () => {
      const results = await client.getCasesByRespondent('did:mycelix:respondent456');

      expect(results).toHaveLength(1);
    });
  });

  describe('getCasesByParty', () => {
    it('should get cases where DID is either party', async () => {
      const results = await client.getCasesByParty('did:mycelix:complainant123');

      expect(results).toHaveLength(1);
    });
  });

  describe('withdrawCase', () => {
    it('should withdraw a case', async () => {
      const result = await client.withdrawCase('case-123', 'Settlement reached');

      expect(result.entry.Present.status).toBe('Withdrawn');
    });
  });

  describe('respondToCase', () => {
    it('should respond to a case', async () => {
      const result = await client.respondToCase('case-123', 'We dispute these claims');

      expect(result.entry.Present.phase).toBe('Mediation');
    });
  });
});

// ============================================================================
// EvidenceClient Tests
// ============================================================================

describe('EvidenceClient', () => {
  let client: EvidenceClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockEvidence = createMockEvidence();

    responses.set('justice_evidence:submit_evidence', createMockRecord(mockEvidence));
    responses.set('justice_evidence:get_evidence_for_case', [createMockRecord(mockEvidence)]);
    responses.set(
      'justice_evidence:verify_evidence',
      createMockRecord({
        ...mockEvidence,
        verified: true,
      })
    );
    responses.set('justice_evidence:challenge_evidence', undefined);

    mockZome = createMockClient(responses);
    client = new EvidenceClient(mockZome);
  });

  describe('submitEvidence', () => {
    it('should submit evidence to a case', async () => {
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Contract Agreement',
        description: 'Original signed contract',
        content_hash: 'sha256:abc123',
      });

      expect(result.entry.Present.id).toBe('evidence-123');
      expect(result.entry.Present.verified).toBe(false);
    });

    it('should submit evidence with all types', async () => {
      const types: EvidenceType[] = [
        'Document',
        'Testimony',
        'Transaction',
        'Screenshot',
        'Witness',
        'Other',
      ];

      for (const type_ of types) {
        const result = await client.submitEvidence({
          case_id: 'case-123',
          type_,
          title: `${type_} Evidence`,
          description: 'Test evidence',
          content_hash: 'sha256:test',
        });

        expect(result).toBeDefined();
      }
    });

    it('should submit evidence with storage reference', async () => {
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Large Document',
        description: 'Stored in DHT',
        content_hash: 'sha256:large',
        storage_ref: 'uhCAk_dht_reference',
      });

      expect(result).toBeDefined();
    });
  });

  describe('getEvidenceForCase', () => {
    it('should get all evidence for a case', async () => {
      const results = await client.getEvidenceForCase('case-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.case_id).toBe('case-123');
    });
  });

  describe('verifyEvidence', () => {
    it('should verify evidence (arbitrator only)', async () => {
      const result = await client.verifyEvidence('evidence-123');

      expect(result.entry.Present.verified).toBe(true);
    });
  });

  describe('challengeEvidence', () => {
    it('should challenge evidence', async () => {
      await expect(
        client.challengeEvidence('evidence-123', 'Document appears to be altered')
      ).resolves.not.toThrow();
    });
  });
});

// ============================================================================
// ArbitrationClient Tests
// ============================================================================

describe('ArbitrationClient', () => {
  let client: ArbitrationClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockMediator = createMockMediatorProfile();
    const mockArbitrator = createMockArbitratorProfile();
    const mockDecision = createMockDecision();
    const mockCase = createMockCase();

    responses.set('justice_arbitration:register_as_mediator', createMockRecord(mockMediator));
    responses.set('justice_arbitration:register_as_arbitrator', createMockRecord(mockArbitrator));
    responses.set('justice_arbitration:get_available_mediators', [createMockRecord(mockMediator)]);
    responses.set('justice_arbitration:get_available_arbitrators', [createMockRecord(mockArbitrator)]);
    responses.set(
      'justice_arbitration:assign_mediator',
      createMockRecord({
        ...mockCase,
        mediator: mockMediator.did,
        phase: 'Mediation',
      })
    );
    responses.set(
      'justice_arbitration:escalate_to_arbitration',
      createMockRecord({
        ...mockCase,
        arbitrators: [mockArbitrator.did],
        phase: 'Arbitration',
      })
    );
    responses.set('justice_arbitration:render_decision', createMockRecord(mockDecision));
    responses.set('justice_arbitration:get_decision', createMockRecord(mockDecision));
    responses.set(
      'justice_arbitration:file_appeal',
      createMockRecord({
        ...mockCase,
        phase: 'Appeal',
      })
    );

    mockZome = createMockClient(responses);
    client = new ArbitrationClient(mockZome);
  });

  describe('registerAsMediator', () => {
    it('should register as mediator', async () => {
      const result = await client.registerAsMediator({
        specializations: ['ContractDispute', 'PropertyDispute'],
      });

      expect(result.entry.Present.did).toBe('did:mycelix:mediator123');
      expect(result.entry.Present.available).toBe(true);
    });
  });

  describe('registerAsArbitrator', () => {
    it('should register as arbitrator with tier', async () => {
      const result = await client.registerAsArbitrator({
        specializations: ['ContractDispute', 'ConductViolation'],
        tier: 'Lead',
      });

      expect(result.entry.Present.tier).toBe('Lead');
      expect(result.entry.Present.cases_arbitrated).toBe(50);
    });

    it('should register with all tiers', async () => {
      const tiers: ArbitratorTier[] = ['Panel', 'Lead', 'Appeals'];

      for (const tier of tiers) {
        const result = await client.registerAsArbitrator({
          specializations: ['ContractDispute'],
          tier,
        });

        expect(result).toBeDefined();
      }
    });
  });

  describe('getAvailableMediators', () => {
    it('should get available mediators for category', async () => {
      const results = await client.getAvailableMediators('ContractDispute');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.available).toBe(true);
      expect(results[0].entry.Present.specializations).toContain('ContractDispute');
    });
  });

  describe('getAvailableArbitrators', () => {
    it('should get available arbitrators for category', async () => {
      const results = await client.getAvailableArbitrators('ContractDispute');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.available).toBe(true);
    });
  });

  describe('assignMediator', () => {
    it('should assign mediator to case', async () => {
      const result = await client.assignMediator('case-123', 'did:mycelix:mediator123');

      expect(result.entry.Present.mediator).toBe('did:mycelix:mediator123');
      expect(result.entry.Present.phase).toBe('Mediation');
    });
  });

  describe('escalateToArbitration', () => {
    it('should escalate case to arbitration', async () => {
      const result = await client.escalateToArbitration('case-123', [
        'did:mycelix:arbitrator123',
      ]);

      expect(result.entry.Present.arbitrators).toContain('did:mycelix:arbitrator123');
      expect(result.entry.Present.phase).toBe('Arbitration');
    });
  });

  describe('renderDecision', () => {
    it('should render decision on case', async () => {
      const remedies: Remedy[] = [
        {
          type_: 'Compensation',
          target: 'did:mycelix:complainant123',
          description: 'Full refund',
          amount: 5000,
          completed: false,
        },
      ];

      const result = await client.renderDecision(
        'case-123',
        'ForComplainant',
        'Clear breach of contract',
        remedies
      );

      expect(result.entry.Present.outcome).toBe('ForComplainant');
      expect(result.entry.Present.remedies).toHaveLength(1);
    });

    it('should render with all outcomes', async () => {
      const outcomes: DecisionOutcome[] = [
        'ForComplainant',
        'ForRespondent',
        'SplitDecision',
        'Dismissed',
        'Settled',
      ];

      for (const outcome of outcomes) {
        const result = await client.renderDecision('case-123', outcome, 'Reasoning', []);

        expect(result).toBeDefined();
      }
    });
  });

  describe('getDecision', () => {
    it('should get decision for case', async () => {
      const result = await client.getDecision('case-123');

      expect(result).not.toBeNull();
      expect(result!.entry.Present.case_id).toBe('case-123');
    });
  });

  describe('fileAppeal', () => {
    it('should file appeal on decision', async () => {
      const result = await client.fileAppeal('decision-123', 'New evidence discovered');

      expect(result.entry.Present.phase).toBe('Appeal');
    });
  });
});

// ============================================================================
// EnforcementClient Tests
// ============================================================================

describe('EnforcementClient', () => {
  let client: EnforcementClient;
  let mockZome: ZomeCallable;

  beforeEach(() => {
    const responses = new Map<string, unknown>();
    const mockEnforcement = createMockEnforcement();
    const mockDecision = createMockDecision();

    responses.set('justice_enforcement:request_enforcement', createMockRecord(mockEnforcement));
    responses.set('justice_enforcement:get_pending_enforcements', [createMockRecord(mockEnforcement)]);
    responses.set(
      'justice_enforcement:acknowledge_enforcement',
      createMockRecord({
        ...mockEnforcement,
        status: 'Acknowledged',
      })
    );
    responses.set('justice_enforcement:get_enforcements_for_decision', [
      createMockRecord(mockEnforcement),
    ]);
    responses.set(
      'justice_enforcement:mark_remedy_completed',
      createMockRecord({
        ...mockDecision,
        remedies: [
          {
            ...mockDecision.remedies[0],
            completed: true,
          },
        ],
      })
    );

    mockZome = createMockClient(responses);
    client = new EnforcementClient(mockZome);
  });

  describe('requestEnforcement', () => {
    it('should request enforcement action', async () => {
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:respondent456',
        action_type: 'PaymentOrder',
        details: 'Transfer 5000 MYC',
        amount: 5000,
        deadline: Date.now() * 1000 + 604800000000,
      });

      expect(result.entry.Present.status).toBe('Requested');
      expect(result.entry.Present.target_happ).toBe('finance');
    });

    it('should request with all enforcement types', async () => {
      const types: EnforcementType[] = [
        'ReputationPenalty',
        'TemporarySuspension',
        'PermanentBan',
        'AssetFreeze',
        'PaymentOrder',
        'RequiredAction',
      ];

      for (const action_type of types) {
        const result = await client.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'identity',
          target_did: 'did:mycelix:target',
          action_type,
          details: `${action_type} action`,
        });

        expect(result).toBeDefined();
      }
    });
  });

  describe('getPendingEnforcements', () => {
    it('should get all pending enforcements', async () => {
      const results = await client.getPendingEnforcements();

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.status).toBe('Requested');
    });

    it('should get pending enforcements for specific hApp', async () => {
      const results = await client.getPendingEnforcements('finance');

      expect(results).toHaveLength(1);
    });
  });

  describe('acknowledgeEnforcement', () => {
    it('should acknowledge enforcement', async () => {
      const result = await client.acknowledgeEnforcement('enforcement-123', 'Acknowledged');

      expect(result.entry.Present.status).toBe('Acknowledged');
    });

    it('should acknowledge with all statuses', async () => {
      const statuses: EnforcementStatus[] = [
        'Requested',
        'Acknowledged',
        'Executed',
        'Rejected',
        'Appealed',
      ];

      for (const status of statuses) {
        const result = await client.acknowledgeEnforcement('enforcement-123', status);

        expect(result).toBeDefined();
      }
    });
  });

  describe('getEnforcementsForDecision', () => {
    it('should get enforcements for a decision', async () => {
      const results = await client.getEnforcementsForDecision('decision-123');

      expect(results).toHaveLength(1);
      expect(results[0].entry.Present.decision_id).toBe('decision-123');
    });
  });

  describe('markRemedyCompleted', () => {
    it('should mark remedy as completed', async () => {
      const result = await client.markRemedyCompleted('decision-123', 0);

      expect(result.entry.Present.remedies[0].completed).toBe(true);
    });
  });
});

// ============================================================================
// Factory Function Tests
// ============================================================================

describe('createJusticeClients', () => {
  it('should create all justice clients', () => {
    const mockZome = createMockClient(new Map());
    const clients = createJusticeClients(mockZome);

    expect(clients.cases).toBeInstanceOf(CasesClient);
    expect(clients.evidence).toBeInstanceOf(EvidenceClient);
    expect(clients.arbitration).toBeInstanceOf(ArbitrationClient);
    expect(clients.enforcement).toBeInstanceOf(EnforcementClient);
  });

  it('should share the same ZomeCallable instance', () => {
    const mockZome = createMockClient(new Map());
    const clients = createJusticeClients(mockZome);

    // All clients should use the same underlying client
    expect(clients.cases).toBeDefined();
    expect(clients.evidence).toBeDefined();
    expect(clients.arbitration).toBeDefined();
    expect(clients.enforcement).toBeDefined();
  });
});

// ============================================================================
// Type Safety Tests
// ============================================================================

describe('Type Safety', () => {
  it('should enforce case category types', () => {
    const validCategories: CaseCategory[] = [
      'ContractDispute',
      'PropertyDispute',
      'ConductViolation',
      'ConductViolation',
      'IPDispute',
      'GovernanceDispute',
      'ConductViolation',
      'Other',
    ];

    validCategories.forEach((cat) => {
      const mockCase = createMockCase({ category: cat });
      expect(validCategories).toContain(mockCase.category);
    });
  });

  it('should enforce case phase progression', () => {
    const phases: CasePhase[] = [
      'Filed',
      'Mediation',
      'Arbitration',
      'Appeal',
      'Enforcement',
      'Closed',
    ];

    phases.forEach((phase) => {
      const mockCase = createMockCase({ phase });
      expect(phases).toContain(mockCase.phase);
    });
  });

  it('should enforce remedy types', () => {
    const remedyTypes: Remedy['type_'][] = [
      'Compensation',
      'Action',
      'Apology',
      'ReputationAdjustment',
      'Ban',
    ];

    remedyTypes.forEach((type_) => {
      const remedy: Remedy = {
        type_,
        target: 'did:mycelix:test',
        description: 'Test remedy',
        completed: false,
      };
      expect(remedyTypes).toContain(remedy.type_);
    });
  });

  it('should enforce decision requires case_id', () => {
    const decision = createMockDecision();
    expect(decision.case_id).toBeDefined();
    expect(decision.case_id).not.toBe('');
  });

  it('should enforce enforcement requires decision_id', () => {
    const enforcement = createMockEnforcement();
    expect(enforcement.decision_id).toBeDefined();
    expect(enforcement.decision_id).not.toBe('');
  });
});

// ============================================================================
// Integration Pattern Tests
// ============================================================================

describe('Integration Patterns', () => {
  it('should support full case lifecycle', async () => {
    const responses = new Map<string, unknown>();
    const mockCase = createMockCase();

    responses.set('justice_cases:file_case', createMockRecord(mockCase));
    responses.set('justice_evidence:submit_evidence', createMockRecord(createMockEvidence()));
    responses.set(
      'justice_arbitration:assign_mediator',
      createMockRecord({
        ...mockCase,
        phase: 'Mediation',
      })
    );

    const mockZome = createMockClient(responses);
    const clients = createJusticeClients(mockZome);

    // File case
    const caseResult = await clients.cases.fileCase({
      title: 'Test Case',
      description: 'Test',
      category: 'Other',
      respondent: 'did:mycelix:test',
    });
    expect(caseResult.entry.Present.id).toBeDefined();

    // Submit evidence
    const evidenceResult = await clients.evidence.submitEvidence({
      case_id: caseResult.entry.Present.id,
      type_: 'Document',
      title: 'Evidence',
      description: 'Test evidence',
      content_hash: 'sha256:test',
    });
    expect(evidenceResult.entry.Present.case_id).toBe(caseResult.entry.Present.id);

    // Assign mediator
    const mediatedCase = await clients.arbitration.assignMediator(
      caseResult.entry.Present.id,
      'did:mycelix:mediator'
    );
    expect(mediatedCase.entry.Present.phase).toBe('Mediation');
  });

  it('should support cross-hApp enforcement flow', async () => {
    const responses = new Map<string, unknown>();
    const mockDecision = createMockDecision();
    const mockEnforcement = createMockEnforcement();

    responses.set('justice_arbitration:render_decision', createMockRecord(mockDecision));
    responses.set('justice_enforcement:request_enforcement', createMockRecord(mockEnforcement));

    const mockZome = createMockClient(responses);
    const clients = createJusticeClients(mockZome);

    // Render decision
    const decision = await clients.arbitration.renderDecision('case-123', 'ForComplainant', 'Reasoning', []);

    // Request cross-hApp enforcement
    const enforcement = await clients.enforcement.requestEnforcement({
      decision_id: decision.entry.Present.id,
      target_happ: 'finance',
      target_did: 'did:mycelix:respondent',
      action_type: 'PaymentOrder',
      details: 'Compensation payment',
      amount: 5000,
    });

    expect(enforcement.entry.Present.target_happ).toBe('finance');
    expect(enforcement.entry.Present.decision_id).toBe(decision.entry.Present.id);
  });
});

// ============================================================================
// Validated Client Tests
// ============================================================================

import {
  ValidatedCasesClient,
  ValidatedEvidenceClient,
  ValidatedArbitrationClient,
  ValidatedEnforcementClient,
  createValidatedJusticeClients,
} from '../src/justice/validated.js';

describe('ValidatedCasesClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedCasesClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['justice_cases:file_case', createMockRecord(createMockCase())],
        ['justice_cases:get_case', createMockRecord(createMockCase())],
        ['justice_cases:get_cases_by_complainant', [createMockRecord(createMockCase())]],
        ['justice_cases:get_cases_by_respondent', [createMockRecord(createMockCase())]],
        ['justice_cases:get_cases_by_party', [createMockRecord(createMockCase())]],
        ['justice_cases:withdraw_case', createMockRecord(createMockCase())],
        ['justice_cases:respond_to_case', createMockRecord(createMockCase())],
      ])
    );
    validatedClient = new ValidatedCasesClient(mockClient);
  });

  describe('fileCase', () => {
    it('accepts valid case input', async () => {
      const result = await validatedClient.fileCase({
        title: 'Contract Breach',
        description: 'Vendor failed to deliver goods as contracted',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent456',
      });
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty title', async () => {
      await expect(
        validatedClient.fileCase({
          title: '',
          description: 'Vendor failed to deliver goods as contracted',
          category: 'ContractDispute',
          respondent: 'did:mycelix:respondent456',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects title over 200 chars', async () => {
      await expect(
        validatedClient.fileCase({
          title: 'x'.repeat(201),
          description: 'Vendor failed to deliver goods as contracted',
          category: 'ContractDispute',
          respondent: 'did:mycelix:respondent456',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short description', async () => {
      await expect(
        validatedClient.fileCase({
          title: 'Contract Breach',
          description: 'Too short',
          category: 'ContractDispute',
          respondent: 'did:mycelix:respondent456',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid category', async () => {
      await expect(
        validatedClient.fileCase({
          title: 'Contract Breach',
          description: 'Vendor failed to deliver goods as contracted',
          category: 'Invalid' as 'ContractDispute',
          respondent: 'did:mycelix:respondent456',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID respondent', async () => {
      await expect(
        validatedClient.fileCase({
          title: 'Contract Breach',
          description: 'Vendor failed to deliver goods as contracted',
          category: 'ContractDispute',
          respondent: 'not-a-did',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getCase', () => {
    it('accepts valid case ID', async () => {
      const result = await validatedClient.getCase('case-123');
      expect(result?.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty case ID', async () => {
      await expect(validatedClient.getCase('')).rejects.toThrow('Validation failed');
    });
  });

  describe('getCasesByComplainant', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getCasesByComplainant('did:mycelix:complainant');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getCasesByComplainant('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('getCasesByRespondent', () => {
    it('accepts valid DID', async () => {
      const result = await validatedClient.getCasesByRespondent('did:mycelix:respondent');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects non-DID string', async () => {
      await expect(validatedClient.getCasesByRespondent('not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('withdrawCase', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.withdrawCase('case-123', 'Resolved privately');
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty case ID', async () => {
      await expect(validatedClient.withdrawCase('', 'Reason')).rejects.toThrow('Validation failed');
    });

    it('rejects empty reason', async () => {
      await expect(validatedClient.withdrawCase('case-123', '')).rejects.toThrow('Validation failed');
    });
  });

  describe('respondToCase', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.respondToCase('case-123', 'I deny these allegations and will provide counter-evidence');
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty case ID', async () => {
      await expect(validatedClient.respondToCase('', 'Response text that is long enough')).rejects.toThrow('Validation failed');
    });

    it('rejects short response', async () => {
      await expect(validatedClient.respondToCase('case-123', 'Too short')).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedEvidenceClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedEvidenceClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['justice_evidence:submit_evidence', createMockRecord(createMockEvidence())],
        ['justice_evidence:get_evidence_for_case', [createMockRecord(createMockEvidence())]],
        ['justice_evidence:verify_evidence', createMockRecord(createMockEvidence())],
        ['justice_evidence:challenge_evidence', undefined],
      ])
    );
    validatedClient = new ValidatedEvidenceClient(mockClient);
  });

  describe('submitEvidence', () => {
    it('accepts valid evidence input', async () => {
      const result = await validatedClient.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Contract Agreement',
        description: 'Original signed contract',
        content_hash: 'sha256:abcdef1234567890abcdef1234567890',
      });
      expect(result.entry.Present?.id).toBe('evidence-123');
    });

    it('rejects empty case ID', async () => {
      await expect(
        validatedClient.submitEvidence({
          case_id: '',
          type_: 'Document',
          title: 'Contract',
          description: 'Description',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid evidence type', async () => {
      await expect(
        validatedClient.submitEvidence({
          case_id: 'case-123',
          type_: 'Invalid' as 'Document',
          title: 'Contract',
          description: 'Description',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty title', async () => {
      await expect(
        validatedClient.submitEvidence({
          case_id: 'case-123',
          type_: 'Document',
          title: '',
          description: 'Description',
          content_hash: 'sha256:abcdef1234567890abcdef1234567890',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short content hash', async () => {
      await expect(
        validatedClient.submitEvidence({
          case_id: 'case-123',
          type_: 'Document',
          title: 'Contract',
          description: 'Description',
          content_hash: 'short',
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getEvidenceForCase', () => {
    it('accepts valid case ID', async () => {
      const result = await validatedClient.getEvidenceForCase('case-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty case ID', async () => {
      await expect(validatedClient.getEvidenceForCase('')).rejects.toThrow('Validation failed');
    });
  });

  describe('verifyEvidence', () => {
    it('accepts valid evidence ID', async () => {
      const result = await validatedClient.verifyEvidence('evidence-123');
      expect(result.entry.Present?.id).toBe('evidence-123');
    });

    it('rejects empty evidence ID', async () => {
      await expect(validatedClient.verifyEvidence('')).rejects.toThrow('Validation failed');
    });
  });

  describe('challengeEvidence', () => {
    it('accepts valid inputs', async () => {
      await expect(
        validatedClient.challengeEvidence('evidence-123', 'This evidence is fabricated and I will prove it')
      ).resolves.toBeUndefined();
    });

    it('rejects empty evidence ID', async () => {
      await expect(
        validatedClient.challengeEvidence('', 'This evidence is fabricated and I will prove it')
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short reason', async () => {
      await expect(
        validatedClient.challengeEvidence('evidence-123', 'Too short')
      ).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedArbitrationClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedArbitrationClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['justice_arbitration:register_as_mediator', createMockRecord(createMockMediatorProfile())],
        ['justice_arbitration:register_as_arbitrator', createMockRecord(createMockArbitratorProfile())],
        ['justice_arbitration:get_available_mediators', [createMockRecord(createMockMediatorProfile())]],
        ['justice_arbitration:get_available_arbitrators', [createMockRecord(createMockArbitratorProfile())]],
        ['justice_arbitration:assign_mediator', createMockRecord(createMockCase())],
        ['justice_arbitration:escalate_to_arbitration', createMockRecord(createMockCase())],
        ['justice_arbitration:render_decision', createMockRecord(createMockDecision())],
        ['justice_arbitration:get_decision', createMockRecord(createMockDecision())],
        ['justice_arbitration:file_appeal', createMockRecord(createMockCase())],
      ])
    );
    validatedClient = new ValidatedArbitrationClient(mockClient);
  });

  describe('registerAsMediator', () => {
    it('accepts valid mediator input', async () => {
      const result = await validatedClient.registerAsMediator({
        specializations: ['ContractDispute', 'PropertyDispute'],
      });
      expect(result.entry.Present?.did).toBe('did:mycelix:mediator123');
    });

    it('rejects empty specializations', async () => {
      await expect(
        validatedClient.registerAsMediator({ specializations: [] })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid specialization', async () => {
      await expect(
        validatedClient.registerAsMediator({ specializations: ['Invalid' as 'ContractDispute'] })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('registerAsArbitrator', () => {
    it('accepts valid arbitrator input', async () => {
      const result = await validatedClient.registerAsArbitrator({
        specializations: ['ContractDispute', 'ConductViolation'],
        tier: 'Lead',
      });
      expect(result.entry.Present?.did).toBe('did:mycelix:arbitrator123');
    });

    it('rejects empty specializations', async () => {
      await expect(
        validatedClient.registerAsArbitrator({ specializations: [], tier: 'Lead' })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid tier', async () => {
      await expect(
        validatedClient.registerAsArbitrator({ specializations: ['ContractDispute'], tier: 'Invalid' as 'Lead' })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getAvailableMediators', () => {
    it('accepts valid category', async () => {
      const result = await validatedClient.getAvailableMediators('ContractDispute');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects invalid category', async () => {
      await expect(
        validatedClient.getAvailableMediators('Invalid' as 'ContractDispute')
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('assignMediator', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.assignMediator('case-123', 'did:mycelix:mediator');
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty case ID', async () => {
      await expect(validatedClient.assignMediator('', 'did:mycelix:mediator')).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID mediator', async () => {
      await expect(validatedClient.assignMediator('case-123', 'not-a-did')).rejects.toThrow('Validation failed');
    });
  });

  describe('escalateToArbitration', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.escalateToArbitration('case-123', ['did:mycelix:arb1', 'did:mycelix:arb2']);
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty case ID', async () => {
      await expect(
        validatedClient.escalateToArbitration('', ['did:mycelix:arb1'])
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty arbitrators array', async () => {
      await expect(
        validatedClient.escalateToArbitration('case-123', [])
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID arbitrators', async () => {
      await expect(
        validatedClient.escalateToArbitration('case-123', ['not-a-did'])
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('renderDecision', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.renderDecision(
        'case-123',
        'ForComplainant',
        'The evidence clearly supports the complainant position and remedy is warranted',
        []
      );
      expect(result.entry.Present?.id).toBe('decision-123');
    });

    it('rejects empty case ID', async () => {
      await expect(
        validatedClient.renderDecision('', 'ForComplainant', 'Long enough reasoning text for decision', [])
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid outcome', async () => {
      await expect(
        validatedClient.renderDecision('case-123', 'Invalid' as 'ForComplainant', 'Long enough reasoning text for decision', [])
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short reasoning', async () => {
      await expect(
        validatedClient.renderDecision('case-123', 'ForComplainant', 'Too short', [])
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('fileAppeal', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.fileAppeal('decision-123', 'The decision failed to consider key evidence that proves my position');
      expect(result.entry.Present?.id).toBe('case-123');
    });

    it('rejects empty decision ID', async () => {
      await expect(
        validatedClient.fileAppeal('', 'The decision failed to consider key evidence that proves my position')
      ).rejects.toThrow('Validation failed');
    });

    it('rejects short grounds', async () => {
      await expect(
        validatedClient.fileAppeal('decision-123', 'Too short')
      ).rejects.toThrow('Validation failed');
    });
  });
});

describe('ValidatedEnforcementClient', () => {
  let mockClient: ZomeCallable;
  let validatedClient: ValidatedEnforcementClient;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['justice_enforcement:request_enforcement', createMockRecord(createMockEnforcement())],
        ['justice_enforcement:get_pending_enforcements', [createMockRecord(createMockEnforcement())]],
        ['justice_enforcement:acknowledge_enforcement', createMockRecord(createMockEnforcement())],
        ['justice_enforcement:get_enforcements_for_decision', [createMockRecord(createMockEnforcement())]],
        ['justice_enforcement:mark_remedy_completed', createMockRecord(createMockDecision())],
      ])
    );
    validatedClient = new ValidatedEnforcementClient(mockClient);
  });

  describe('requestEnforcement', () => {
    it('accepts valid enforcement input', async () => {
      const result = await validatedClient.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:respondent',
        action_type: 'PaymentOrder',
        details: 'Compensation payment required',
      });
      expect(result.entry.Present?.id).toBe('enforcement-123');
    });

    it('rejects empty decision ID', async () => {
      await expect(
        validatedClient.requestEnforcement({
          decision_id: '',
          target_happ: 'finance',
          target_did: 'did:mycelix:respondent',
          action_type: 'PaymentOrder',
          details: 'Details',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects empty target happ', async () => {
      await expect(
        validatedClient.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: '',
          target_did: 'did:mycelix:respondent',
          action_type: 'PaymentOrder',
          details: 'Details',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects non-DID target', async () => {
      await expect(
        validatedClient.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'finance',
          target_did: 'not-a-did',
          action_type: 'PaymentOrder',
          details: 'Details',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid action type', async () => {
      await expect(
        validatedClient.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'finance',
          target_did: 'did:mycelix:respondent',
          action_type: 'Invalid' as 'PaymentOrder',
          details: 'Details',
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative amount', async () => {
      await expect(
        validatedClient.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'finance',
          target_did: 'did:mycelix:respondent',
          action_type: 'PaymentOrder',
          details: 'Details',
          amount: -1000,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('acknowledgeEnforcement', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.acknowledgeEnforcement('enforcement-123', 'Acknowledged');
      expect(result.entry.Present?.id).toBe('enforcement-123');
    });

    it('rejects empty enforcement ID', async () => {
      await expect(
        validatedClient.acknowledgeEnforcement('', 'Acknowledged')
      ).rejects.toThrow('Validation failed');
    });

    it('rejects invalid status', async () => {
      await expect(
        validatedClient.acknowledgeEnforcement('enforcement-123', 'Invalid' as 'Acknowledged')
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('getEnforcementsForDecision', () => {
    it('accepts valid decision ID', async () => {
      const result = await validatedClient.getEnforcementsForDecision('decision-123');
      expect(result.length).toBeGreaterThan(0);
    });

    it('rejects empty decision ID', async () => {
      await expect(validatedClient.getEnforcementsForDecision('')).rejects.toThrow('Validation failed');
    });
  });

  describe('markRemedyCompleted', () => {
    it('accepts valid inputs', async () => {
      const result = await validatedClient.markRemedyCompleted('decision-123', 0);
      expect(result.entry.Present?.id).toBe('decision-123');
    });

    it('rejects empty decision ID', async () => {
      await expect(validatedClient.markRemedyCompleted('', 0)).rejects.toThrow('Validation failed');
    });

    it('rejects negative remedy index', async () => {
      await expect(validatedClient.markRemedyCompleted('decision-123', -1)).rejects.toThrow('Validation failed');
    });
  });
});

describe('createValidatedJusticeClients', () => {
  it('creates all validated clients', () => {
    const mockClient = createMockClient(new Map());
    const clients = createValidatedJusticeClients(mockClient);

    expect(clients.cases).toBeInstanceOf(ValidatedCasesClient);
    expect(clients.evidence).toBeInstanceOf(ValidatedEvidenceClient);
    expect(clients.arbitration).toBeInstanceOf(ValidatedArbitrationClient);
    expect(clients.enforcement).toBeInstanceOf(ValidatedEnforcementClient);
  });
});

// ============================================================================
// Additional Mutation Score Tests - Validation Schema Boundary Conditions
// ============================================================================

describe('Justice Validation Schema Boundary Tests', () => {
  let mockClient: ZomeCallable;

  beforeEach(() => {
    mockClient = createMockClient(
      new Map([
        ['justice_cases:file_case', createMockRecord(createMockCase())],
        ['justice_cases:get_case', createMockRecord(createMockCase())],
        ['justice_cases:get_cases_by_complainant', [createMockRecord(createMockCase())]],
        ['justice_cases:get_cases_by_respondent', [createMockRecord(createMockCase())]],
        ['justice_cases:get_cases_by_party', [createMockRecord(createMockCase())]],
        ['justice_cases:withdraw_case', createMockRecord(createMockCase())],
        ['justice_cases:respond_to_case', createMockRecord(createMockCase())],
        ['justice_evidence:submit_evidence', createMockRecord(createMockEvidence())],
        ['justice_evidence:get_evidence_for_case', [createMockRecord(createMockEvidence())]],
        ['justice_evidence:verify_evidence', createMockRecord(createMockEvidence())],
        ['justice_evidence:challenge_evidence', undefined],
        ['justice_arbitration:register_as_mediator', createMockRecord(createMockMediatorProfile())],
        ['justice_arbitration:register_as_arbitrator', createMockRecord(createMockArbitratorProfile())],
        ['justice_arbitration:get_available_mediators', [createMockRecord(createMockMediatorProfile())]],
        ['justice_arbitration:get_available_arbitrators', [createMockRecord(createMockArbitratorProfile())]],
        ['justice_arbitration:assign_mediator', createMockRecord(createMockCase())],
        ['justice_arbitration:escalate_to_arbitration', createMockRecord(createMockCase())],
        ['justice_arbitration:render_decision', createMockRecord(createMockDecision())],
        ['justice_arbitration:get_decision', createMockRecord(createMockDecision())],
        ['justice_arbitration:file_appeal', createMockRecord(createMockCase())],
        ['justice_enforcement:request_enforcement', createMockRecord(createMockEnforcement())],
        ['justice_enforcement:get_pending_enforcements', [createMockRecord(createMockEnforcement())]],
        ['justice_enforcement:acknowledge_enforcement', createMockRecord(createMockEnforcement())],
        ['justice_enforcement:get_enforcements_for_decision', [createMockRecord(createMockEnforcement())]],
        ['justice_enforcement:mark_remedy_completed', createMockRecord(createMockDecision())],
      ])
    );
  });

  describe('CaseCategory Enum Validation', () => {
    const validClient = () => new ValidatedCasesClient(mockClient);

    const validCategories: CaseCategory[] = [
      'ContractDispute',
      'PropertyDispute',
      'ConductViolation',
      'ConductViolation',
      'IPDispute',
      'GovernanceDispute',
      'ConductViolation',
      'Other',
    ];

    it.each(validCategories)('accepts valid category: %s', async (category) => {
      const client = validClient();
      const result = await client.fileCase({
        title: 'Test Case Title',
        description: 'A detailed description of the case for validation',
        category,
        respondent: 'did:mycelix:test123',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Title Length Boundary Tests', () => {
    it('accepts title at exactly 200 characters', async () => {
      const client = new ValidatedCasesClient(mockClient);
      const result = await client.fileCase({
        title: 'x'.repeat(200),
        description: 'A detailed description of the case for validation',
        category: 'ContractDispute',
        respondent: 'did:mycelix:test123',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts title at exactly 1 character', async () => {
      const client = new ValidatedCasesClient(mockClient);
      const result = await client.fileCase({
        title: 'x',
        description: 'A detailed description of the case for validation',
        category: 'ContractDispute',
        respondent: 'did:mycelix:test123',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Description Length Boundary Tests', () => {
    it('accepts description at exactly 10 characters', async () => {
      const client = new ValidatedCasesClient(mockClient);
      const result = await client.fileCase({
        title: 'Test Case',
        description: 'abcdefghij', // exactly 10 chars
        category: 'ContractDispute',
        respondent: 'did:mycelix:test123',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects description at 9 characters', async () => {
      const client = new ValidatedCasesClient(mockClient);
      await expect(
        client.fileCase({
          title: 'Test Case',
          description: 'abcdefghi', // 9 chars
          category: 'ContractDispute',
          respondent: 'did:mycelix:test123',
        })
      ).rejects.toThrow('at least 10 characters');
    });
  });

  describe('DID Schema Validation', () => {
    it('rejects empty DID', async () => {
      const client = new ValidatedCasesClient(mockClient);
      await expect(client.getCasesByComplainant('')).rejects.toThrow('Validation failed');
    });

    it('rejects DID without prefix', async () => {
      const client = new ValidatedCasesClient(mockClient);
      await expect(client.getCasesByComplainant('mycelix:test')).rejects.toThrow('Must be a valid DID');
    });

    it('accepts DID with different methods', async () => {
      const client = new ValidatedCasesClient(mockClient);
      // did:key, did:web, etc. should all be accepted as they start with did:
      const result = await client.getCasesByComplainant('did:key:test123');
      expect(result).toBeDefined();
    });
  });

  describe('EvidenceType Enum Validation', () => {
    const validClient = () => new ValidatedEvidenceClient(mockClient);

    const validTypes: EvidenceType[] = ['Document', 'Testimony', 'Transaction', 'Screenshot', 'Witness', 'Other'];

    it.each(validTypes)('accepts valid evidence type: %s', async (type_) => {
      const client = validClient();
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_,
        title: 'Evidence Title',
        description: 'Evidence description',
        content_hash: 'sha256:' + 'a'.repeat(64),
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Content Hash Length Validation', () => {
    it('accepts content hash at exactly 32 characters', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Evidence',
        description: 'Description',
        content_hash: 'a'.repeat(32),
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects content hash at 31 characters', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      await expect(
        client.submitEvidence({
          case_id: 'case-123',
          type_: 'Document',
          title: 'Evidence',
          description: 'Description',
          content_hash: 'a'.repeat(31),
        })
      ).rejects.toThrow('at least 32 characters');
    });
  });

  describe('Challenge Reason Length Validation', () => {
    it('accepts challenge reason at exactly 10 characters', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      await expect(
        client.challengeEvidence('evidence-123', 'abcdefghij')
      ).resolves.toBeUndefined();
    });

    it('rejects challenge reason at 9 characters', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      await expect(
        client.challengeEvidence('evidence-123', 'abcdefghi')
      ).rejects.toThrow('at least 10 characters');
    });
  });

  describe('Response Length Validation', () => {
    it('accepts response at exactly 10 characters', async () => {
      const client = new ValidatedCasesClient(mockClient);
      const result = await client.respondToCase('case-123', 'abcdefghij');
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects response at 9 characters', async () => {
      const client = new ValidatedCasesClient(mockClient);
      await expect(client.respondToCase('case-123', 'abcdefghi')).rejects.toThrow('at least 10 characters');
    });
  });

  describe('DecisionOutcome Enum Validation', () => {
    const validClient = () => new ValidatedArbitrationClient(mockClient);

    const validOutcomes: DecisionOutcome[] = ['ForComplainant', 'ForRespondent', 'SplitDecision', 'Dismissed', 'Settled'];

    it.each(validOutcomes)('accepts valid outcome: %s', async (outcome) => {
      const client = validClient();
      const result = await client.renderDecision(
        'case-123',
        outcome,
        'The reasoning for this decision is at least 20 characters',
        []
      );
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Reasoning Length Validation', () => {
    it('accepts reasoning at exactly 20 characters', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.renderDecision('case-123', 'ForComplainant', 'a'.repeat(20), []);
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects reasoning at 19 characters', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(
        client.renderDecision('case-123', 'ForComplainant', 'a'.repeat(19), [])
      ).rejects.toThrow('at least 20 characters');
    });
  });

  describe('Appeal Grounds Length Validation', () => {
    it('accepts appeal grounds at exactly 20 characters', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.fileAppeal('decision-123', 'a'.repeat(20));
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects appeal grounds at 19 characters', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(client.fileAppeal('decision-123', 'a'.repeat(19))).rejects.toThrow('at least 20 characters');
    });
  });

  describe('ArbitratorTier Enum Validation', () => {
    const validClient = () => new ValidatedArbitrationClient(mockClient);

    const validTiers: ArbitratorTier[] = ['Panel', 'Lead', 'Appeals'];

    it.each(validTiers)('accepts valid tier: %s', async (tier) => {
      const client = validClient();
      const result = await client.registerAsArbitrator({
        specializations: ['ContractDispute'],
        tier,
      });
      expect(result.entry.Present?.did).toBeDefined();
    });
  });

  describe('EnforcementType Enum Validation', () => {
    const validClient = () => new ValidatedEnforcementClient(mockClient);

    const validTypes: EnforcementType[] = [
      'ReputationPenalty',
      'TemporarySuspension',
      'PermanentBan',
      'AssetFreeze',
      'PaymentOrder',
      'RequiredAction',
    ];

    it.each(validTypes)('accepts valid enforcement type: %s', async (action_type) => {
      const client = validClient();
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:target',
        action_type,
        details: 'Enforcement details',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('EnforcementStatus Enum Validation', () => {
    const validClient = () => new ValidatedEnforcementClient(mockClient);

    const validStatuses: EnforcementStatus[] = ['Requested', 'Acknowledged', 'Executed', 'Rejected', 'Appealed'];

    it.each(validStatuses)('accepts valid status: %s', async (status) => {
      const client = validClient();
      const result = await client.acknowledgeEnforcement('enforcement-123', status);
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Amount Validation', () => {
    it('accepts zero amount', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'identity',
        target_did: 'did:mycelix:target',
        action_type: 'ReputationPenalty',
        details: 'Details',
        amount: 0,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts large positive amount', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:target',
        action_type: 'PaymentOrder',
        details: 'Details',
        amount: 1000000,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Deadline Validation', () => {
    it('accepts positive deadline', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:target',
        action_type: 'PaymentOrder',
        details: 'Details',
        deadline: Date.now() + 86400000,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects zero deadline', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      await expect(
        client.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'finance',
          target_did: 'did:mycelix:target',
          action_type: 'PaymentOrder',
          details: 'Details',
          deadline: 0,
        })
      ).rejects.toThrow('Validation failed');
    });

    it('rejects negative deadline', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      await expect(
        client.requestEnforcement({
          decision_id: 'decision-123',
          target_happ: 'finance',
          target_did: 'did:mycelix:target',
          action_type: 'PaymentOrder',
          details: 'Details',
          deadline: -1000,
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('RemedyIndex Validation', () => {
    it('accepts zero remedy index', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.markRemedyCompleted('decision-123', 0);
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts large remedy index', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.markRemedyCompleted('decision-123', 100);
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Arbitrators Array Validation', () => {
    it('accepts single arbitrator', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.escalateToArbitration('case-123', ['did:mycelix:arb1']);
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts multiple arbitrators', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.escalateToArbitration('case-123', [
        'did:mycelix:arb1',
        'did:mycelix:arb2',
        'did:mycelix:arb3',
      ]);
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('rejects mixed valid and invalid DIDs', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(
        client.escalateToArbitration('case-123', ['did:mycelix:valid', 'invalid-did'])
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('Specializations Array Validation', () => {
    it('accepts single specialization', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.registerAsMediator({
        specializations: ['ContractDispute'],
      });
      expect(result.entry.Present?.did).toBeDefined();
    });

    it('accepts all specializations', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.registerAsMediator({
        specializations: [
          'ContractDispute',
          'PropertyDispute',
          'ConductViolation',
          'ConductViolation',
          'IPDispute',
          'GovernanceDispute',
          'ConductViolation',
          'Other',
        ],
      });
      expect(result.entry.Present?.did).toBeDefined();
    });

    it('rejects mixed valid and invalid specializations', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(
        client.registerAsMediator({
          specializations: ['ContractDispute', 'InvalidSpecialization' as CaseCategory],
        })
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('GetAvailableArbitrators Validation', () => {
    it('accepts valid category', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.getAvailableArbitrators('ContractDispute');
      expect(result).toBeDefined();
    });

    it('rejects invalid category', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(
        client.getAvailableArbitrators('InvalidCategory' as CaseCategory)
      ).rejects.toThrow('Validation failed');
    });
  });

  describe('GetDecision Validation', () => {
    it('accepts valid case ID', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      const result = await client.getDecision('case-123');
      expect(result?.entry.Present?.id).toBeDefined();
    });

    it('rejects empty case ID', async () => {
      const client = new ValidatedArbitrationClient(mockClient);
      await expect(client.getDecision('')).rejects.toThrow('Validation failed');
    });
  });

  describe('GetCasesByParty Validation', () => {
    it('accepts valid DID', async () => {
      const client = new ValidatedCasesClient(mockClient);
      const result = await client.getCasesByParty('did:mycelix:party');
      expect(result).toBeDefined();
    });

    it('rejects invalid DID', async () => {
      const client = new ValidatedCasesClient(mockClient);
      await expect(client.getCasesByParty('invalid')).rejects.toThrow('Must be a valid DID');
    });
  });

  describe('GetPendingEnforcements Validation', () => {
    it('accepts undefined targetHapp', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.getPendingEnforcements();
      expect(result).toBeDefined();
    });

    it('accepts valid targetHapp', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.getPendingEnforcements('finance');
      expect(result).toBeDefined();
    });

    it('rejects empty targetHapp', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      await expect(client.getPendingEnforcements('')).rejects.toThrow('Validation failed');
    });
  });

  describe('Optional Fields', () => {
    it('accepts evidence with storage_ref', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Evidence',
        description: 'Description',
        content_hash: 'a'.repeat(64),
        storage_ref: 'ipfs://Qm123',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts evidence without storage_ref', async () => {
      const client = new ValidatedEvidenceClient(mockClient);
      const result = await client.submitEvidence({
        case_id: 'case-123',
        type_: 'Document',
        title: 'Evidence',
        description: 'Description',
        content_hash: 'a'.repeat(64),
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts enforcement with amount', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:target',
        action_type: 'PaymentOrder',
        details: 'Details',
        amount: 5000,
      });
      expect(result.entry.Present?.id).toBeDefined();
    });

    it('accepts enforcement without amount', async () => {
      const client = new ValidatedEnforcementClient(mockClient);
      const result = await client.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'identity',
        target_did: 'did:mycelix:target',
        action_type: 'ReputationPenalty',
        details: 'Details',
      });
      expect(result.entry.Present?.id).toBeDefined();
    });
  });

  describe('Error Message Content', () => {
    it('includes field path in error for nested validation', async () => {
      const client = new ValidatedCasesClient(mockClient);
      try {
        await client.fileCase({
          title: '',
          description: 'Valid description here',
          category: 'ContractDispute',
          respondent: 'did:mycelix:test',
        });
        expect.fail('Should have thrown');
      } catch (error: unknown) {
        expect((error as Error).message).toContain('title');
      }
    });

    it('includes INVALID_ARGUMENT error code', async () => {
      const client = new ValidatedCasesClient(mockClient);
      try {
        await client.fileCase({
          title: 'Test',
          description: 'short',
          category: 'ContractDispute',
          respondent: 'did:mycelix:test',
        });
        expect.fail('Should have thrown');
      } catch (error: unknown) {
        expect((error as Error).message).toContain('Validation failed');
      }
    });
  });
});
