/**
 * Justice hApp Conductor Integration Tests
 *
 * These tests verify the Justice clients work correctly with a real
 * Holochain conductor. They require the conductor harness to be available.
 *
 * Run with: npm run test:conductor
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  createJusticeClients,
  type ZomeCallable,
  type Case,
  type Evidence,
  type MediatorProfile,
  type ArbitratorProfile,
  type Decision,
  type Enforcement,
} from '../../src/justice/index.js';
import {
  createValidatedJusticeClients,
  ValidatedCasesClient,
  ValidatedEvidenceClient,
  ValidatedArbitrationClient,
  ValidatedEnforcementClient,
} from '../../src/justice/validated.js';
import { MycelixError, ErrorCode } from '../../src/errors.js';
import { CONDUCTOR_ENABLED, generateTestAgentId, waitForSync } from './conductor-harness.js';

// Mock client for testing without conductor
function createMockClient(): ZomeCallable {
  const cases = new Map<string, Case>();
  const evidence = new Map<string, Evidence[]>();
  const mediators = new Map<string, MediatorProfile>();
  const arbitrators = new Map<string, ArbitratorProfile>();
  const decisions = new Map<string, Decision>();
  const enforcements = new Map<string, Enforcement[]>();
  // Counter to ensure unique IDs (Date.now() can collide in fast-running tests)
  let idCounter = 0;

  return {
    async callZome({
      fn_name,
      payload,
    }: {
      role_name: string;
      zome_name: string;
      fn_name: string;
      payload: unknown;
    }) {
      const agentId = `did:mycelix:${generateTestAgentId()}`;

      switch (fn_name) {
        case 'file_case': {
          const input = payload as {
            title: string;
            description: string;
            category: string;
            respondent: string;
          };
          const caseRecord: Case = {
            id: `case-${Date.now()}-${++idCounter}`,
            complainant: agentId,
            respondent: input.respondent,
            title: input.title,
            description: input.description,
            category: input.category as Case['category'],
            status: 'Filed',
            mediator: undefined,
            arbitrators: [],
            created_at: Date.now(),
            updated_at: Date.now(),
          };
          cases.set(caseRecord.id, caseRecord);
          return {
            signed_action: { hashed: { hash: caseRecord.id, content: {} }, signature: 'sig' },
            entry: { Present: caseRecord },
          };
        }

        case 'get_case': {
          const id = payload as string;
          const c = cases.get(id);
          return c
            ? {
                signed_action: { hashed: { hash: id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'get_cases_by_complainant': {
          const did = payload as string;
          return Array.from(cases.values())
            .filter((c) => c.complainant === did)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'get_cases_by_respondent': {
          const did = payload as string;
          return Array.from(cases.values())
            .filter((c) => c.respondent === did)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'get_cases_by_party': {
          const did = payload as string;
          return Array.from(cases.values())
            .filter((c) => c.complainant === did || c.respondent === did)
            .map((c) => ({
              signed_action: { hashed: { hash: c.id, content: {} }, signature: 'sig' },
              entry: { Present: c },
            }));
        }

        case 'withdraw_case': {
          const { case_id, reason } = payload as { case_id: string; reason: string };
          const c = cases.get(case_id);
          if (c) {
            c.status = 'Withdrawn';
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: case_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'respond_to_case': {
          const { case_id } = payload as { case_id: string; response: string };
          const c = cases.get(case_id);
          if (c) {
            c.status = 'Responded';
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: case_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'submit_evidence': {
          const input = payload as {
            case_id: string;
            type_: string;
            title: string;
            description: string;
            content_hash: string;
            storage_ref?: string;
          };
          const ev: Evidence = {
            id: `ev-${Date.now()}-${++idCounter}`,
            case_id: input.case_id,
            submitter: agentId,
            type_: input.type_ as Evidence['type_'],
            title: input.title,
            description: input.description,
            content_hash: input.content_hash,
            storage_ref: input.storage_ref,
            verified: false,
            submitted_at: Date.now(),
          };
          const existing = evidence.get(input.case_id) || [];
          existing.push(ev);
          evidence.set(input.case_id, existing);
          return {
            signed_action: { hashed: { hash: ev.id, content: {} }, signature: 'sig' },
            entry: { Present: ev },
          };
        }

        case 'get_evidence_for_case': {
          const caseId = payload as string;
          const evs = evidence.get(caseId) || [];
          return evs.map((e) => ({
            signed_action: { hashed: { hash: e.id, content: {} }, signature: 'sig' },
            entry: { Present: e },
          }));
        }

        case 'verify_evidence': {
          const evId = payload as string;
          for (const evs of evidence.values()) {
            const ev = evs.find((e) => e.id === evId);
            if (ev) {
              ev.verified = true;
              return {
                signed_action: { hashed: { hash: evId, content: {} }, signature: 'sig' },
                entry: { Present: ev },
              };
            }
          }
          return null;
        }

        case 'register_as_mediator': {
          const input = payload as { specializations: string[] };
          const profile: MediatorProfile = {
            did: agentId,
            specializations: input.specializations as MediatorProfile['specializations'],
            cases_mediated: 0,
            success_rate: 0,
            available: true,
            registered_at: Date.now(),
          };
          mediators.set(agentId, profile);
          return {
            signed_action: { hashed: { hash: agentId, content: {} }, signature: 'sig' },
            entry: { Present: profile },
          };
        }

        case 'register_as_arbitrator': {
          const input = payload as { specializations: string[]; tier: string };
          const profile: ArbitratorProfile = {
            did: agentId,
            specializations: input.specializations as ArbitratorProfile['specializations'],
            tier: input.tier as ArbitratorProfile['tier'],
            cases_arbitrated: 0,
            reputation_score: 0.5,
            available: true,
            registered_at: Date.now(),
          };
          arbitrators.set(agentId, profile);
          return {
            signed_action: { hashed: { hash: agentId, content: {} }, signature: 'sig' },
            entry: { Present: profile },
          };
        }

        case 'get_available_mediators': {
          const category = payload as string;
          return Array.from(mediators.values())
            .filter((m) => m.available && m.specializations.includes(category as any))
            .map((m) => ({
              signed_action: { hashed: { hash: m.did, content: {} }, signature: 'sig' },
              entry: { Present: m },
            }));
        }

        case 'get_available_arbitrators': {
          const category = payload as string;
          return Array.from(arbitrators.values())
            .filter((a) => a.available && a.specializations.includes(category as any))
            .map((a) => ({
              signed_action: { hashed: { hash: a.did, content: {} }, signature: 'sig' },
              entry: { Present: a },
            }));
        }

        case 'assign_mediator': {
          const { case_id, mediator_did } = payload as { case_id: string; mediator_did: string };
          const c = cases.get(case_id);
          if (c) {
            c.mediator = mediator_did;
            c.status = 'InMediation';
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: case_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'escalate_to_arbitration': {
          const { case_id, arbitrator_dids } = payload as {
            case_id: string;
            arbitrator_dids: string[];
          };
          const c = cases.get(case_id);
          if (c) {
            c.arbitrators = arbitrator_dids;
            c.status = 'InArbitration';
            c.updated_at = Date.now();
          }
          return c
            ? {
                signed_action: { hashed: { hash: case_id, content: {} }, signature: 'sig' },
                entry: { Present: c },
              }
            : null;
        }

        case 'render_decision': {
          const { case_id, outcome, reasoning, remedies } = payload as {
            case_id: string;
            outcome: string;
            reasoning: string;
            remedies: any[];
          };
          const decision: Decision = {
            id: `decision-${Date.now()}-${++idCounter}`,
            case_id,
            outcome: outcome as Decision['outcome'],
            reasoning,
            remedies,
            rendered_by: agentId,
            rendered_at: Date.now(),
            appealable_until: Date.now() + 30 * 24 * 60 * 60 * 1000,
          };
          decisions.set(case_id, decision);
          const c = cases.get(case_id);
          if (c) {
            c.status = 'Decided';
            c.updated_at = Date.now();
          }
          return {
            signed_action: { hashed: { hash: decision.id, content: {} }, signature: 'sig' },
            entry: { Present: decision },
          };
        }

        case 'get_decision': {
          const caseId = payload as string;
          const d = decisions.get(caseId);
          return d
            ? {
                signed_action: { hashed: { hash: d.id, content: {} }, signature: 'sig' },
                entry: { Present: d },
              }
            : null;
        }

        case 'request_enforcement': {
          const input = payload as {
            decision_id: string;
            target_happ: string;
            target_did: string;
            action_type: string;
            details: string;
            amount?: number;
            deadline?: number;
          };
          const enforcement: Enforcement = {
            id: `enf-${Date.now()}-${++idCounter}`,
            decision_id: input.decision_id,
            target_happ: input.target_happ,
            target_did: input.target_did,
            action_type: input.action_type as Enforcement['action_type'],
            details: input.details,
            amount: input.amount,
            deadline: input.deadline,
            status: 'Requested',
            requested_at: Date.now(),
          };
          const existing = enforcements.get(input.decision_id) || [];
          existing.push(enforcement);
          enforcements.set(input.decision_id, existing);
          return {
            signed_action: { hashed: { hash: enforcement.id, content: {} }, signature: 'sig' },
            entry: { Present: enforcement },
          };
        }

        case 'get_pending_enforcements': {
          const targetHapp = payload as string | undefined;
          const all: Enforcement[] = [];
          for (const enfs of enforcements.values()) {
            for (const e of enfs) {
              if (e.status === 'Requested' && (!targetHapp || e.target_happ === targetHapp)) {
                all.push(e);
              }
            }
          }
          return all.map((e) => ({
            signed_action: { hashed: { hash: e.id, content: {} }, signature: 'sig' },
            entry: { Present: e },
          }));
        }

        case 'acknowledge_enforcement': {
          const { enforcement_id, status } = payload as { enforcement_id: string; status: string };
          for (const enfs of enforcements.values()) {
            const e = enfs.find((ef) => ef.id === enforcement_id);
            if (e) {
              e.status = status as Enforcement['status'];
              return {
                signed_action: { hashed: { hash: enforcement_id, content: {} }, signature: 'sig' },
                entry: { Present: e },
              };
            }
          }
          return null;
        }

        case 'get_enforcements_for_decision': {
          const decisionId = payload as string;
          const enfs = enforcements.get(decisionId) || [];
          return enfs.map((e) => ({
            signed_action: { hashed: { hash: e.id, content: {} }, signature: 'sig' },
            entry: { Present: e },
          }));
        }

        default:
          throw new Error(`Unknown function: ${fn_name}`);
      }
    },
  };
}

// Use mock client for unit tests, real conductor for integration tests
const describeConductor = CONDUCTOR_ENABLED ? describe : describe.skip;
const describeUnit = describe;

describeUnit('Justice Clients (Mock)', () => {
  let mockClient: ZomeCallable;
  let casesClient: CasesClient;
  let evidenceClient: EvidenceClient;
  let arbitrationClient: ArbitrationClient;
  let enforcementClient: EnforcementClient;

  beforeAll(() => {
    mockClient = createMockClient();
    const clients = createJusticeClients(mockClient);
    casesClient = clients.cases;
    evidenceClient = clients.evidence;
    arbitrationClient = clients.arbitration;
    enforcementClient = clients.enforcement;
  });

  describe('CasesClient', () => {
    it('should file a case', async () => {
      const result = await casesClient.fileCase({
        title: 'Test Case',
        description: 'This is a test case for contract breach',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent123',
      });

      expect(result).toBeDefined();
      expect(result.entry.Present).toBeDefined();
      const caseData = result.entry.Present as Case;
      expect(caseData.title).toBe('Test Case');
      expect(caseData.category).toBe('ContractDispute');
      expect(caseData.status).toBe('Filed');
    });

    it('should get a case by ID', async () => {
      const filed = await casesClient.fileCase({
        title: 'Retrieval Test',
        description: 'Test case for retrieval',
        category: 'PropertyDispute',
        respondent: 'did:mycelix:respondent456',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await casesClient.getCase(caseId);

      expect(result).toBeDefined();
      expect((result!.entry.Present as Case).id).toBe(caseId);
    });

    it('should respond to a case', async () => {
      const filed = await casesClient.fileCase({
        title: 'Response Test',
        description: 'Test case for response',
        category: 'ConductViolation',
        respondent: 'did:mycelix:respondent789',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await casesClient.respondToCase(
        caseId,
        'I dispute these allegations and will provide evidence.'
      );

      expect(result).toBeDefined();
      expect((result.entry.Present as Case).status).toBe('Responded');
    });

    it('should withdraw a case', async () => {
      const filed = await casesClient.fileCase({
        title: 'Withdrawal Test',
        description: 'Test case for withdrawal',
        category: 'ConductViolation',
        respondent: 'did:mycelix:respondent000',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await casesClient.withdrawCase(caseId, 'Resolved privately');

      expect(result).toBeDefined();
      expect((result.entry.Present as Case).status).toBe('Withdrawn');
    });
  });

  describe('EvidenceClient', () => {
    it('should submit evidence', async () => {
      const filed = await casesClient.fileCase({
        title: 'Evidence Test',
        description: 'Case with evidence',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent111',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await evidenceClient.submitEvidence({
        case_id: caseId,
        type_: 'Document',
        title: 'Contract Copy',
        description: 'The original signed contract',
        content_hash: 'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
        storage_ref: 'ipfs://QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG',
      });

      expect(result).toBeDefined();
      const ev = result.entry.Present as Evidence;
      expect(ev.type_).toBe('Document');
      expect(ev.verified).toBe(false);
    });

    it('should get evidence for a case', async () => {
      const filed = await casesClient.fileCase({
        title: 'Multi Evidence Test',
        description: 'Case with multiple evidence',
        category: 'ConductViolation',
        respondent: 'did:mycelix:respondent222',
      });
      const caseId = (filed.entry.Present as Case).id;

      await evidenceClient.submitEvidence({
        case_id: caseId,
        type_: 'Screenshot',
        title: 'Transaction Screenshot',
        description: 'Screenshot of fraudulent transaction',
        content_hash: 'hash1',
      });

      await evidenceClient.submitEvidence({
        case_id: caseId,
        type_: 'Transaction',
        title: 'Blockchain Record',
        description: 'On-chain transaction record',
        content_hash: 'hash2',
      });

      const result = await evidenceClient.getEvidenceForCase(caseId);

      expect(result).toHaveLength(2);
    });

    it('should verify evidence', async () => {
      const filed = await casesClient.fileCase({
        title: 'Verify Evidence Test',
        description: 'Test evidence verification',
        category: 'PropertyDispute',
        respondent: 'did:mycelix:respondent333',
      });
      const caseId = (filed.entry.Present as Case).id;

      const submitted = await evidenceClient.submitEvidence({
        case_id: caseId,
        type_: 'Document',
        title: 'Deed',
        description: 'Property deed',
        content_hash: 'hash3',
      });
      const evId = (submitted.entry.Present as Evidence).id;

      const verified = await evidenceClient.verifyEvidence(evId);

      expect((verified.entry.Present as Evidence).verified).toBe(true);
    });
  });

  describe('ArbitrationClient', () => {
    it('should register as a mediator', async () => {
      const result = await arbitrationClient.registerAsMediator({
        specializations: ['ContractDispute', 'PropertyDispute'],
      });

      expect(result).toBeDefined();
      const profile = result.entry.Present as MediatorProfile;
      expect(profile.specializations).toContain('ContractDispute');
      expect(profile.available).toBe(true);
    });

    it('should register as an arbitrator', async () => {
      const result = await arbitrationClient.registerAsArbitrator({
        specializations: ['ConductViolation', 'IPDispute'],
        tier: 'Panel',
      });

      expect(result).toBeDefined();
      const profile = result.entry.Present as ArbitratorProfile;
      expect(profile.tier).toBe('Panel');
      expect(profile.available).toBe(true);
    });

    it('should assign a mediator to a case', async () => {
      await arbitrationClient.registerAsMediator({
        specializations: ['ConductViolation'],
      });

      const filed = await casesClient.fileCase({
        title: 'Mediation Test',
        description: 'Test mediation assignment',
        category: 'ConductViolation',
        respondent: 'did:mycelix:respondent444',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await arbitrationClient.assignMediator(caseId, 'did:mycelix:mediator1');

      expect(result).toBeDefined();
      expect((result.entry.Present as Case).status).toBe('InMediation');
    });

    it('should escalate to arbitration', async () => {
      const filed = await casesClient.fileCase({
        title: 'Arbitration Escalation',
        description: 'Test arbitration escalation',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent555',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await arbitrationClient.escalateToArbitration(caseId, [
        'did:mycelix:arb1',
        'did:mycelix:arb2',
        'did:mycelix:arb3',
      ]);

      expect(result).toBeDefined();
      expect((result.entry.Present as Case).status).toBe('InArbitration');
      expect((result.entry.Present as Case).arbitrators).toHaveLength(3);
    });

    it('should render a decision', async () => {
      const filed = await casesClient.fileCase({
        title: 'Decision Test',
        description: 'Test decision rendering',
        category: 'ContractDispute',
        respondent: 'did:mycelix:respondent666',
      });
      const caseId = (filed.entry.Present as Case).id;

      const result = await arbitrationClient.renderDecision(
        caseId,
        'ForComplainant',
        'The evidence clearly shows a breach of contract. The respondent failed to deliver services as agreed.',
        [
          {
            type_: 'Compensation',
            target: 'did:mycelix:respondent666',
            description: 'Pay damages',
            amount: 5000,
            deadline: Date.now() + 30 * 24 * 60 * 60 * 1000,
            completed: false,
          },
        ]
      );

      expect(result).toBeDefined();
      const decision = result.entry.Present as Decision;
      expect(decision.outcome).toBe('ForComplainant');
      expect(decision.remedies).toHaveLength(1);
    });
  });

  describe('EnforcementClient', () => {
    it('should request enforcement', async () => {
      const result = await enforcementClient.requestEnforcement({
        decision_id: 'decision-123',
        target_happ: 'finance',
        target_did: 'did:mycelix:target',
        action_type: 'PaymentOrder',
        details: 'Transfer 5000 MYC to complainant',
        amount: 5000,
        deadline: Date.now() + 7 * 24 * 60 * 60 * 1000,
      });

      expect(result).toBeDefined();
      const enforcement = result.entry.Present as Enforcement;
      expect(enforcement.action_type).toBe('PaymentOrder');
      expect(enforcement.status).toBe('Requested');
    });

    it('should get pending enforcements', async () => {
      await enforcementClient.requestEnforcement({
        decision_id: 'decision-pending',
        target_happ: 'governance',
        target_did: 'did:mycelix:pending',
        action_type: 'TemporarySuspension',
        details: 'Suspend voting rights for 30 days',
      });

      const result = await enforcementClient.getPendingEnforcements();

      expect(result.length).toBeGreaterThan(0);
    });

    it('should acknowledge enforcement', async () => {
      const requested = await enforcementClient.requestEnforcement({
        decision_id: 'decision-ack',
        target_happ: 'identity',
        target_did: 'did:mycelix:ack',
        action_type: 'ReputationPenalty',
        details: 'Reduce reputation score by 20%',
      });
      const enfId = (requested.entry.Present as Enforcement).id;

      const result = await enforcementClient.acknowledgeEnforcement(enfId, 'Executed');

      expect(result).toBeDefined();
      expect((result.entry.Present as Enforcement).status).toBe('Executed');
    });
  });
});

describe('Validated Justice Clients', () => {
  let mockClient: ZomeCallable;
  let validatedClients: ReturnType<typeof createValidatedJusticeClients>;

  beforeAll(() => {
    mockClient = createMockClient();
    validatedClients = createValidatedJusticeClients(mockClient);
  });

  it('should reject invalid case input', async () => {
    await expect(
      validatedClients.cases.fileCase({
        title: '', // Empty title should fail
        description: 'short', // Too short description
        category: 'ContractDispute',
        respondent: 'not-a-did', // Invalid DID
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid evidence input', async () => {
    await expect(
      validatedClients.evidence.submitEvidence({
        case_id: '', // Empty case ID
        type_: 'Document',
        title: '',
        description: '',
        content_hash: 'short', // Too short hash
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid arbitrator registration', async () => {
    await expect(
      validatedClients.arbitration.registerAsArbitrator({
        specializations: [], // Empty specializations
        tier: 'Panel',
      })
    ).rejects.toThrow(MycelixError);
  });

  it('should reject invalid enforcement request', async () => {
    await expect(
      validatedClients.enforcement.requestEnforcement({
        decision_id: '',
        target_happ: '',
        target_did: 'invalid',
        action_type: 'PaymentOrder',
        details: '',
      })
    ).rejects.toThrow(MycelixError);
  });
});

describeConductor('Justice Conductor Integration Tests', () => {
  // Real conductor tests would go here
  // They follow the same patterns but use real Holochain conductor
  it.todo('should file a case on the real conductor');
  it.todo('should submit evidence with actual content addressing');
  it.todo('should render binding decisions');
  it.todo('should enforce decisions across hApps');
});
