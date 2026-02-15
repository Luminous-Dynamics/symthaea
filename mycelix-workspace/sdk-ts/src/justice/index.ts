/**
 * Mycelix Justice Module
 *
 * TypeScript client for the mycelix-justice hApp.
 * Provides dispute resolution, arbitration, mediation, and enforcement.
 *
 * @module @mycelix/sdk/justice
 */

// ============================================================================
// Types
// ============================================================================

/** Holochain record wrapper */
export interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
  entry: { Present: T };
}

/** Generic zome call interface */
export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Case Types
// ============================================================================

/** Case phase (matches Rust CasePhase in justice-cases integrity) */
export type CasePhase =
  | 'Filed'
  | 'Negotiation'
  | 'Mediation'
  | 'Arbitration'
  | 'Appeal'
  | 'Enforcement'
  | 'Closed';

/** Case status (matches Rust CaseStatus in justice-cases integrity) */
export type CaseStatus =
  | 'Active'
  | 'OnHold'
  | 'AwaitingResponse'
  | 'InDeliberation'
  | 'DecisionRendered'
  | 'Enforcing'
  | 'Resolved'
  | 'Dismissed'
  | 'Withdrawn';

/** Case category (matches Rust CaseType in justice-cases integrity) */
export type CaseCategory =
  | 'ContractDispute'
  | 'ConductViolation'
  | 'PropertyDispute'
  | 'FinancialDispute'
  | 'GovernanceDispute'
  | 'IdentityDispute'
  | 'IPDispute'
  | 'Other';

/** Evidence type */
export type EvidenceType =
  | 'Document'
  | 'Testimony'
  | 'Transaction'
  | 'Screenshot'
  | 'Witness'
  | 'Other';

/** Decision outcome (matches Rust DecisionOutcome in justice-cases integrity) */
export type DecisionOutcome =
  | 'ForComplainant'
  | 'ForRespondent'
  | 'SplitDecision'
  | 'Dismissed'
  | 'Settled';

/** A dispute case */
export interface Case {
  /** Case ID */
  id: string;
  /** Case title */
  title: string;
  /** Description */
  description: string;
  /** Category */
  category: CaseCategory;
  /** Complainant's DID */
  complainant: string;
  /** Respondent's DID */
  respondent: string;
  /** Current phase */
  phase: CasePhase;
  /** Status */
  status: CaseStatus;
  /** Assigned mediator (if any) */
  mediator?: string;
  /** Assigned arbitrators */
  arbitrators: string[];
  /** Evidence count */
  evidence_count: number;
  /** Filed timestamp */
  filed_at: number;
  /** Last update timestamp */
  updated_at: number;
  /** Resolution timestamp */
  resolved_at?: number;
}

/** Input for filing a case */
export interface FileCaseInput {
  title: string;
  description: string;
  category: CaseCategory;
  respondent: string;
}

/** Evidence submission */
export interface Evidence {
  /** Evidence ID */
  id: string;
  /** Case ID */
  case_id: string;
  /** Submitter's DID */
  submitter: string;
  /** Evidence type */
  type_: EvidenceType;
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** Content hash */
  content_hash: string;
  /** Storage reference */
  storage_ref?: string;
  /** Submission timestamp */
  submitted_at: number;
  /** Verified by arbitrator */
  verified: boolean;
}

/** Input for submitting evidence */
export interface SubmitEvidenceInput {
  case_id: string;
  type_: EvidenceType;
  title: string;
  description: string;
  content_hash: string;
  storage_ref?: string;
}

/** Decision record */
export interface Decision {
  /** Decision ID */
  id: string;
  /** Case ID */
  case_id: string;
  /** Outcome */
  outcome: DecisionOutcome;
  /** Reasoning */
  reasoning: string;
  /** Remedies */
  remedies: Remedy[];
  /** Arbitrators who decided */
  arbitrators: string[];
  /** Can be appealed */
  appealable: boolean;
  /** Appeal deadline */
  appeal_deadline?: number;
  /** Issued timestamp */
  issued_at: number;
  /** Finalized */
  finalized: boolean;
}

/** A remedy in a decision */
export interface Remedy {
  /** Remedy type */
  type_: 'Compensation' | 'Action' | 'Apology' | 'ReputationAdjustment' | 'Ban';
  /** Target DID */
  target: string;
  /** Description */
  description: string;
  /** Amount (if compensation) */
  amount?: number;
  /** Deadline */
  deadline?: number;
  /** Completed */
  completed: boolean;
}

// ============================================================================
// Arbitration Types
// ============================================================================

/** Arbitrator tier */
export type ArbitratorTier = 'Panel' | 'Lead' | 'Appeals';

/** Mediator profile */
export interface MediatorProfile {
  /** DID */
  did: string;
  /** MATL reputation score */
  reputation_score: number;
  /** Cases mediated */
  cases_mediated: number;
  /** Success rate (0-1) */
  success_rate: number;
  /** Specializations */
  specializations: CaseCategory[];
  /** Currently available */
  available: boolean;
  /** Certification timestamp */
  certified_at: number;
}

/** Arbitrator profile */
export interface ArbitratorProfile {
  /** DID */
  did: string;
  /** MATL reputation score */
  reputation_score: number;
  /** Cases arbitrated */
  cases_arbitrated: number;
  /** Appeals overturned */
  appeals_overturned: number;
  /** Specializations */
  specializations: CaseCategory[];
  /** Tier */
  tier: ArbitratorTier;
  /** Currently available */
  available: boolean;
  /** Certification timestamp */
  certified_at: number;
}

/** Input for registering as mediator */
export interface RegisterMediatorInput {
  specializations: CaseCategory[];
}

/** Input for registering as arbitrator */
export interface RegisterArbitratorInput {
  specializations: CaseCategory[];
  tier: ArbitratorTier;
}

// ============================================================================
// Enforcement Types
// ============================================================================

/** Enforcement action type */
export type EnforcementType =
  | 'ReputationPenalty'
  | 'TemporarySuspension'
  | 'PermanentBan'
  | 'AssetFreeze'
  | 'PaymentOrder'
  | 'RequiredAction';

/** Enforcement status */
export type EnforcementStatus = 'Requested' | 'Acknowledged' | 'Executed' | 'Rejected' | 'Appealed';

/** Enforcement action */
export interface Enforcement {
  /** Enforcement ID */
  id: string;
  /** Decision ID */
  decision_id: string;
  /** Target hApp */
  target_happ: string;
  /** Target DID */
  target_did: string;
  /** Action type */
  action_type: EnforcementType;
  /** Details */
  details: string;
  /** Amount */
  amount?: number;
  /** Deadline */
  deadline?: number;
  /** Status */
  status: EnforcementStatus;
  /** Requested timestamp */
  requested_at: number;
  /** Executed timestamp */
  executed_at?: number;
}

/** Input for requesting enforcement */
export interface RequestEnforcementInput {
  decision_id: string;
  target_happ: string;
  target_did: string;
  action_type: EnforcementType;
  details: string;
  amount?: number;
  deadline?: number;
}

// ============================================================================
// Cases Client
// ============================================================================

const JUSTICE_ROLE = 'civic';
const CASES_ZOME = 'justice_cases';

/**
 * Cases Client - File and manage dispute cases
 *
 * @example
 * ```typescript
 * const cases = new CasesClient(conductor);
 *
 * const myCase = await cases.fileCase({
 *   title: 'Contract Breach',
 *   description: 'Vendor failed to deliver...',
 *   category: 'ContractDispute',
 *   respondent: 'did:mycelix:vendor123',
 * });
 * ```
 */
export class CasesClient {
  constructor(private readonly client: ZomeCallable) {}

  /** File a new case */
  async fileCase(input: FileCaseInput): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'file_case',
      payload: input,
    }) as Promise<HolochainRecord<Case>>;
  }

  /** Get case by ID */
  async getCase(caseId: string): Promise<HolochainRecord<Case> | null> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'get_case',
      payload: caseId,
    }) as Promise<HolochainRecord<Case> | null>;
  }

  /** Get cases by complainant */
  async getCasesByComplainant(complainantDid: string): Promise<HolochainRecord<Case>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'get_cases_by_complainant',
      payload: complainantDid,
    }) as Promise<HolochainRecord<Case>[]>;
  }

  /** Get cases by respondent */
  async getCasesByRespondent(respondentDid: string): Promise<HolochainRecord<Case>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'get_cases_by_respondent',
      payload: respondentDid,
    }) as Promise<HolochainRecord<Case>[]>;
  }

  /** Get cases by party (as either complainant or respondent) */
  async getCasesByParty(did: string): Promise<HolochainRecord<Case>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'get_cases_by_party',
      payload: did,
    }) as Promise<HolochainRecord<Case>[]>;
  }

  /** Withdraw a case */
  async withdrawCase(caseId: string, reason: string): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'withdraw_case',
      payload: { case_id: caseId, reason },
    }) as Promise<HolochainRecord<Case>>;
  }

  /** Respond to a case */
  async respondToCase(caseId: string, response: string): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: CASES_ZOME,
      fn_name: 'respond_to_case',
      payload: { case_id: caseId, response },
    }) as Promise<HolochainRecord<Case>>;
  }
}

// ============================================================================
// Evidence Client
// ============================================================================

const EVIDENCE_ZOME = 'justice_evidence';

/**
 * Evidence Client - Submit and manage case evidence
 */
export class EvidenceClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Submit evidence */
  async submitEvidence(input: SubmitEvidenceInput): Promise<HolochainRecord<Evidence>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: EVIDENCE_ZOME,
      fn_name: 'submit_evidence',
      payload: input,
    }) as Promise<HolochainRecord<Evidence>>;
  }

  /** Get evidence for a case */
  async getEvidenceForCase(caseId: string): Promise<HolochainRecord<Evidence>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: EVIDENCE_ZOME,
      fn_name: 'get_evidence_for_case',
      payload: caseId,
    }) as Promise<HolochainRecord<Evidence>[]>;
  }

  /** Verify evidence (arbitrator only) */
  async verifyEvidence(evidenceId: string): Promise<HolochainRecord<Evidence>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: EVIDENCE_ZOME,
      fn_name: 'verify_evidence',
      payload: evidenceId,
    }) as Promise<HolochainRecord<Evidence>>;
  }

  /** Challenge evidence */
  async challengeEvidence(evidenceId: string, reason: string): Promise<void> {
    await this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: EVIDENCE_ZOME,
      fn_name: 'challenge_evidence',
      payload: { evidence_id: evidenceId, reason },
    });
  }
}

// ============================================================================
// Arbitration Client
// ============================================================================

const ARBITRATION_ZOME = 'justice_arbitration';

/**
 * Arbitration Client - Manage mediators, arbitrators, and decisions
 */
export class ArbitrationClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Register as mediator */
  async registerAsMediator(
    input: RegisterMediatorInput
  ): Promise<HolochainRecord<MediatorProfile>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'register_as_mediator',
      payload: input,
    }) as Promise<HolochainRecord<MediatorProfile>>;
  }

  /** Register as arbitrator */
  async registerAsArbitrator(
    input: RegisterArbitratorInput
  ): Promise<HolochainRecord<ArbitratorProfile>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'register_as_arbitrator',
      payload: input,
    }) as Promise<HolochainRecord<ArbitratorProfile>>;
  }

  /** Get available mediators for a category */
  async getAvailableMediators(category: CaseCategory): Promise<HolochainRecord<MediatorProfile>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'get_available_mediators',
      payload: category,
    }) as Promise<HolochainRecord<MediatorProfile>[]>;
  }

  /** Get available arbitrators for a category */
  async getAvailableArbitrators(
    category: CaseCategory
  ): Promise<HolochainRecord<ArbitratorProfile>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'get_available_arbitrators',
      payload: category,
    }) as Promise<HolochainRecord<ArbitratorProfile>[]>;
  }

  /** Assign mediator to case */
  async assignMediator(caseId: string, mediatorDid: string): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'assign_mediator',
      payload: { case_id: caseId, mediator_did: mediatorDid },
    }) as Promise<HolochainRecord<Case>>;
  }

  /** Escalate to arbitration */
  async escalateToArbitration(
    caseId: string,
    arbitratorDids: string[]
  ): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'escalate_to_arbitration',
      payload: { case_id: caseId, arbitrator_dids: arbitratorDids },
    }) as Promise<HolochainRecord<Case>>;
  }

  /** Render decision */
  async renderDecision(
    caseId: string,
    outcome: DecisionOutcome,
    reasoning: string,
    remedies: Remedy[]
  ): Promise<HolochainRecord<Decision>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'render_decision',
      payload: { case_id: caseId, outcome, reasoning, remedies },
    }) as Promise<HolochainRecord<Decision>>;
  }

  /** Get decision for a case */
  async getDecision(caseId: string): Promise<HolochainRecord<Decision> | null> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'get_decision',
      payload: caseId,
    }) as Promise<HolochainRecord<Decision> | null>;
  }

  /** File appeal */
  async fileAppeal(decisionId: string, grounds: string): Promise<HolochainRecord<Case>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ARBITRATION_ZOME,
      fn_name: 'file_appeal',
      payload: { decision_id: decisionId, grounds },
    }) as Promise<HolochainRecord<Case>>;
  }
}

// ============================================================================
// Enforcement Client
// ============================================================================

const ENFORCEMENT_ZOME = 'justice_enforcement';

/**
 * Enforcement Client - Request and track enforcement actions
 */
export class EnforcementClient {
  constructor(private readonly client: ZomeCallable) {}

  /** Request enforcement */
  async requestEnforcement(input: RequestEnforcementInput): Promise<HolochainRecord<Enforcement>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ENFORCEMENT_ZOME,
      fn_name: 'request_enforcement',
      payload: input,
    }) as Promise<HolochainRecord<Enforcement>>;
  }

  /** Get pending enforcements for a hApp */
  async getPendingEnforcements(targetHapp?: string): Promise<HolochainRecord<Enforcement>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ENFORCEMENT_ZOME,
      fn_name: 'get_pending_enforcements',
      payload: targetHapp ?? null,
    }) as Promise<HolochainRecord<Enforcement>[]>;
  }

  /** Acknowledge enforcement */
  async acknowledgeEnforcement(
    enforcementId: string,
    status: EnforcementStatus
  ): Promise<HolochainRecord<Enforcement>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ENFORCEMENT_ZOME,
      fn_name: 'acknowledge_enforcement',
      payload: { enforcement_id: enforcementId, status },
    }) as Promise<HolochainRecord<Enforcement>>;
  }

  /** Get enforcements for a decision */
  async getEnforcementsForDecision(decisionId: string): Promise<HolochainRecord<Enforcement>[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ENFORCEMENT_ZOME,
      fn_name: 'get_enforcements_for_decision',
      payload: decisionId,
    }) as Promise<HolochainRecord<Enforcement>[]>;
  }

  /** Mark remedy as completed */
  async markRemedyCompleted(
    decisionId: string,
    remedyIndex: number
  ): Promise<HolochainRecord<Decision>> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: ENFORCEMENT_ZOME,
      fn_name: 'mark_remedy_completed',
      payload: { decision_id: decisionId, remedy_index: remedyIndex },
    }) as Promise<HolochainRecord<Decision>>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create all Justice hApp clients
 *
 * @example
 * ```typescript
 * const { cases, evidence, arbitration, enforcement } = createJusticeClients(conductor);
 * ```
 */
export function createJusticeClients(client: ZomeCallable) {
  return {
    cases: new CasesClient(client),
    evidence: new EvidenceClient(client),
    arbitration: new ArbitrationClient(client),
    enforcement: new EnforcementClient(client),
  };
}

// Unified client
export { MycelixJusticeClient, JusticeSdkError } from './client';
export type { JusticeClientConfig, JusticeConnectionOptions } from './client';

export default {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  createJusticeClients,
};
