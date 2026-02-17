/**
 * @mycelix/sdk Justice Integration
 *
 * hApp-specific adapter for Mycelix-Justice providing:
 * - Dispute resolution with tiered escalation
 * - Community mediation and arbitration
 * - Restorative justice circles
 * - Cross-hApp enforcement coordination via Bridge
 * - Reputation-based arbitrator selection
 *
 * @packageDocumentation
 * @module integrations/justice
 */


import { LocalBridge } from '../../bridge/index.js';
import { type MycelixClient } from '../../client/index.js';
import {
  createReputation,
  recordPositive,
  isTrustworthy,
  type ReputationScore,
} from '../../matl/index.js';
import { JusticeValidators } from '../../utils/validation.js';

// ============================================================================
// Bridge Zome Types (matching Rust justice_bridge zome)
// ============================================================================

/** Dispute types for cross-hApp disputes */
export type CrossHappDisputeType =
  | 'ContractBreach'
  | 'Fraud'
  | 'PropertyDispute'
  | 'Defamation'
  | 'CopyrightViolation'
  | 'TokenDispute'
  | 'GovernanceViolation'
  | 'Other';

/** Enforcement action types */
export type EnforcementType =
  | 'ReputationPenalty'
  | 'TemporarySuspension'
  | 'PermanentBan'
  | 'AssetFreeze'
  | 'PaymentOrder'
  | 'RequiredAction';

/** Enforcement status */
export type EnforcementStatus = 'Requested' | 'Acknowledged' | 'Executed' | 'Rejected' | 'Appealed';

/** Cross-hApp dispute */
export interface CrossHappDispute {
  id: string;
  dispute_type: CrossHappDisputeType;
  complainant_did: string;
  respondent_did: string;
  source_happ: string;
  related_happs: string[];
  title: string;
  description: string;
  evidence_hashes: string[];
  status: CaseStatus;
  filed_at: number;
}

/** Enforcement action */
export interface EnforcementAction {
  id: string;
  dispute_id: string;
  target_happ: string;
  target_did: string;
  action_type: EnforcementType;
  details: string;
  amount?: number;
  deadline?: number;
  status: EnforcementStatus;
  requested_at: number;
  executed_at?: number;
}

/** Justice bridge event types */
export type JusticeBridgeEventType =
  | 'DisputeFiled'
  | 'DisputeResolved'
  | 'EnforcementRequested'
  | 'EnforcementExecuted'
  | 'AppealFiled'
  | 'MediationInitiated';

/** Justice bridge event */
export interface JusticeBridgeEvent {
  id: string;
  event_type: JusticeBridgeEventType;
  dispute_id?: string;
  enforcement_id?: string;
  payload: string;
  source_happ: string;
  timestamp: number;
}

/** File cross-hApp dispute input */
export interface FileCrossHappDisputeInput {
  dispute_type: CrossHappDisputeType;
  respondent_did: string;
  related_happs: string[];
  title: string;
  description: string;
  evidence_hashes: string[];
}

/** Request enforcement input */
export interface RequestEnforcementInput {
  dispute_id: string;
  target_happ: string;
  target_did: string;
  action_type: EnforcementType;
  details: string;
  amount?: number;
  deadline?: number;
}

/** Dispute history query */
export interface DisputeHistoryQuery {
  did: string;
  role: 'complainant' | 'respondent' | 'both';
  limit?: number;
}

// ============================================================================
// Justice-Specific Types
// ============================================================================

/** Case phase in justice process (matches Rust CasePhase in justice-cases integrity) */
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

/** Decision outcome (matches Rust DecisionOutcome in justice-cases integrity) */
export type DecisionOutcome = 'ForComplainant' | 'ForRespondent' | 'SplitDecision' | 'Dismissed' | 'Settled';

/** Dispute case */
export interface Case {
  id: string;
  title: string;
  description: string;
  category: CaseCategory;
  complainantId: string;
  respondentId: string;
  phase: CasePhase;
  status: CaseStatus;
  evidence: Evidence[];
  timeline: TimelineEntry[];
  filedAt: number;
  updatedAt: number;
  resolvedAt?: number;
}

/** Evidence submission */
export interface Evidence {
  id: string;
  caseId: string;
  submitterId: string;
  type: 'document' | 'testimony' | 'transaction' | 'screenshot' | 'witness';
  title: string;
  description: string;
  contentHash: string;
  storageRef?: string;
  submittedAt: number;
  verified: boolean;
}

/** Timeline entry */
export interface TimelineEntry {
  timestamp: number;
  action: string;
  actor: string;
  details?: string;
}

/** Mediator profile */
export interface MediatorProfile {
  did: string;
  reputation: ReputationScore;
  casesMediated: number;
  successRate: number;
  specializations: CaseCategory[];
  available: boolean;
  certifiedAt: number;
}

/** Arbitrator profile */
export interface ArbitratorProfile {
  did: string;
  reputation: ReputationScore;
  casesArbitrated: number;
  appealsOverturned: number;
  specializations: CaseCategory[];
  tier: 1 | 2 | 3; // 1 = panel member, 2 = lead arbitrator, 3 = appeals
  available: boolean;
  certifiedAt: number;
}

/** Decision record */
export interface Decision {
  id: string;
  caseId: string;
  outcome: DecisionOutcome;
  reasoning: string;
  remedies: Remedy[];
  arbitrators: string[];
  appealable: boolean;
  appealDeadline?: number;
  issuedAt: number;
  finalized: boolean;
}

/** Remedy specification */
export interface Remedy {
  type: 'compensation' | 'action' | 'apology' | 'reputation_adjustment' | 'ban';
  targetId: string;
  description: string;
  amount?: number;
  deadline?: number;
  completed: boolean;
}

/** Restorative circle */
export interface RestorativeCircle {
  id: string;
  caseId: string;
  facilitatorId: string;
  participants: CircleParticipant[];
  sessions: CircleSession[];
  agreements: string[];
  status: 'forming' | 'active' | 'completed' | 'abandoned';
  createdAt: number;
}

/** Circle participant */
export interface CircleParticipant {
  did: string;
  role: 'harmed' | 'responsible' | 'supporter' | 'community';
  consented: boolean;
  attendedSessions: number[];
}

/** Circle session */
export interface CircleSession {
  number: number;
  scheduledAt: number;
  completedAt?: number;
  notes?: string;
}

// ============================================================================
// Justice Service
// ============================================================================

/**
 * Justice service for dispute resolution and enforcement
 */
export class JusticeService {
  private cases = new Map<string, Case>();
  private evidence = new Map<string, Evidence[]>();
  private mediators = new Map<string, MediatorProfile>();
  private arbitrators = new Map<string, ArbitratorProfile>();
  private decisions = new Map<string, Decision>();
  private circles = new Map<string, RestorativeCircle>();
  private bridge: LocalBridge;

  constructor() {
    this.bridge = new LocalBridge();
    this.bridge.registerHapp('justice');
  }

  /**
   * File a new case
   */
  fileCase(
    complainantId: string,
    respondentId: string,
    title: string,
    description: string,
    category: CaseCategory
  ): Case {
    // Validate complainant and respondent DIDs are not empty
    JusticeValidators.case(complainantId, respondentId);

    const id = `case-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const caseRecord: Case = {
      id,
      title,
      description,
      category,
      complainantId,
      respondentId,
      phase: 'Filed',
      status: 'Active',
      evidence: [],
      timeline: [
        {
          timestamp: Date.now(),
          action: 'Case filed',
          actor: complainantId,
        },
      ],
      filedAt: Date.now(),
      updatedAt: Date.now(),
    };

    this.cases.set(id, caseRecord);
    this.evidence.set(id, []);
    return caseRecord;
  }

  /**
   * Submit evidence for a case
   */
  submitEvidence(
    caseId: string,
    submitterId: string,
    type: Evidence['type'],
    title: string,
    description: string,
    contentHash: string,
    storageRef?: string
  ): Evidence {
    // Validate content hash is not empty
    JusticeValidators.evidence(contentHash);

    const caseRecord = this.cases.get(caseId);
    if (!caseRecord) throw new Error('Case not found');

    const evidenceItem: Evidence = {
      id: `evidence-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      caseId,
      submitterId,
      type,
      title,
      description,
      contentHash,
      storageRef,
      submittedAt: Date.now(),
      verified: false,
    };

    const caseEvidence = this.evidence.get(caseId) || [];
    caseEvidence.push(evidenceItem);
    this.evidence.set(caseId, caseEvidence);

    caseRecord.evidence.push(evidenceItem);
    caseRecord.timeline.push({
      timestamp: Date.now(),
      action: 'Evidence submitted',
      actor: submitterId,
      details: title,
    });

    return evidenceItem;
  }

  /**
   * Initiate mediation
   */
  initiateMediation(caseId: string, mediatorId: string): Case {
    const caseRecord = this.cases.get(caseId);
    if (!caseRecord) throw new Error('Case not found');

    const mediator = this.mediators.get(mediatorId);
    if (!mediator || !mediator.available) {
      throw new Error('Mediator not available');
    }

    caseRecord.phase = 'Mediation';
    caseRecord.status = 'Active';
    caseRecord.timeline.push({
      timestamp: Date.now(),
      action: 'Mediation initiated',
      actor: mediatorId,
    });
    caseRecord.updatedAt = Date.now();

    return caseRecord;
  }

  /**
   * Escalate to arbitration
   */
  escalateToArbitration(caseId: string, arbitratorIds: string[]): Case {
    const caseRecord = this.cases.get(caseId);
    if (!caseRecord) throw new Error('Case not found');

    // Verify arbitrators
    for (const arbId of arbitratorIds) {
      const arb = this.arbitrators.get(arbId);
      if (!arb || !arb.available) {
        throw new Error(`Arbitrator ${arbId} not available`);
      }
    }

    caseRecord.phase = 'Arbitration';
    caseRecord.timeline.push({
      timestamp: Date.now(),
      action: 'Escalated to arbitration',
      actor: 'system',
      details: `Panel: ${arbitratorIds.join(', ')}`,
    });
    caseRecord.updatedAt = Date.now();

    return caseRecord;
  }

  /**
   * Render a decision
   */
  renderDecision(
    caseId: string,
    outcome: DecisionOutcome,
    reasoning: string,
    remedies: Remedy[],
    arbitrators: string[]
  ): Decision {
    const caseRecord = this.cases.get(caseId);
    if (!caseRecord) throw new Error('Case not found');

    const decision: Decision = {
      id: `decision-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      caseId,
      outcome,
      reasoning,
      remedies,
      arbitrators,
      appealable: true,
      appealDeadline: Date.now() + 7 * 24 * 60 * 60 * 1000, // 7 days
      issuedAt: Date.now(),
      finalized: false,
    };

    this.decisions.set(decision.id, decision);

    caseRecord.phase = 'Enforcement';
    caseRecord.timeline.push({
      timestamp: Date.now(),
      action: 'Decision rendered',
      actor: arbitrators[0],
      details: outcome,
    });

    // Update arbitrator stats
    for (const arbId of arbitrators) {
      const arb = this.arbitrators.get(arbId);
      if (arb) {
        arb.casesArbitrated++;
        arb.reputation = recordPositive(arb.reputation);
      }
    }

    return decision;
  }

  /**
   * Create a restorative circle
   */
  createRestorativeCircle(
    caseId: string,
    facilitatorId: string,
    participants: Omit<CircleParticipant, 'consented' | 'attendedSessions'>[]
  ): RestorativeCircle {
    const caseRecord = this.cases.get(caseId);
    if (!caseRecord) throw new Error('Case not found');

    const circle: RestorativeCircle = {
      id: `circle-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      caseId,
      facilitatorId,
      participants: participants.map((p) => ({
        ...p,
        consented: false,
        attendedSessions: [],
      })),
      sessions: [],
      agreements: [],
      status: 'forming',
      createdAt: Date.now(),
    };

    this.circles.set(circle.id, circle);
    return circle;
  }

  /**
   * Record circle consent
   */
  recordConsent(circleId: string, participantDid: string): RestorativeCircle {
    const circle = this.circles.get(circleId);
    if (!circle) throw new Error('Circle not found');

    const participant = circle.participants.find((p) => p.did === participantDid);
    if (participant) {
      participant.consented = true;
    }

    // Activate if all consented
    if (circle.participants.every((p) => p.consented)) {
      circle.status = 'active';
    }

    return circle;
  }

  /**
   * Register as mediator
   */
  registerMediator(did: string, specializations: CaseCategory[]): MediatorProfile {
    // Validate mediator has at least one specialization
    JusticeValidators.mediator(specializations);

    const profile: MediatorProfile = {
      did,
      reputation: createReputation(did),
      casesMediated: 0,
      successRate: 1.0,
      specializations,
      available: true,
      certifiedAt: Date.now(),
    };
    this.mediators.set(did, profile);
    return profile;
  }

  /**
   * Register as arbitrator
   */
  registerArbitrator(did: string, specializations: CaseCategory[], tier: 1 | 2 | 3 = 1): ArbitratorProfile {
    const profile: ArbitratorProfile = {
      did,
      reputation: createReputation(did),
      casesArbitrated: 0,
      appealsOverturned: 0,
      specializations,
      tier,
      available: true,
      certifiedAt: Date.now(),
    };
    this.arbitrators.set(did, profile);
    return profile;
  }

  /**
   * Get case by ID
   */
  getCase(caseId: string): Case | undefined {
    return this.cases.get(caseId);
  }

  /**
   * Get decision for a case
   */
  getDecision(caseId: string): Decision | undefined {
    return Array.from(this.decisions.values()).find((d) => d.caseId === caseId);
  }

  /**
   * Get available mediators for a category
   */
  getAvailableMediators(category: CaseCategory): MediatorProfile[] {
    return Array.from(this.mediators.values()).filter(
      (m) => m.available && m.specializations.includes(category) && isTrustworthy(m.reputation)
    );
  }

  /**
   * Cross-hApp enforcement coordination
   */
  async executeEnforcement(decisionId: string, targetHapp: string): Promise<{ success: boolean; message: string }> {
    const decision = this.decisions.get(decisionId);
    if (!decision) return { success: false, message: 'Decision not found' };

    // In production, would coordinate with other hApps via bridge
    try {
      // Note: In production, would use bridge for enforcement
      // For now, just log the enforcement action
      console.log(`Enforcement ${decisionId} -> ${targetHapp}`);
      return { success: true, message: `Enforcement coordinated with ${targetHapp}` };
    } catch (error) {
      return { success: false, message: `Enforcement failed: ${error}` };
    }
  }
}

// Singleton
let instance: JusticeService | null = null;

export function getJusticeService(): JusticeService {
  if (!instance) instance = new JusticeService();
  return instance;
}

export function resetJusticeService(): void {
  instance = null;
}

// ============================================================================
// Justice Bridge Client (Holochain Zome Calls)
// ============================================================================

const JUSTICE_ROLE = 'civic';
const BRIDGE_ZOME = 'civic_bridge';

/**
 * Justice Bridge Client - Direct Holochain zome calls for cross-hApp dispute resolution
 */
export class JusticeBridgeClient {
  constructor(private client: MycelixClient) {}

  /**
   * File a cross-hApp dispute
   */
  async fileCrossHappDispute(input: FileCrossHappDisputeInput): Promise<CrossHappDispute> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'file_cross_happ_dispute',
      payload: input,
    });
  }

  /**
   * Request enforcement action on another hApp
   */
  async requestEnforcement(input: RequestEnforcementInput): Promise<EnforcementAction> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'request_enforcement',
      payload: input,
    });
  }

  /**
   * Get pending enforcement actions for a hApp
   */
  async getPendingEnforcements(targetHapp: string): Promise<EnforcementAction[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_pending_enforcements',
      payload: targetHapp,
    });
  }

  /**
   * Acknowledge enforcement execution
   */
  async acknowledgeEnforcement(enforcementId: string, status: EnforcementStatus): Promise<EnforcementAction> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'acknowledge_enforcement',
      payload: { enforcement_id: enforcementId, status },
    });
  }

  /**
   * Get dispute history for a DID
   */
  async getDisputeHistory(query: DisputeHistoryQuery): Promise<CrossHappDispute[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_dispute_history',
      payload: query,
    });
  }

  /**
   * Broadcast a justice event
   */
  async broadcastJusticeEvent(eventType: JusticeBridgeEventType, disputeId?: string, enforcementId?: string, payload: string = '{}'): Promise<JusticeBridgeEvent> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'broadcast_justice_event',
      payload: { event_type: eventType, dispute_id: disputeId, enforcement_id: enforcementId, payload },
    });
  }

  /**
   * Get recent justice events
   */
  async getRecentEvents(limit?: number): Promise<JusticeBridgeEvent[]> {
    return this.client.callZome({
      role_name: JUSTICE_ROLE,
      zome_name: BRIDGE_ZOME,
      fn_name: 'get_recent_events',
      payload: limit ?? 50,
    });
  }
}

// Bridge client singleton
let bridgeInstance: JusticeBridgeClient | null = null;

export function getJusticeBridgeClient(client: MycelixClient): JusticeBridgeClient {
  if (!bridgeInstance) bridgeInstance = new JusticeBridgeClient(client);
  return bridgeInstance;
}

export function resetJusticeBridgeClient(): void {
  bridgeInstance = null;
}

// ============================================================================
// Unified Client (re-exported from src/justice)
// ============================================================================

/**
 * Unified Mycelix Justice Client
 *
 * Use this client for most use cases. Re-exported from the main justice module.
 *
 * @example
 * ```typescript
 * import { MycelixJusticeClient } from '@mycelix/sdk/integrations/justice';
 *
 * const justice = await MycelixJusticeClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // File a case
 * const caseRecord = await justice.cases.fileCase({
 *   title: 'Contract Breach',
 *   description: 'Vendor failed to deliver...',
 *   category: 'ContractDispute',
 *   respondent: 'did:mycelix:vendor123',
 * });
 * ```
 */
export {
  MycelixJusticeClient,
  JusticeSdkError,
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  createJusticeClients,
} from '../../justice/index';
export type {
  JusticeClientConfig,
  JusticeConnectionOptions,
  HolochainRecord,
  ZomeCallable,
  Case as JusticeCase,
  CasePhase as JusticeCasePhase,
  CaseStatus as JusticeCaseStatus,
  CaseCategory as JusticeCaseCategory,
  Evidence as JusticeEvidence,
  EvidenceType as JusticeEvidenceType,
  Decision as JusticeDecision,
  DecisionOutcome as JusticeDecisionOutcome,
  Remedy as JusticeRemedy,
  MediatorProfile as JusticeMediatorProfile,
  ArbitratorProfile as JusticeArbitratorProfile,
  Enforcement as JusticeEnforcement,
  EnforcementType as JusticeEnforcementType,
  EnforcementStatus as JusticeEnforcementStatus,
  FileCaseInput,
  SubmitEvidenceInput,
  RegisterMediatorInput,
  RegisterArbitratorInput,
  RequestEnforcementInput as JusticeRequestEnforcementInput,
} from '../../justice/index';
