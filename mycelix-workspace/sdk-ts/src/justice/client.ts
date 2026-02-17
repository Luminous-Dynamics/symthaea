/**
 * Mycelix Justice Client
 *
 * Unified client for the Justice hApp SDK, providing access to
 * dispute resolution, arbitration, mediation, and enforcement.
 *
 * @module @mycelix/sdk/justice
 */

import {
  type AppClient,
  AppWebsocket,
} from '@holochain/client';

import { RetryPolicy, RetryPolicies, type RetryOptions } from '../common/retry';

import {
  CasesClient,
  EvidenceClient,
  ArbitrationClient,
  EnforcementClient,
  type Case,
  type Evidence,
  type Decision,
  type Enforcement,
  type MediatorProfile,
  type ArbitratorProfile,
  type CasePhase,
  type CaseStatus,
  type CaseCategory,
  type DecisionOutcome,
} from './index';

/**
 * Justice SDK Error
 */
export class JusticeSdkError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly cause?: unknown
  ) {
    super(message);
    this.name = 'JusticeSdkError';
  }
}

/**
 * Configuration for the Justice client
 */
export interface JusticeClientConfig {
  /** Role ID for the justice DNA */
  roleId: string;
  /** Retry configuration for zome calls */
  retry?: RetryOptions | RetryPolicy;
}

const DEFAULT_CONFIG: JusticeClientConfig = {
  roleId: 'justice',
};

/**
 * Connection options for creating a new client
 */
export interface JusticeConnectionOptions {
  /** WebSocket URL to connect to */
  url: string;
  /** Optional timeout in milliseconds */
  timeout?: number;
  /** Retry configuration for connection and zome calls */
  retry?: RetryOptions | RetryPolicy;
}

/**
 * Unified Mycelix Justice Client
 *
 * Provides access to all justice functionality through a single interface.
 *
 * @example
 * ```typescript
 * import { MycelixJusticeClient } from '@mycelix/sdk/justice';
 *
 * // Connect to Holochain
 * const justice = await MycelixJusticeClient.connect({
 *   url: 'ws://localhost:8888',
 * });
 *
 * // File a case
 * const case = await justice.cases.fileCase({
 *   title: 'Contract Breach',
 *   description: 'Failure to deliver agreed services',
 *   category: 'ContractDispute',
 *   respondent: 'did:mycelix:respondent',
 * });
 *
 * // Submit evidence
 * await justice.evidence.submitEvidence({
 *   case_id: case.entry.Present.id,
 *   type_: 'Document',
 *   title: 'Contract Agreement',
 *   description: 'Original signed contract',
 *   content_hash: 'abc123...',
 * });
 *
 * // Escalate to arbitration
 * await justice.arbitration.escalateToArbitration(
 *   case.entry.Present.id,
 *   ['did:mycelix:arbitrator1', 'did:mycelix:arbitrator2']
 * );
 * ```
 */
export class MycelixJusticeClient {
  /** Case filing and management */
  public readonly cases: CasesClient;

  /** Evidence submission and verification */
  public readonly evidence: EvidenceClient;

  /** Mediation and arbitration */
  public readonly arbitration: ArbitrationClient;

  /** Cross-hApp enforcement */
  public readonly enforcement: EnforcementClient;

  private readonly config: JusticeClientConfig;

  /** Retry policy for zome calls */
  private readonly retryPolicy: RetryPolicy;

  /**
   * Create a justice client from an existing Holochain client
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   */
  constructor(
    private readonly client: AppClient,
    config: Partial<JusticeClientConfig> = {}
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Set up retry policy
    if (config.retry instanceof RetryPolicy) {
      this.retryPolicy = config.retry;
    } else if (config.retry) {
      this.retryPolicy = new RetryPolicy(config.retry);
    } else {
      this.retryPolicy = RetryPolicies.standard;
    }

    // Initialize sub-clients
    this.cases = new CasesClient(client);
    this.evidence = new EvidenceClient(client);
    this.arbitration = new ArbitrationClient(client);
    this.enforcement = new EnforcementClient(client);
  }

  /**
   * Connect to Holochain and create a justice client
   *
   * @param options - Connection options
   * @returns Connected justice client
   */
  static async connect(
    options: JusticeConnectionOptions
  ): Promise<MycelixJusticeClient> {
    // Set up retry for connection
    const retryPolicy = options.retry instanceof RetryPolicy
      ? options.retry
      : options.retry
        ? new RetryPolicy(options.retry)
        : RetryPolicies.network;

    const connectFn = async () => {
      try {
        const client = await AppWebsocket.connect({
          url: new URL(options.url),
          wsClientOptions: { origin: 'mycelix-justice-sdk' },
        });

        return new MycelixJusticeClient(client, { retry: retryPolicy });
      } catch (error) {
        throw new JusticeSdkError(
          'CONNECTION_ERROR',
          `Failed to connect to Holochain: ${error instanceof Error ? error.message : String(error)}`,
          error
        );
      }
    };

    return retryPolicy.execute(connectFn);
  }

  /**
   * Create a justice client from an existing AppClient
   *
   * @param client - Existing AppClient instance
   * @param config - Optional configuration overrides
   * @returns Justice client
   */
  static fromClient(
    client: AppClient,
    config: Partial<JusticeClientConfig> = {}
  ): MycelixJusticeClient {
    return new MycelixJusticeClient(client, config);
  }

  // ============================================================================
  // Convenience Methods
  // ============================================================================

  /**
   * Get complete case details with all related data
   *
   * @param caseId - Case identifier
   * @returns Complete case information
   */
  async getCaseComplete(caseId: string): Promise<{
    case_: Case | null;
    evidence: Evidence[];
    decision: Decision | null;
    enforcements: Enforcement[];
  } | null> {
    const caseRecord = await this.cases.getCase(caseId);
    if (!caseRecord) {
      return null;
    }

    const case_ = caseRecord.entry.Present;

    const [evidenceRecords, decisionRecord] = await Promise.all([
      this.evidence.getEvidenceForCase(caseId),
      this.arbitration.getDecision(caseId).catch(() => null),
    ]);

    const enforcementRecords = decisionRecord
      ? await this.enforcement.getEnforcementsForDecision(caseId)
      : [];

    return {
      case_,
      evidence: evidenceRecords.map(r => r.entry.Present),
      decision: decisionRecord ? decisionRecord.entry.Present : null,
      enforcements: enforcementRecords.map(r => r.entry.Present),
    };
  }

  /**
   * Get all cases for a party (as complainant or respondent)
   *
   * @param did - Party's DID
   * @returns Cases where party is involved
   */
  async getAllCasesForParty(did: string): Promise<{
    asComplainant: Case[];
    asRespondent: Case[];
    total: number;
  }> {
    const [complainantRecords, respondentRecords] = await Promise.all([
      this.cases.getCasesByComplainant(did),
      this.cases.getCasesByRespondent(did),
    ]);

    const asComplainant = complainantRecords.map(r => r.entry.Present);
    const asRespondent = respondentRecords.map(r => r.entry.Present);

    return {
      asComplainant,
      asRespondent,
      total: asComplainant.length + asRespondent.length,
    };
  }

  /**
   * Get case statistics for a party
   *
   * @param did - Party's DID
   * @returns Case statistics
   */
  async getCaseStatistics(did: string): Promise<{
    totalCases: number;
    casesAsComplainant: number;
    casesAsRespondent: number;
    wonAsComplainant: number;
    wonAsRespondent: number;
    pending: number;
    resolved: number;
    byCategory: Record<CaseCategory, number>;
    byPhase: Record<CasePhase, number>;
  }> {
    const { asComplainant, asRespondent, total } = await this.getAllCasesForParty(did);
    const allCases = [...asComplainant, ...asRespondent];

    const byCategory: Record<string, number> = {};
    const byPhase: Record<string, number> = {};
    let pending = 0;
    let resolved = 0;
    const wonAsComplainant = 0;
    const wonAsRespondent = 0;

    for (const case_ of allCases) {
      byCategory[case_.category] = (byCategory[case_.category] || 0) + 1;
      byPhase[case_.phase] = (byPhase[case_.phase] || 0) + 1;

      if (case_.status === 'Active' || case_.status === 'AwaitingResponse' || case_.status === 'InDeliberation') {
        pending++;
      } else if (case_.status === 'Resolved') {
        resolved++;
        // Check if they won (would need decision data)
      }
    }

    return {
      totalCases: total,
      casesAsComplainant: asComplainant.length,
      casesAsRespondent: asRespondent.length,
      wonAsComplainant,
      wonAsRespondent,
      pending,
      resolved,
      byCategory: byCategory as Record<CaseCategory, number>,
      byPhase: byPhase as Record<CasePhase, number>,
    };
  }

  /**
   * Find available dispute resolution professionals
   *
   * @param category - Case category
   * @returns Available mediators and arbitrators
   */
  async findDisputeResolvers(category: CaseCategory): Promise<{
    mediators: MediatorProfile[];
    arbitrators: ArbitratorProfile[];
  }> {
    const [mediatorRecords, arbitratorRecords] = await Promise.all([
      this.arbitration.getAvailableMediators(category),
      this.arbitration.getAvailableArbitrators(category),
    ]);

    return {
      mediators: mediatorRecords.map(r => r.entry.Present),
      arbitrators: arbitratorRecords.map(r => r.entry.Present),
    };
  }

  /**
   * Quick case filing with initial evidence
   *
   * @param caseInput - Case filing input
   * @param initialEvidence - Optional initial evidence
   * @returns Filed case and evidence
   */
  async fileWithEvidence(
    caseInput: Parameters<CasesClient['fileCase']>[0],
    initialEvidence?: Array<Omit<Parameters<EvidenceClient['submitEvidence']>[0], 'case_id'>>
  ): Promise<{
    case_: Case;
    evidence: Evidence[];
  }> {
    const caseRecord = await this.cases.fileCase(caseInput);
    const case_ = caseRecord.entry.Present;

    const evidence: Evidence[] = [];
    if (initialEvidence && initialEvidence.length > 0) {
      for (const ev of initialEvidence) {
        const evRecord = await this.evidence.submitEvidence({
          ...ev,
          case_id: case_.id,
        });
        evidence.push(evRecord.entry.Present);
      }
    }

    return { case_, evidence };
  }

  /**
   * Get phase description
   *
   * @param phase - Case phase
   * @returns Human-readable description
   */
  getPhaseDescription(phase: CasePhase): string {
    const descriptions: Record<CasePhase, string> = {
      Filed: 'Case has been filed and is awaiting response',
      Negotiation: 'Parties are in direct negotiation',
      Mediation: 'Case is in mediation with a neutral mediator',
      Arbitration: 'Case is being heard by arbitrators',
      Appeal: 'Decision is being appealed',
      Enforcement: 'Decision is being enforced',
      Closed: 'Case has been closed',
    };
    return descriptions[phase];
  }

  /**
   * Get status description
   *
   * @param status - Case status
   * @returns Human-readable description
   */
  getStatusDescription(status: CaseStatus): string {
    const descriptions: Record<CaseStatus, string> = {
      Active: 'Case is open and active',
      OnHold: 'Case is on hold pending further action',
      AwaitingResponse: 'Awaiting response from respondent',
      InDeliberation: 'Case is under deliberation',
      DecisionRendered: 'A decision has been rendered',
      Enforcing: 'Decision is being enforced',
      Resolved: 'Case has been resolved',
      Dismissed: 'Case was dismissed',
      Withdrawn: 'Case was withdrawn by complainant',
    };
    return descriptions[status];
  }

  /**
   * Get outcome description
   *
   * @param outcome - Decision outcome
   * @returns Human-readable description
   */
  getOutcomeDescription(outcome: DecisionOutcome): string {
    const descriptions: Record<DecisionOutcome, string> = {
      ForComplainant: 'Decision favors the complainant',
      ForRespondent: 'Decision favors the respondent',
      SplitDecision: 'Decision assigns responsibility to both parties',
      Dismissed: 'Case was dismissed',
      Settled: 'Parties reached a settlement',
    };
    return descriptions[outcome];
  }

  /**
   * Get the arbitration timeline for a case
   *
   * Returns a chronological timeline of all events in a case.
   *
   * @param caseId - Case identifier
   * @returns Arbitration timeline with all events
   *
   * @example
   * ```typescript
   * const timeline = await justice.getArbitrationTimeline('case-001');
   * for (const event of timeline.events) {
   *   console.log(`${event.timestamp}: ${event.type} - ${event.description}`);
   * }
   * ```
   */
  async getArbitrationTimeline(caseId: string): Promise<{
    caseId: string;
    events: Array<{
      type: 'filed' | 'evidence' | 'response' | 'mediation' | 'arbitration' | 'decision' | 'enforcement' | 'appeal';
      timestamp: number;
      description: string;
      actor?: string;
    }>;
    currentPhase: CasePhase;
    currentStatus: CaseStatus;
  } | null> {
    const complete = await this.getCaseComplete(caseId);
    if (!complete) return null;

    const { case_, evidence, decision, enforcements } = complete;
    if (!case_) return null;
    const events: Array<{
      type: 'filed' | 'evidence' | 'response' | 'mediation' | 'arbitration' | 'decision' | 'enforcement' | 'appeal';
      timestamp: number;
      description: string;
      actor?: string;
    }> = [];

    // Filing event
    events.push({
      type: 'filed',
      timestamp: case_.filed_at,
      description: `Case filed: ${case_.title}`,
      actor: case_.complainant,
    });

    // Evidence events
    for (const ev of evidence) {
      events.push({
        type: 'evidence',
        timestamp: ev.submitted_at,
        description: `Evidence submitted: ${ev.title}`,
        actor: ev.submitter,
      });
    }

    // Mediation assignment
    if (case_.mediator) {
      events.push({
        type: 'mediation',
        timestamp: case_.updated_at, // Approximation
        description: 'Mediator assigned',
        actor: case_.mediator,
      });
    }

    // Arbitration escalation
    if (case_.arbitrators.length > 0) {
      events.push({
        type: 'arbitration',
        timestamp: case_.updated_at,
        description: `Escalated to arbitration with ${case_.arbitrators.length} arbitrators`,
      });
    }

    // Decision
    if (decision) {
      events.push({
        type: 'decision',
        timestamp: decision.issued_at,
        description: `Decision rendered: ${decision.outcome}`,
      });
    }

    // Enforcement actions
    for (const enf of enforcements) {
      events.push({
        type: 'enforcement',
        timestamp: enf.requested_at,
        description: `Enforcement ${enf.action_type}: ${enf.details}`,
      });
    }

    // Sort by timestamp
    events.sort((a, b) => a.timestamp - b.timestamp);

    return {
      caseId,
      events,
      currentPhase: case_.phase,
      currentStatus: case_.status,
    };
  }

  /**
   * Get the underlying Holochain client
   */
  getClient(): AppClient {
    return this.client;
  }

  /**
   * Get the current retry policy
   */
  getRetryPolicy(): RetryPolicy {
    return this.retryPolicy;
  }

  /**
   * Create a new client with a different retry policy
   *
   * @param retry - New retry configuration
   * @returns New client instance with updated retry policy
   */
  withRetry(retry: RetryOptions | RetryPolicy): MycelixJusticeClient {
    return new MycelixJusticeClient(this.client, { ...this.config, retry });
  }
}

export default MycelixJusticeClient;
