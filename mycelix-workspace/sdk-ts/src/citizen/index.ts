/**
 * Citizen API Module - MY GOV Dashboard Backend
 *
 * Revolutionary Architecture: This module doesn't build new infrastructure.
 * It provides a unified view into existing Mycelix modules:
 * - Identity (DIDs, credentials, verification)
 * - Governance (proposals, voting, delegation)
 * - MATL (trust scoring, reputation)
 * - Justice (appeals, disputes, restorative processes)
 * - Finance (CGC balance, HEARTH participation)
 */

import type { MycelixClient } from '../client';
import type { EpistemicClaim } from '../epistemic';

// ============================================================================
// Types - Views into existing Mycelix data
// ============================================================================

export interface CitizenProfile {
  id: string;
  did: string; // Decentralized Identifier - citizen owns this
  name: string;
  trustScore: number;
  memberSince: Date;
  verificationLevel: 'basic' | 'verified' | 'enhanced' | 'full';
  credentials: VerifiableCredential[];
  delegations: Delegation[];
}

export interface VerifiableCredential {
  type: string; // 'drivers_license' | 'voter_registration' | 'benefits_card' | etc
  issuer: string;
  issuedAt: Date;
  expiresAt?: Date;
  verified: boolean;
}

export interface Delegation {
  domain: string; // 'fiscal' | 'environmental' | 'social' | etc
  delegateTo: string;
  delegatedAt: Date;
  revocable: boolean;
}

export interface PendingDecision {
  id: string;
  type: 'proposal' | 'application' | 'case' | 'appeal';
  title: string;
  submittedAt: Date;
  estimatedCompletion: Date;
  status: 'queued' | 'in_review' | 'pending_documents' | 'final_review';
  progress: number;
  assignedReviewer?: string;
  // Transparency: show the algorithm
  algorithm?: string;
  factors?: { name: string; value: number; weight: number }[];
}

export interface RecentDecision {
  id: string;
  type: string;
  title: string;
  decidedAt: Date;
  outcome: 'approved' | 'denied' | 'partial';
  explanation: string;
  appealDeadline?: Date;
  // Algorithmic transparency
  algorithmUsed: string;
  inputData: Record<string, unknown>;
  outputScore: number;
  epistemicClaim?: EpistemicClaim; // Link to verifiable claim
}

export interface TrustBreakdown {
  overall: number;
  components: TrustComponent[];
  history: TrustHistoryEntry[];
  // MATL-specific
  byzantineTolerance: number; // 0.34 = 34% validated
  lastVerification: Date;
}

export interface TrustComponent {
  name: string;
  score: number;
  weight: number;
  explanation: string;
  // How to improve
  improvementActions?: string[];
}

export interface TrustHistoryEntry {
  date: Date;
  score: number;
  event?: string;
  delta: number;
}

export interface AppealableDecision {
  id: string;
  title: string;
  outcome: string;
  deadline: Date;
  grounds: string[];
  // Justice module integration
  mediationAvailable: boolean;
  restorativeOption: boolean;
}

export interface CivicCredits {
  balance: number;
  monthlyAllocation: number;
  lastAllocation: Date;
  expiresAt: Date;
  spentThisMonth: CreditSpend[];
}

export interface CreditSpend {
  amount: number;
  recipient: string;
  purpose: string;
  timestamp: Date;
}

export interface DashboardData {
  profile: CitizenProfile;
  pendingDecisions: PendingDecision[];
  recentDecisions: RecentDecision[];
  trustBreakdown: TrustBreakdown;
  appealableDecisions: AppealableDecision[];
  civicCredits: CivicCredits;
  activeBenefits: Benefit[];
  upcomingVotes: Vote[];
}

export interface Benefit {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'pending' | 'expiring';
  value: number;
  renewalDate?: Date;
}

export interface Vote {
  proposalId: string;
  title: string;
  deadline: Date;
  yourDelegateVoted?: boolean;
  delegateName?: string;
}

// ============================================================================
// Service - Unified view into Mycelix
// ============================================================================

export interface CitizenServiceConfig {
  client?: MycelixClient;
  agentId: string;
  // Fallback for non-Holochain environments
  apiBaseUrl?: string;
}

export class CitizenDashboardService {
  private client?: MycelixClient;
  private agentId: string;
  private apiBaseUrl?: string;

  constructor(config: CitizenServiceConfig) {
    this.client = config.client;
    this.agentId = config.agentId;
    this.apiBaseUrl = config.apiBaseUrl;
  }

  /**
   * Get citizen profile from Identity module
   */
  async getProfile(): Promise<CitizenProfile> {
    if (this.client) {
      // Direct Holochain access
      const identity = await this.client.callZome({
        role_name: 'identity',
        zome_name: 'identity',
        fn_name: 'get_profile',
        payload: { agent_id: this.agentId },
      });
      return this.mapIdentityToProfile(identity as Record<string, unknown>);
    }
    return this.fetchApi('/citizen/profile');
  }

  /**
   * Get pending decisions from Governance module
   */
  async getPendingDecisions(): Promise<PendingDecision[]> {
    if (this.client) {
      const [proposals, applications, cases] = await Promise.all([
        this.client.callZome({
          role_name: 'governance',
          zome_name: 'proposals',
          fn_name: 'get_pending_for_agent',
          payload: { agent_id: this.agentId },
        }),
        this.client.callZome({
          role_name: 'governance',
          zome_name: 'applications',
          fn_name: 'get_pending',
          payload: { agent_id: this.agentId },
        }),
        this.client.callZome({
          role_name: 'justice',
          zome_name: 'cases',
          fn_name: 'get_active',
          payload: { agent_id: this.agentId },
        }),
      ]);
      return this.mapToDecisions([...(proposals as Record<string, unknown>[]), ...(applications as Record<string, unknown>[]), ...(cases as Record<string, unknown>[])]);
    }
    return this.fetchApi('/citizen/decisions/pending');
  }

  /**
   * Get recent decisions with algorithmic transparency
   */
  async getRecentDecisions(days = 30): Promise<RecentDecision[]> {
    if (this.client) {
      const decisions = await this.client.callZome({
        role_name: 'governance',
        zome_name: 'decisions',
        fn_name: 'get_recent',
        payload: { agent_id: this.agentId, days },
      });
      return (decisions as Record<string, unknown>[]).map(this.mapToRecentDecision);
    }
    return this.fetchApi(`/citizen/decisions/recent?days=${days}`);
  }

  /**
   * Get MATL trust breakdown with full transparency
   */
  async getTrustBreakdown(): Promise<TrustBreakdown> {
    if (this.client) {
      const matlRaw = await this.client.callZome({
        role_name: 'bridge',
        zome_name: 'matl',
        fn_name: 'get_trust_breakdown',
        payload: { agent_id: this.agentId },
      });
      const matl = matlRaw as Record<string, unknown>;
      return {
        overall: matl.composite_score as number,
        byzantineTolerance: 0.34, // MATL constant (34% validated)
        lastVerification: new Date(matl.last_verified as number),
        components: [
          {
            name: 'Proof of Gradient Quality',
            score: matl.pogq as number,
            weight: 0.4,
            explanation: 'Quality of contributions to federated learning',
            improvementActions: ['Participate in more FL rounds', 'Improve gradient accuracy'],
          },
          {
            name: 'Consistency',
            score: matl.consistency as number,
            weight: 0.3,
            explanation: 'Reliability of behavior over time',
            improvementActions: ['Maintain regular participation', 'Honor commitments'],
          },
          {
            name: 'Reputation',
            score: matl.reputation as number,
            weight: 0.3,
            explanation: 'Peer assessments and endorsements',
            improvementActions: ['Build positive interactions', 'Contribute to community'],
          },
        ],
        history: (matl.history as { timestamp: number; score: number; event?: string; prev_score: number }[]).map((h: { timestamp: number; score: number; event?: string; prev_score: number }) => ({
          date: new Date(h.timestamp),
          score: h.score,
          event: h.event,
          delta: h.score - h.prev_score,
        })),
      };
    }
    return this.fetchApi('/citizen/trust/breakdown');
  }

  /**
   * Get appealable decisions from Justice module
   */
  async getAppealableDecisions(): Promise<AppealableDecision[]> {
    if (this.client) {
      const appeals = await this.client.callZome({
        role_name: 'justice',
        zome_name: 'appeals',
        fn_name: 'get_eligible',
        payload: { agent_id: this.agentId },
      });
      return (appeals as Record<string, unknown>[]).map(this.mapToAppealable);
    }
    return this.fetchApi('/citizen/appeals/eligible');
  }

  /**
   * Get CGC (Civic Gifting Credits) balance from Finance module
   */
  async getCivicCredits(): Promise<CivicCredits> {
    if (this.client) {
      const cgc = await this.client.callZome({
        role_name: 'finance',
        zome_name: 'cgc',
        fn_name: 'get_balance',
        payload: { agent_id: this.agentId },
      });
      return this.mapToCivicCredits(cgc as Record<string, unknown>);
    }
    return this.fetchApi('/citizen/credits');
  }

  /**
   * Get active benefits
   */
  async getActiveBenefits(): Promise<Benefit[]> {
    if (this.client) {
      const benefits = await this.client.callZome({
        role_name: 'governance',
        zome_name: 'benefits',
        fn_name: 'get_active',
        payload: { agent_id: this.agentId },
      });
      return benefits as Benefit[];
    }
    return this.fetchApi('/citizen/benefits');
  }

  /**
   * Get upcoming votes where citizen can participate
   */
  async getUpcomingVotes(): Promise<Vote[]> {
    if (this.client) {
      const votes = await this.client.callZome({
        role_name: 'governance',
        zome_name: 'voting',
        fn_name: 'get_upcoming',
        payload: { agent_id: this.agentId },
      });
      return votes as Vote[];
    }
    return this.fetchApi('/citizen/votes/upcoming');
  }

  /**
   * Get complete dashboard data - single call for efficiency
   */
  async getDashboardData(): Promise<DashboardData> {
    const [
      profile,
      pendingDecisions,
      recentDecisions,
      trustBreakdown,
      appealableDecisions,
      civicCredits,
      activeBenefits,
      upcomingVotes,
    ] = await Promise.all([
      this.getProfile(),
      this.getPendingDecisions(),
      this.getRecentDecisions(),
      this.getTrustBreakdown(),
      this.getAppealableDecisions(),
      this.getCivicCredits(),
      this.getActiveBenefits(),
      this.getUpcomingVotes(),
    ]);

    return {
      profile,
      pendingDecisions,
      recentDecisions,
      trustBreakdown,
      appealableDecisions,
      civicCredits,
      activeBenefits,
      upcomingVotes,
    };
  }

  // ============================================================================
  // Private helpers
  // ============================================================================

  private async fetchApi<T>(endpoint: string): Promise<T> {
    if (!this.apiBaseUrl) {
      throw new Error('No Mycelix client or API URL configured');
    }
    const res = await fetch(`${this.apiBaseUrl}${endpoint}`, {
      headers: { 'X-Agent-Id': this.agentId },
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
  }

  private mapIdentityToProfile(identity: Record<string, unknown>): CitizenProfile {
    return {
      id: identity.agent_id as string,
      did: identity.did as string,
      name: identity.display_name as string,
      trustScore: identity.trust_score as number,
      memberSince: new Date(identity.created_at as number),
      verificationLevel: identity.verification_level as CitizenProfile['verificationLevel'],
      credentials: (identity.credentials as VerifiableCredential[]) || [],
      delegations: (identity.delegations as Delegation[]) || [],
    };
  }

  private mapToDecisions(items: Record<string, unknown>[]): PendingDecision[] {
    return items.map((item) => ({
      id: item.id as string,
      type: item.type as PendingDecision['type'],
      title: item.title as string,
      submittedAt: new Date(item.submitted_at as number),
      estimatedCompletion: new Date(item.estimated_completion as number),
      status: item.status as PendingDecision['status'],
      progress: item.progress as number,
      assignedReviewer: item.assigned_reviewer as string | undefined,
      algorithm: item.algorithm as string | undefined,
      factors: item.factors as PendingDecision['factors'],
    }));
  }

  private mapToRecentDecision(item: Record<string, unknown>): RecentDecision {
    return {
      id: item.id as string,
      type: item.type as string,
      title: item.title as string,
      decidedAt: new Date(item.decided_at as number),
      outcome: item.outcome as RecentDecision['outcome'],
      explanation: item.explanation as string,
      appealDeadline: item.appeal_deadline ? new Date(item.appeal_deadline as number) : undefined,
      algorithmUsed: item.algorithm_used as string,
      inputData: item.input_data as Record<string, unknown>,
      outputScore: item.output_score as number,
      epistemicClaim: item.epistemic_claim as EpistemicClaim | undefined,
    };
  }

  private mapToAppealable(item: Record<string, unknown>): AppealableDecision {
    return {
      id: item.id as string,
      title: item.title as string,
      outcome: item.outcome as string,
      deadline: new Date(item.deadline as number),
      grounds: item.grounds as string[],
      mediationAvailable: item.mediation_available as boolean,
      restorativeOption: item.restorative_option as boolean,
    };
  }

  private mapToCivicCredits(cgc: Record<string, unknown>): CivicCredits {
    return {
      balance: cgc.balance as number,
      monthlyAllocation: cgc.monthly_allocation as number,
      lastAllocation: new Date(cgc.last_allocation as number),
      expiresAt: new Date(cgc.expires_at as number),
      spentThisMonth: (cgc.spent_this_month as CreditSpend[]) || [],
    };
  }
}

// ============================================================================
// Utilities
// ============================================================================

export function calculateTrustScore(components: { score: number; weight: number }[]): number {
  const totalWeight = components.reduce((s, c) => s + c.weight, 0);
  return totalWeight ? components.reduce((s, c) => s + c.score * c.weight, 0) / totalWeight : 0;
}

export function canAppeal(decision: RecentDecision): boolean {
  return !!decision.appealDeadline && decision.appealDeadline.getTime() > Date.now();
}

export function getDaysUntilDeadline(deadline: Date): number {
  return Math.ceil((deadline.getTime() - Date.now()) / (1000 * 60 * 60 * 24));
}

export function createCitizenService(config: CitizenServiceConfig): CitizenDashboardService {
  return new CitizenDashboardService(config);
}

// ============================================================================
// Proactive Outreach (Revolutionary Feature)
// ============================================================================

export interface LifeEvent {
  type: 'birth' | 'death' | 'marriage' | 'divorce' | 'job_change' | 'address_change' | 'income_change';
  citizenId: string;
  timestamp: Date;
  data: Record<string, unknown>;
}

export interface EligibilityResult {
  program: string;
  eligible: boolean;
  matchScore: number;
  requirements: { name: string; met: boolean; value?: unknown }[];
  estimatedValue?: number;
}

export interface OutreachMessage {
  citizenId: string;
  channel: 'sms' | 'email' | 'dashboard' | 'all';
  subject: string;
  body: string;
  actions: { label: string; endpoint: string }[];
  expiresAt?: Date;
}

/**
 * Proactive outreach engine - government that finds citizens
 * Instead of: citizen discovers program → applies → waits
 * Now: life event → eligibility calculated → benefit offered
 */
export class ProactiveOutreachEngine {
  constructor(
    private client: MycelixClient,
    private notificationService: { send: (msg: OutreachMessage) => Promise<void> }
  ) {}

  /**
   * Handle life event and proactively offer eligible benefits
   */
  async handleLifeEvent(event: LifeEvent): Promise<void> {
    const eligibility = await this.calculateEligibility(event);
    const eligible = eligibility.filter((e) => e.eligible);

    if (eligible.length === 0) return;

    const message = this.composeOutreachMessage(event, eligible);
    await this.notificationService.send(message);

    // Log the outreach for transparency
    await this.client.callZome({
      role_name: 'governance',
      zome_name: 'outreach',
      fn_name: 'log_proactive_contact',
      payload: {
        citizen_id: event.citizenId,
        event_type: event.type,
        programs_offered: eligible.map((e) => e.program),
        timestamp: Date.now(),
      },
    });
  }

  private async calculateEligibility(event: LifeEvent): Promise<EligibilityResult[]> {
    const programsToCheck = this.getProgramsForEvent(event.type);

    const results = await Promise.all(
      programsToCheck.map(async (program) => {
        const result = await this.client.callZome({
          role_name: 'governance',
          zome_name: 'eligibility',
          fn_name: 'check',
          payload: { citizen_id: event.citizenId, program, event_data: event.data },
        });
        return result as EligibilityResult;
      })
    );

    return results;
  }

  private getProgramsForEvent(eventType: LifeEvent['type']): string[] {
    const programMap: Record<LifeEvent['type'], string[]> = {
      birth: ['child_tax_credit', 'parental_leave', 'childcare_assistance', 'health_coverage_expansion'],
      death: ['survivor_benefits', 'estate_tax_exemption', 'grief_counseling'],
      marriage: ['tax_filing_status_change', 'health_coverage_update', 'name_change_assistance'],
      divorce: ['single_parent_support', 'housing_assistance', 'legal_aid'],
      job_change: ['unemployment_insurance', 'health_coverage_transition', 'retirement_rollover'],
      address_change: ['voter_registration_update', 'utility_assistance', 'local_services'],
      income_change: ['benefit_recalculation', 'tax_adjustment', 'assistance_programs'],
    };
    return programMap[eventType] || [];
  }

  private composeOutreachMessage(event: LifeEvent, eligible: EligibilityResult[]): OutreachMessage {
    const programList = eligible.map((e) => `• ${e.program} (${Math.round(e.matchScore * 100)}% match)`).join('\n');

    return {
      citizenId: event.citizenId,
      channel: 'all',
      subject: `You may be eligible for ${eligible.length} program${eligible.length > 1 ? 's' : ''}`,
      body: `Based on your recent ${event.type.replace('_', ' ')}, you may qualify for:\n\n${programList}\n\nReply YES to learn more, or visit your dashboard.`,
      actions: eligible.map((e) => ({
        label: `Apply for ${e.program}`,
        endpoint: `/apply/${e.program}`,
      })),
      expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
    };
  }
}
