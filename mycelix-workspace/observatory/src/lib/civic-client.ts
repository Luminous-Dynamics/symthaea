/**
 * Civic Client - Live Holochain Data for MY GOV Dashboard
 *
 * Provides real-time access to civic data from Holochain conductor:
 * - Justice cases and decisions
 * - Governance proposals and votes
 * - Trust/reputation scores (MATL)
 * - Benefits and applications
 *
 * Uses the SDK's justice and governance modules to fetch data,
 * then transforms it into the DashboardData format.
 */

import { browser } from '$app/environment';
import { callZome, isConnected, connectToConductor, getClient } from './conductor';
import type {
  DashboardData,
  CitizenProfile,
  PendingDecision,
  RecentDecision,
  TrustBreakdown,
  TrustComponent,
  TrustHistoryEntry,
  AppealableDecision,
  CivicCredits,
  Benefit,
  Vote,
} from './mycelix';

// ============================================================================
// Types from SDK (mirrored to avoid import issues)
// ============================================================================

interface HolochainRecord<T = unknown> {
  signed_action: {
    hashed: { hash: string; content: unknown };
    signature: string;
  };
  entry: { Present: T };
}

interface Case {
  id: string;
  title: string;
  description: string;
  category: string;
  complainant: string;
  respondent: string;
  phase: string;
  status: string;
  filed_at: number;
  updated_at: number;
  resolved_at?: number;
}

interface Decision {
  id: string;
  case_id: string;
  outcome: string;
  reasoning: string;
  remedies: Array<{ type_: string; description: string; completed: boolean }>;
  appealable: boolean;
  appeal_deadline?: number;
  issued_at: number;
  finalized: boolean;
}

interface Proposal {
  id: string;
  dao_id: string;
  title: string;
  description: string;
  proposer: string;
  status: string;
  voting_starts: number;
  voting_ends: number;
  created_at: number;
}

interface DAOMember {
  did: string;
  voting_power: number;
  role: string;
  joined_at: number;
}

interface ReputationScore {
  agent_id: string;
  quality: number;
  consistency: number;
  reputation: number;
  composite: number;
  byzantine_flag: boolean;
  last_updated: number;
}

// ============================================================================
// Zome Role/Name Constants
// ============================================================================

const JUSTICE_ROLE = 'justice';
const GOVERNANCE_ROLE = 'governance';
const MATL_ROLE = 'mycelix-core';
const IDENTITY_ROLE = 'identity';
const BENEFITS_ROLE = 'benefits';

// ============================================================================
// Low-level Zome Call Helpers
// ============================================================================

async function safeCallZome<T>(
  roleName: string,
  zomeName: string,
  fnName: string,
  payload: unknown
): Promise<T | null> {
  try {
    return await callZome<T>({
      role_name: roleName,
      zome_name: zomeName,
      fn_name: fnName,
      payload,
    });
  } catch (error) {
    console.warn(`[CivicClient] Zome call failed: ${roleName}/${zomeName}/${fnName}`, error);
    return null;
  }
}

function extractEntry<T>(record: HolochainRecord<T> | null): T | null {
  if (!record) return null;
  return record.entry?.Present ?? null;
}

// ============================================================================
// Citizen Profile
// ============================================================================

async function fetchCitizenProfile(agentId: string): Promise<CitizenProfile | null> {
  // Get identity data
  const identityResult = await safeCallZome<HolochainRecord<{
    did: string;
    name: string;
    verification_level: string;
    created_at: number;
  }>>(IDENTITY_ROLE, 'identity', 'get_my_profile', null);

  const identity = extractEntry(identityResult);

  // Get reputation from MATL
  const reputationResult = await safeCallZome<ReputationScore>(
    MATL_ROLE,
    'matl',
    'get_my_reputation',
    null
  );

  // Get credentials
  const credentialsResult = await safeCallZome<HolochainRecord<{
    type: string;
    issuer: string;
    issued_at: number;
    expires_at?: number;
    verified: boolean;
  }>[]>(IDENTITY_ROLE, 'credentials', 'get_my_credentials', null);

  // Get delegations
  const delegationsResult = await safeCallZome<HolochainRecord<{
    domain: string;
    delegate_to: string;
    delegated_at: number;
    revocable: boolean;
  }>[]>(GOVERNANCE_ROLE, 'delegation', 'get_delegations_from', agentId);

  const credentials = (credentialsResult || []).map(r => {
    const entry = extractEntry(r);
    return entry ? {
      type: entry.type,
      issuer: entry.issuer,
      issuedAt: new Date(entry.issued_at),
      expiresAt: entry.expires_at ? new Date(entry.expires_at) : undefined,
      verified: entry.verified,
    } : null;
  }).filter((c): c is NonNullable<typeof c> => c !== null);

  const delegations = (delegationsResult || []).map(r => {
    const entry = extractEntry(r);
    return entry ? {
      domain: entry.domain,
      delegateTo: entry.delegate_to,
      delegatedAt: new Date(entry.delegated_at),
      revocable: entry.revocable,
    } : null;
  }).filter((d): d is NonNullable<typeof d> => d !== null);

  return {
    id: `CIT-${agentId.slice(0, 12)}`,
    did: identity?.did || `did:mycelix:${agentId.slice(0, 16)}`,
    name: identity?.name || 'Citizen',
    trustScore: reputationResult?.composite ?? 0.5,
    memberSince: new Date(identity?.created_at || Date.now() - 365 * 24 * 60 * 60 * 1000),
    verificationLevel: (identity?.verification_level as CitizenProfile['verificationLevel']) || 'basic',
    credentials,
    delegations,
  };
}

// ============================================================================
// Pending Decisions (Applications, Cases in Progress)
// ============================================================================

async function fetchPendingDecisions(agentId: string): Promise<PendingDecision[]> {
  const pending: PendingDecision[] = [];

  // Get my cases (as complainant or respondent) that are in progress
  const casesResult = await safeCallZome<HolochainRecord<Case>[]>(
    JUSTICE_ROLE,
    'cases',
    'get_cases_by_party',
    agentId
  );

  if (casesResult) {
    for (const record of casesResult) {
      const caseData = extractEntry(record);
      if (caseData && !['Closed', 'Resolved'].includes(caseData.status)) {
        // Calculate progress based on phase
        const phaseProgress: Record<string, number> = {
          Filing: 10,
          Mediation: 40,
          Arbitration: 60,
          Appeal: 80,
          Enforcement: 90,
        };

        pending.push({
          id: caseData.id,
          type: 'case',
          title: caseData.title,
          submittedAt: new Date(caseData.filed_at),
          estimatedCompletion: new Date(caseData.filed_at + 30 * 24 * 60 * 60 * 1000), // +30 days estimate
          status: caseData.phase.toLowerCase() as PendingDecision['status'],
          progress: phaseProgress[caseData.phase] ?? 50,
        });
      }
    }
  }

  // Get pending benefit applications
  const applicationsResult = await safeCallZome<HolochainRecord<{
    id: string;
    benefit_type: string;
    title: string;
    submitted_at: number;
    status: string;
    progress: number;
  }>[]>(BENEFITS_ROLE, 'applications', 'get_my_applications', null);

  if (applicationsResult) {
    for (const record of applicationsResult) {
      const app = extractEntry(record);
      if (app && app.status !== 'completed' && app.status !== 'denied') {
        pending.push({
          id: app.id,
          type: 'application',
          title: app.title,
          submittedAt: new Date(app.submitted_at),
          estimatedCompletion: new Date(app.submitted_at + 14 * 24 * 60 * 60 * 1000),
          status: app.status as PendingDecision['status'],
          progress: app.progress,
        });
      }
    }
  }

  return pending.sort((a, b) => b.submittedAt.getTime() - a.submittedAt.getTime());
}

// ============================================================================
// Recent Decisions
// ============================================================================

async function fetchRecentDecisions(agentId: string, days: number = 30): Promise<RecentDecision[]> {
  const recent: RecentDecision[] = [];
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;

  // Get decided cases
  const casesResult = await safeCallZome<HolochainRecord<Case>[]>(
    JUSTICE_ROLE,
    'cases',
    'get_cases_by_party',
    agentId
  );

  if (casesResult) {
    for (const record of casesResult) {
      const caseData = extractEntry(record);
      if (caseData && caseData.resolved_at && caseData.resolved_at > cutoff) {
        // Get the decision for this case
        const decisionResult = await safeCallZome<HolochainRecord<Decision>>(
          JUSTICE_ROLE,
          'arbitration',
          'get_decision',
          caseData.id
        );

        const decision = extractEntry(decisionResult);

        recent.push({
          id: caseData.id,
          type: caseData.category,
          title: caseData.title,
          decidedAt: new Date(caseData.resolved_at),
          outcome: mapOutcome(decision?.outcome || caseData.status),
          explanation: decision?.reasoning || 'Case resolved.',
          appealDeadline: decision?.appeal_deadline ? new Date(decision.appeal_deadline) : undefined,
          algorithmUsed: 'arbitration_v1',
          inputData: { category: caseData.category },
          outputScore: 0.8,
        });
      }
    }
  }

  // Get completed benefit decisions
  const benefitsResult = await safeCallZome<HolochainRecord<{
    id: string;
    title: string;
    decided_at: number;
    outcome: string;
    explanation: string;
    algorithm: string;
  }>[]>(BENEFITS_ROLE, 'decisions', 'get_my_recent_decisions', { days });

  if (benefitsResult) {
    for (const record of benefitsResult) {
      const decision = extractEntry(record);
      if (decision && decision.decided_at > cutoff) {
        recent.push({
          id: decision.id,
          type: 'Benefit',
          title: decision.title,
          decidedAt: new Date(decision.decided_at),
          outcome: mapOutcome(decision.outcome),
          explanation: decision.explanation,
          algorithmUsed: decision.algorithm,
          inputData: {},
          outputScore: 0.75,
        });
      }
    }
  }

  return recent.sort((a, b) => b.decidedAt.getTime() - a.decidedAt.getTime());
}

function mapOutcome(outcome: string): 'approved' | 'denied' | 'partial' {
  switch (outcome?.toLowerCase()) {
    case 'approved':
    case 'resolved':
    case 'complainantfavor':
    case 'settled':
      return 'approved';
    case 'denied':
    case 'dismissed':
    case 'respondentfavor':
      return 'denied';
    case 'partial':
    case 'split':
    default:
      return 'partial';
  }
}

// ============================================================================
// Trust Breakdown (MATL)
// ============================================================================

async function fetchTrustBreakdown(agentId: string): Promise<TrustBreakdown | null> {
  // Get detailed MATL breakdown
  const matlResult = await safeCallZome<{
    quality: number;
    consistency: number;
    reputation: number;
    composite: number;
    byzantine_tolerance: number;
    last_verified: number;
    history: Array<{
      timestamp: number;
      score: number;
      event?: string;
      delta: number;
    }>;
  }>(MATL_ROLE, 'matl', 'get_detailed_breakdown', agentId);

  if (!matlResult) {
    return null;
  }

  const components: TrustComponent[] = [
    {
      name: 'Quality (PoGQ)',
      score: matlResult.quality,
      weight: 0.40,
      explanation: 'Quality of contributions verified through Proof-of-Gradient-Quality',
      improvementActions: ['Submit high-quality proposals', 'Provide accurate information'],
    },
    {
      name: 'Consistency',
      score: matlResult.consistency,
      weight: 0.30,
      explanation: 'Consistency of behavior across interactions',
      improvementActions: ['Respond promptly to requests', 'Follow through on commitments'],
    },
    {
      name: 'Cross-hApp Reputation',
      score: matlResult.reputation,
      weight: 0.30,
      explanation: 'Reputation aggregated from other Mycelix applications',
      improvementActions: ['Participate in governance', 'Build positive track record'],
    },
  ];

  const history: TrustHistoryEntry[] = (matlResult.history || []).map(h => ({
    date: new Date(h.timestamp),
    score: h.score,
    event: h.event,
    delta: h.delta,
  }));

  return {
    overall: matlResult.composite,
    components,
    history,
    byzantineTolerance: matlResult.byzantine_tolerance,
    lastVerification: new Date(matlResult.last_verified),
  };
}

// ============================================================================
// Appealable Decisions
// ============================================================================

async function fetchAppealableDecisions(agentId: string): Promise<AppealableDecision[]> {
  const appealable: AppealableDecision[] = [];
  const now = Date.now();

  // Get decisions that can still be appealed
  const decisionsResult = await safeCallZome<HolochainRecord<Decision>[]>(
    JUSTICE_ROLE,
    'arbitration',
    'get_appealable_decisions',
    agentId
  );

  if (decisionsResult) {
    for (const record of decisionsResult) {
      const decision = extractEntry(record);
      if (decision && decision.appealable && decision.appeal_deadline && decision.appeal_deadline > now) {
        // Get case details for title
        const caseResult = await safeCallZome<HolochainRecord<Case>>(
          JUSTICE_ROLE,
          'cases',
          'get_case',
          decision.case_id
        );
        const caseData = extractEntry(caseResult);

        appealable.push({
          id: decision.id,
          title: caseData?.title || 'Decision',
          outcome: decision.outcome,
          deadline: new Date(decision.appeal_deadline),
          grounds: getAppealGrounds(decision.outcome),
          mediationAvailable: true,
          restorativeOption: decision.remedies?.some(r => r.type_ === 'Apology') || false,
        });
      }
    }
  }

  // Also get appealable benefit decisions
  const benefitsResult = await safeCallZome<HolochainRecord<{
    id: string;
    title: string;
    outcome: string;
    appeal_deadline: number;
    grounds: string[];
  }>[]>(BENEFITS_ROLE, 'decisions', 'get_appealable_decisions', null);

  if (benefitsResult) {
    for (const record of benefitsResult) {
      const decision = extractEntry(record);
      if (decision && decision.appeal_deadline > now) {
        appealable.push({
          id: decision.id,
          title: decision.title,
          outcome: decision.outcome,
          deadline: new Date(decision.appeal_deadline),
          grounds: decision.grounds,
          mediationAvailable: false,
          restorativeOption: false,
        });
      }
    }
  }

  return appealable.sort((a, b) => a.deadline.getTime() - b.deadline.getTime());
}

function getAppealGrounds(outcome: string): string[] {
  // Standard appeal grounds based on outcome type
  const baseGrounds = ['new_evidence', 'procedural_error', 'bias'];
  if (outcome === 'Dismissed') {
    return [...baseGrounds, 'standing_dispute'];
  }
  return baseGrounds;
}

// ============================================================================
// Civic Credits
// ============================================================================

async function fetchCivicCredits(): Promise<CivicCredits | null> {
  const creditsResult = await safeCallZome<{
    balance: number;
    monthly_allocation: number;
    last_allocation: number;
    expires_at: number;
    spent_this_month: Array<{
      amount: number;
      recipient: string;
      purpose: string;
      timestamp: number;
    }>;
  }>(BENEFITS_ROLE, 'credits', 'get_my_credits', null);

  if (!creditsResult) return null;

  return {
    balance: creditsResult.balance,
    monthlyAllocation: creditsResult.monthly_allocation,
    lastAllocation: new Date(creditsResult.last_allocation),
    expiresAt: new Date(creditsResult.expires_at),
    spentThisMonth: creditsResult.spent_this_month.map(s => ({
      amount: s.amount,
      recipient: s.recipient,
      purpose: s.purpose,
      timestamp: new Date(s.timestamp),
    })),
  };
}

// ============================================================================
// Active Benefits
// ============================================================================

async function fetchActiveBenefits(): Promise<Benefit[]> {
  const benefitsResult = await safeCallZome<HolochainRecord<{
    id: string;
    name: string;
    type: string;
    status: string;
    value: number;
    renewal_date?: number;
  }>[]>(BENEFITS_ROLE, 'benefits', 'get_my_active_benefits', null);

  if (!benefitsResult) return [];

  return benefitsResult
    .map(record => {
      const benefit = extractEntry(record);
      if (!benefit) return null;
      return {
        id: benefit.id,
        name: benefit.name,
        type: benefit.type,
        status: benefit.status as Benefit['status'],
        value: benefit.value,
        renewalDate: benefit.renewal_date ? new Date(benefit.renewal_date) : undefined,
      } satisfies Benefit;
    })
    .filter((b): b is NonNullable<typeof b> => b !== null);
}

// ============================================================================
// Upcoming Votes
// ============================================================================

async function fetchUpcomingVotes(agentId: string): Promise<Vote[]> {
  // Get active proposals from DAOs the user is a member of
  const proposalsResult = await safeCallZome<HolochainRecord<Proposal>[]>(
    GOVERNANCE_ROLE,
    'proposals',
    'get_active_proposals_for_member',
    agentId
  );

  if (!proposalsResult) return [];

  const votes: Vote[] = [];

  for (const record of proposalsResult) {
    const proposal = extractEntry(record);
    if (!proposal || proposal.voting_ends < Date.now()) continue;

    // Check if delegate has voted
    const delegateVoteResult = await safeCallZome<{
      voted: boolean;
      delegate_name?: string;
    }>(GOVERNANCE_ROLE, 'voting', 'check_delegate_voted', {
      proposal_id: proposal.id,
      delegator: agentId,
    });

    votes.push({
      proposalId: proposal.id,
      title: proposal.title,
      deadline: new Date(proposal.voting_ends),
      yourDelegateVoted: delegateVoteResult?.voted,
      delegateName: delegateVoteResult?.delegate_name,
    });
  }

  return votes.sort((a, b) => a.deadline.getTime() - b.deadline.getTime());
}

// ============================================================================
// Main Export: Load Live Dashboard Data
// ============================================================================

/**
 * Load dashboard data from live Holochain conductor
 *
 * @param agentId - The citizen's agent/DID
 * @returns Dashboard data or null if not connected
 */
export async function loadLiveDashboardData(agentId: string): Promise<DashboardData | null> {
  if (!browser) {
    console.log('[CivicClient] Skipping in SSR');
    return null;
  }

  // Ensure connected to conductor
  if (!isConnected()) {
    const connected = await connectToConductor();
    if (!connected) {
      console.warn('[CivicClient] Could not connect to conductor');
      return null;
    }
  }

  console.log('[CivicClient] Loading live dashboard data for', agentId.slice(0, 12) + '...');

  // Fetch all data in parallel where possible
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
    fetchCitizenProfile(agentId),
    fetchPendingDecisions(agentId),
    fetchRecentDecisions(agentId, 30),
    fetchTrustBreakdown(agentId),
    fetchAppealableDecisions(agentId),
    fetchCivicCredits(),
    fetchActiveBenefits(),
    fetchUpcomingVotes(agentId),
  ]);

  // If we couldn't get basic profile, return null to trigger fallback
  if (!profile) {
    console.warn('[CivicClient] Could not fetch citizen profile');
    return null;
  }

  return {
    profile,
    pendingDecisions,
    recentDecisions,
    trustBreakdown: trustBreakdown || {
      overall: 0.5,
      components: [],
      history: [],
      byzantineTolerance: 0.34,
      lastVerification: new Date(),
    },
    appealableDecisions,
    civicCredits: civicCredits || {
      balance: 0,
      monthlyAllocation: 100,
      lastAllocation: new Date(),
      expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      spentThisMonth: [],
    },
    activeBenefits,
    upcomingVotes,
  };
}

/**
 * Check if civic client can connect to conductor.
 * Returns true only when a live conductor connection is available.
 */
export async function checkCivicConnection(): Promise<boolean> {
  if (!browser) return false;
  if (isConnected()) return true;

  try {
    const ok = await connectToConductor();
    return ok;
  } catch {
    return false;
  }
}

/**
 * Get the Holochain client for advanced operations
 */
export function getCivicClient() {
  return getClient();
}
