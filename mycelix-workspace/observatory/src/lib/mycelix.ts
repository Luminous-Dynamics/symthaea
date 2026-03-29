// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix Service Integration for Observatory
 *
 * Provides unified access to Mycelix backend services.
 * Self-contained types - doesn't require @mycelix/sdk import.
 */

// ============================================================================
// Types (mirror of @mycelix/sdk/citizen types)
// ============================================================================

export interface CitizenProfile {
  id: string;
  did: string;
  name: string;
  trustScore: number;
  memberSince: Date;
  verificationLevel: 'basic' | 'verified' | 'enhanced' | 'full';
  credentials: VerifiableCredential[];
  delegations: Delegation[];
}

export interface VerifiableCredential {
  type: string;
  issuer: string;
  issuedAt: Date;
  expiresAt?: Date;
  verified: boolean;
}

export interface Delegation {
  domain: string;
  delegateTo: string;
  delegatedAt: Date;
  revocable: boolean;
}

export interface PendingDecision {
  id: string;
  type: 'proposal' | 'application' | 'case' | 'appeal' | string;
  title: string;
  submittedAt: Date;
  estimatedCompletion: Date;
  status: 'queued' | 'in_review' | 'pending_documents' | 'final_review' | string;
  progress: number;
  assignedReviewer?: string;
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
  algorithmUsed: string;
  inputData: Record<string, unknown>;
  outputScore: number;
}

export interface TrustBreakdown {
  overall: number;
  components: TrustComponent[];
  history: TrustHistoryEntry[];
  byzantineTolerance: number;
  lastVerification: Date;
}

export interface TrustComponent {
  name: string;
  score: number;
  weight: number;
  explanation?: string;
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

// ============================================================================
// Environment configuration
// ============================================================================

const MYCELIX_API_URL = import.meta.env.VITE_MYCELIX_API_URL || 'http://localhost:3333';

// ============================================================================
// API Client
// ============================================================================

async function fetchApi<T>(endpoint: string, agentId: string): Promise<T> {
  const res = await fetch(`${MYCELIX_API_URL}${endpoint}`, {
    headers: {
      'X-Agent-Id': agentId,
      'Content-Type': 'application/json',
    },
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

/**
 * Load dashboard data from the Mycelix API
 */
async function loadLiveDashboardData(agentId: string): Promise<DashboardData> {
  // Fetch all dashboard data in parallel
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
    fetchApi<CitizenProfile>('/citizen/profile', agentId),
    fetchApi<PendingDecision[]>('/citizen/decisions/pending', agentId),
    fetchApi<RecentDecision[]>('/citizen/decisions/recent?days=30', agentId),
    fetchApi<TrustBreakdown>('/citizen/trust/breakdown', agentId),
    fetchApi<AppealableDecision[]>('/citizen/appeals/eligible', agentId),
    fetchApi<CivicCredits>('/citizen/credits', agentId),
    fetchApi<Benefit[]>('/citizen/benefits', agentId),
    fetchApi<Vote[]>('/citizen/votes/upcoming', agentId),
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
// Mock Data for Demo Mode
// ============================================================================

export function getMockDashboardData(): DashboardData {
  const now = Date.now();
  const day = 24 * 60 * 60 * 1000;

  return {
    profile: {
      id: 'CIT-2024-001234',
      did: 'did:mycelix:demo123456789',
      name: 'Demo Citizen',
      trustScore: 0.78,
      memberSince: new Date(now - 365 * day),
      verificationLevel: 'verified',
      credentials: [
        { type: 'voter_registration', issuer: 'County Clerk', issuedAt: new Date(now - 180 * day), verified: true },
        { type: 'drivers_license', issuer: 'DMV', issuedAt: new Date(now - 90 * day), verified: true },
      ],
      delegations: [
        { domain: 'environmental', delegateTo: 'did:mycelix:expert789', delegatedAt: new Date(now - 30 * day), revocable: true },
      ],
    },
    pendingDecisions: [
      {
        id: 'BP-2024-123456',
        type: 'application',
        title: 'Residential Addition',
        submittedAt: new Date(now - 14 * day),
        estimatedCompletion: new Date(now + 7 * day),
        status: 'in_review',
        progress: 45,
        algorithm: 'permit_review_v2',
        factors: [
          { name: 'Zoning Compliance', value: 0.95, weight: 0.4 },
          { name: 'Structural Review', value: 0.8, weight: 0.3 },
          { name: 'Neighbor Notification', value: 1.0, weight: 0.2 },
          { name: 'Environmental Review', value: 0.6, weight: 0.1 },
        ],
      },
      {
        id: 'BN-2024-789012',
        type: 'application',
        title: 'SNAP Application',
        submittedAt: new Date(now - 5 * day),
        estimatedCompletion: new Date(now + 10 * day),
        status: 'pending_documents',
        progress: 30,
      },
    ],
    recentDecisions: [
      {
        id: 'BL-2024-567890',
        type: 'Business License',
        title: 'Food Vendor License',
        decidedAt: new Date(now - 7 * day),
        outcome: 'approved',
        explanation: 'All requirements met. Health inspection passed. Insurance verified.',
        appealDeadline: new Date(now + 23 * day),
        algorithmUsed: 'business_license_v1',
        inputData: { health_score: 95, insurance_valid: true, training_complete: true },
        outputScore: 0.92,
      },
      {
        id: 'HA-2024-345678',
        type: 'Housing',
        title: 'Emergency Rental Assistance',
        decidedAt: new Date(now - 3 * day),
        outcome: 'partial',
        explanation: 'Approved for 2 months based on income verification. Full term requires additional documentation.',
        appealDeadline: new Date(now + 27 * day),
        algorithmUsed: 'era_eligibility_v3',
        inputData: { income_ratio: 0.35, household_size: 3, covid_impact: true },
        outputScore: 0.75,
      },
    ],
    trustBreakdown: {
      overall: 0.78,
      byzantineTolerance: 0.34,
      lastVerification: new Date(now - 2 * day),
      components: [
        {
          name: 'Document Accuracy',
          score: 0.95,
          weight: 0.30,
          explanation: 'Accuracy of documents submitted across all applications',
          improvementActions: ['Ensure all documents are current', 'Double-check information before submitting'],
        },
        {
          name: 'Response Timeliness',
          score: 0.82,
          weight: 0.20,
          explanation: 'Speed of responding to requests for additional information',
          improvementActions: ['Respond to requests within 48 hours', 'Set up notifications'],
        },
        {
          name: 'Compliance History',
          score: 0.88,
          weight: 0.25,
          explanation: 'History of following through on commitments and requirements',
        },
        {
          name: 'Community Standing',
          score: 0.55,
          weight: 0.15,
          explanation: 'Participation in civic activities and community programs',
          improvementActions: ['Participate in public comment periods', 'Vote in local elections'],
        },
        {
          name: 'Verification Status',
          score: 0.70,
          weight: 0.10,
          explanation: 'Level of identity and credential verification',
          improvementActions: ['Complete enhanced verification', 'Link additional credentials'],
        },
      ],
      history: [
        { date: new Date(now - 30 * day), score: 0.75, delta: 0.03 },
        { date: new Date(now - 23 * day), score: 0.76, event: 'Document verified', delta: 0.01 },
        { date: new Date(now - 14 * day), score: 0.77, event: 'Application submitted', delta: 0.01 },
        { date: new Date(now - 7 * day), score: 0.78, event: 'License approved', delta: 0.01 },
      ],
    },
    appealableDecisions: [
      {
        id: 'HA-2024-345678',
        title: 'Emergency Rental Assistance',
        outcome: 'Partial Approval',
        deadline: new Date(now + 27 * day),
        grounds: ['income_verification', 'documentation_incomplete', 'program_limit_reached'],
        mediationAvailable: true,
        restorativeOption: false,
      },
    ],
    civicCredits: {
      balance: 150,
      monthlyAllocation: 100,
      lastAllocation: new Date(now - 15 * day),
      expiresAt: new Date(now + 15 * day),
      spentThisMonth: [
        { amount: 30, recipient: 'Local Food Bank', purpose: 'donation', timestamp: new Date(now - 10 * day) },
        { amount: 20, recipient: 'Community Garden', purpose: 'membership', timestamp: new Date(now - 5 * day) },
      ],
    },
    activeBenefits: [
      { id: 'BEN-001', name: 'Healthcare Subsidy', type: 'health', status: 'active', value: 250, renewalDate: new Date(now + 180 * day) },
      { id: 'BEN-002', name: 'Transit Pass', type: 'transport', status: 'active', value: 75 },
    ],
    upcomingVotes: [
      {
        proposalId: 'PROP-2024-042',
        title: 'Community Solar Initiative',
        deadline: new Date(now + 5 * day),
        yourDelegateVoted: false,
      },
      {
        proposalId: 'PROP-2024-043',
        title: 'Parks Budget Allocation',
        deadline: new Date(now + 12 * day),
        yourDelegateVoted: true,
        delegateName: 'Environmental Expert Council',
      },
    ],
  };
}

// ============================================================================
// Main Export
// ============================================================================

/**
 * Load dashboard data - tries Holochain conductor first, then REST API, finally mock data
 *
 * Priority:
 * 1. Holochain conductor (civic-client) - direct DHT access
 * 2. REST API (loadLiveDashboardData) - HTTP fallback
 * 3. Mock data (getMockDashboardData) - demo mode
 */
export async function loadDashboardData(agentId: string): Promise<{ data: DashboardData; isLive: boolean; source: 'conductor' | 'api' | 'mock' }> {
  // Try 1: Holochain conductor via civic-client
  try {
    const { loadLiveDashboardData: loadFromConductor, checkCivicConnection } = await import('./civic-client');

    const canConnect = await checkCivicConnection();
    if (canConnect) {
      const conductorData = await loadFromConductor(agentId);
      if (conductorData) {
        console.log('[Mycelix] Loaded data from Holochain conductor');
        return { data: conductorData, isLive: true, source: 'conductor' };
      }
    }
  } catch (error) {
    console.warn('[Mycelix] Holochain conductor not available:', error);
  }

  // Try 2: REST API
  try {
    const data = await loadLiveDashboardData(agentId);
    console.log('[Mycelix] Loaded data from REST API');
    return { data, isLive: true, source: 'api' };
  } catch (error) {
    console.warn('[Mycelix] REST API not available:', error);
  }

  // Try 3: Mock data (demo mode)
  console.log('[Mycelix] Using demo mode with mock data');
  return { data: getMockDashboardData(), isLive: false, source: 'mock' };
}
