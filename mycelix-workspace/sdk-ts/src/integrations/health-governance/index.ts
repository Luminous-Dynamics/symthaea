/**
 * Health ↔ Governance Integration Module
 *
 * Provides cross-module bridges between mycelix-health and mycelix-governance.
 * Key use cases:
 * - Public health policy voting (clinical guidelines, resource allocation)
 * - Clinical trial oversight with patient advocacy
 * - Provider credentialing governance
 * - Pandemic response coordination
 * - Health data privacy policy decisions
 *
 * @module @mycelix/sdk/integrations/health-governance
 */

import type { ActionHash } from '@holochain/client';

// ============================================================================
// Types
// ============================================================================

/**
 * Public health policy categories
 */
export type HealthPolicyCategory =
  | 'ClinicalGuidelines' // Treatment protocols, screening recommendations
  | 'ResourceAllocation' // Hospital beds, equipment, staffing
  | 'DataPrivacy' // Patient data sharing, consent policies
  | 'PandemicResponse' // Emergency protocols, quarantine rules
  | 'ProviderCredentialing' // Licensing, certification standards
  | 'ClinicalTrialOversight' // Trial ethics, patient safety
  | 'CommunityHealth' // Public health initiatives, prevention programs
  | 'MentalHealth' // Mental health parity, access policies
  | 'Other';

/**
 * Voter eligibility types for health policy decisions
 */
export type HealthVoterType =
  | 'Patient' // Affected patients
  | 'Provider' // Healthcare providers (doctors, nurses)
  | 'Researcher' // Clinical researchers
  | 'PublicHealthOfficial' // Government health officials
  | 'CommunityAdvocate' // Patient advocacy groups
  | 'TrialParticipant' // Clinical trial participants
  | 'Caregiver'; // Family caregivers

/**
 * Health policy proposal
 */
export interface HealthPolicyProposal {
  proposalId: string;
  category: HealthPolicyCategory;
  title: string;
  summary: string;
  fullText: string;
  evidenceLinks: string[]; // Citations, research papers
  impactedConditions?: string[]; // ICD-10 codes or condition names
  impactedPopulations?: string[];
  proposerId: string;
  proposerType: HealthVoterType;
  sponsoringOrganization?: string;
  jurisdictionLevel: 'Local' | 'Regional' | 'National' | 'International';
  status: 'Draft' | 'Review' | 'Voting' | 'Passed' | 'Rejected' | 'Implemented';
  votingStartsAt?: number;
  votingEndsAt?: number;
  quorumRequired: number; // Percentage of eligible voters
  approvalThreshold: number; // Percentage for passage
  createdAt: number;
  updatedAt: number;
}

/**
 * Health policy vote
 */
export interface HealthPolicyVote {
  voteId: string;
  proposalId: string;
  voterId: string;
  voterType: HealthVoterType;
  patientHash?: ActionHash; // If voter is a patient
  providerHash?: ActionHash; // If voter is a provider
  vote: 'For' | 'Against' | 'Abstain';
  weight: number; // Based on stakeholder relevance
  rationale?: string;
  declaredConflicts?: string[]; // Conflict of interest disclosures
  castAt: number;
}

/**
 * Clinical trial governance oversight
 */
export interface TrialOversight {
  oversightId: string;
  trialId: string;
  trialTitle: string;
  sponsor: string;
  principalInvestigator: string;
  patientAdvocates: string[];
  safetyMonitors: string[];
  irb: string; // Institutional Review Board
  status: 'Pending' | 'Approved' | 'Active' | 'Suspended' | 'Completed' | 'Terminated';
  participantCount: number;
  adverseEventCount: number;
  lastReviewDate: number;
  nextReviewDate: number;
  patientVoiceEnabled: boolean; // Whether participants can vote on trial modifications
}

/**
 * Trial participant voice (voting on trial decisions)
 */
export interface TrialParticipantVoice {
  decisionId: string;
  trialId: string;
  decisionType: 'ProtocolModification' | 'EarlyTermination' | 'SafetyConcern' | 'ConsentUpdate';
  description: string;
  participantVotes: {
    for: number;
    against: number;
    abstain: number;
  };
  participantComments: Array<{
    participantId: string;
    comment: string;
    submittedAt: number;
  }>;
  researcherResponse?: string;
  resolution?: string;
  status: 'Open' | 'UnderReview' | 'Resolved';
  createdAt: number;
  resolvedAt?: number;
}

/**
 * Provider credentialing decision
 */
export interface CredentialingDecision {
  decisionId: string;
  providerId: string;
  providerName: string;
  credentialType: 'License' | 'Certification' | 'Privileges' | 'Specialty';
  requestedAction: 'Grant' | 'Renew' | 'Suspend' | 'Revoke';
  reviewCommittee: string[];
  patientRepresentatives: string[];
  status: 'Pending' | 'UnderReview' | 'Approved' | 'Denied' | 'Deferred';
  voteTally?: {
    approve: number;
    deny: number;
    defer: number;
  };
  conditions?: string[];
  effectiveDate?: number;
  expirationDate?: number;
  createdAt: number;
  decidedAt?: number;
}

/**
 * Pandemic response decision
 */
export interface PandemicResponseDecision {
  decisionId: string;
  pathogenId: string;
  pathogenName: string;
  threatLevel: 'Low' | 'Moderate' | 'High' | 'Critical';
  proposedMeasures: string[];
  affectedRegions: string[];
  estimatedImpact: {
    populationAffected: number;
    resourcesRequired: string[];
    economicCost?: number;
  };
  scientificAdvisory: string;
  publicComment: Array<{
    commenterId: string;
    comment: string;
    submittedAt: number;
  }>;
  decisionOutcome?: 'Implement' | 'Modify' | 'Reject';
  votingOpen: boolean;
  voteTally?: {
    implement: number;
    modify: number;
    reject: number;
  };
  implementationDate?: number;
  createdAt: number;
  decidedAt?: number;
}

/**
 * Health data privacy policy
 */
export interface DataPrivacyPolicy {
  policyId: string;
  title: string;
  scope: 'Research' | 'Treatment' | 'PublicHealth' | 'Commercial' | 'All';
  dataCategories: string[];
  allowedUses: string[];
  prohibitedUses: string[];
  consentRequirements: 'OptIn' | 'OptOut' | 'Informed' | 'Waived';
  retentionPeriod: number; // Days
  deidentificationLevel: 'None' | 'Limited' | 'Safe Harbor' | 'Expert Determination';
  patientRights: string[];
  enforcementMechanisms: string[];
  status: 'Draft' | 'Review' | 'Active' | 'Superseded';
  approvedBy?: string[];
  effectiveDate?: number;
  createdAt: number;
  updatedAt: number;
}

// ============================================================================
// Zome Callable Interface
// ============================================================================

export interface ZomeCallable {
  callZome(args: {
    role_name: string;
    zome_name: string;
    fn_name: string;
    payload: unknown;
  }): Promise<unknown>;
}

// ============================================================================
// Client Classes
// ============================================================================

const GOVERNANCE_ROLE = 'governance';

/**
 * Client for health policy proposal management
 */
export class HealthPolicyClient {
  constructor(readonly client: ZomeCallable) {}

  /**
   * Create a health policy proposal
   */
  async createProposal(
    input: Omit<HealthPolicyProposal, 'proposalId' | 'status' | 'createdAt' | 'updatedAt'>
  ): Promise<HealthPolicyProposal> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'create_health_proposal',
      payload: input,
    }) as Promise<HealthPolicyProposal>;
  }

  /**
   * Get active proposals by category
   */
  async getProposalsByCategory(category: HealthPolicyCategory): Promise<HealthPolicyProposal[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'get_proposals_by_category',
      payload: category,
    }) as Promise<HealthPolicyProposal[]>;
  }

  /**
   * Get proposals affecting a specific condition
   */
  async getProposalsByCondition(conditionCode: string): Promise<HealthPolicyProposal[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'get_proposals_by_condition',
      payload: conditionCode,
    }) as Promise<HealthPolicyProposal[]>;
  }

  /**
   * Cast a vote on a health policy proposal
   */
  async castVote(
    vote: Omit<HealthPolicyVote, 'voteId' | 'castAt'>
  ): Promise<HealthPolicyVote> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'cast_health_policy_vote',
      payload: vote,
    }) as Promise<HealthPolicyVote>;
  }

  /**
   * Get vote results for a proposal
   */
  async getVoteResults(proposalId: string): Promise<{
    totalVotes: number;
    forVotes: number;
    againstVotes: number;
    abstainVotes: number;
    weightedFor: number;
    weightedAgainst: number;
    quorumMet: boolean;
    passed: boolean;
    votesByType: Record<HealthVoterType, { for: number; against: number; abstain: number }>;
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'get_vote_results',
      payload: proposalId,
    }) as Promise<{
      totalVotes: number;
      forVotes: number;
      againstVotes: number;
      abstainVotes: number;
      weightedFor: number;
      weightedAgainst: number;
      quorumMet: boolean;
      passed: boolean;
      votesByType: Record<HealthVoterType, { for: number; against: number; abstain: number }>;
    }>;
  }

  /**
   * Get voter eligibility for a proposal
   */
  async checkVoterEligibility(
    proposalId: string,
    voterId: string
  ): Promise<{
    eligible: boolean;
    voterType: HealthVoterType | null;
    weight: number;
    reason?: string;
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'health_policy',
      fn_name: 'check_voter_eligibility',
      payload: { proposal_id: proposalId, voter_id: voterId },
    }) as Promise<{
      eligible: boolean;
      voterType: HealthVoterType | null;
      weight: number;
      reason?: string;
    }>;
  }
}

/**
 * Client for clinical trial oversight governance
 */
export class TrialOversightClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Get trial oversight status
   */
  async getTrialOversight(trialId: string): Promise<TrialOversight | null> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'trial_oversight',
      fn_name: 'get_trial_oversight',
      payload: trialId,
    }) as Promise<TrialOversight | null>;
  }

  /**
   * Submit a participant voice decision request
   */
  async submitParticipantVoice(
    input: Omit<TrialParticipantVoice, 'decisionId' | 'participantVotes' | 'participantComments' | 'status' | 'createdAt'>
  ): Promise<TrialParticipantVoice> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'trial_oversight',
      fn_name: 'submit_participant_voice',
      payload: input,
    }) as Promise<TrialParticipantVoice>;
  }

  /**
   * Vote on a trial participant voice decision
   */
  async voteOnTrialDecision(
    decisionId: string,
    participantId: string,
    vote: 'for' | 'against' | 'abstain',
    comment?: string
  ): Promise<void> {
    await this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'trial_oversight',
      fn_name: 'vote_on_trial_decision',
      payload: {
        decision_id: decisionId,
        participant_id: participantId,
        vote,
        comment,
      },
    });
  }

  /**
   * Report adverse event requiring oversight review
   */
  async reportAdverseEvent(
    trialId: string,
    eventDescription: string,
    severity: 'Minor' | 'Moderate' | 'Serious' | 'LifeThreatening'
  ): Promise<{
    acknowledged: boolean;
    reviewRequired: boolean;
    oversightNotified: boolean;
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'trial_oversight',
      fn_name: 'report_adverse_event',
      payload: { trial_id: trialId, description: eventDescription, severity },
    }) as Promise<{
      acknowledged: boolean;
      reviewRequired: boolean;
      oversightNotified: boolean;
    }>;
  }
}

/**
 * Client for pandemic response governance
 */
export class PandemicResponseClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Get current active pandemic response decisions
   */
  async getActiveDecisions(): Promise<PandemicResponseDecision[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'pandemic_response',
      fn_name: 'get_active_decisions',
      payload: null,
    }) as Promise<PandemicResponseDecision[]>;
  }

  /**
   * Submit public comment on pandemic response decision
   */
  async submitPublicComment(
    decisionId: string,
    comment: string
  ): Promise<void> {
    await this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'pandemic_response',
      fn_name: 'submit_public_comment',
      payload: { decision_id: decisionId, comment },
    });
  }

  /**
   * Vote on pandemic response decision (for eligible voters)
   */
  async voteOnDecision(
    decisionId: string,
    vote: 'implement' | 'modify' | 'reject',
    voterRole: HealthVoterType
  ): Promise<void> {
    await this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'pandemic_response',
      fn_name: 'vote_on_decision',
      payload: { decision_id: decisionId, vote, voter_role: voterRole },
    });
  }
}

/**
 * Client for data privacy policy governance
 */
export class DataPrivacyPolicyClient {
  constructor(private readonly client: ZomeCallable) {}

  /**
   * Get active privacy policies
   */
  async getActivePolicies(scope?: DataPrivacyPolicy['scope']): Promise<DataPrivacyPolicy[]> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'data_privacy',
      fn_name: 'get_active_policies',
      payload: scope || null,
    }) as Promise<DataPrivacyPolicy[]>;
  }

  /**
   * Check if a data use is permitted under current policies
   */
  async checkDataUsePermission(
    dataCategory: string,
    intendedUse: string,
    consentStatus: 'OptedIn' | 'OptedOut' | 'NotAsked'
  ): Promise<{
    permitted: boolean;
    applicablePolicies: string[];
    conditions?: string[];
    reason?: string;
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'data_privacy',
      fn_name: 'check_data_use_permission',
      payload: {
        data_category: dataCategory,
        intended_use: intendedUse,
        consent_status: consentStatus,
      },
    }) as Promise<{
      permitted: boolean;
      applicablePolicies: string[];
      conditions?: string[];
      reason?: string;
    }>;
  }

  /**
   * Submit a data use request for governance review
   */
  async submitDataUseRequest(
    dataCategory: string,
    intendedUse: string,
    justification: string,
    dataMinimization: string[]
  ): Promise<{
    requestId: string;
    status: 'Submitted' | 'UnderReview' | 'Approved' | 'Denied';
  }> {
    return this.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'data_privacy',
      fn_name: 'submit_data_use_request',
      payload: {
        data_category: dataCategory,
        intended_use: intendedUse,
        justification,
        data_minimization: dataMinimization,
      },
    }) as Promise<{
      requestId: string;
      status: 'Submitted' | 'UnderReview' | 'Approved' | 'Denied';
    }>;
  }
}

// ============================================================================
// Unified Health-Governance Bridge
// ============================================================================

/**
 * Unified interface for health-governance integration
 */
export class HealthGovernanceBridge {
  readonly policy: HealthPolicyClient;
  readonly trialOversight: TrialOversightClient;
  readonly pandemicResponse: PandemicResponseClient;
  readonly dataPrivacy: DataPrivacyPolicyClient;

  constructor(client: ZomeCallable) {
    this.policy = new HealthPolicyClient(client);
    this.trialOversight = new TrialOversightClient(client);
    this.pandemicResponse = new PandemicResponseClient(client);
    this.dataPrivacy = new DataPrivacyPolicyClient(client);
  }

  /**
   * Get all proposals that might affect a specific patient
   *
   * Aggregates proposals based on patient's conditions, medications, and location
   */
  async getPatientRelevantProposals(
    _patientHash: ActionHash,
    patientConditions: string[],
    patientLocation: string
  ): Promise<{
    conditionRelated: HealthPolicyProposal[];
    regionRelated: HealthPolicyProposal[];
    privacyRelated: HealthPolicyProposal[];
  }> {
    // Fetch all active proposals in parallel
    const [conditionProposals, pandemicDecisions, privacyPolicies] = await Promise.all([
      // Get proposals for each condition
      Promise.all(
        patientConditions.map((c) => this.policy.getProposalsByCondition(c))
      ),
      this.pandemicResponse.getActiveDecisions(),
      this.policy.getProposalsByCategory('DataPrivacy'),
    ]);

    // Filter pandemic decisions by region
    const regionRelated: HealthPolicyProposal[] = pandemicDecisions
      .filter((d) => d.affectedRegions.some((r) => patientLocation.includes(r)))
      .map((d) => ({
        proposalId: d.decisionId,
        category: 'PandemicResponse' as HealthPolicyCategory,
        title: `Pandemic Response: ${d.pathogenName}`,
        summary: d.proposedMeasures.join(', '),
        fullText: d.scientificAdvisory,
        evidenceLinks: [],
        proposerId: 'system',
        proposerType: 'PublicHealthOfficial' as HealthVoterType,
        jurisdictionLevel: 'Regional' as const,
        status: d.votingOpen ? 'Voting' as const : 'Review' as const,
        quorumRequired: 0.5,
        approvalThreshold: 0.6,
        createdAt: d.createdAt,
        updatedAt: d.createdAt,
      }));

    return {
      conditionRelated: conditionProposals.flat(),
      regionRelated,
      privacyRelated: privacyPolicies,
    };
  }

  /**
   * Check patient's voting eligibility across all governance areas
   */
  async getPatientVotingRights(
    patientHash: ActionHash,
    patientDid: string
  ): Promise<{
    eligibleProposals: string[];
    eligibleTrialDecisions: string[];
    eligiblePandemicVotes: string[];
    voterTypes: HealthVoterType[];
  }> {
    // This would query across governance modules to find where patient can vote
    return this.policy.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'voter_registry',
      fn_name: 'get_patient_voting_rights',
      payload: { patient_hash: patientHash, patient_did: patientDid },
    }) as Promise<{
      eligibleProposals: string[];
      eligibleTrialDecisions: string[];
      eligiblePandemicVotes: string[];
      voterTypes: HealthVoterType[];
    }>;
  }

  /**
   * Get provider governance responsibilities
   */
  async getProviderGovernanceRoles(
    providerHash: ActionHash
  ): Promise<{
    committeesMembership: string[];
    pendingReviews: number;
    openVotes: number;
    credentialingDecisions: CredentialingDecision[];
  }> {
    return this.policy.client.callZome({
      role_name: GOVERNANCE_ROLE,
      zome_name: 'provider_governance',
      fn_name: 'get_provider_roles',
      payload: providerHash,
    }) as Promise<{
      committeesMembership: string[];
      pendingReviews: number;
      openVotes: number;
      credentialingDecisions: CredentialingDecision[];
    }>;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

/**
 * Create a Health-Governance bridge instance
 */
export function createHealthGovernanceBridge(client: ZomeCallable): HealthGovernanceBridge {
  return new HealthGovernanceBridge(client);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate voter weight based on stakeholder type and relevance
 */
export function calculateVoterWeight(
  voterType: HealthVoterType,
  proposalCategory: HealthPolicyCategory,
  isDirectlyAffected: boolean
): number {
  let baseWeight = 1.0;

  // Stakeholder type weights
  switch (voterType) {
    case 'Patient':
      baseWeight = 1.0;
      break;
    case 'Provider':
      baseWeight = 1.2;
      break;
    case 'Researcher':
      baseWeight = 1.1;
      break;
    case 'PublicHealthOfficial':
      baseWeight = 1.3;
      break;
    case 'CommunityAdvocate':
      baseWeight = 0.9;
      break;
    case 'TrialParticipant':
      baseWeight = 1.5; // High weight for directly affected
      break;
    case 'Caregiver':
      baseWeight = 0.8;
      break;
  }

  // Category-specific adjustments
  if (proposalCategory === 'ClinicalGuidelines' && voterType === 'Provider') {
    baseWeight *= 1.3; // Providers have more weight on clinical matters
  }
  if (proposalCategory === 'DataPrivacy' && voterType === 'Patient') {
    baseWeight *= 1.4; // Patients have more weight on privacy
  }
  if (proposalCategory === 'ClinicalTrialOversight' && voterType === 'TrialParticipant') {
    baseWeight *= 1.5;
  }

  // Direct impact multiplier
  if (isDirectlyAffected) {
    baseWeight *= 1.25;
  }

  return Math.round(baseWeight * 100) / 100;
}

/**
 * Determine if a proposal has reached quorum and passed
 */
export function evaluateProposalOutcome(
  proposal: HealthPolicyProposal,
  voteResults: {
    weightedFor: number;
    weightedAgainst: number;
    totalWeight: number;
    eligibleWeight: number;
  }
): {
  quorumMet: boolean;
  passed: boolean;
  participationRate: number;
  approvalRate: number;
} {
  const participationRate = voteResults.totalWeight / voteResults.eligibleWeight;
  const quorumMet = participationRate >= proposal.quorumRequired;

  const approvalRate =
    voteResults.totalWeight > 0
      ? voteResults.weightedFor / voteResults.totalWeight
      : 0;

  const passed = quorumMet && approvalRate >= proposal.approvalThreshold;

  return {
    quorumMet,
    passed,
    participationRate: Math.round(participationRate * 1000) / 10, // Percentage with 1 decimal
    approvalRate: Math.round(approvalRate * 1000) / 10,
  };
}

/**
 * Check if a voter has a conflict of interest
 */
export function detectConflictOfInterest(
  voterType: HealthVoterType,
  voterId: string,
  proposal: HealthPolicyProposal
): { hasConflict: boolean; conflictType?: string; mustRecuse?: boolean } {
  // Provider voting on their own credentialing
  if (
    voterType === 'Provider' &&
    proposal.category === 'ProviderCredentialing' &&
    proposal.fullText.includes(voterId)
  ) {
    return {
      hasConflict: true,
      conflictType: 'Direct personal interest',
      mustRecuse: true,
    };
  }

  // Researcher voting on their own trial
  if (
    voterType === 'Researcher' &&
    proposal.category === 'ClinicalTrialOversight' &&
    proposal.proposerId === voterId
  ) {
    return {
      hasConflict: true,
      conflictType: 'Principal investigator on trial',
      mustRecuse: true,
    };
  }

  return { hasConflict: false };
}

// ============================================================================
// Exports
// ============================================================================

export default {
  HealthGovernanceBridge,
  HealthPolicyClient,
  TrialOversightClient,
  PandemicResponseClient,
  DataPrivacyPolicyClient,
  createHealthGovernanceBridge,
  calculateVoterWeight,
  evaluateProposalOutcome,
  detectConflictOfInterest,
};
