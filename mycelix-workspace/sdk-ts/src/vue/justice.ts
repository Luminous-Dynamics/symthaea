/**
 * @mycelix/sdk Vue 3 Composables for Justice Module
 *
 * Provides Vue 3 Composition API composables for the Justice hApp integration.
 *
 * @packageDocumentation
 * @module vue/justice
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface Case {
  id: string;
  complainant_did: string;
  respondent_did: string;
  title: string;
  description: string;
  category: string;
  tier: 'mediation' | 'arbitration' | 'appeal';
  status: 'filed' | 'mediation' | 'arbitration' | 'deliberation' | 'judgment' | 'appeal' | 'closed';
  filed_at: number;
  closed_at?: number;
  judgment?: Judgment;
}

export interface Evidence {
  id: string;
  case_id: string;
  submitter_did: string;
  type: 'document' | 'testimony' | 'record' | 'media' | 'expert_opinion';
  title: string;
  description: string;
  content_hash: string;
  storage_uri?: string;
  epistemic_classification: {
    empirical: number;
    normative: number;
    metaphysical: number;
  };
  admitted: boolean;
  submitted_at: number;
}

export interface Judgment {
  id: string;
  case_id: string;
  verdict: 'favor_complainant' | 'favor_respondent' | 'dismissed' | 'settled';
  reasoning: string;
  remedies: Remedy[];
  jurors: string[];
  vote_breakdown: {
    for_complainant: number;
    for_respondent: number;
    abstain: number;
  };
  issued_at: number;
}

export interface Remedy {
  type: 'monetary' | 'behavioral' | 'restoration' | 'apology';
  description: string;
  target_did: string;
  amount?: number;
  currency?: string;
  deadline?: number;
  verified: boolean;
}

export interface Juror {
  did: string;
  qualifications: string[];
  cases_served: number;
  matl_score: number;
  available: boolean;
}

// ============================================================================
// Case Composables
// ============================================================================

export interface UseCaseOptions extends UseQueryOptions {
  /** Include evidence */
  includeEvidence?: boolean;
  /** Include judgment */
  includeJudgment?: boolean;
}

/**
 * Composable to file a case
 */
export function useFileCase(): UseMutationReturn<
  HolochainRecord<Case>,
  {
    respondent_did: string;
    title: string;
    description: string;
    category: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a case
 */
export function useCase(
  _caseId: string,
  _options?: UseCaseOptions
): UseQueryReturn<HolochainRecord<Case> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get cases by complainant
 */
export function useCasesByComplainant(
  _complainantDid: string,
  _options?: UseCaseOptions
): UseQueryReturn<HolochainRecord<Case>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get cases by respondent
 */
export function useCasesByRespondent(
  _respondentDid: string,
  _options?: UseCaseOptions
): UseQueryReturn<HolochainRecord<Case>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get cases by party (either complainant or respondent)
 */
export function useCasesByParty(
  _partyDid: string,
  _options?: UseCaseOptions
): UseQueryReturn<HolochainRecord<Case>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to escalate case to next tier
 */
export function useEscalateCase(): UseMutationReturn<
  HolochainRecord<Case>,
  {
    case_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to request mediation
 */
export function useRequestMediation(): UseMutationReturn<HolochainRecord<Case>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to settle case
 */
export function useSettleCase(): UseMutationReturn<
  HolochainRecord<Case>,
  {
    case_id: string;
    settlement_terms: string;
    agreed_remedies: Remedy[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Evidence Composables
// ============================================================================

/**
 * Composable to submit evidence
 */
export function useSubmitEvidence(): UseMutationReturn<
  HolochainRecord<Evidence>,
  {
    case_id: string;
    type: Evidence['type'];
    title: string;
    description: string;
    content_hash: string;
    storage_uri?: string;
    epistemic_classification?: {
      empirical: number;
      normative: number;
      metaphysical: number;
    };
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get evidence for a case
 */
export function useEvidenceForCase(_caseId: string): UseQueryReturn<HolochainRecord<Evidence>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get evidence by submitter
 */
export function useEvidenceBySubmitter(
  _submitterDid: string
): UseQueryReturn<HolochainRecord<Evidence>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to challenge evidence
 */
export function useChallengeEvidence(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    evidence_id: string;
    challenge_reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to admit/reject evidence (juror only)
 */
export function useRuleOnEvidence(): UseMutationReturn<
  HolochainRecord<Evidence>,
  {
    evidence_id: string;
    admitted: boolean;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Arbitration Composables
// ============================================================================

/**
 * Composable to get available jurors
 */
export function useAvailableJurors(_category: string): UseQueryReturn<Juror[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to assign jurors
 */
export function useAssignJurors(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    juror_count: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to recuse as juror
 */
export function useRecuseAsJuror(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    reason: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to submit juror vote
 */
export function useSubmitJurorVote(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    verdict: 'favor_complainant' | 'favor_respondent' | 'abstain';
    reasoning: string;
    recommended_remedies?: Remedy[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Judgment Composables
// ============================================================================

/**
 * Composable to get judgment for a case
 */
export function useJudgment(_caseId: string): UseQueryReturn<HolochainRecord<Judgment> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to appeal judgment
 */
export function useAppealJudgment(): UseMutationReturn<
  HolochainRecord<Case>,
  {
    case_id: string;
    grounds_for_appeal: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to verify remedy completion
 */
export function useVerifyRemedyCompletion(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    remedy_index: number;
    verification_evidence: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Enforcement Composables
// ============================================================================

/**
 * Composable to request enforcement
 */
export function useRequestEnforcement(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    unmet_remedy_indices: number[];
    description: string;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to execute cross-hApp enforcement
 */
export function useExecuteEnforcement(): UseMutationReturn<
  HolochainRecord<unknown>,
  {
    case_id: string;
    enforcement_actions: Array<{
      target_happ: string;
      action_type: string;
      parameters: Record<string, unknown>;
    }>;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
