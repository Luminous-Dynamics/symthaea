/**
 * React Hooks for Mycelix Justice Module
 *
 * Provides React hooks for dispute resolution, arbitration, and enforcement.
 *
 * @module @mycelix/sdk/react/justice
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Case,
  FileCaseInput,
  Evidence,
  SubmitEvidenceInput,
  Decision,
  DecisionOutcome,
  Remedy,
  MediatorProfile,
  ArbitratorProfile,
  RegisterMediatorInput,
  RegisterArbitratorInput,
  Enforcement,
  RequestEnforcementInput,
  EnforcementStatus,
  CaseCategory,
  HolochainRecord,
} from '../justice/index.js';

// ============================================================================
// Case Hooks
// ============================================================================

/**
 * Hook to fetch a case by ID
 */
export function useCaseById(_caseId: string): QueryState<HolochainRecord<Case> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch cases by complainant
 */
export function useCasesByComplainant(
  _complainantDid: string
): QueryState<HolochainRecord<Case>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch cases by respondent
 */
export function useCasesByRespondent(_respondentDid: string): QueryState<HolochainRecord<Case>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch cases where user is a party
 */
export function useMyCases(_did: string): QueryState<HolochainRecord<Case>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to file a case
 */
export function useFileCaseMutation(): MutationState<HolochainRecord<Case>, FileCaseInput> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to withdraw a case
 */
export function useWithdrawCase(): MutationState<
  HolochainRecord<Case>,
  { caseId: string; reason: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to respond to a case
 */
export function useRespondToCase(): MutationState<
  HolochainRecord<Case>,
  { caseId: string; response: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Evidence Hooks
// ============================================================================

/**
 * Hook to fetch evidence for a case
 */
export function useEvidenceForCase(_caseId: string): QueryState<HolochainRecord<Evidence>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to submit evidence
 */
export function useSubmitEvidenceMutation(): MutationState<
  HolochainRecord<Evidence>,
  SubmitEvidenceInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to verify evidence (arbitrator only)
 */
export function useVerifyEvidence(): MutationState<HolochainRecord<Evidence>, string> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to challenge evidence
 */
export function useChallengeEvidence(): MutationState<
  void,
  { evidenceId: string; reason: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Arbitration Hooks
// ============================================================================

/**
 * Hook to fetch available mediators for a category
 */
export function useAvailableMediators(
  _category: CaseCategory
): QueryState<HolochainRecord<MediatorProfile>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch available arbitrators for a category
 */
export function useAvailableArbitrators(
  _category: CaseCategory
): QueryState<HolochainRecord<ArbitratorProfile>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to register as mediator
 */
export function useRegisterAsMediator(): MutationState<
  HolochainRecord<MediatorProfile>,
  RegisterMediatorInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to register as arbitrator
 */
export function useRegisterAsArbitrator(): MutationState<
  HolochainRecord<ArbitratorProfile>,
  RegisterArbitratorInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to assign mediator to case
 */
export function useAssignMediator(): MutationState<
  HolochainRecord<Case>,
  { caseId: string; mediatorDid: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to escalate to arbitration
 */
export function useEscalateToArbitration(): MutationState<
  HolochainRecord<Case>,
  { caseId: string; arbitratorDids: string[] }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch decision for a case
 */
export function useDecisionForCase(_caseId: string): QueryState<HolochainRecord<Decision> | null> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to render a decision
 */
export function useRenderDecision(): MutationState<
  HolochainRecord<Decision>,
  { caseId: string; outcome: DecisionOutcome; reasoning: string; remedies: Remedy[] }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to file an appeal
 */
export function useFileAppeal(): MutationState<
  HolochainRecord<Case>,
  { decisionId: string; grounds: string }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

// ============================================================================
// Enforcement Hooks
// ============================================================================

/**
 * Hook to fetch pending enforcements
 */
export function usePendingEnforcementsJustice(
  _targetHapp?: string
): QueryState<HolochainRecord<Enforcement>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to fetch enforcements for a decision
 */
export function useEnforcementsForDecision(
  _decisionId: string
): QueryState<HolochainRecord<Enforcement>[]> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to request enforcement
 */
export function useRequestEnforcementMutation(): MutationState<
  HolochainRecord<Enforcement>,
  RequestEnforcementInput
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to acknowledge enforcement
 */
export function useAcknowledgeEnforcementJustice(): MutationState<
  HolochainRecord<Enforcement>,
  { enforcementId: string; status: EnforcementStatus }
> {
  throw new Error('React hooks require React. This is a type stub.');
}

/**
 * Hook to mark remedy as completed
 */
export function useMarkRemedyCompleted(): MutationState<
  HolochainRecord<Decision>,
  { decisionId: string; remedyIndex: number }
> {
  throw new Error('React hooks require React. This is a type stub.');
}
