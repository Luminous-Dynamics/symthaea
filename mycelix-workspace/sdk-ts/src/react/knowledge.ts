// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk React Hooks for Knowledge Module
 *
 * Provides React hooks for the Knowledge hApp integration.
 * Works with React 18+ and supports Suspense, concurrent features.
 *
 * @packageDocumentation
 * @module react/knowledge
 */

import type { QueryState, MutationState } from './index.js';
import type {
  Claim,
  UpdateClaimInput,
  Evidence,
  ChallengeClaimInput,
  EpistemicRangeSearch,
  Relationship,
  FindPathInput,
  Ontology,
  Concept,
  SavedQuery,
  ExecuteQueryInput,
  QueryExecutionResult,
  Inference,
  AssessCredibilityInput,
  DetectPatternsInput,
  VerifyInferenceInput,
  CredibilityScore,
  Pattern,
  QueryKnowledgeInput,
  QueryKnowledgeResult,
  FactCheckInput,
  FactCheckResult,
  RegisterClaimInput,
  BroadcastKnowledgeEventInput,
  KnowledgeBridgeEvent,
  ClaimChallenge,
  GraphStats,
  HolochainRecord,
} from '../knowledge/index.js';

// ============================================================================
// Claims Hooks
// ============================================================================

export interface UseClaimOptions {
  /** Auto-fetch on mount */
  enabled?: boolean;
  /** Refetch interval in ms */
  refetchInterval?: number;
}

/**
 * Hook to submit a new claim
 *
 * @example
 * ```tsx
 * function SubmitClaim() {
 *   const { mutate, loading, error } = useSubmitClaim();
 *
 *   const handleSubmit = async (claim: Claim) => {
 *     await mutate(claim);
 *   };
 *
 *   return <ClaimForm onSubmit={handleSubmit} loading={loading} />;
 * }
 * ```
 */
export function useSubmitClaim(): MutationState<HolochainRecord<Claim>, Claim> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get a claim by ID
 */
export function useClaim(
  _claimId: string,
  _options?: UseClaimOptions
): QueryState<HolochainRecord<Claim> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get claims by author
 */
export function useClaimsByAuthor(
  _authorDid: string,
  _options?: UseClaimOptions
): QueryState<HolochainRecord<Claim>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get claims by tag
 */
export function useClaimsByTag(
  _tag: string,
  _options?: UseClaimOptions
): QueryState<HolochainRecord<Claim>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to update a claim
 */
export function useUpdateClaim(): MutationState<HolochainRecord<Claim>, UpdateClaimInput> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to add evidence to a claim
 */
export function useAddEvidence(): MutationState<HolochainRecord<Evidence>, Evidence> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get evidence for a claim
 */
export function useClaimEvidence(_claimId: string): QueryState<HolochainRecord<Evidence>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to challenge a claim
 */
export function useChallengeClaim(): MutationState<
  HolochainRecord<ClaimChallenge>,
  ChallengeClaimInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get challenges for a claim
 */
export function useClaimChallenges(
  _claimId: string
): QueryState<HolochainRecord<ClaimChallenge>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to search claims by epistemic range
 *
 * @example
 * ```tsx
 * function HighEmpiricalClaims() {
 *   const { data, loading } = useSearchByEpistemicRange({
 *     e_min: 0.8, e_max: 1.0,
 *     n_min: 0.0, n_max: 1.0,
 *     m_min: 0.0, m_max: 1.0,
 *   });
 *
 *   return <ClaimList claims={data || []} loading={loading} />;
 * }
 * ```
 */
export function useSearchByEpistemicRange(
  _input: EpistemicRangeSearch,
  _options?: UseClaimOptions
): QueryState<HolochainRecord<Claim>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Graph Hooks
// ============================================================================

/**
 * Hook to create a relationship between claims
 */
export function useCreateRelationship(): MutationState<
  HolochainRecord<Relationship>,
  Relationship
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get outgoing relationships from a claim
 */
export function useOutgoingRelationships(
  _claimId: string
): QueryState<HolochainRecord<Relationship>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get incoming relationships to a claim
 */
export function useIncomingRelationships(
  _claimId: string
): QueryState<HolochainRecord<Relationship>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to find path between claims
 */
export function useFindPath(_input: FindPathInput): QueryState<string[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to create an ontology
 */
export function useCreateOntology(): MutationState<HolochainRecord<Ontology>, Ontology> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to create a concept in an ontology
 */
export function useCreateConcept(): MutationState<HolochainRecord<Concept>, Concept> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get concepts in an ontology
 */
export function useOntologyConcepts(_ontologyId: string): QueryState<HolochainRecord<Concept>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get child concepts
 */
export function useChildConcepts(_conceptId: string): QueryState<HolochainRecord<Concept>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get graph statistics
 */
export function useGraphStats(): QueryState<GraphStats> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Query Hooks
// ============================================================================

/**
 * Hook to execute a query on the knowledge graph
 *
 * @example
 * ```tsx
 * function QueryRunner() {
 *   const { mutate, data, loading } = useExecuteQuery();
 *
 *   const runQuery = async () => {
 *     const results = await mutate({
 *       query: "SELECT * FROM claims WHERE tag = 'climate'",
 *       use_cache: true,
 *       limit: 100,
 *     });
 *   };
 *
 *   return <QueryResults results={data} loading={loading} />;
 * }
 * ```
 */
export function useExecuteQuery(): MutationState<QueryExecutionResult, ExecuteQueryInput> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to save a query for reuse
 */
export function useSaveQuery(): MutationState<HolochainRecord<SavedQuery>, SavedQuery> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get user's saved queries
 */
export function useMyQueries(_creatorDid: string): QueryState<HolochainRecord<SavedQuery>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get public queries
 */
export function usePublicQueries(): QueryState<HolochainRecord<SavedQuery>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Inference Hooks
// ============================================================================

/**
 * Hook to create an inference
 */
export function useCreateInference(): MutationState<HolochainRecord<Inference>, Inference> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get inferences for a claim
 */
export function useClaimInferences(_claimId: string): QueryState<HolochainRecord<Inference>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to assess credibility of a subject
 */
export function useAssessCredibility(): MutationState<
  HolochainRecord<CredibilityScore>,
  AssessCredibilityInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get credibility score for a subject
 */
export function useCredibility(
  _subject: string
): QueryState<HolochainRecord<CredibilityScore> | null> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to detect patterns in the knowledge graph
 */
export function useDetectPatterns(): MutationState<
  HolochainRecord<Pattern>[],
  DetectPatternsInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to find contradictions between claims
 */
export function useFindContradictions(
  _claimIds: string[]
): QueryState<HolochainRecord<Inference>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to verify an inference
 */
export function useVerifyInference(): MutationState<
  HolochainRecord<Inference>,
  VerifyInferenceInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

// ============================================================================
// Knowledge Bridge Hooks
// ============================================================================

/**
 * Hook to query knowledge from another hApp
 */
export function useQueryKnowledge(): MutationState<QueryKnowledgeResult, QueryKnowledgeInput> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to fact-check a statement
 *
 * @example
 * ```tsx
 * function FactChecker() {
 *   const { mutate, data, loading } = useFactCheck();
 *
 *   const checkFact = async (statement: string) => {
 *     const result = await mutate({
 *       source_happ: 'my-news-app',
 *       statement,
 *     });
 *     console.log(`Verdict: ${result.verdict}`);
 *   };
 *
 *   return <FactCheckUI onCheck={checkFact} result={data} />;
 * }
 * ```
 */
export function useFactCheck(): MutationState<FactCheckResult, FactCheckInput> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to register an external claim
 */
export function useRegisterExternalClaim(): MutationState<
  HolochainRecord<unknown>,
  RegisterClaimInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to broadcast a knowledge event
 */
export function useBroadcastKnowledgeEvent(): MutationState<
  HolochainRecord<KnowledgeBridgeEvent>,
  BroadcastKnowledgeEventInput
> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}

/**
 * Hook to get recent knowledge events
 */
export function useRecentKnowledgeEvents(
  _since?: number
): QueryState<HolochainRecord<KnowledgeBridgeEvent>[]> {
  throw new Error(
    'React hooks require the @mycelix/react package. ' +
      'Install it with: npm install @mycelix/react'
  );
}
