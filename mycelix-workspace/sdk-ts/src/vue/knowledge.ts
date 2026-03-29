// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * @mycelix/sdk Vue 3 Composables for Knowledge Module
 *
 * Provides Vue 3 Composition API composables for the Knowledge hApp integration.
 *
 * @packageDocumentation
 * @module vue/knowledge
 */

import type { UseQueryReturn, UseMutationReturn, UseQueryOptions } from './index.js';
import type { HolochainRecord } from '../identity/index.js';

// ============================================================================
// Types
// ============================================================================

export interface Claim {
  id: string;
  author_did: string;
  title: string;
  content: string;
  empirical: number; // 0-1 scale
  normative: number; // 0-1 scale
  metaphysical: number; // 0-1 scale
  tags: string[];
  created_at: number;
  updated_at: number;
  endorsements: number;
}

export interface ClaimRelation {
  id: string;
  source_claim_id: string;
  target_claim_id: string;
  relation_type: 'supports' | 'contradicts' | 'extends' | 'cites' | 'supersedes';
  strength: number;
  created_by: string;
  created_at: number;
}

export interface Ontology {
  id: string;
  name: string;
  version: string;
  description: string;
  schema: Record<string, unknown>;
  author_did: string;
  created_at: number;
}

// ============================================================================
// Claim Composables
// ============================================================================

export interface UseClaimOptions extends UseQueryOptions {
  /** Include relations */
  includeRelations?: boolean;
  /** Include endorsement details */
  includeEndorsements?: boolean;
}

/**
 * Composable to submit a claim
 */
export function useSubmitClaim(): UseMutationReturn<
  HolochainRecord<Claim>,
  {
    title: string;
    content: string;
    empirical: number;
    normative: number;
    metaphysical?: number;
    tags?: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get a claim by ID
 */
export function useClaim(
  _claimId: string,
  _options?: UseClaimOptions
): UseQueryReturn<HolochainRecord<Claim> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get claims by author
 */
export function useClaimsByAuthor(
  _authorDid: string,
  _options?: UseClaimOptions
): UseQueryReturn<HolochainRecord<Claim>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get claims by tag
 */
export function useClaimsByTag(
  _tag: string,
  _options?: UseClaimOptions
): UseQueryReturn<HolochainRecord<Claim>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to search claims
 */
export function useSearchClaims(): UseMutationReturn<
  HolochainRecord<Claim>[],
  {
    query: string;
    filters?: {
      empirical_min?: number;
      empirical_max?: number;
      normative_min?: number;
      normative_max?: number;
      tags?: string[];
      author?: string;
    };
    limit?: number;
    offset?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to update a claim
 */
export function useUpdateClaim(): UseMutationReturn<
  HolochainRecord<Claim>,
  {
    claim_id: string;
    title?: string;
    content?: string;
    empirical?: number;
    normative?: number;
    metaphysical?: number;
    tags?: string[];
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to endorse a claim
 */
export function useEndorseClaim(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to remove claim endorsement
 */
export function useRemoveClaimEndorsement(): UseMutationReturn<HolochainRecord<unknown>, string> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Relation Composables
// ============================================================================

/**
 * Composable to create a claim relation
 */
export function useCreateRelation(): UseMutationReturn<
  HolochainRecord<ClaimRelation>,
  {
    source_claim_id: string;
    target_claim_id: string;
    relation_type: 'supports' | 'contradicts' | 'extends' | 'cites' | 'supersedes';
    strength: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get relations for a claim
 */
export function useClaimRelations(
  _claimId: string
): UseQueryReturn<HolochainRecord<ClaimRelation>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get claims that support a claim
 */
export function useSupportingClaims(_claimId: string): UseQueryReturn<HolochainRecord<Claim>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get claims that contradict a claim
 */
export function useContradictingClaims(_claimId: string): UseQueryReturn<HolochainRecord<Claim>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Ontology Composables
// ============================================================================

/**
 * Composable to register an ontology
 */
export function useRegisterOntology(): UseMutationReturn<
  HolochainRecord<Ontology>,
  {
    name: string;
    version: string;
    description: string;
    schema: Record<string, unknown>;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get an ontology
 */
export function useOntology(_ontologyId: string): UseQueryReturn<HolochainRecord<Ontology> | null> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to get ontologies by author
 */
export function useOntologiesByAuthor(
  _authorDid: string
): UseQueryReturn<HolochainRecord<Ontology>[]> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to query knowledge graph
 */
export function useQueryGraph(): UseMutationReturn<
  unknown[],
  {
    query: string; // SPARQL-like query
    ontology_id?: string;
    limit?: number;
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

// ============================================================================
// Inference Composables
// ============================================================================

/**
 * Composable to request inference on claims
 */
export function useRequestInference(): UseMutationReturn<
  {
    inferred_claims: Claim[];
    confidence: number;
    reasoning_path: string[];
  },
  {
    source_claims: string[];
    inference_type: 'deductive' | 'inductive' | 'abductive';
  }
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}

/**
 * Composable to verify claim consistency
 */
export function useVerifyConsistency(): UseMutationReturn<
  {
    consistent: boolean;
    conflicts: Array<{ claim_a: string; claim_b: string; reason: string }>;
  },
  string[]
> {
  throw new Error('Vue composables require Vue 3. This is a type stub.');
}
